#!/usr/bin/env python3
"""Exercise GLM-5.2's DCP2 CKV-prefetch collective pattern.

This is intentionally weight-free. Two vLLM PYNCCL communicators use the
image-owned NCCL library: one issues the full-CKV all-gather on a side stream
while the other performs the sparse indexer's DCP candidate all-gather on the
main stream. This is the actual overlap when transient CKV gather is active;
that path does not perform the attention LSE all-gather/output reduce-scatter.
"""

from __future__ import annotations

import json
import math
import os
import time

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator


WORLD = 2
LAYERS = int(os.environ.get("BENCH_LAYERS", "75"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "5"))
MODE = os.environ.get("BENCH_MODE", "concurrent")
RECORD_BYTES = 656
BLOCK_SIZE = 64
TOPK_TOKENS = 2048

if MODE not in {"concurrent", "ckv", "indexer"}:
    raise ValueError(f"invalid BENCH_MODE {MODE!r}")


def _padded_local_tokens(global_tokens: int) -> int:
    local = math.ceil(global_tokens / WORLD)
    return math.ceil(local / BLOCK_SIZE) * BLOCK_SIZE


def _buffers(global_tokens: int, query_tokens: int, device: torch.device):
    local_tokens = _padded_local_tokens(global_tokens)
    ckv_in = torch.zeros(local_tokens * RECORD_BYTES, dtype=torch.uint8, device=device)
    ckv_out = torch.empty(WORLD * ckv_in.numel(), dtype=torch.uint8, device=device)

    # Each DCP rank contributes local top-k positions and FP32 score bits,
    # represented in production as two contiguous int32 planes. The result is
    # rank-major and then merged locally by the tiled top-k kernel.
    candidates_in = torch.zeros(
        (query_tokens, 2, TOPK_TOKENS), dtype=torch.int32, device=device
    )
    candidates_out = torch.empty(
        (WORLD * query_tokens, 2, TOPK_TOKENS), dtype=torch.int32, device=device
    )
    return local_tokens, ckv_in, ckv_out, candidates_in, candidates_out


def _run_case(
    *,
    global_tokens: int,
    query_tokens: int,
    main_comm: PyNcclCommunicator,
    prefetch_comm: PyNcclCommunicator,
    device: torch.device,
) -> dict[str, float | int]:
    (
        local_tokens,
        ckv_in,
        ckv_out,
        candidates_in,
        candidates_out,
    ) = _buffers(global_tokens, query_tokens, device)
    side = torch.cuda.Stream(device=device)
    current = torch.cuda.current_stream(device)

    def layer() -> None:
        ready = None
        if MODE in {"concurrent", "ckv"}:
            side.wait_stream(current)
            with torch.cuda.stream(side):
                prefetch_comm.all_gather(ckv_out, ckv_in, stream=side)
                ready = torch.cuda.Event(blocking=False)
                ready.record(side)
        if MODE in {"concurrent", "indexer"}:
            main_comm.all_gather(candidates_out, candidates_in, stream=current)
        if ready is not None:
            current.wait_event(ready)

    for _ in range(WARMUP):
        layer()
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(LAYERS):
        layer()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "global_tokens": global_tokens,
        "query_tokens": query_tokens,
        "local_ckv_tokens": local_tokens,
        "ckv_input_bytes": ckv_in.numel(),
        "indexer_candidate_input_bytes": (
            candidates_in.numel() * candidates_in.element_size()
        ),
        "layers": LAYERS,
        "warmup": WARMUP,
        "mode": MODE,
        "total_ms": elapsed * 1000.0,
        "per_layer_us": elapsed * 1.0e6 / LAYERS,
    }


def main() -> None:
    # PYNCCL uses a non-NCCL process group only to exchange NCCL unique IDs.
    # The data-plane collectives below go through the exact library selected
    # by VLLM_NCCL_SO_PATH, just as they do in the production worker.
    dist.init_process_group("gloo")
    if dist.get_world_size() != WORLD:
        raise RuntimeError(f"expected world size {WORLD}")
    rank = dist.get_rank()
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)

    # Creation order must be identical on both ranks. These correspond to
    # vLLM's main DCP and dedicated dcp_ckv_prefetch communicators.
    main_group = dist.new_group(ranks=list(range(WORLD)), backend="gloo")
    prefetch_group = dist.new_group(ranks=list(range(WORLD)), backend="gloo")
    library_path = os.environ.get("VLLM_NCCL_SO_PATH")
    if not library_path or not os.path.isfile(library_path):
        raise RuntimeError(
            "VLLM_NCCL_SO_PATH must name the image-owned NCCL library, got "
            f"{library_path!r}"
        )
    main_comm = PyNcclCommunicator(main_group, device, library_path=library_path)
    prefetch_comm = PyNcclCommunicator(
        prefetch_group, device, library_path=library_path
    )
    if not main_comm.available or not prefetch_comm.available:
        raise RuntimeError("PYNCCL communicator initialization failed")
    cases = [
        (8192, 3072),
        (32768, 3072),
        (64256, 222),
    ]
    case_index = os.environ.get("BENCH_CASE_INDEX")
    if case_index is not None:
        cases = [cases[int(case_index)]]
    if rank == 0:
        print(
            json.dumps(
                {
                    "event": "start",
                    "cases": cases,
                    "layers": LAYERS,
                    "warmup": WARMUP,
                    "mode": MODE,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    results = [
        _run_case(
            global_tokens=context,
            query_tokens=query,
            main_comm=main_comm,
            prefetch_comm=prefetch_comm,
            device=device,
        )
        for context, query in cases
    ]
    if rank == 0:
        print(
            json.dumps(
                {
                    "torch": torch.__version__,
                    "nccl_library": library_path,
                    "pynccl_raw_version": main_comm.nccl_version,
                    "pynccl_version": main_comm.nccl.ncclGetVersion(),
                    "channels": {
                        "min": os.environ.get("NCCL_MIN_NCHANNELS"),
                        "max": os.environ.get("NCCL_MAX_NCHANNELS"),
                    },
                    "results": results,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.barrier()
    prefetch_comm.destroy()
    main_comm.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
