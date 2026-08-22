#!/usr/bin/env python3
"""Time the exact GLM-5.2 TP4 mixed-Trellis MoE geometry.

This is a cross-generation diagnostic for the GG r34 and II r17/r18 B12X
APIs.  It deliberately uses the production launch capacities and routing
geometry while synthesizing deterministic K3/K4 weights, so it does not need
the checkpoint and cannot perturb a serving node.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from b12x.moe._shared.kernels.w4a16.host import max_packed_route_slots
from b12x.moe._shared.kernels.w4a16 import mixed_trellis as mixed_api
from b12x.moe._shared.kernels.w4a16.prepare import (
    prepare_trellis256_moe_weights,
)


HIDDEN = 6144
INTERMEDIATE = 512
TOPK = 8
TOTAL_EXPERTS = 256
TILE_CONFIG = (128, 128, 32, 512)
DECODE_CAPACITY = 32
PREFILL_CAPACITY = 3072
DECODE_BLOCK_M = 8
PREFILL_BLOCK_M = 32


@dataclass
class Result:
    m: int
    plan: str
    median_us: float
    minimum_us: float
    maximum_us: float
    output_norm: float
    output_finite: bool
    output_sha256: str
    reference_max_abs: float | None
    reference_mean_abs: float | None
    reference_rel_l2: float | None
    reference_exact_fraction: float | None
    reference_allclose: bool | None


def _prepare_tier(*, bits: int, experts: int, seed: int, device: torch.device) -> Any:
    # The BTX generation renamed the native layout without changing its wire
    # geometry.  Bound execution is the reliable cross-version discriminator.
    w13_layout = (
        "trellis_t256_proj"
        if hasattr(mixed_api, "bind_mixed_trellis")
        else "trellis3_t256_proj"
    )
    ones_h = torch.ones((1, HIDDEN), dtype=torch.float16, device=device)
    ones_i = torch.ones((experts, 3 * INTERMEDIATE), dtype=torch.float16, device=device)
    return prepare_trellis256_moe_weights(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_experts=experts,
        activation="silu",
        fc1_tile_n=TILE_CONFIG[1],
        fc2_tile_n=TILE_CONFIG[3],
        device=device,
        seed=seed,
        params_dtype=torch.float16,
        w13_layout=w13_layout,
        trellis_bits=bits,
        codebook="mcg",
        gate_suh=ones_h,
        up_suh=ones_h,
        intermediate_rotations=ones_i,
        down_svh=ones_h,
        tile_config=TILE_CONFIG,
    )


def _make_runner(
    *,
    capacity: int,
    block_m: int,
    tier0: Any,
    tier1: Any,
    global_map: torch.Tensor,
    descriptor: torch.Tensor,
    rotations: Any,
    device: torch.device,
) -> tuple[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor], Any]:
    props = torch.cuda.get_device_properties(device)
    route_slots = max_packed_route_slots(capacity * TOPK, block_m, TOTAL_EXPERTS)
    compile_kwargs = {
        "size_m": capacity,
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "tier0_num_experts": int(tier0.num_experts),
        "tier1_num_experts": int(tier1.num_experts),
        "tier0_bits": 3,
        "tier1_bits": 4,
        "top_k": TOPK,
        "max_m_blocks": (route_slots + block_m - 1) // block_m,
        "moe_block_size": block_m,
        "sms": int(props.multi_processor_count),
        "max_shared_mem": int(props.shared_memory_per_block_optin),
        "force_tile_config": TILE_CONFIG,
        "rotation_input_dtype": "bf16",
        "route_ids_dtype": torch.int32,
        "broadcast_suh": True,
        "broadcast_svh": True,
    }
    compile_parameters = inspect.signature(mixed_api.compile_mixed_trellis).parameters
    if "trellis_codebook" in compile_parameters:
        compile_kwargs["trellis_codebook"] = "mcg"
    if "route_num_experts" in compile_parameters:
        compile_kwargs["route_num_experts"] = TOTAL_EXPERTS
    launch = mixed_api.compile_mixed_trellis(**compile_kwargs)
    buffers = mixed_api.make_mixed_trellis_buffers(
        launch, device=device, sms=int(props.multi_processor_count)
    )
    if hasattr(mixed_api, "bind_mixed_trellis"):
        binding = mixed_api.bind_mixed_trellis(
            tier0,
            tier1,
            global_map,
            descriptor,
            rotations,
            launch,
        )

        def run(x, weights, ids):
            return mixed_api.run_bound_mixed_trellis(x, weights, ids, binding, buffers)

    else:

        def run(x, weights, ids):
            return mixed_api.run_mixed_trellis(
                x,
                tier0,
                tier1,
                weights,
                ids,
                global_map,
                descriptor,
                rotations,
                launch,
                buffers,
            )

    return run, launch


def _inputs(m: int, device: torch.device):
    cpu_generator = torch.Generator(device="cpu").manual_seed(20260819 + m)
    ids = torch.stack(
        [
            torch.randperm(TOTAL_EXPERTS, generator=cpu_generator)[:TOPK]
            for _ in range(m)
        ]
    ).to(dtype=torch.int32, device=device)
    cuda_generator = torch.Generator(device=device).manual_seed(20260819 + m)
    weights = torch.softmax(
        torch.randn(
            (m, TOPK),
            dtype=torch.float32,
            device=device,
            generator=cuda_generator,
        ),
        dim=-1,
    )
    x = (
        torch.randn(
            (m, HIDDEN),
            dtype=torch.float32,
            device=device,
            generator=cuda_generator,
        )
        * 1.0e-3
    ).to(torch.bfloat16)
    return x, weights, ids


def _time_calls(fn: Callable[[], torch.Tensor], warmup: int, repeats: int):
    for _ in range(warmup):
        output = fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        output = fn()
        end.record()
    torch.cuda.synchronize()
    elapsed = [
        start.elapsed_time(end) * 1000.0
        for start, end in zip(starts, ends, strict=True)
    ]
    finite = bool(torch.isfinite(output).all().item())
    norm = float(output.float().norm().item())
    return elapsed, finite, norm, output.detach().cpu().contiguous()


def _output_path(directory: Path, tier0_experts: int, tier1_experts: int, m: int):
    return directory / f"tier-{tier0_experts}-{tier1_experts}-m-{m}.pt"


def _compare_reference(
    output: torch.Tensor,
    reference: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> tuple[float, float, float, float]:
    if output.shape != reference.shape or output.dtype != reference.dtype:
        raise AssertionError(
            "mixed-Trellis output metadata differs from reference: "
            f"output={tuple(output.shape)}/{output.dtype}, "
            f"reference={tuple(reference.shape)}/{reference.dtype}"
        )
    if not bool(torch.isfinite(reference).all().item()):
        raise AssertionError("mixed-Trellis reference contains non-finite values")
    output_f32 = output.float()
    reference_f32 = reference.float()
    difference = (output_f32 - reference_f32).abs()
    max_abs = float(difference.max().item())
    mean_abs = float(difference.mean().item())
    reference_norm = float(reference_f32.norm().item())
    rel_l2 = float(difference.norm().item()) / max(reference_norm, 1.0e-12)
    exact_fraction = float((output == reference).float().mean().item())
    torch.testing.assert_close(
        output_f32,
        reference_f32,
        rtol=rtol,
        atol=atol,
        equal_nan=False,
    )
    return max_abs, mean_abs, rel_l2, exact_fraction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--tier0-experts", type=int, default=148)
    parser.add_argument("--tier1-experts", type=int, default=108)
    parser.add_argument(
        "--m", type=int, nargs="+", default=[1, 4, 16, 32, 33, 221, 222, 3072]
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--save-output-dir", type=Path)
    parser.add_argument("--reference-output-dir", type=Path)
    parser.add_argument("--reference-rtol", type=float, default=1.0e-2)
    parser.add_argument("--reference-atol", type=float, default=1.0e-2)
    args = parser.parse_args()
    if args.tier0_experts + args.tier1_experts != TOTAL_EXPERTS:
        raise ValueError("tier counts must sum to 256")
    if args.save_output_dir is not None:
        args.save_output_dir.mkdir(parents=True, exist_ok=True)
    if args.reference_output_dir is not None and not args.reference_output_dir.is_dir():
        raise FileNotFoundError(
            f"reference output directory not found: {args.reference_output_dir}"
        )

    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    started = time.monotonic()
    tier0 = _prepare_tier(bits=3, experts=args.tier0_experts, seed=301, device=device)
    tier1 = _prepare_tier(bits=4, experts=args.tier1_experts, seed=401, device=device)
    global_map, descriptor = mixed_api.build_tiered_maps(
        range(args.tier0_experts),
        range(args.tier0_experts, TOTAL_EXPERTS),
        device=device,
    )
    rotations = mixed_api.MixedTrellisRotations(
        intermediate=torch.cat(
            (tier0.intermediate_rotations, tier1.intermediate_rotations), dim=0
        ).contiguous(),
        gate_suh=torch.ones((1, HIDDEN), dtype=torch.float16, device=device),
        up_suh=torch.ones((1, HIDDEN), dtype=torch.float16, device=device),
        down_svh=torch.ones((1, HIDDEN), dtype=torch.float16, device=device),
    )
    decode, decode_launch = _make_runner(
        capacity=DECODE_CAPACITY,
        block_m=DECODE_BLOCK_M,
        tier0=tier0,
        tier1=tier1,
        global_map=global_map,
        descriptor=descriptor,
        rotations=rotations,
        device=device,
    )
    prefill, prefill_launch = _make_runner(
        capacity=PREFILL_CAPACITY,
        block_m=PREFILL_BLOCK_M,
        tier0=tier0,
        tier1=tier1,
        global_map=global_map,
        descriptor=descriptor,
        rotations=rotations,
        device=device,
    )
    print(
        json.dumps(
            {
                "label": args.label,
                "device": props.name,
                "capability": list(torch.cuda.get_device_capability(device)),
                "sms": int(props.multi_processor_count),
                "torch": torch.__version__,
                "b12x_api": (
                    "bound" if hasattr(mixed_api, "bind_mixed_trellis") else "legacy"
                ),
                "tier_counts": [args.tier0_experts, args.tier1_experts],
                "tile": TILE_CONFIG,
                "decode_registers": decode_launch.registers_per_thread,
                "prefill_registers": prefill_launch.registers_per_thread,
                "setup_seconds": time.monotonic() - started,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    results = []
    for m in args.m:
        if m <= 0 or m > PREFILL_CAPACITY:
            raise ValueError(f"M must be in [1, {PREFILL_CAPACITY}], got {m}")
        run = decode if m <= DECODE_CAPACITY else prefill
        plan = "decode" if m <= DECODE_CAPACITY else "prefill"
        x, weights, ids = _inputs(m, device)
        elapsed, finite, norm, output_cpu = _time_calls(
            lambda: run(x, weights, ids), args.warmup, args.repeats
        )
        output_sha256 = hashlib.sha256(
            output_cpu.view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        reference_max_abs = None
        reference_mean_abs = None
        reference_rel_l2 = None
        reference_exact_fraction = None
        reference_allclose = None
        output_path = (
            _output_path(
                args.save_output_dir, args.tier0_experts, args.tier1_experts, m
            )
            if args.save_output_dir is not None
            else None
        )
        if args.reference_output_dir is not None:
            reference_path = _output_path(
                args.reference_output_dir,
                args.tier0_experts,
                args.tier1_experts,
                m,
            )
            if not reference_path.is_file():
                raise FileNotFoundError(f"reference output not found: {reference_path}")
            reference = torch.load(
                reference_path, map_location="cpu", weights_only=True
            )
            (
                reference_max_abs,
                reference_mean_abs,
                reference_rel_l2,
                reference_exact_fraction,
            ) = _compare_reference(
                output_cpu,
                reference,
                rtol=args.reference_rtol,
                atol=args.reference_atol,
            )
            reference_allclose = True
        if output_path is not None:
            torch.save(output_cpu, output_path)
        result = Result(
            m=m,
            plan=plan,
            median_us=statistics.median(elapsed),
            minimum_us=min(elapsed),
            maximum_us=max(elapsed),
            output_norm=norm,
            output_finite=finite,
            output_sha256=output_sha256,
            reference_max_abs=reference_max_abs,
            reference_mean_abs=reference_mean_abs,
            reference_rel_l2=reference_rel_l2,
            reference_exact_fraction=reference_exact_fraction,
            reference_allclose=reference_allclose,
        )
        results.append(result)
        print(json.dumps(asdict(result), sort_keys=True), flush=True)
        if not finite:
            raise RuntimeError(f"non-finite mixed-Trellis output at M={m}")
    print(
        json.dumps(
            {"label": args.label, "results": [asdict(result) for result in results]},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
