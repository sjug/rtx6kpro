#!/usr/bin/env python3
"""Screen remaining prepared-W4A8 kernel policies at DS4 TP2 geometry.

This is a single-GPU, weight-free discriminator. It prepares one deterministic
E=256, K=4096, N=1024, top-k=6 MXFP4 expert package, then times CUDA-graph
replay for the stock policy and four supported policy alternatives. Every arm
uses identical inputs and is checked against the stock output before its timing
is accepted. A positive microbenchmark result is only permission for a serving
A/B; it is not deployment qualification.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch


E = 256
K = 4096
N = 1024
TOPK = 6
M_VALUES = (1, 3, 4, 5, 6, 24, 256, 1024)
POLICY_ENVS = (
    "B12X_DYNAMIC_WORK_SOURCE",
    "B12X_DYNAMIC_W4A8_MATERIALIZED",
    "B12X_DYNAMIC_W4A8_SHARE_INPUT",
    "B12X_W4A8_TINY_DECODE",
)
ARMS = (
    ("stock", {}),
    ("persistent_grid", {"B12X_DYNAMIC_WORK_SOURCE": "persistent_grid"}),
    ("no_materialized", {"B12X_DYNAMIC_W4A8_MATERIALIZED": "0"}),
    ("no_share_input", {"B12X_DYNAMIC_W4A8_SHARE_INPUT": "0"}),
    ("no_tiny", {"B12X_W4A8_TINY_DECODE": "0"}),
)


def _set_policy(values: dict[str, str]) -> None:
    for name in POLICY_ENVS:
        os.environ.pop(name, None)
    os.environ.update(values)


def _inputs(m: int, device: torch.device):
    generator = torch.Generator(device=device).manual_seed(20260822 + m)
    x = (torch.randn(m, K, generator=generator, device=device) * 2.0).to(torch.bfloat16)
    logits = torch.randn(m, E, generator=generator, device=device)
    topk_logits, topk_ids = torch.topk(logits, TOPK, dim=-1)
    topk_weights = torch.softmax(topk_logits, dim=-1).float()
    return x, topk_ids.to(torch.int32).contiguous(), topk_weights.contiguous()


def _iters(m: int) -> int:
    if m <= 24:
        return 100
    if m <= 256:
        return 30
    return 10


def _digest(tensor: torch.Tensor) -> str:
    raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _compare(
    output: torch.Tensor,
    reference: torch.Tensor,
    *,
    minimum_cosine: float = 0.999,
    norm_ratio_bounds: tuple[float, float] = (0.99, 1.01),
    require_same_dtype: bool = True,
) -> dict[str, float]:
    if output.shape != reference.shape or (
        require_same_dtype and output.dtype != reference.dtype
    ):
        raise AssertionError(
            "output metadata changed: "
            f"{tuple(output.shape)}/{output.dtype} versus "
            f"{tuple(reference.shape)}/{reference.dtype}"
        )
    if not bool(torch.isfinite(output).all().item()):
        raise AssertionError("candidate output contains a non-finite value")
    output_f32 = output.float()
    reference_f32 = reference.float()
    delta = (output_f32 - reference_f32).abs()
    cosine = float(
        torch.nn.functional.cosine_similarity(
            output_f32.flatten(), reference_f32.flatten(), dim=0
        ).item()
    )
    norm_ratio = float(output_f32.norm().item()) / max(
        float(reference_f32.norm().item()), 1.0e-12
    )
    if cosine < minimum_cosine or not (
        norm_ratio_bounds[0] <= norm_ratio <= norm_ratio_bounds[1]
    ):
        raise AssertionError(
            f"candidate output diverged: cosine={cosine}, norm_ratio={norm_ratio}"
        )
    return {
        "cosine": cosine,
        "norm_ratio": norm_ratio,
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
        "exact_fraction": float((output == reference).float().mean().item()),
    }


def _bench_one(*, experts, m: int, device: torch.device):
    from b12x.moe.fused_moe._impl import (
        allocate_tp_moe_workspace_pool,
        b12x_moe_fp4,
        build_tp_moe_fp4_binding,
        clear_tp_moe_caches,
    )

    clear_tp_moe_caches()
    x, topk_ids, topk_weights = _inputs(m, device)
    output = torch.empty(m, K, dtype=torch.bfloat16, device=device)
    binding = build_tp_moe_fp4_binding(
        scratch=allocate_tp_moe_workspace_pool(),
        a=x,
        experts=experts,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output=output,
        input_scales_static=True,
        quant_mode="w4a8_mx",
    )

    def launch() -> None:
        b12x_moe_fp4(binding=binding)

    for _ in range(3):
        launch()
    torch.cuda.synchronize(device)

    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(capture_stream):
        launch()
    capture_stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        launch()
    torch.cuda.synchronize(device)

    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize(device)

    iterations = _iters(m)
    samples_ms = []
    for _ in range(7):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        samples_ms.append(float(start.elapsed_time(end)) / iterations)

    graph.replay()
    torch.cuda.synchronize(device)
    result_output = output.detach().cpu().contiguous()
    result = {
        "m": m,
        "median_ms": statistics.median(samples_ms),
        "minimum_ms": min(samples_ms),
        "maximum_ms": max(samples_ms),
        "samples_ms": samples_ms,
        "iterations_per_sample": iterations,
        "output_sha256": _digest(result_output),
        "output_finite": bool(torch.isfinite(result_output).all().item()),
    }
    del graph, output, x, topk_ids, topk_weights
    gc.collect()
    torch.cuda.empty_cache()
    return result, result_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b12x-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--arms",
        default=",".join(label for label, _ in ARMS),
        help="comma-separated arm labels",
    )
    parser.add_argument(
        "--m-values",
        default=",".join(str(m) for m in M_VALUES),
        help="comma-separated positive token counts",
    )
    args = parser.parse_args()

    requested_arms = args.arms.split(",")
    policies = dict(ARMS)
    unknown_arms = sorted(set(requested_arms) - policies.keys())
    if unknown_arms:
        raise ValueError(f"unknown arms: {unknown_arms}")
    if not requested_arms or requested_arms[0] != "stock":
        raise ValueError("the first selected arm must be stock")
    selected_arms = [(label, policies[label]) for label in requested_arms]
    selected_m_values = tuple(int(value) for value in args.m_values.split(","))
    if not selected_m_values or any(m <= 0 for m in selected_m_values):
        raise ValueError("m-values must be positive")

    source_root = args.b12x_root.resolve()
    if not (source_root / "b12x" / "__init__.py").is_file():
        raise FileNotFoundError(f"not a B12X source root: {source_root}")
    sys.path.insert(0, str(source_root))

    from benchmarks.benchmark_ds4_moe import (
        _prepare_b12x_experts,
        make_synthetic_mxfp4_moe,
    )
    from b12x.moe._shared.kernels.reference import moe_reference_w4a8_mx

    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    if (props.major, props.minor) != (12, 1):
        raise RuntimeError(
            f"this screen is SM121-specific, got {(props.major, props.minor)}"
        )

    started = time.monotonic()
    _set_policy({})
    source_weights = make_synthetic_mxfp4_moe(E, K, N, seed=20260822, device=device)
    oracle_outputs: dict[int, torch.Tensor] = {}
    for m in selected_m_values:
        if m > 4:
            continue
        x, topk_ids, topk_weights = _inputs(m, device)
        oracle_outputs[m] = (
            moe_reference_w4a8_mx(
                x.float(),
                source_weights["w13_fp4"],
                source_weights["w13_mx"],
                None,
                source_weights["alphas"],
                source_weights["w2_fp4"],
                source_weights["w2_mx"],
                None,
                source_weights["alphas"],
                topk_ids,
                topk_weights,
                E,
                K,
                N,
                activation="silu",
            )
            .detach()
            .cpu()
            .contiguous()
        )
        del x, topk_ids, topk_weights
    experts = _prepare_b12x_experts("w4a8_mx", source_weights)
    del source_weights
    gc.collect()
    torch.cuda.empty_cache()

    document = {
        "schema": 1,
        "gpu": props.name,
        "compute_capability": [props.major, props.minor],
        "geometry": {"experts": E, "k": K, "n": N, "topk": TOPK},
        "m_values": list(selected_m_values),
        "arms": [],
    }
    references: dict[int, torch.Tensor] = {}
    for label, policy in selected_arms:
        print(f"arm={label} env={policy}", flush=True)
        _set_policy(policy)
        arm = {"label": label, "env": policy, "results": []}
        for m in selected_m_values:
            result, output = _bench_one(experts=experts, m=m, device=device)
            if label == "stock":
                references[m] = output
                result["comparison_to_stock"] = {
                    "cosine": 1.0,
                    "norm_ratio": 1.0,
                    "max_abs": 0.0,
                    "mean_abs": 0.0,
                    "exact_fraction": 1.0,
                }
            else:
                result["comparison_to_stock"] = _compare(
                    output,
                    references[m],
                    minimum_cosine=0.998 if label == "no_tiny" else 0.999,
                )
            if m in oracle_outputs:
                result["comparison_to_oracle"] = _compare(
                    output,
                    oracle_outputs[m],
                    minimum_cosine=0.998,
                    norm_ratio_bounds=(0.98, 1.02),
                    require_same_dtype=False,
                )
            arm["results"].append(result)
            print(
                f"  m={m:4d} median={result['median_ms'] * 1000.0:9.3f} us "
                f"sha={result['output_sha256'][:12]}",
                flush=True,
            )
            del output
        document["arms"].append(arm)
        document["elapsed_s"] = time.monotonic() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, indent=2) + "\n")

    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
