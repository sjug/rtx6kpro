#!/usr/bin/env python3
"""Compare the two DS4 dense block-FP8 launcher selections on SM121.

The ``b12x-a8`` launcher selects B12X for dense block-FP8 linear layers.
The ``b12x-a8-dglin`` launcher leaves the linear backend on ``auto`` and sets
``VLLM_USE_B12X_FP8_GEMM=0``. This benchmark constructs the cached DS4
``ModelConfig``, asks vLLM's real kernel-selection API for both
implementations, and invokes each selected kernel through ``apply_weights``.
It does not substitute a guessed low-level GEMM. The model config is required:
without it, DeepGEMM rejects the standalone context and auto falls through to
CUTLASS even though live DGLIN serving selects DeepGEMM.

Shapes are the TP2 rank-local dense projections in DeepSeek-V4-Flash-0731:

* fused_wqa_wkv: 4096 -> 1536, replicated
* wq_b: 1024 -> 16384, column parallel
* wo_b: 4096 -> 4096, row parallel
* shared_gate_up: 4096 -> 2048, merged column parallel
* shared_down: 1024 -> 4096, row parallel
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass

import torch

import vllm.envs as envs
from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    create_fp8_quant_key,
)


SHAPES = {
    "fused_wqa_wkv": (1536, 4096),
    "wq_b": (16384, 1024),
    "wo_b": (4096, 4096),
    "shared_gate_up": (2048, 4096),
    "shared_down": (4096, 1024),
}


@dataclass
class ArmResult:
    projection: str
    m: int
    backend: str
    kernel: str
    median_us: float
    minimum_us: float
    maximum_us: float
    output_finite: bool
    output_norm: float


class _BlockFp8Layer(torch.nn.Module):
    def __init__(
        self,
        *,
        prefix: str,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        linear_backend: str,
        model_config: ModelConfig,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
        self.weight_scale_inv = torch.nn.Parameter(
            weight_scale.clone(), requires_grad=False
        )
        self.weight_scale = None
        self.input_scale = None
        self.input_scale_ub = None
        self.weight_block_size = [128, 128]

        activation_key = create_fp8_quant_key(
            static=False,
            group_shape=GroupShape(1, 128),
        )
        weight_key = create_fp8_quant_key(
            static=True,
            group_shape=GroupShape(128, 128),
        )
        config = VllmConfig(model_config=model_config)
        config.kernel_config.linear_backend = linear_backend
        with set_current_vllm_config(config):
            self.kernel = init_fp8_linear_kernel(
                activation_quant_key=activation_key,
                weight_quant_key=weight_key,
                input_dtype=torch.bfloat16,
                out_dtype=torch.bfloat16,
                weight_shape=tuple(weight.shape),
                module_name=prefix,
            )
            self.kernel.process_weights_after_loading(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kernel.apply_weights(self, x)


def _make_weight(
    n: int,
    k: int,
    *,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    weight = (
        torch.randn((n, k), dtype=torch.float32, device=device, generator=generator)
        * 0.125
    ).to(torch.float8_e4m3fn)
    scale = torch.full(
        (n // 128, k // 128),
        0.0625,
        dtype=torch.float32,
        device=device,
    )
    return weight, scale


def _time_batches(
    fn,
    *,
    warmup: int,
    samples: int,
    iterations: int,
) -> tuple[list[float], torch.Tensor]:
    output = None
    for _ in range(warmup):
        output = fn()
    torch.cuda.synchronize()

    elapsed = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            output = fn()
        end.record()
        end.synchronize()
        elapsed.append(start.elapsed_time(end) * 1000.0 / iterations)
    assert output is not None
    return elapsed, output


def _result(
    projection: str,
    m: int,
    backend: str,
    layer: _BlockFp8Layer,
    elapsed: list[float],
    output: torch.Tensor,
) -> ArmResult:
    return ArmResult(
        projection=projection,
        m=m,
        backend=backend,
        kernel=type(layer.kernel).__name__,
        median_us=statistics.median(elapsed),
        minimum_us=min(elapsed),
        maximum_us=max(elapsed),
        output_finite=bool(torch.isfinite(output).all().item()),
        output_norm=float(output.float().norm().item()),
    )


def main() -> None:
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", nargs="+", type=int, default=[6, 12, 18, 24, 4096, 8192])
    parser.add_argument("--projection", choices=sorted(SHAPES), nargs="+")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash-0731")
    parser.add_argument(
        "--revision",
        default="9e165c30e2704aec5d9d593cce3eebd58bbef1cb",
    )
    args = parser.parse_args()

    if envs.VLLM_USE_B12X_FP8_GEMM:
        raise RuntimeError(
            "run with VLLM_USE_B12X_FP8_GEMM=0 so the auto arm matches DGLIN"
        )
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    projections = args.projection or list(SHAPES)
    model_config = ModelConfig(
        model=args.model,
        revision=args.revision,
        tokenizer_revision=args.revision,
        tokenizer_mode="deepseek_v4",
        trust_remote_code=True,
        dtype=torch.bfloat16,
        max_model_len=524288,
    )
    print(
        json.dumps(
            {
                "device": properties.name,
                "capability": list(torch.cuda.get_device_capability(device)),
                "sms": int(properties.multi_processor_count),
                "torch": torch.__version__,
                "vllm_use_b12x_fp8_gemm": envs.VLLM_USE_B12X_FP8_GEMM,
                "model": args.model,
                "revision": args.revision,
                "model_type": model_config.hf_text_config.model_type,
                "projections": projections,
                "m": args.m,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    all_results: list[ArmResult] = []
    for projection_index, projection in enumerate(projections):
        n, k = SHAPES[projection]
        weight, scale = _make_weight(
            n,
            k,
            device=device,
            seed=20260823 + projection_index,
        )
        layers = {
            "b12x": _BlockFp8Layer(
                prefix=f"bench.{projection}.b12x",
                weight=weight,
                weight_scale=scale,
                linear_backend="b12x",
                model_config=model_config,
            ),
            "auto": _BlockFp8Layer(
                prefix=f"bench.{projection}.auto",
                weight=weight,
                weight_scale=scale,
                linear_backend="auto",
                model_config=model_config,
            ),
        }
        print(
            json.dumps(
                {
                    "projection": projection,
                    "shape_nk": [n, k],
                    "selected_kernels": {
                        name: type(layer.kernel).__name__
                        for name, layer in layers.items()
                    },
                },
                sort_keys=True,
            ),
            flush=True,
        )
        auto_kernel = type(layers["auto"].kernel).__name__
        if auto_kernel != "DeepGemmFp8BlockScaledMMKernel":
            raise RuntimeError(
                "DGLIN discriminator did not select the serving kernel: "
                f"got {auto_kernel}, expected DeepGemmFp8BlockScaledMMKernel"
            )

        for m in args.m:
            generator = torch.Generator(device=device).manual_seed(
                20260823 + projection_index * 10000 + m
            )
            x = (
                torch.randn(
                    (m, k),
                    dtype=torch.float32,
                    device=device,
                    generator=generator,
                )
                * 0.125
            ).to(torch.bfloat16)
            outputs = {}
            timings = {"b12x": [], "auto": []}
            for backend in ("b12x", "auto", "auto", "b12x"):
                layer = layers[backend]
                elapsed, output = _time_batches(
                    lambda layer=layer: layer(x),
                    warmup=args.warmup,
                    samples=args.samples,
                    iterations=args.iterations,
                )
                outputs[backend] = output
                timings[backend].extend(elapsed)

            for backend in ("b12x", "auto"):
                layer = layers[backend]
                output = outputs[backend]
                result = _result(
                    projection,
                    m,
                    backend,
                    layer,
                    timings[backend],
                    output,
                )
                all_results.append(result)
                print(json.dumps(asdict(result), sort_keys=True), flush=True)
                if not result.output_finite:
                    raise RuntimeError(
                        f"non-finite output for {projection} M={m} {backend}"
                    )

            b12x = outputs["b12x"].float()
            auto = outputs["auto"].float()
            difference = (b12x - auto).abs()
            rel_l2 = float(difference.norm().item()) / max(
                float(auto.norm().item()), 1.0e-12
            )
            print(
                json.dumps(
                    {
                        "projection": projection,
                        "m": m,
                        "comparison": "b12x_vs_auto",
                        "max_abs": float(difference.max().item()),
                        "mean_abs": float(difference.mean().item()),
                        "relative_l2": rel_l2,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        del layers, weight, scale
        torch.cuda.empty_cache()

    print(
        json.dumps(
            {
                "elapsed_seconds": time.monotonic() - started,
                "results": [asdict(result) for result in all_results],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
