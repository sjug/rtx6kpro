#!/usr/bin/env python3
"""GPU admission of the pinned ARM64 foundation, not model qualification."""
import importlib.metadata
import json
import platform
import subprocess


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    import torch
    import triton
    import triton.language as tl

    require(platform.machine() == "aarch64", "ARM64 foundation required")
    require(torch.__version__.startswith("2.14.0a0+4fdf77b"), "Torch pin mismatch")
    require(torch.version.cuda == "13.4", "CUDA pin mismatch")
    require(torch.cuda.is_available(), "GPU required; no skipped gate")
    require(torch.cuda.get_device_capability() == (12, 1), "GB10 required")
    require(torch._C._GLIBCXX_USE_CXX11_ABI, "C++11 ABI required")

    @triton.jit
    def increment(x, y, N: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(y + i, tl.load(x + i, i < N, 0) + 1, i < N)

    x = torch.arange(1024, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    increment[(4,)](x, y, N=x.numel(), BLOCK=256)
    torch.cuda.synchronize()
    require(torch.equal(y, x + 1), "Triton JIT output mismatch")
    matrix = torch.eye(64, device="cuda", dtype=torch.bfloat16)
    require(torch.equal(matrix @ matrix, matrix), "BF16 GEMM mismatch")
    torch.cuda.synchronize()
    print(json.dumps({
        "gate": "FOUNDATION_GPU_PASS",
        "scope": "Torch, cuBLAS and Triton only; not Karmic native/model qualification",
        "torch": torch.__version__,
        "torch_distribution": importlib.metadata.version("torch"),
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "torch_path": torch.__file__,
        "gpu": torch.cuda.get_device_name(),
        "driver": subprocess.check_output([
            "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
        ], text=True).strip(),
        "nccl": torch.cuda.nccl.version(),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
