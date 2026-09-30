#!/usr/bin/env python3
import argparse
import importlib.metadata as metadata
from pathlib import Path
import platform
import zipfile


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--foundation-only", action="store_true")
    args = parser.parse_args()
    import torch
    require(platform.machine() == "aarch64", "ARM64 required")
    require(torch.__version__ == "2.14.0a0+4fdf77b940.nv26.08", "Torch runtime mismatch")
    require(torch.version.cuda == "13.4", "CUDA mismatch")
    require(torch._C._GLIBCXX_USE_CXX11_ABI, "C++ ABI mismatch")
    for package, version in {
        "torch": "2.14.0a0+4fdf77b940.nv26.8.63802676",
        "nvidia-cutlass-dsl": "4.6.2",
        "nvidia-cutlass-dsl-libs-cu13": "4.6.2",
        "apache-tvm-ffi": "0.1.13.post3",
        "setuptools": "81.0.0", "wheel": "0.48.0", "packaging": "26.2",
        "ninja": "1.13.0", "numpy": "2.1.0", "tqdm": "4.70.0",
        "requests": "2.34.2", "nvidia-ml-py": "13.610.43",
    }.items():
        require(metadata.version(package) == version, f"Dependency mismatch: {package}")
    if args.foundation_only:
        print("FLASHINFER_FOUNDATION_METADATA_PASS")
        return
    import flashinfer
    import flashinfer_jit_cache
    commit = "2206a14e46387a56c093860a46bbbdd00596b75b"
    version = "0.6.18+lil.cu134.sm121.g2206a14e4638"
    require(flashinfer.__git_commit__ == commit, "FlashInfer commit mismatch")
    require(flashinfer_jit_cache.__git_version__ == commit, "JIT cache commit mismatch")
    require(flashinfer.__version__ == version, "FlashInfer version mismatch")
    require(flashinfer_jit_cache.__version__ == version, "JIT cache version mismatch")
    require(Path(flashinfer.__file__).is_relative_to("/opt/flashinfer-check"), "Wrong imported FlashInfer")
    wheels = list(Path("/artifacts/flashinfer").glob("*.whl"))
    require(len(wheels) == 2, "Expected exactly two FlashInfer wheels")
    require(sum("aarch64.whl" in p.name for p in wheels) == 1, "ARM64 JIT wheel required")
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            require(not any(n.startswith(("b12x/", "flashinfer/b12x/")) for n in archive.namelist()),
                    "FlashInfer wheel must not ship B12X")
    print("FLASHINFER_COMPONENT_METADATA_PASS; GPU kernels still require runtime gates")


if __name__ == "__main__":
    main()
