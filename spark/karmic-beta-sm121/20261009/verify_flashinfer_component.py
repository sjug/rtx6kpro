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
        "nvidia-cutlass-dsl": "4.7.1",
        "nvidia-cutlass-dsl-libs-cu13": "4.7.1",
        "nvidia-cutlass-dsl-libs-base": "4.7.1", "nvidia-cutlass-dsl-libs-core": "4.7.1",
        "nvidia-cutlass-dsl-libs-cu12": "4.7.1", "quack-kernels": "0.6.5",
        "torch-c-dlpack-ext": "0.1.5",
        "apache-tvm-ffi": "0.1.13.post3",
        "setuptools": "81.0.0", "wheel": "0.48.0", "packaging": "26.2",
        "ninja": "1.13.0", "numpy": "2.1.0", "tqdm": "4.70.0",
        "requests": "2.34.2", "nvidia-ml-py": "13.610.43",
    }.items():
        require(metadata.version(package) == version, f"Dependency mismatch: {package}")
    import cutlass
    require(cutlass.__version__ == "4.7.1", "Imported CUTLASS version mismatch")
    require(Path(cutlass.__file__).is_relative_to("/usr"), "Component compiler must come from NGC system Python")
    if args.foundation_only:
        print("FLASHINFER_FOUNDATION_METADATA_PASS")
        return
    import flashinfer
    import flashinfer_jit_cache
    commit = "b4c35e6ec1d712f6b832679c08ce5719fae57cd6"
    version = "0.7.1+lil.cu134.sm121.gb4c35e6ec1d7"
    require(flashinfer.__git_commit__ == commit, "FlashInfer commit mismatch")
    require(flashinfer_jit_cache.__git_version__ == commit, "JIT cache commit mismatch")
    require(flashinfer.__version__ == version, "FlashInfer version mismatch")
    require(flashinfer_jit_cache.__version__ == version, "JIT cache version mismatch")
    require(metadata.version("flashinfer-jit-cache-sm121a") == version, "JIT cache provider version mismatch")
    require(Path(flashinfer.__file__).is_relative_to("/opt/flashinfer-check"), "Wrong imported FlashInfer")
    providers = flashinfer_jit_cache.get_jit_cache_providers()
    require(len(providers) == 1 and providers[0].distribution == "flashinfer-jit-cache-sm121a"
            and providers[0].cuda_architectures == frozenset({"sm121a"}), f"Unexpected JIT cache providers: {providers}")
    require(Path(providers[0].jit_cache_dir).is_relative_to("/opt/flashinfer-check"), "Provider outside the checked install")
    wheels = list(Path("/artifacts/flashinfer").glob("*.whl"))
    require(len(wheels) == 3, "Expected exactly three FlashInfer wheels")
    # Python package: pure (py3-none-any); JIT-cache shim: tagged manylinux aarch64 by its backend;
    # provider: the native sm121a modules, linux_aarch64. Identify each by distribution name.
    (python_wheel,) = [p for p in wheels if p.name.startswith("flashinfer_python-")]
    (shim_wheel,) = [p for p in wheels if p.name.startswith("flashinfer_jit_cache-")]
    (provider_wheel,) = [p for p in wheels if p.name.startswith("flashinfer_jit_cache_sm121a-")]
    require(python_wheel.name.endswith("-py3-none-any.whl"), f"Unexpected Python wheel tag: {python_wheel.name}")
    require("aarch64" in provider_wheel.name and "aarch64" in shim_wheel.name, "ARM64 shim and provider wheels required")
    with zipfile.ZipFile(python_wheel) as archive:
        names = set(archive.namelist())
    # B12X ships inside FlashInfer; the kit's B12X root is refreshed from the same subtree.
    require("b12x/__init__.py" in names and "flashinfer/experimental/b12x/_api.py" in names, "FlashInfer wheel lacks B12X")
    print("FLASHINFER_COMPONENT_METADATA_PASS; GPU kernels still require runtime gates")


if __name__ == "__main__":
    main()
