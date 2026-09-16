#!/usr/bin/env python3
"""Check installed R37 dependency bytes, versions and coherent NCCL loading."""

import hashlib
import importlib.metadata as metadata
import json
import os
import subprocess
from pathlib import Path
from contracts import require_sm121a

if not __debug__:
    raise RuntimeError("R37 gates require assertions enabled")

import torch
import flashinfer
from flashinfer import _build_meta as fi_meta
from flashinfer_jit_cache import _build_meta as cache_meta

lock = json.loads(Path("/opt/glm53-flash/source.lock").read_text())
for entry in json.loads(Path("/build/r37/inputs/upstream/dependency-python-patches.json").read_text())["patches"]:
    path = Path("/") / entry["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["after_sha256"], path
assert metadata.version("torch") == "2.13.0"
assert metadata.version("flashinfer-python") == lock["flashinfer"]["version"]
assert metadata.version("flashinfer-jit-cache") == lock["flashinfer"]["version"]
assert fi_meta.__git_version__ == cache_meta.__git_version__ == lock["flashinfer"]["commit"]
assert torch.version.cuda == "13.3"
assert os.environ["LD_PRELOAD"] == os.environ["VLLM_NCCL_SO_PATH"]
expected = Path(os.environ["VLLM_NCCL_SO_PATH"]).resolve()
assert expected.name == "libnccl.so.2.31.2"
mapped = {Path(line.split()[-1]).resolve() for line in Path("/proc/self/maps").read_text().splitlines()
          if "/" in line and "libnccl.so" in line}
assert mapped == {expected}, mapped
assert torch.cuda.get_device_capability() == (12, 1)
cache = Path(metadata.distribution("flashinfer-jit-cache").locate_file("flashinfer_jit_cache"))
sparse = list(cache.rglob("sparse_mla_sm120.so"))
assert len(sparse) == 1, sparse
require_sm121a(subprocess.check_output(["cuobjdump", "--list-elf", str(sparse[0])], text=True))
print(f"R37 FlashInfer sparse MLA AOT: {hashlib.sha256(sparse[0].read_bytes()).hexdigest()}", flush=True)
print(f"R37 dependencies and coherent NCCL: PASS; FlashInfer={flashinfer.__file__}", flush=True)
