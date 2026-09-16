#!/usr/bin/env python3
"""Adapt the pinned upstream CUDA-reuse packer to ARM and tracked-index trees."""

import importlib.util
import subprocess
from pathlib import Path

from contracts import FROZEN_TREES  # Also rejects optimized Python.

UPSTREAM = Path("/build/r37/inputs/upstream/pack_lmcache_native_reuse.py")


def native_identity(root, mode="reuse-all"):
    tree = subprocess.check_output(["git", "-C", str(root), "write-tree"], text=True).strip()
    assert tree in (FROZEN_TREES["lmcache"], "dc1dad48e97cb849fcccf4be289e86bf8449988d")
    result = {}
    for path in module.NATIVE_INPUTS:
        if mode == "cpu-rebuild-cuda-reuse" and path == "csrc":
            continue
        result[path] = subprocess.check_output(["git", "-C", str(root), "rev-parse", f"{tree}:{path}"], text=True).strip()
    if mode == "cpu-rebuild-cuda-reuse":
        sources = subprocess.check_output(["git", "-C", str(root), "ls-tree", "-r", tree, "csrc"], text=True)
        for line in sources.splitlines():
            metadata, path = line.split("\t", 1)
            if not path.startswith(module.CPU_SOURCE_DIRS):
                result[path] = metadata
    return result


# Architecture and tracked-tree addressing are the only adaptations. Preserve
# the upstream CPU-module set, CUDA-only selection, wheel RECORD and hash checks.
code = UPSTREAM.read_text().replace("linux_x86_64", "linux_aarch64").replace("Linux x86_64", "Linux aarch64")
spec = importlib.util.spec_from_loader("r37_lmcache_reuse", loader=None)
module = importlib.util.module_from_spec(spec)
exec(compile(code, str(UPSTREAM), "exec"), module.__dict__)
module.native_identity = native_identity
module.main()
