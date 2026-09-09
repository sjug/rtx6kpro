#!/usr/bin/env python3
"""Install rebuilt artifacts and metadata without changing frozen Torch."""

if not __debug__:
    raise RuntimeError("R28 verification requires Python assertions enabled")

import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

root = Path("/opt/jovian-judgement/vllm")
native = Path("/opt/local-inference/r28-native")
assert (
    hashlib.sha256((native / "_flashkda_C.abi3.so").read_bytes()).hexdigest()
    == os.environ["FLASHKDA_SHA256"]
)
for manifest in ("flashkda.sha256", "lmcache-native.sha256"):
    subprocess.run(["sha256sum", "-c", str(native / manifest)], cwd=native, check=True)
lock_file = Path("/opt/glm53-flash/source.lock")
assert (
    hashlib.sha256(lock_file.read_bytes()).hexdigest()
    == os.environ["SOURCE_LOCK_SHA256"]
)
lock = json.loads(lock_file.read_text())
before = json.loads(Path("/opt/local-inference/r28-reused-products.json").read_text())
torch_version = importlib.metadata.version("torch")


def run(*args, **kwargs):
    subprocess.run(args, check=True, **kwargs)


run(
    "uv",
    "pip",
    "install",
    "--python",
    sys.executable,
    "--no-deps",
    "--no-build-isolation",
    "--reinstall",
    str(root),
    env={
        **os.environ,
        "VLLM_TARGET_DEVICE": "empty",
        "VLLM_REQUIRE_RUST_FRONTEND": "1",
        "VLLM_VERSION_OVERRIDE": os.environ["VLLM_VERSION"],
    },
)
wheels = list(native.glob("lmcache-*.whl"))
assert len(wheels) == 1
run(
    "uv",
    "pip",
    "install",
    "--python",
    sys.executable,
    "--no-deps",
    "--reinstall",
    str(wheels[0]),
)
# Package data completeness follows the source-locked upstream recipe.
lmcache_package = Path(
    importlib.metadata.distribution("lmcache").locate_file("lmcache")
)
# Preserve tracked symlinks too. Dereferencing the bitmap-ops README changes
# Git mode 120000 to 100644 and fails the exact tracked-content gate.
source_package = Path("/opt/lmcache/source-r28/lmcache")
for source_path in source_package.rglob("*"):
    if source_path.is_symlink():
        installed_path = lmcache_package / source_path.relative_to(source_package)
        # Wheel installation materializes documentation links as regular files.
        # Replace only these source-declared paths, never arbitrary package data.
        if installed_path.is_symlink() or installed_path.is_file():
            installed_path.unlink()
shutil.copytree(
    source_package, lmcache_package, dirs_exist_ok=True, symlinks=True
)
run(
    "git",
    "--git-dir=/opt/lmcache/source-r28/.git",
    "diff",
    "--quiet",
    "HEAD",
    "--",
    "lmcache",
    env={**os.environ, "GIT_WORK_TREE": str(lmcache_package.parent)},
)
for relative, digest in before.items():
    assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest, (
        relative
    )
shutil.copy2(native / "_flashkda_C.abi3.so", root / "vllm/_flashkda_C.abi3.so")
shutil.copy2(
    native / "liblmcache_cumem_shareable.so",
    "/opt/lmcache/lib/liblmcache_cumem_shareable.so",
)
assert importlib.metadata.version("torch") == torch_version
assert importlib.metadata.version("vllm") == os.environ["VLLM_VERSION"]
assert importlib.metadata.version("lmcache") == lock["lmcache"]["version"]
for relative, digest in before.items():
    if relative != "vllm/_flashkda_C.abi3.so":
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest, (
            relative
        )
Path("/opt/local-inference/r28-native/flashkda-source.sha256").write_text(
    lock["flashkda"]["patch_sha256"] + "\n"
)
print("R28 runtime installation, preserved natives, and metadata: PASS")
