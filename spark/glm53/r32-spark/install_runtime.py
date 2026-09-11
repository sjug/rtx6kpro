#!/usr/bin/env python3
"""Install Python metadata and sources while preserving every native artifact."""

if not __debug__:
    raise RuntimeError("R32 verification requires Python assertions enabled")

import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from native_reuse import lmcache_preinstall_inventory, native_manifest, verify_manifest

root = Path("/opt/jovian-judgement/vllm")
native = Path("/opt/local-inference/r32-native")
assert (
    hashlib.sha256((root / "vllm/_flashkda_C.abi3.so").read_bytes()).hexdigest()
    == os.environ["FLASHKDA_SHA256"]
)
assert (
    hashlib.sha256((root / "vllm/_C_stable_libtorch.abi3.so").read_bytes()).hexdigest()
    == os.environ["STABLE_NATIVE_SHA256"]
)
for manifest in ("lmcache-wheel.sha256",):
    subprocess.run(["sha256sum", "-c", str(native / manifest)], cwd=native, check=True)
lock_file = Path("/opt/glm53-flash/source.lock")
assert (
    hashlib.sha256(lock_file.read_bytes()).hexdigest()
    == os.environ["SOURCE_LOCK_SHA256"]
)
lock = json.loads(lock_file.read_text())
before = json.loads(Path("/opt/local-inference/r32-reused-products.json").read_text())
native_before = json.loads(Path("/opt/local-inference/r32-native-manifest.json").read_text())
torch_version = importlib.metadata.version("torch")
wheels = list(native.glob("lmcache-*-cp312-cp312-linux_aarch64.whl"))
assert len(wheels) == 1
# Complete the inventory and wheel coverage check BEFORE either reinstall.
# A future base with unhandled native payloads must stop with its files intact.
verify_manifest(native_before, native_manifest())
inventory = lmcache_preinstall_inventory(
    Path("/opt/lmcache/source-r32"), lock["lmcache"]["base_tree"],
    lock["lmcache"]["tree"], native_before, wheels[0],
)
Path("/opt/local-inference/r32-lmcache-preinstall-inventory.json").write_text(
    json.dumps(inventory, indent=2) + "\n"
)


def run(*args, **kwargs):
    subprocess.run(args, check=True, **kwargs)


run(
    "uv",
    "pip",
    "install",
    "--python",
    sys.executable,
    "--no-deps",
    "--offline",
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
run(
    "uv",
    "pip",
    "install",
    "--python",
    sys.executable,
    "--no-deps",
    "--offline",
    "--reinstall",
    str(wheels[0]),
)
# Copy only tracked package data. Old generated _version.py and build outputs
# in the R29 source tree must not replace the newly installed wheel metadata.
lmcache_package = Path(
    importlib.metadata.distribution("lmcache").locate_file("lmcache")
)
source_root = Path("/opt/lmcache/source-r32")
tracked_paths = subprocess.check_output(
    ["git", "-C", str(source_root), "ls-tree", "-rz", "--name-only",
     lock["lmcache"]["tree"], "--", "lmcache"],
).decode().split("\0")
for relative in filter(None, tracked_paths):
    source_path = source_root / relative
    installed_path = lmcache_package.parent / relative
    installed_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep Git mode 120000 for documentation links, not dereferenced data.
    if source_path.is_symlink():
        if installed_path.is_symlink() or installed_path.is_file():
            installed_path.unlink()
        installed_path.symlink_to(os.readlink(source_path))
    else:
        shutil.copy2(source_path, installed_path)
run(
    "git",
    "--git-dir=/opt/lmcache/source-r32/.git",
    "diff",
    "--quiet",
    lock["lmcache"]["tree"],
    "--",
    "lmcache",
    env={**os.environ, "GIT_WORK_TREE": str(lmcache_package.parent)},
)
for relative, digest in before.items():
    assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest, (
        relative
    )
verify_manifest(native_before, native_manifest())
assert importlib.metadata.version("torch") == torch_version
assert importlib.metadata.version("vllm") == os.environ["VLLM_VERSION"]
assert importlib.metadata.version("lmcache") == lock["lmcache"]["version"]
Path("/opt/local-inference/r32-native/flashkda-source.sha256").write_text(
    lock["flashkda"]["patch_sha256"] + "\n"
)
print("R32 runtime installation, preserved natives, and metadata: PASS")
