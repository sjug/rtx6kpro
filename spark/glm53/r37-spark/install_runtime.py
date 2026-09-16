#!/usr/bin/env python3
"""Install R37 Python and rebuilt CPU payloads with a before/after inventory."""

import hashlib
import importlib.metadata as metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from contracts import FROZEN_TREES
from native_reuse import lmcache_preinstall_inventory, native_manifest, verify_manifest


def run(*args, **kwargs):
    subprocess.run(args, check=True, **kwargs)


def main():
    root = Path("/opt/jovian-judgement/vllm")
    receipt = Path("/opt/local-inference/r37-native")
    lock_path = Path("/opt/glm53-flash/source.lock")
    assert hashlib.sha256(lock_path.read_bytes()).hexdigest() == os.environ["SOURCE_LOCK_SHA256"]
    lock = json.loads(lock_path.read_text())
    before = json.loads(Path("/opt/local-inference/r37-base-native-manifest.json").read_text())
    verify_manifest(before, native_manifest())
    for name, expected in (("_flashkda_C", lock["flashkda"]["spark_extension_sha256"]),
                           ("_C_stable_libtorch", lock["stable_native_sha256"])):
        assert hashlib.sha256((root / f"vllm/{name}.abi3.so").read_bytes()).hexdigest() == expected
    for directory in (receipt / "lmcache", receipt / "flashinfer"):
        run("sha256sum", "-c", "SHA256SUMS", cwd=directory)
    wheels = list((receipt / "lmcache").glob("lmcache-*-cp312-cp312-linux_aarch64.whl"))
    assert len(wheels) == 1
    source = Path("/opt/lmcache/source-r37")
    inventory = lmcache_preinstall_inventory(
        source, lock["lmcache"]["base_tree"], lock["lmcache"]["tree"], before,
        wheels[0], rebuilt_cpu=True,
    )
    Path("/opt/local-inference/r37-lmcache-preinstall-inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    torch_before = metadata.version("torch")
    run("uv", "pip", "install", "--python", sys.executable, "--no-deps", "--offline",
        "--no-build-isolation", "--reinstall", str(root),
        env={**os.environ, "VLLM_TARGET_DEVICE": "empty", "VLLM_REQUIRE_RUST_FRONTEND": "1",
             "VLLM_VERSION_OVERRIDE": lock["vllm"]["version"]})
    run("uv", "pip", "install", "--python", sys.executable, "--no-deps", "--offline", "--reinstall", str(wheels[0]))
    fi = list((receipt / "flashinfer").glob("*.whl"))
    assert len(fi) == 2
    run("uv", "pip", "install", "--python", sys.executable, "--no-deps", "--offline", "--reinstall", *map(str, fi))
    package = Path(metadata.distribution("lmcache").locate_file("lmcache"))
    paths = subprocess.check_output(["git", "-C", str(source), "ls-tree", "-rz", "--name-only", lock["lmcache"]["tree"], "--", "lmcache"]).decode().split("\0")
    for relative in filter(None, paths):
        src, dst = source / relative, package.parent / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_symlink():
            if dst.is_file() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(os.readlink(src))
        else:
            shutil.copy2(src, dst)
    run("git", f"--git-dir={source}/.git", "diff", "--quiet", lock["lmcache"]["tree"], "--", "lmcache",
        env={**os.environ, "GIT_WORK_TREE": str(package.parent)})
    expected = {**before, **inventory["rebuilt_native_sha256"]}
    verify_manifest(expected, native_manifest())
    Path("/opt/local-inference/r37-native-manifest.json").write_text(json.dumps(expected, indent=2) + "\n")
    for path, expected_digest in json.loads(Path("/opt/local-inference/r37-reused-products.json").read_text()).items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected_digest, path
    assert metadata.version("torch") == torch_before == "2.13.0"
    assert metadata.version("vllm") == lock["vllm"]["version"]
    assert metadata.version("lmcache") == lock["lmcache"]["version"]
    assert metadata.version("flashinfer-python") == lock["flashinfer"]["version"]
    for name in FROZEN_TREES:
        assert lock[name]["tree"] == FROZEN_TREES[name]
    print("R37 runtime install and classified native replacement: PASS", flush=True)


if __name__ == "__main__":
    main()
