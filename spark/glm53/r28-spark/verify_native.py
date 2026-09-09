#!/usr/bin/env python3
"""CPU-side native artifact checks, also used after runtime installation."""

if not __debug__:
    raise RuntimeError("R28 verification requires Python assertions enabled")

import argparse
import hashlib
import importlib
import json
import subprocess
from pathlib import Path

from contracts import require_sm121a


def check_binary(path, cuda=False):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"Missing native object: {path}")
    linkage = subprocess.check_output(["ldd", str(path)], text=True)
    if "not found" in linkage:
        raise RuntimeError(linkage)
    if cuda:
        cubins = subprocess.check_output(
            ["cuobjdump", "--list-elf", str(path)], text=True
        )
        require_sm121a(cubins)
        print(cubins, flush=True)
    print(
        f"NATIVE-OK {hashlib.sha256(path.read_bytes()).hexdigest()} {path}", flush=True
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["flashkda", "lmcache", "proxy", "bundle"])
    parser.add_argument("path", nargs="?")
    args = parser.parse_args()
    if args.mode == "flashkda":
        check_binary(args.path, cuda=True)
    elif args.mode == "lmcache":
        import torch  # noqa: F401  Load Torch's shared libraries before its extensions.

        for name in ("cuda_ops", "lmcache_native", "lmcache_fs", "lmcache_redis"):
            module = importlib.import_module(f"lmcache.{name}")
            check_binary(module.__file__, cuda=name == "cuda_ops")
        check_binary(args.path)
    elif args.mode == "proxy":
        from b12x.comm.roce import _proxy

        if _proxy.load().roce_abi_version() != 3:
            raise RuntimeError("RoCEnante proxy must implement ABI 3")
    else:
        lock = json.loads(Path("/build/r28/source.lock.json").read_text())
        if (
            hashlib.sha256(Path(args.path).read_bytes()).hexdigest()
            != lock["lmcache"]["bundle_sha256"]
        ):
            raise RuntimeError("LMCache source bundle digest mismatch")


if __name__ == "__main__":
    main()
