#!/usr/bin/env python3
"""CPU-side native artifact checks, also used after runtime installation."""

if not __debug__:
    raise RuntimeError("R32 verification requires Python assertions enabled")

import argparse
import hashlib
import importlib
import subprocess
from pathlib import Path

from contracts import require_linkage, require_sm121a


def check_binary(path, cuda=False, *, allow_missing_driver=False):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"Missing native object: {path}")
    linkage = subprocess.check_output(["ldd", str(path)], text=True)
    require_linkage(linkage, allow_missing_driver=allow_missing_driver)
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
    parser.add_argument("mode", choices=["stable", "flashkda", "lmcache", "proxy"])
    parser.add_argument("path", nargs="?")
    args = parser.parse_args()
    if args.mode == "stable":
        # Build RUN has no CDI device mount. Only the host-provided driver may
        # be absent; the GPU-attached runtime gate calls check_binary strictly.
        check_binary(args.path, cuda=True, allow_missing_driver=True)
    elif args.mode == "flashkda":
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


if __name__ == "__main__":
    main()
