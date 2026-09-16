#!/usr/bin/env python3
"""Verify the FlashInfer component's actual input receipt before reuse."""

import hashlib
import json
import sys
from pathlib import Path

if not __debug__:
    raise RuntimeError("Component verification requires assertions enabled")


def verify(directory, lock):
    directory = Path(directory)
    inputs = dict(line.split("=", 1) for line in (directory / "inputs.txt").read_text().splitlines())
    expected = {"base": lock["base_image_id"], "arch": lock["flashinfer"]["arch"],
                "commit": lock["flashinfer"]["commit"]}
    if inputs != expected:
        raise RuntimeError(f"FlashInfer component inputs differ: {inputs} != {expected}")
    recorded = dict(reversed(line.split("  ", 1)) for line in (directory / "recipe.sha256").read_text().splitlines())
    files = ("Dockerfile.flashinfer", "build-flashinfer.sh")
    if set(recorded) != set(files):
        raise RuntimeError(f"Unexpected component recipe manifest: {sorted(recorded)}")
    for name in files:
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual != recorded[name] or actual != lock["native_build_inputs"][name]:
            raise RuntimeError(f"FlashInfer {name} differs from the component receipt or current lock")


if __name__ == "__main__":
    verify(sys.argv[1], json.loads(Path(sys.argv[2]).read_text()))
    print("FlashInfer component recipe and inputs: PASS", flush=True)
