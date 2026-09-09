#!/usr/bin/env python3
"""Strengthen both upstream graph replays without changing their numerics."""

if not __debug__:
    raise RuntimeError("R28 verification requires Python assertions enabled")

import hashlib
import sys
from pathlib import Path

path = Path(sys.argv[1])
original = path.read_text()
needle = '        saved.fill_(float("nan"))\n        graph.replay()'
replacement = (
    '        saved.fill_(float("nan"))\n'
    '        output.fill_(float("nan"))\n'
    '        final.fill_(float("nan"))\n'
    "        graph.replay()"
)
if original.count(needle) != 2:
    raise RuntimeError("Expected exactly two upstream checkpoint graph replay loops")
updated = original.replace(needle, replacement)
path.write_text(updated)
path.with_suffix(".provenance.txt").write_text(
    f"upstream_patched_sha256={hashlib.sha256(original.encode()).hexdigest()}\n"
    f"strengthened_sha256={hashlib.sha256(updated.encode()).hexdigest()}\n"
    "delta=poison output and final state before both graph replay loops\n"
)
