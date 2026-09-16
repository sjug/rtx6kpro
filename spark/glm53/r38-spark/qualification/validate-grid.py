#!/usr/bin/env python3
"""Fail on incomplete or flagged standard grids before changing the serving arm."""
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

module_path = Path(__file__).with_name("compare-qwen-grids.py")
spec = importlib.util.spec_from_file_location("grid_compare", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
path = Path(sys.argv[1])
data, cells = module.load(path)
if set(data["metadata"]["concurrency_levels"]) != {1, 2, 4} or len(cells) != 15:
    raise ValueError("Expected the complete standard 15-cell grid")
if set(data["prefill"]) != {"8192", "16384", "32768", "65536", "131072"}:
    raise ValueError("Expected all five prefill scouts")
if not all(math.isfinite(row["tok_per_sec"]) and row["tok_per_sec"] > 0
           for row in data["prefill"].values()):
    raise ValueError("Invalid prefill timing")
print(json.dumps({"valid": True, "path": str(path), "cells": len(cells),
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))
