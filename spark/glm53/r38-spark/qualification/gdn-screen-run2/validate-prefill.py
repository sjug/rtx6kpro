#!/usr/bin/env python3
"""Validate this screen's immutable run_bench prefill receipts."""
import json
from pathlib import Path
import sys

record = json.loads(Path(sys.argv[1]).read_text())
prefill = record.get("prefill_summary", record.get("prefill", {}))
if set(prefill) != {"8192", "16384", "32768", "65536", "131072"}:
    raise SystemExit(f"Unexpected prefill cells: {list(prefill)}")
for context, row in prefill.items():
    server = row["server_validation"]
    if row["samples"] < 2 or row["tok_per_sec"] <= 0:
        raise SystemExit(f"Invalid client result at {context}: {row}")
    if server["cached_tokens"] != 0 or server["invalid_reason"] or server["samples"] < 2:
        raise SystemExit(f"Invalid server comparison at {context}: {server}")
    if server["tok_per_sec"] <= 0:
        raise SystemExit(f"Missing server rate at {context}")
print(json.dumps({"valid": True, "path": sys.argv[1], "prefill": prefill}, indent=2))
