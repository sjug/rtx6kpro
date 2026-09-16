#!/usr/bin/env python3
"""Verify the pinned HF snapshot, optionally including payload SHA256."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--cache", type=Path, required=True)
p.add_argument("--manifest", type=Path, required=True)
p.add_argument("--hash", action="store_true")
a = p.parse_args()
manifest = json.loads(a.manifest.read_text())
revision = "6821d6ad3681a4b137b066b76094fa82ebd0a380"
if manifest["revision"] != revision:
    raise RuntimeError("Manifest revision mismatch")
snapshot = a.cache / "hub/models--deepseek-ai--DeepSeek-V4-Flash-Vision-Exp/snapshots" / revision
for item in manifest["files"]:
    path = snapshot / item["path"]
    if not path.is_file() or path.stat().st_size != item["size"]:
        raise RuntimeError(f"Missing or truncated: {path}")
    if a.hash and item["sha256"]:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != item["sha256"]:
            raise RuntimeError(f"Hash mismatch: {path}")
config = json.loads((snapshot / "config.json").read_text())
index = json.loads((snapshot / "model.safetensors.index.json").read_text())
if config["model_type"] != "deepseek_v4" or config.get("vision_n_layers", 0) <= 0:
    raise RuntimeError("Not the DeepSeek V4 Vision checkpoint")
if len(set(index["weight_map"].values())) != 48:
    raise RuntimeError("Wrong shard inventory")
print(json.dumps({"status": "MODEL-VERIFIED", "revision": revision,
                  "hashed": a.hash, "snapshot": str(snapshot)}))
