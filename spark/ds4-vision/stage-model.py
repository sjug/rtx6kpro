#!/usr/bin/env python3
"""Download the pinned public checkpoint and inventory it without changing caches."""
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

MODEL = "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"
REVISION = "6821d6ad3681a4b137b066b76094fa82ebd0a380"
receipts = Path("/receipts")
info = HfApi().model_info(MODEL, revision=REVISION, files_metadata=True)
if info.sha != REVISION:
    raise RuntimeError("Model revision mismatch")
files = [
    {"path": f.rfilename, "size": f.size,
     "sha256": f.lfs.sha256 if f.lfs else None}
    for f in info.siblings
]
manifest = {"model": MODEL, "revision": REVISION, "files": files}
(receipts / "model-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"revision": REVISION, "bytes": sum(f["size"] for f in files),
                  "files": len(files)}), flush=True)
snapshot = Path(snapshot_download(MODEL, revision=REVISION, max_workers=16))
for item in files:
    path = snapshot / item["path"]
    if not path.is_file() or path.stat().st_size != item["size"]:
        raise RuntimeError(f"Missing or wrong-size file: {path}")
    if item["sha256"]:
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != item["sha256"]:
            raise RuntimeError(f"SHA256 mismatch: {path}")
    print(f"VERIFIED {item['path']}", flush=True)
index = json.loads((snapshot / "model.safetensors.index.json").read_text())
shards = set(index["weight_map"].values())
if len(shards) != 48 or not all((snapshot / f).is_file() for f in shards):
    raise RuntimeError("Weight index completeness failed")
print(f"MODEL-STAGED revision={REVISION} shards={len(shards)} path={snapshot}", flush=True)
