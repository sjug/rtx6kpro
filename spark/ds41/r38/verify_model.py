"""Read-only HF-cache checks; payload hashing is explicit and can be expensive."""

import argparse
import hashlib
import json
from pathlib import Path

from contract import REPO_DIR, REVISION, ROOT, require, sha256


def verify(cache, full_hash=False):
    cache = Path(cache).resolve()
    snapshot = cache / "hub" / REPO_DIR / "snapshots" / REVISION
    manifest = json.loads((ROOT / "model-manifest.json").read_text())
    require(manifest["revision"] == REVISION, "Model manifest revision mismatch")
    for item in manifest["files"]:
        path = snapshot / item["path"]
        for part in (path, *path.parents):
            if part == cache:
                break
            require(not part.is_symlink() or not part.readlink().is_absolute(),
                    f"Absolute HF link cannot survive container remapping: {part}")
        require(path.resolve().is_relative_to(cache), f"Snapshot escapes mounted HF cache: {path}")
        require(path.is_file() and path.stat().st_size == item["size"],
                f"Missing, broken or truncated checkpoint file: {path}")
        if "git_blob" in item:
            data = path.read_bytes()
            digest = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            require(digest == item["git_blob"], f"Metadata identity mismatch: {path}")
        elif full_hash:
            require(sha256(path) == item["sha256"], f"Checkpoint SHA256 mismatch: {path}")
    config = json.loads((snapshot / "config.json").read_text())
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    require(config["architectures"] == ["DeepseekV41ForCausalLM"], "Wrong model architecture")
    require(config["model_type"] == "deepseek_v41", "Wrong model family")
    require(config["text_config"]["max_position_embeddings"] == 1048576, "Wrong context geometry")
    require(config["text_config"]["num_nextn_predict_layers"] == 3, "Missing embedded DSpark")
    require(index["metadata"]["total_size"] == 510286023000, "Wrong tensor payload size")
    expected = {x["path"] for x in manifest["files"] if x["path"].endswith(".safetensors")}
    require(len(expected) == 48 and set(index["weight_map"].values()) == expected,
            "Wrong shard inventory")
    return {"snapshot": str(snapshot), "revision": REVISION, "shards": len(expected),
            "payload_hashes_checked": full_hash, "metadata_identity_checked": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--hash", action="store_true", help="Read and SHA256 every checkpoint payload")
    args = parser.parse_args()
    print(json.dumps(verify(args.cache, args.hash), indent=2))
