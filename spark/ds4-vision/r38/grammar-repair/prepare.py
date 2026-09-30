#!/usr/bin/env python3
"""Reconstruct R38 and compose only the pinned grammar repair, without checkout."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[2] / "glm53/r38-spark"
REPO = Path("/home/jugs/git/vllm")
FIX = "8e1f1e587f8d24faf606f334a1c4bdaaa6bd4368"


def git(*args, data=None, env=None):
    return subprocess.check_output(["git", "-C", str(REPO), *args], input=data, env=env)


def main():
    lock = json.loads((BASE / "source.lock.json").read_text())
    patch = (ROOT / "upstream.patch").read_bytes()
    if hashlib.sha256(patch).hexdigest() != "e14a19f9441a0eeb305abe0bfe9fccc8d4520354973b2eea98bd365ae05d784c":
        raise RuntimeError("Unexpected upstream patch")
    if patch != git("show", "--format=", FIX):
        raise RuntimeError("Patch differs from pinned upstream commit")
    with tempfile.TemporaryDirectory(prefix="ds4-grammar-index-") as tmp:
        env = dict(os.environ, GIT_INDEX_FILE=f"{tmp}/index")
        git("read-tree", lock["vllm"]["upstream_tree"], env=env)
        for path in ("patches/spark-overlay.patch", "patches/vllm-pr756.patch"):
            git("apply", "--cached", "-", data=(BASE / path).read_bytes(), env=env)
        before = git("write-tree", env=env).decode().strip()
        if before != lock["vllm"]["tree"]:
            raise RuntimeError(f"R38 reconstruction mismatch: {before}")
        git("apply", "--cached", "-", data=patch, env=env)
        after = git("write-tree", env=env).decode().strip()
        paths = git("diff", "--name-only", before, after).decode().splitlines()
        if len(paths) != 7 or not all(p.endswith(".py") for p in paths):
            raise RuntimeError(f"Unexpected changed paths: {paths}")
        files = {}
        for path in paths:
            files[path] = {side: hashlib.sha256(git("show", f"{tree}:{path}")).hexdigest()
                           for side, tree in (("before", before), ("after", after))}
        result = {
            "base_image_id": "ea031e1d3d051033f077fc986bf6f8fce04cf9ab52483d5a719ba13114567fc5",
            "base_tree": before, "tree": after,
            "package_tree": git("rev-parse", f"{after}:vllm").decode().strip(),
            "fix": FIX, "patch_sha256": hashlib.sha256(patch).hexdigest(), "files": files,
        }
        (ROOT / "source.lock.json").write_text(json.dumps(result, indent=2) + "\n")
        # Retain composed objects against pruning; do not change any branch.
        git("update-ref", "refs/spark/ds4-r38-grammar/tree", after)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
