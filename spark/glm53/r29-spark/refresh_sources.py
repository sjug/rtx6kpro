#!/usr/bin/env python3
"""Build-time tracked-content refresh. Never replace untracked native products."""

if not __debug__:
    raise RuntimeError("R29 verification requires Python assertions enabled")

import hashlib
import json
import subprocess
from pathlib import Path

from contracts import FROZEN_TREES, validate_paths

ROOT = Path("/build/r29")
LOCK = json.loads((ROOT / "source.lock.json").read_text())


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args]).decode().strip()


def products(root):
    result = {}
    for file in (root / "vllm").rglob("*"):
        if not file.is_file() or "__pycache__" in file.parts or file.suffix == ".pyc":
            continue
        relative = file.relative_to(root).as_posix()
        if (
            file.name.endswith(".abi3.so")
            or file.name == "vllm-rs"
            or any(
                fragment in relative
                for fragment in (
                    "vllm_flash_attn/cute/",
                    "vllm_flash_attn/layers/",
                    "vllm_flash_attn/ops/",
                    "third_party/triton_kernels/",
                )
            )
        ):
            result[relative] = hashlib.sha256(file.read_bytes()).hexdigest()
    assert sum(k.endswith(".abi3.so") for k in result) == 10
    return result


vllm = Path("/opt/jovian-judgement/vllm")
before = products(vllm)
for name in ("vllm", "b12x"):
    root = Path("/opt/jovian-judgement") / name
    entry = LOCK[name]
    assert entry["tree"] == FROZEN_TREES[name]
    assert git(root, "write-tree") == entry["base_tree"]
    # Assert tracked working content, not just the index.
    subprocess.run(["git", "-C", str(root), "diff", "--quiet"], check=True)
    for key, sha in (
        ("patch", "patch_sha256"),
        ("changed_paths", "changed_paths_sha256"),
    ):
        assert (
            hashlib.sha256((ROOT / entry[key]).read_bytes()).hexdigest() == entry[sha]
        )
    validate_paths(name, (ROOT / entry["changed_paths"]).read_text().splitlines())
    if (ROOT / entry["patch"]).stat().st_size:
        subprocess.run(
            ["git", "-C", str(root), "apply", "--index", str(ROOT / entry["patch"])],
            check=True,
        )
    else:
        assert entry["base_tree"] == entry["tree"]
        assert entry["changed_paths_count"] == 0
    assert git(root, "write-tree") == entry["tree"]
    assert git(root, "rev-parse", f"{entry['tree']}:{name}") == entry["package_tree"]
    actual = git(
        root, "diff", "--name-only", "--no-renames", entry["base_tree"], entry["tree"]
    )
    assert actual == (ROOT / entry["changed_paths"]).read_text().strip()
assert before == products(vllm)
Path("/opt/local-inference").mkdir(exist_ok=True)
Path("/opt/local-inference/r29-reused-products.json").write_text(
    json.dumps(before, indent=2) + "\n"
)
print("R29 tracked source refresh and preserved products: PASS")
