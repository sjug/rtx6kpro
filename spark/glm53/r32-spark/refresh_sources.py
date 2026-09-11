#!/usr/bin/env python3
"""Build-time tracked-content refresh. Never replace untracked native products."""

if not __debug__:
    raise RuntimeError("R32 verification requires Python assertions enabled")

import hashlib
import json
import subprocess
from pathlib import Path

from contracts import FROZEN_TREES, validate_paths
from native_reuse import native_manifest

ROOT = Path("/build/r32")
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
all_native = native_manifest()
base_lock = Path("/opt/glm53-flash/source.lock")
assert hashlib.sha256(base_lock.read_bytes()).hexdigest() == LOCK["base_source_lock_sha256"]
for name in ("vllm", "b12x", "lmcache"):
    root = (Path("/opt/lmcache/source-r32") if name == "lmcache"
            else Path("/opt/jovian-judgement") / name)
    entry = LOCK[name]
    assert entry["tree"] == FROZEN_TREES[name]
    assert git(root, "write-tree") == entry["base_tree"]
    # Assert tracked working content, not just the index.
    subprocess.run(["git", "-C", str(root), "diff", "--quiet"], check=True)
    for path, identity in entry.get("native_inputs", {}).items():
        assert git(root, "rev-parse", f"{entry['base_tree']}:{path}") == identity
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
    for path, identity in entry.get("native_inputs", {}).items():
        assert git(root, "rev-parse", f"{entry['tree']}:{path}") == identity
    assert git(root, "rev-parse", f"{entry['tree']}:{name}") == entry["package_tree"]
    actual = git(
        root, "diff", "--name-only", "--no-renames", entry["base_tree"], entry["tree"]
    )
    assert actual == (ROOT / entry["changed_paths"]).read_text().strip()
assert before == products(vllm)
Path("/opt/local-inference").mkdir(exist_ok=True)
Path("/opt/local-inference/r32-reused-products.json").write_text(
    json.dumps(before, indent=2) + "\n"
)
Path("/opt/local-inference/r32-native-manifest.json").write_text(json.dumps(all_native, indent=2) + "\n")
print("R32 tracked source refresh and preserved products: PASS")
