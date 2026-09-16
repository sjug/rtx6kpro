#!/usr/bin/env python3
"""Refresh tracked sources, proving native-reuse and patch identities first."""

import hashlib
import json
import subprocess
from pathlib import Path

from contracts import FROZEN_TREES
from native_reuse import native_manifest

ROOT = Path("/build/r37")
LOCK = json.loads((ROOT / "source.lock.json").read_text())


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def verify(entry):
    path = ROOT / entry["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], path
    return path


def products():
    root = Path("/opt/jovian-judgement/vllm/vllm")
    selected = {}
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix == ".pyc":
            continue
        relative = p.relative_to(root).as_posix()
        if (p.name.endswith(".so") or p.name == "vllm-rs"
                or relative.startswith(("vllm_flash_attn/cute/", "vllm_flash_attn/layers/",
                                        "vllm_flash_attn/ops/", "third_party/triton_kernels/"))):
            selected[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    assert sum(p.endswith(".abi3.so") for p in selected) == 10
    return selected


def main():
    verify(LOCK["base_lock"])
    verify(LOCK["upstream_lock"])
    verify(LOCK["spark_overlay"])
    for entry in LOCK["upstream_inputs"].values():
        verify(entry)
    base = Path("/opt/glm53-flash/source.lock")
    assert hashlib.sha256(base.read_bytes()).hexdigest() == LOCK["base_source_lock_sha256"]
    before = native_manifest()
    before_products = products()
    for name in FROZEN_TREES:
        entry = LOCK[name]
        root = (Path("/opt/lmcache/source-r37") if name == "lmcache"
                else Path("/opt/jovian-judgement") / name)
        assert git(root, "write-tree") == entry["base_tree"]
        subprocess.run(["git", "-C", str(root), "diff", "--quiet"], check=True)
        patch = verify(entry["patch"])
        paths = verify(entry["changed_paths"])
        assert len(paths.read_text().splitlines()) == entry["changed_paths_count"]
        subprocess.run(["git", "-C", str(root), "apply", "--index", str(patch)], check=True)
        assert git(root, "write-tree") == entry["tree"] == FROZEN_TREES[name]
        assert git(root, "rev-parse", f"{entry['tree']}:{name}") == entry["package_tree"]
        assert git(root, "diff", "--name-only", "--no-renames", entry["base_tree"], entry["tree"]) == paths.read_text().strip()
        if name == "vllm":
            for path, identity in entry["native_inputs"].items():
                assert git(root, "rev-parse", f"{entry['base_tree']}:{path}") == identity
                assert git(root, "rev-parse", f"{entry['tree']}:{path}") == identity
        if name == "b12x":
            allowed = verify(entry["non_python_allowlist"])
            assert [p for p in paths.read_text().splitlines() if not p.endswith(".py")] == allowed.read_text().splitlines()
        # Inherited bytecode for a changed module must not survive source refresh.
        for name_changed in paths.read_text().splitlines():
            file = root / name_changed
            if file.suffix == ".py":
                for cached in (file.parent / "__pycache__").glob(file.stem + ".*.pyc"):
                    cached.unlink()
    assert products() == before_products
    assert native_manifest() == before
    destination = Path("/opt/local-inference")
    (destination / "r37-base-native-manifest.json").write_text(json.dumps(before, indent=2) + "\n")
    (destination / "r37-reused-products.json").write_text(json.dumps(before_products, indent=2) + "\n")
    print("R37 tracked-source replay and retained build products: PASS", flush=True)


if __name__ == "__main__":
    main()
