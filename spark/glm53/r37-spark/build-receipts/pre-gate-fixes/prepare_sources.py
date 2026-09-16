#!/usr/bin/env python3
"""Compose pinned R37 sources against the independently reconstructed R32 base."""

import hashlib
import json
import os
import subprocess
from pathlib import Path
from contracts import FROZEN_TREES

if not __debug__:
    raise RuntimeError("Build verification requires assertions enabled")

ROOT = Path(__file__).resolve().parent
REPOS = ROOT / ".compose/repos"
DOC_COMMIT = "7a36aecdf0627f4e1efbd878532d630b3489a494"
RECIPE_COMMIT = "9ae14bdf2f483656726f2fa0a83b9e03d2159422"
BASE_IMAGE = "74e53e710bef141f6f68e722582569f9c6aa388bce405ad6f1423566a2300c9c"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(repo, *args, data=None, env=None):
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], input=data, env=env
    )


def identity(repo, rev):
    return git(repo, "rev-parse", rev).decode().strip()


def write(path, data):
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return {"file": path, "sha256": sha(data)}


def main():
    raw = git(ROOT, "show", f"{DOC_COMMIT}:models/deepseek-v4.1-flash/r37/source.lock")
    upstream = dict(line.split("=", 1) for line in raw.decode().splitlines())
    previous_raw = (ROOT.parent / "r32-spark/source.lock.json").read_bytes()
    previous = json.loads(previous_raw)
    assert sha(previous_raw) == "c28c09ef0fabe37d5ebd46a35aceee875a996cd3d5dafd813976050ce3940207"
    overlay = (ROOT.parent / "r27-spark/patches/vllm/spark-overlay.patch").read_bytes()
    assert sha(overlay) == previous["vllm"]["spark_overlay_sha256"]
    lock = {
        "release": "jj-r37-spark-sm121",
        "upstream_document_commit": DOC_COMMIT,
        "upstream_recipe_commit": RECIPE_COMMIT,
        "upstream_lock": write("upstream-r37.source.lock", raw),
        "base_image_id": BASE_IMAGE,
        "base_source_lock_sha256": sha(previous_raw),
        "base_lock": write("inputs/r32-source.lock.json", previous_raw),
        "spark_overlay": write("patches/spark-overlay.patch", overlay),
        "runtime": {"torch": "2.13.0", "cuda": "13.3", "arch": "12.1a",
                    "lmcache_enabled": False, "checkpoint_policy": "aligned",
                    "loader": "instanttensor", "glm_draft_head": "bf16"},
        "flashkda": previous["flashkda"],
        "stable_native_sha256": previous["native_reuse"]["stable_sha256"],
    }
    for name in ("vllm", "b12x", "lmcache"):
        repo = REPOS / f"{name}.git"
        commit = upstream[f"{name}.commit"]
        tree = identity(repo, f"{commit}^{{tree}}")
        assert tree == upstream[f"{name}.tree"]
        assert identity(repo, f"{tree}:{name}") == upstream[f"{name}.package.tree"]
        env = {**os.environ, "GIT_INDEX_FILE": str(ROOT / f".compose/{name}.index")}
        base = previous[name]["tree"]
        if name == "vllm":
            for original, expected in ((previous[name]["upstream_tree"], base), (tree, None)):
                git(repo, "read-tree", original, env=env)
                git(repo, "apply", "--cached", "-", data=overlay, env=env)
                target = git(repo, "write-tree", env=env).decode().strip()
                if expected is not None:
                    assert target == expected
            for path, expected in previous[name]["native_inputs"].items():
                assert identity(repo, f"{target}:{path}") == expected, path
            native_changes = []
        else:
            target = tree
            assert identity(repo, f"{previous[name]['commit']}^{{tree}}") == base
            native_changes = git(repo, "diff", "--name-only", base, target, "--",
                                 "csrc", "rust", "setup_extensions", "CMakeLists.txt",
                                 "setup.py", "pyproject.toml", "requirements").decode().splitlines()
            if name == "lmcache":
                assert native_changes == ["csrc/storage_backends/fs/connector.cpp"]
            else:
                assert not native_changes
                assert not git(repo, "diff", base, target, "--", "b12x/comm/roce")
                native_changes = git(repo, "diff", "--name-only", base, target, "--",
                                     "b12x/**/*.c", "b12x/**/*.cpp", "b12x/**/*.h",
                                     "b12x/**/*.cu").decode().splitlines()
                assert native_changes == ["b12x/loader/_ple_reader.c", "b12x/loader/_storage.c"]
        assert target == FROZEN_TREES[name]
        patch = git(repo, "diff", "--binary", "--full-index", "--no-renames", base, target)
        paths = git(repo, "diff", "--name-only", "--no-renames", base, target)
        git(repo, "read-tree", base, env=env)
        if patch:
            git(repo, "apply", "--cached", "-", data=patch, env=env)
        assert git(repo, "write-tree", env=env).decode().strip() == target
        git(repo, "update-ref", "refs/r37/spark-tree", target)
        lock[name] = {
            "commit": commit, "upstream_tree": tree, "base_tree": base,
            "tree": target, "package_tree": identity(repo, f"{target}:{name}"),
            "patch": write(f"patches/{name}.patch", patch),
            "changed_paths": write(f"patches/{name}-changed-paths.txt", paths),
            "changed_paths_count": len(paths.splitlines()),
            "native_changed_paths": native_changes,
        }
        if name == "vllm":
            lock[name]["native_inputs"] = previous[name]["native_inputs"]
            lock[name]["version"] = "0.26.1rc0+jj.r37.spark.vllm" + lock[name]["package_tree"][:7]
        if name == "lmcache":
            lock[name]["version"] = upstream["lmcache.version"]
            lock[name]["native_policy"] = "rebuild CPU extensions; retain source-identical CUDA and interposer"
    recipe = REPOS / "recipe.git"
    assert identity(recipe, "r37") == RECIPE_COMMIT
    files = ["dependency-python-patches.json", "install_dependency_python_patches.py",
             "torch-schema-enumeration.patch", "torch-mutable-argument-metadata.patch",
             "cutlass-sentinel-identity.patch", "pack_lmcache_native_reuse.py",
             "tests/test_dependency_python_patches.py"]
    lock["upstream_inputs"] = {}
    for path in files:
        content = git(recipe, "show", f"{RECIPE_COMMIT}:recipes/glm53/{path}")
        if not path.startswith("tests/"):
            assert sha(content) == upstream[f"input.{path}.sha256"], path
        lock["upstream_inputs"][path] = write(f"inputs/upstream/{path}", content)
    fi = REPOS / "flashinfer.git"
    fi_commit = upstream["flashinfer.commit"]
    lock["flashinfer"] = {"commit": fi_commit, "tree": identity(fi, f"{fi_commit}^{{tree}}"),
                         "arch": "12.1a", "version": "0.6.18+cu133"}
    lock["flashinfer"]["regression_test"] = write(
        "tests/test_sparse_mla_sm120.py",
        git(fi, "show", f"{fi_commit}:tests/attention/test_sparse_mla_sm120.py"),
    )
    lock["launchers"] = {p.name: sha(p.read_bytes()) for p in sorted((ROOT / "launchers").glob("*.sh"))}
    lock["native_build_inputs"] = {
        name: sha((ROOT / name).read_bytes())
        for name in ("Dockerfile", "Dockerfile.flashinfer", "repack_lmcache.py")
    }
    lock["cache_fingerprint"] = "cu133-torch213-jj-r37-sm121-" + sha(json.dumps(lock, sort_keys=True).encode())[:20]
    (ROOT / "source.lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    print(json.dumps({n: lock[n]["tree"] for n in ("vllm", "b12x", "lmcache")}, indent=2))
    print("R37-SOURCE-REPLAY-PASS; vLLM native reuse proven; LMCache CPU rebuild required")


if __name__ == "__main__":
    main()
