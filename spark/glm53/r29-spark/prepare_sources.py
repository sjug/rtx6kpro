#!/usr/bin/env python3
"""Replay R29 from frozen source inputs using private indices and persistent refs."""

if not __debug__:
    raise RuntimeError("R29 verification requires Python assertions enabled")

import hashlib
import json
import os
import subprocess
from pathlib import Path

from contracts import FROZEN_TREES, NATIVE_VLLM, cache_fingerprint, validate_paths

ROOT = Path(__file__).resolve().parent
REPOS = Path("/home/jugs/git")
UPSTREAM_SHA = "3307f3372213496e5b7de4fc485ef5b8f7fc43ff99df896ac40fa68d4dd3f80c"
BASE_IMAGE = "cd93d80b3f9547f70e1e4608cd42af7c4b0f2ff33d92d5e520913cc9bab2d8d1"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(repo, *args, env=None, data=None):
    return subprocess.check_output(["git", "-C", str(repo), *args], env=env, input=data)


def main():
    raw = git(ROOT, "show", "0df5fbc8ef198ace5fff4a277fc0f4876316558a:models/glm-5.3-flash/validation/shared-serving-r29.source.lock")
    assert sha(raw) == UPSTREAM_SHA
    (ROOT / "upstream-r29.source.lock").write_bytes(raw)
    upstream = dict(line.split("=", 1) for line in raw.decode().splitlines())
    previous = json.loads((ROOT.parent / "r28-spark/source.lock.json").read_text())
    scratch = ROOT / ".compose"
    scratch.mkdir(exist_ok=True)
    result = {
        "release": "jj-r29-spark-sm121",
        "status": "authored, not built or qualified",
        "upstream_lock_sha256": UPSTREAM_SHA,
        "base_image_id": BASE_IMAGE,
        "runtime": {
            "checkpoint_policy": "aligned", "lmcache_enabled": False,
            "glm_mtp_head": "bf16", "loader": "instanttensor",
            "cuda_arch": "12.1a", "nccl": "retain R28 image and runner contract",
        },
        "flashinfer": {
            "retained_spark_commit": "1ac6942776b383c6b03c7a5805a22e72a3e3349f",
            "upstream_commit": upstream["flashinfer.commit"],
            "reason": "reuse qualified SM121 build; upstream artifact is not an ARM wheel",
        },
    }
    overlay = (ROOT.parent / "r27-spark/patches/vllm/spark-overlay.patch").read_bytes()
    assert sha(overlay) == previous["vllm"]["spark_overlay_sha256"]
    replacements = {}
    for name, dirname in (("vllm", "vllm"), ("b12x", "b12x"), ("lmcache", "LMCache")):
        repo = REPOS / dirname
        commit = upstream[f"{name}.commit"]
        tree = git(repo, "rev-parse", f"{commit}^{{tree}}").decode().strip()
        assert tree == upstream[f"{name}.tree"]
        assert git(repo, "rev-parse", f"{tree}:{name}").decode().strip() == upstream[f"{name}.package.tree"]
        base = previous[name]["tree"]
        target = tree
        entry = {"commit": commit, "upstream_tree": tree,
                 "upstream_package_tree": upstream[f"{name}.package.tree"]}
        env = {**os.environ, "GIT_INDEX_FILE": str(scratch / f"{name}.index")}
        if name == "vllm":
            for original, expected in ((previous[name]["upstream_tree"], base), (tree, None)):
                git(repo, "read-tree", original, env=env)
                git(repo, "apply", "--cached", "-", env=env, data=overlay)
                target = git(repo, "write-tree", env=env).decode().strip()
                if expected:
                    assert target == expected
            entry["spark_overlay_sha256"] = sha(overlay)
            native = git(repo, "diff", "--name-only", base, target, "--", "csrc", "cmake", "CMakeLists.txt", "setup.py", "requirements").decode().splitlines()
            assert set(native) == NATIVE_VLLM, native
            entry["native_changed_paths"] = native
            entry["rebuild_native"] = ["_C_stable_libtorch"]
        elif name == "b12x":
            assert base == target, "R29 is expected to retain R28 B12X exactly"
            entry["rebuild_native"] = []
        else:
            entry["version"] = upstream["lmcache.version"]
            entry["rebuild_native"] = ["LMCache wheel and interposer from unchanged native source"]
        assert target == FROZEN_TREES[name], (name, target)
        paths = git(repo, "diff", "--name-only", "--no-renames", base, target)
        validate_paths(name, paths.decode().splitlines())
        patch = git(repo, "diff", "--binary", "--full-index", "--no-renames", base, target)
        directory = ROOT / "patches" / name
        directory.mkdir(parents=True, exist_ok=True)
        patch_file, paths_file = directory / "r28-to-r29.patch", directory / "changed-paths.txt"
        patch_file.write_bytes(patch)
        paths_file.write_bytes(paths)
        git(repo, "read-tree", base, env=env)
        if patch:
            git(repo, "apply", "--cached", "-", env=env, data=patch)
        else:
            assert base == target and not paths
        assert git(repo, "write-tree", env=env).decode().strip() == target
        for suffix, identity in (("r28-base", base), ("r29-candidate", target)):
            git(repo, "update-ref", f"refs/spark/jj-r29/{suffix}", identity)
        entry.update(base_tree=base, tree=target,
                     package_tree=git(repo, "rev-parse", f"{target}:{name}").decode().strip(),
                     patch=str(patch_file.relative_to(ROOT)), patch_sha256=sha(patch),
                     changed_paths=str(paths_file.relative_to(ROOT)), changed_paths_sha256=sha(paths),
                     changed_paths_count=len(paths.splitlines()))
        result[name] = entry
        for field in ("tree", "upstream_tree", "package_tree"):
            replacements[previous[name][field]] = entry[field]
    for path in [*ROOT.glob("run-*.sh"), *(ROOT / "launchers").glob("*.sh"), *(ROOT / "tests").glob("*runner.sh")]:
        text = path.read_text()
        for old, new in replacements.items():
            text = text.replace(old, new)
        path.write_text(text)
    result["flashkda"] = {
        "base_commit": upstream["flashkda.base.commit"],
        "patch_sha256": upstream["flashkda.patch.sha256"],
        "spark_extension_sha256": "69347e79224ee3a48d1d3604d302dc275a595188b89db53be2a09593dbd07d98",
        "spark_digest_source": "qualified R28 image label and runtime byte verification",
    }
    assert result["flashkda"]["base_commit"] == previous["flashkda"]["base_commit"]
    assert result["flashkda"]["patch_sha256"] == previous["flashkda"]["patch_sha256"]
    inputs = ROOT / "inputs"
    inputs.mkdir(exist_ok=True)
    repo = REPOS / "LMCache"
    git(repo, "update-ref", "refs/spark/jj-r29/lmcache-source", result["lmcache"]["commit"])
    bundle = inputs / "lmcache.bundle"
    git(repo, "bundle", "create", str(bundle), "refs/spark/jj-r29/lmcache-source")
    result["lmcache"].update(bundle="inputs/lmcache.bundle", bundle_sha256=sha(bundle.read_bytes()))
    result["native_build_contract"] = {
        "cuda_arch": "12.1a", "cmake_arch": "121a", "torch": "2.13.0",
        "cuda": "13.3", "cxx11_abi": 1,
        "dockerfile_sha256": sha((ROOT / "Dockerfile").read_bytes()),
    }
    result["cache_fingerprint"] = cache_fingerprint(result)
    result["launchers"] = {p.name: sha(p.read_bytes()) for p in sorted((ROOT / "launchers").glob("*.sh"))}
    (ROOT / "source.lock.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({n: result[n]["tree"] for n in FROZEN_TREES}, indent=2))
    print("R29-SOURCE-REPLAY-PASS (not a build or GPU qualification)")


if __name__ == "__main__":
    main()
