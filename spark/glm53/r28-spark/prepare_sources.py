#!/usr/bin/env python3
"""Compose exact R28 Spark trees without checking out or changing any branch.

Uses a private index, never the source repositories' real indices. Generated
patches and manifests are release artifacts; scratch indices live on disk here.
Requires the published commits already fetched in the named local repositories.
"""

if not __debug__:
    raise RuntimeError("R28 verification requires Python assertions enabled")

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from contracts import FROZEN_TREES, cache_fingerprint, validate_paths

ROOT = Path(__file__).resolve().parent
UPSTREAM_SHA = "15a9a649559830822cb943ea0c3c6a644c8b69c9e54d7be3e265801851843932"
BASE_IMAGE = "ef669fa1cde3e99936c02575eca8f610990bb6c64dd1bbfcb04fd42dde87afae"
NATIVE_DELTA = {
    "cmake/external_projects/apply_flashkda_checkpoint_patch.cmake",
    "cmake/external_projects/flashkda.cmake",
    "cmake/external_projects/patches/flashkda-packed-checkpoints.patch",
    "csrc/flashkda_registration.cpp",
}


def git(repo, *args, env=None, data=None):
    return subprocess.check_output(["git", "-C", str(repo), *args], env=env, input=data)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repos", type=Path, default=Path("/home/jugs/git"))
    args = parser.parse_args()
    upstream_bytes = (ROOT / "upstream-r28.source.lock").read_bytes()
    assert digest(upstream_bytes) == UPSTREAM_SHA
    upstream = dict(line.split("=", 1) for line in upstream_bytes.decode().splitlines())
    previous = json.loads((ROOT.parent / "r27-spark/source.lock.json").read_text())
    scratch = ROOT / ".compose"
    scratch.mkdir(exist_ok=True)
    result = {
        "release": "jj-r28-spark-sm121",
        "status": "authored, not built or qualified",
        "upstream_lock_sha256": UPSTREAM_SHA,
        "base_image_id": BASE_IMAGE,
        "upstream_recipe_commit": "53031598b6ba4c176e9a64512c17cad18a8bf829",
        "runtime": {
            "checkpoint_policy": "aligned",
            "lmcache_enabled": False,
            "glm_mtp_head": "bf16",
            "loader": "instanttensor",
            "cuda_arch": "12.1a",
            "nccl": "retain R27 image and runner contract",
        },
    }
    for name, dirname in (("vllm", "vllm"), ("b12x", "b12x"), ("lmcache", "LMCache")):
        repo = args.repos / dirname
        commit = upstream[f"{name}.commit"]
        tree = git(repo, "rev-parse", f"{commit}^{{tree}}").decode().strip()
        assert tree == upstream[f"{name}.tree"]
        assert (
            git(repo, "rev-parse", f"{tree}:{name}").decode().strip()
            == upstream[f"{name}.package.tree"]
        )
        entry = {
            "commit": commit,
            "upstream_tree": tree,
            "upstream_package_tree": upstream[f"{name}.package.tree"],
        }
        if name == "vllm":
            index = scratch / "vllm.index"
            env = {**os.environ, "GIT_INDEX_FILE": str(index)}
            git(repo, "read-tree", commit, env=env)
            # Preserve the already-qualified architecture and draft-head overlay.
            overlay = (
                ROOT.parent / "r27-spark/patches/vllm/spark-overlay.patch"
            ).read_bytes()
            assert (
                digest(overlay)
                == "992cbae5b9f7fd358fe01f20659c9a7985406d37bdafa129940bfe7e33afaea4"
            )
            entry["spark_overlay_sha256"] = digest(overlay)
            git(repo, "apply", "--cached", "--check", "-", env=env, data=overlay)
            git(repo, "apply", "--cached", "-", env=env, data=overlay)
            target = git(repo, "write-tree", env=env).decode().strip()
            base = previous["vllm"]["spark_tree"]
            # Reconstruct the prior overlay tree too; it need not have been
            # persisted in this source checkout by the earlier build.
            git(repo, "read-tree", previous["vllm"]["r27_integration_tree"], env=env)
            git(repo, "apply", "--cached", "-", env=env, data=overlay)
            prior_fix = (
                ROOT.parent / "r27-spark" / previous["vllm"]["pr674"]["patch"]
            ).read_bytes()
            assert digest(prior_fix) == previous["vllm"]["pr674"]["patch_sha256"]
            git(repo, "apply", "--cached", "-", env=env, data=prior_fix)
            assert git(repo, "write-tree", env=env).decode().strip() == base
            # The published restore implementation widens scalar arguments before
            # arithmetic. Do not blindly reapply our superseded PR 674 patch.
            restore = git(
                repo, "show", f"{commit}:vllm/v1/worker/gpu/boundary_checkpoint.py"
            ).decode()
            assert "block = tl.full((), block, tl.int64)" in restore
            assert "slot = tl.full((), slot, tl.int64)" in restore
            native = (
                git(
                    repo,
                    "diff",
                    "--name-only",
                    previous["vllm"]["r27_integration_commit"],
                    commit,
                    "--",
                    "csrc",
                    "cmake",
                    "CMakeLists.txt",
                    "setup.py",
                    "requirements",
                )
                .decode()
                .splitlines()
            )
            assert set(native) == NATIVE_DELTA, native
            entry["rebuild_native"] = ["_flashkda_C"]
            entry["pr674"] = (
                "superseded by widened scalar restore; retain upstream GPU regression"
            )
            entry["native_changed_paths"] = native
            flash_patch = git(
                repo,
                "show",
                f"{commit}:cmake/external_projects/patches/flashkda-packed-checkpoints.patch",
            )
            assert digest(flash_patch) == upstream["flashkda.patch.sha256"]
        elif name == "b12x":
            target = tree
            base = previous["b12x"]["r27_integration_tree"]
            entry["rebuild_native"] = [
                "RoCEnante proxy from composed source, if source digest changes"
            ]
        else:
            target = tree
            base = previous["lmcache"]["integration_tree"]
            entry["rebuild_native"] = ["SM121 LMCache wheel", "cuMem IPC interposer"]
            entry["version"] = upstream["lmcache.version"]
        paths = git(repo, "diff", "--name-only", "--no-renames", base, target)
        assert target == FROZEN_TREES[name]
        validate_paths(name, paths.decode().splitlines())
        patch = git(
            repo, "diff", "--binary", "--full-index", "--no-renames", base, target
        )
        directory = ROOT / "patches" / name
        directory.mkdir(parents=True, exist_ok=True)
        patch_file = directory / "r27-to-r28.patch"
        path_file = directory / "changed-paths.txt"
        patch_file.write_bytes(patch)
        path_file.write_bytes(paths)
        # Replay from the exact source tree shipped in our R27 image.
        env = {**os.environ, "GIT_INDEX_FILE": str(scratch / f"{name}-replay.index")}
        git(repo, "read-tree", base, env=env)
        git(repo, "apply", "--cached", "-", env=env, data=patch)
        assert git(repo, "write-tree", env=env).decode().strip() == target
        for suffix, identity in (("r27-base", base), ("r28-candidate", target)):
            git(repo, "update-ref", f"refs/spark/jj-r28/{suffix}", identity)
        entry.update(
            base_tree=base,
            tree=target,
            package_tree=git(repo, "rev-parse", f"{target}:{name}").decode().strip(),
            patch=str(patch_file.relative_to(ROOT)),
            patch_sha256=digest(patch),
            changed_paths=str(path_file.relative_to(ROOT)),
            changed_paths_sha256=digest(paths),
            changed_paths_count=len(paths.splitlines()),
        )
        result[name] = entry
    result["flashkda"] = {
        "base_commit": upstream["flashkda.base.commit"],
        "patch_sha256": upstream["flashkda.patch.sha256"],
        "upstream_sm120_extension_sha256": upstream["flashkda.extension.sha256"],
        "spark_digest_source": "frozen native-stage receipt before runtime assembly",
    }
    # Bundle LMCache locally. Assembly does not need a live source-repository fetch.
    lm_repo = args.repos / "LMCache"
    git(
        lm_repo,
        "update-ref",
        "refs/spark/jj-r28/lmcache-source",
        result["lmcache"]["commit"],
    )
    inputs = ROOT / "inputs"
    inputs.mkdir(exist_ok=True)
    bundle = inputs / "lmcache.bundle"
    git(lm_repo, "bundle", "create", str(bundle), "refs/spark/jj-r28/lmcache-source")
    result["lmcache"]["bundle"] = "inputs/lmcache.bundle"
    result["lmcache"]["bundle_sha256"] = digest(bundle.read_bytes())
    # Every source component, including the patched native input, affects caches.
    result["native_build_contract"] = {
        "cuda_arch": "12.1a",
        "cmake_arch": "121a",
        "torch": "2.13.0",
        "cuda": "13.3",
        "cxx11_abi": 1,
        "dockerfile_sha256": digest((ROOT / "Dockerfile").read_bytes()),
    }
    result["cache_fingerprint"] = cache_fingerprint(result)
    replacements = {
        previous["vllm"]["r27_integration_tree"]: result["vllm"]["upstream_tree"],
        previous["vllm"]["spark_tree"]: result["vllm"]["tree"],
        previous["vllm"]["spark_package_tree"]: result["vllm"]["package_tree"],
        previous["b12x"]["r27_integration_tree"]: result["b12x"]["tree"],
        previous["b12x"]["r27_package_tree"]: result["b12x"]["package_tree"],
        previous["lmcache"]["integration_tree"]: result["lmcache"]["tree"],
        previous["lmcache"]["package_tree"]: result["lmcache"]["package_tree"],
    }
    templates = [
        "launchers/serve-glm53-flash-jj-r27-spark.sh",
        "launchers/serve-qwen38-flash-next-jj-r27-spark.sh",
        "run-glm53-flash-jj-r27-spark-tp4-node.sh",
        "run-qwen38-flash-next-jj-r27-spark-tp2-node.sh",
        "tests/test-glm53-flash-jj-r27-spark-tp4-runner.sh",
        "tests/test-qwen38-flash-next-jj-r27-spark-runner.sh",
        "tests/verify_jj_r27_runtime.py",
        "tests/verify_glm53_nvfp4_draft_head_sm121.py",
    ]
    result["template_sources"] = {}
    for relative in templates:
        original = (ROOT.parent / "r27-spark" / relative).read_text()
        result["template_sources"][relative] = digest(original.encode())
        contents = original
        for old, new in replacements.items():
            contents = contents.replace(old, new)
        contents = re.sub(
            r"localhost/voipmonitor/vllm:glm53-jj-r27-spark-[A-Za-z0-9.+-]+",
            "localhost/voipmonitor/vllm:jj-r28-spark-sm121",
            contents,
        )
        contents = contents.replace(
            "${EXPECTED_IMAGE_ID:-" + BASE_IMAGE + "}", "${EXPECTED_IMAGE_ID:-}"
        )
        contents = contents.replace("r27", "r28").replace("R27", "R28")
        contents = contents.replace(
            "# r28 removed the InstantTensor copy option",
            "# r27 removed the InstantTensor copy option",
        )
        contents = contents.replace(
            "# uses on its x86 launchers", "# used on its x86 launchers"
        )
        contents = contents.replace(
            "# Qwen-specific: r28's default recurrent checkpoint policy",
            "# R27 qualification found that the default recurrent checkpoint policy",
        )
        contents = contents.replace(
            "# the correctness set. GLM keeps its own policy until its own qualification.",
            "# the R27 correctness set. Both R28 profiles retain aligned pending qualification.",
        )
        contents = contents.replace(
            "# Qualified ef669fa1 profile. Carry the flag explicitly because that image's\n# baked launcher predates this default. Explicit auto is for reproduction only.",
            "# Carry the qualified R27 aligned profile. R28 is not qualified yet.\n# Explicit auto is a separate reproduction control.",
        )
        contents = contents.replace(
            "# Qualified Qwen profile on this image: block-aligned recurrent checkpoints.\n# The launcher baked into image ef669fa1 predates its aligned default, so the\n# runner passes the flag itself; a plain restart keeps the validated profile.\n# Drop this line when a rebuilt image carries the launcher default.",
            "# Carry R27's qualified aligned profile into the R28 candidate.\n# Keep the policy explicit in rank command receipts; R28 is not qualified yet.",
        )
        contents = contents.replace(
            "# R28 source markers: the pinned PR 674 scalar casts, the leading-instruction",
            "# R28 source markers: widened scalar restore, leading-instruction",
        )
        if relative.endswith("verify_jj_r27_runtime.py"):
            contents = contents.replace(
                '"tl.cast(block, tl.int64)"', '"block = tl.full((), block, tl.int64)"'
            )
            contents = contents.replace(
                '"tl.cast(slot, tl.int64)"', '"slot = tl.full((), slot, tl.int64)"'
            )
        destination = ROOT / relative.replace("r27", "r28")
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Existing derivatives are maintained source, not disposable generated
        # files. Source recomposition must not overwrite reviewed gate changes.
        if not destination.exists():
            destination.write_text(contents)
            destination.chmod(0o755)
    result["launchers"] = {
        p.name: digest(p.read_bytes())
        for p in sorted((ROOT / "launchers").glob("*.sh"))
    }
    (ROOT / "source.lock.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {n: result[n]["tree"] for n in ("vllm", "b12x", "lmcache")}, indent=2
        )
    )
    print("R28-SOURCE-REPLAY-PASS (not a build or GPU qualification)")


if __name__ == "__main__":
    main()
