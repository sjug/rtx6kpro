#!/usr/bin/env python3
"""Carry forward the rejected R37 build and prune-exposure regressions."""

import ast
import hashlib
import re
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from flashkda_counts import RequiredMatrix
from verify_component import verify


class GateRepairs(unittest.TestCase):
    def test_launcher_build_args_cannot_be_shadowed_by_runtime_env(self):
        source = (ROOT / "Dockerfile").read_text()
        for family in ("GLM", "QWEN"):
            self.assertIn(f"ARG BUILD_{family}_LAUNCHER_SHA256", source)
            self.assertIn(f"{family}_LAUNCHER_SHA256=${{BUILD_{family}_LAUNCHER_SHA256}}", source)
        self.assertIn("sha256sum -c", source)

    def test_flashinfer_metadata_matches_pinned_generated_module(self):
        source = ast.parse((ROOT / "tests/verify_dependencies.py").read_text())
        check = next(n for n in source.body if isinstance(n, ast.Assert)
                     and "fi_meta." in ast.unparse(n))
        namespace = {"fi_meta": SimpleNamespace(__git_commit__="pinned"),
                     "cache_meta": SimpleNamespace(__git_version__="pinned"),
                     "lock": {"flashinfer": {"commit": "pinned"}}}
        exec(compile(ast.Module(body=[check], type_ignores=[]), "metadata-check", "exec"), namespace)
        namespace["cache_meta"].__git_version__ = "wrong"
        with self.assertRaises(AssertionError):
            exec(compile(ast.Module(body=[check], type_ignores=[]), "metadata-check", "exec"), namespace)

    def test_collection_counts_match_pinned_parametrizations(self):
        script = ROOT / "tests/gate-regressions.sh"
        source = (script if script.exists() else ROOT / "tests/gate-image.sh").read_text()
        for selector, expected in (("full_boundary_hit_is_admitted_while_another_request_decodes", 20),
                                   ("prefill_dsv4_dual", 24)):
            line = next(line for line in source.splitlines() if " -k " in line and selector in line)
            self.assertEqual(int(re.search(r"--count (\d+)", line)[1]), expected)

    def test_builds_tag_outputs_at_commit_not_after_gates(self):
        for name in ("build.sh", "build-flashinfer.sh"):
            source = (ROOT / name).read_text()
            self.assertIn('--tag "$retained_tag"', source, name)
            self.assertIn('retained-tag.txt', source, name)
        runtime = (ROOT / "build.sh").read_text()
        self.assertLess(runtime.index('> "$receipt/image-inspect.json"'),
                        runtime.index('publish_after_gates "$candidate"'))

    def test_effective_nvcc_parallelism_is_one(self):
        source = (ROOT / "Dockerfile.flashinfer").read_text()
        env = "\n".join(line for line in source.replace("\\\n", " ").splitlines() if line.startswith("ENV "))
        self.assertRegex(env, r"\bMAX_JOBS=20\b")
        self.assertRegex(env, r"\bFLASHINFER_NVCC_THREADS=1\b")
        self.assertNotRegex(env, r"\bNVCC_THREADS=")

    def test_collection_fails_closed_without_executing_tests(self):
        gate = RequiredMatrix()
        gate.collected = 24
        gate.verify_collection(0, 24)
        self.assertEqual(gate.passed, 0)
        for count, status in ((16, 0), (0, 0), (24, 2)):
            gate.collected = count
            with self.assertRaises(RuntimeError):
                gate.verify_collection(status, 24)
        gate.collected = 24
        gate.pytest_collectreport(SimpleNamespace(skipped=True, failed=False, nodeid="skipped-module"))
        with self.assertRaises(RuntimeError):
            gate.verify_collection(0, 24)

    def test_collection_precedes_gpu_execution(self):
        source = (ROOT / "tests/gate-image.sh").read_text()
        self.assertLess(source.index("gate-regressions.sh collect"), source.index("/verify_jj_r38_runtime.py"))
        self.assertLess(source.index("gate-regressions.sh collect"), source.index("gate-regressions.sh run"))

    def test_component_receipt_must_match_current_recipe(self):
        (ROOT / ".compose").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / ".compose") as folder:
            directory = Path(folder)
            lock = {"base_image_id": "base", "flashinfer": {"arch": "12.1a", "commit": "pin"},
                    "native_build_inputs": {}}
            (directory / "inputs.txt").write_text("base=base\narch=12.1a\ncommit=pin\n")
            lines = []
            for name in ("Dockerfile.flashinfer", "build-flashinfer.sh"):
                (directory / name).write_text("original")
                digest = hashlib.sha256(b"original").hexdigest()
                lock["native_build_inputs"][name] = digest
                lines.append(f"{digest}  {name}\n")
            (directory / "recipe.sha256").write_text("".join(lines))
            verify(directory, lock)
            lock["native_build_inputs"]["Dockerfile.flashinfer"] = "new-recipe"
            with self.assertRaises(RuntimeError):
                verify(directory, lock)
            lock["native_build_inputs"]["Dockerfile.flashinfer"] = digest
            (directory / "build-flashinfer.sh").write_text("changed")
            with self.assertRaises(RuntimeError):
                verify(directory, lock)

    def test_lmcache_selection_imports_the_installed_distribution(self):
        # Its source tree carries the Python files but never the rebuilt
        # extensions, and every level of its test package has __init__.py.
        lines = (ROOT / "tests/gate-regressions.sh").read_text().splitlines()
        selection = [line for line in lines if "test_fs_native_connector.py" in line]
        self.assertEqual(len(selection), 1)
        self.assertIn("--import-mode=importlib", selection[0])
        self.assertIn("--count 4", selection[0])
        self.assertIn("cd /opt/lmcache/source-r38", lines)

    def test_label_failure_prints_expected_and_actual(self):
        result = subprocess.run(
            ["bash", "-c", 'podman() { echo wrong-label; }; export -f podman; bash tests/gate-image.sh image expected-label'],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Source lock label: expected=expected-label actual=wrong-label", result.stderr)


if __name__ == "__main__":
    unittest.main()
