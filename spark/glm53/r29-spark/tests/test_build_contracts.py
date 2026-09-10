#!/usr/bin/env python3
"""CPU-only failure-path tests. No real Podman, SSH, or GPU calls."""

if not __debug__:
    raise RuntimeError("R29 verification requires Python assertions enabled")

import copy
import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contracts import FROZEN_TREES, cache_fingerprint, require_linkage, require_sm121a, validate_paths
from flashkda_counts import RequiredMatrix


class BuildContracts(unittest.TestCase):
    def shell(self, code):
        return subprocess.run(
            ["bash", "-c", "set -euo pipefail; source ./build_safety.sh; " + code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_failed_or_busy_idle_probe_refuses_build(self):
        for local, remote in [
            ("return 127", "return 0"),
            ("return 0", "return 255"),
            ("echo busy", "return 0"),
            ("return 0", "echo busy"),
        ]:
            with self.subTest(local=local, remote=remote):
                result = self.shell(
                    f"podman() {{ {local}; }}; ssh() {{ {remote}; }}; require_idle_pair"
                )
                self.assertEqual(result.returncode, 78, result.stderr)
        self.assertEqual(
            self.shell("podman() { :; }; ssh() { :; }; require_idle_pair").returncode, 0
        )

    def test_only_passed_gates_publish(self):
        for gate, expected in [("false", 1), ("true", 0)]:
            result = self.shell(
                f'podman() {{ echo "PUBLISH $*"; }}; publish_after_gates candidate final {gate}'
            )
            self.assertEqual(result.returncode, expected)
            self.assertEqual(
                "PUBLISH tag candidate final" in result.stdout, gate == "true"
            )

    def test_strict_boolean(self):
        for value in ("true", "yes", "2", ""):
            self.assertEqual(
                self.shell(f'require_bool DRY_RUN "{value}"').returncode, 2
            )

    def test_optimized_python_fails_before_work(self):
        scripts = [p for p in ROOT.glob("*.py")] + list((ROOT / "tests").glob("*.py"))
        for script in scripts:
            for flag, setting in [(["-O"], ""), ([], "1")]:
                with self.subTest(script=script.name, flag=flag, setting=setting):
                    result = subprocess.run(
                        [sys.executable, *flag, str(script)],
                        cwd=ROOT,
                        env={**os.environ, "PYTHONOPTIMIZE": setting},
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("requires Python assertions enabled", result.stderr)

    def test_frozen_sources_and_paths(self):
        lock = json.loads((ROOT / "source.lock.json").read_text())
        for name, tree in FROZEN_TREES.items():
            self.assertEqual(lock[name]["tree"], tree)
            manifest = (ROOT / lock[name]["changed_paths"]).read_bytes()
            self.assertEqual(
                hashlib.sha256(manifest).hexdigest(), lock[name]["changed_paths_sha256"]
            )
            validate_paths(name, manifest.decode().splitlines())
        for name, path in [
            ("vllm", "csrc/new.cu"),
            ("vllm", "cmake/new.cmake"),
            ("vllm", "setup.py"),
            ("b12x", "b12x/new.cpp"),
            ("lmcache", "csrc/new.cu"),
            ("vllm", "../oops.py"),
        ]:
            with self.subTest(name=name, path=path), self.assertRaises(ValueError):
                validate_paths(name, [path])

    def test_native_inputs_change_cache_identity(self):
        lock = json.loads((ROOT / "source.lock.json").read_text())
        for field in ("vllm", "b12x", "lmcache"):
            variant = copy.deepcopy(lock)
            variant[field]["tree"] = "a-different-full-tree-with-the-same-package"
            self.assertNotEqual(cache_fingerprint(lock), cache_fingerprint(variant))
        variant = copy.deepcopy(lock)
        variant["native_build_contract"]["dockerfile_sha256"] = "changed"
        self.assertNotEqual(cache_fingerprint(lock), cache_fingerprint(variant))
        self.assertEqual(cache_fingerprint(lock), lock["cache_fingerprint"])
        self.assertEqual(
            hashlib.sha256((ROOT / "Dockerfile").read_bytes()).hexdigest(),
            lock["native_build_contract"]["dockerfile_sha256"],
        )

    def test_arch_specific_cubin_required(self):
        require_sm121a("ELF file 1: kernel.sm_121a.cubin")
        for sample in ("kernel.sm_121.cubin", "sm_120a", "sm_121abc", ""):
            with self.subTest(sample=sample), self.assertRaises(RuntimeError):
                require_sm121a(sample)

    def test_driver_only_build_linkage_exception(self):
        missing_driver = "libcuda.so.1 => not found\nlibc.so.6 => /lib/libc.so.6"
        with self.assertRaises(RuntimeError):
            require_linkage(missing_driver)
        require_linkage(missing_driver, allow_missing_driver=True)
        require_linkage("libcuda.so.1 => /usr/local/cuda/compat/lib.real/libcuda.so.1")
        for missing in ("libtorch_cpu.so", "libcudart.so.13", "libcuda.so.2"):
            with self.subTest(missing=missing), self.assertRaises(RuntimeError):
                require_linkage(f"{missing} => not found", allow_missing_driver=True)

    def test_complete_flashkda_matrix_required(self):
        for collected, passed, skipped in [
            (0, 0, 0),
            (12, 0, 12),
            (8, 8, 0),
            (12, 11, 1),
        ]:
            gate = RequiredMatrix()
            gate.collected, gate.passed = collected, passed
            gate.invalid = ["skipped"] * skipped
            with self.assertRaises(RuntimeError):
                gate.verify(0)
        gate = RequiredMatrix()
        gate.pytest_collection_finish(SimpleNamespace(items=[None] * 12))
        for i in range(12):
            gate.pytest_runtest_logreport(
                SimpleNamespace(
                    skipped=False, failed=False, passed=True, when="call", nodeid=str(i)
                )
            )
        gate.verify(0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
