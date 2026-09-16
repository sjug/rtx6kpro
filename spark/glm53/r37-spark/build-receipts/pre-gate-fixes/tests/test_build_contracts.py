#!/usr/bin/env python3
"""Offline recipe checks plus failure-path tests for build publication."""

import hashlib
import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contracts import FROZEN_TREES, require_sm121a


class Contracts(unittest.TestCase):
    def test_flashinfer_preserves_spark_build_parallelism(self):
        dockerfile = (ROOT / "Dockerfile.flashinfer").read_text()
        env = "\n".join(line for line in dockerfile.replace("\\\n", " ").splitlines()
                        if line.startswith("ENV "))
        self.assertEqual(re.findall(r"\bMAX_JOBS=(\d+)(?=\s|$)", env), ["20"])
        self.assertEqual(re.findall(r"\bNVCC_THREADS=(\d+)(?=\s|$)", env), ["4"])
        self.assertNotIn("AOT_MAX_JOBS_CAP", env)
        self.assertNotIn("MAX_JOBS=", (ROOT / "build-flashinfer.sh").read_text())

    def test_lmcache_cpu_build_preserves_upstream_parallelism(self):
        dockerfile = (ROOT / "Dockerfile").read_text()
        stage = dockerfile.split("FROM refreshed AS lmcache-builder\n", 1)[1].split("\nFROM ", 1)[0]
        env = "\n".join(line for line in stage.replace("\\\n", " ").splitlines()
                        if line.startswith("ENV "))
        self.assertRegex(env, r"\bNO_GPU_EXT=1\b")
        self.assertEqual(re.findall(r"\bMAX_JOBS=(\d+)(?=\s|$)", env), ["8"])

    def test_inputs(self):
        lock = json.loads((ROOT / "source.lock.json").read_text())
        entries = [lock[k] for k in ("base_lock", "upstream_lock", "spark_overlay")]
        entries.extend(lock["upstream_inputs"].values())
        entries.append(lock["flashinfer"]["regression_test"])
        for name, tree in FROZEN_TREES.items():
            self.assertEqual(lock[name]["tree"], tree)
            entries.extend([lock[name]["patch"], lock[name]["changed_paths"]])
            paths = (ROOT / lock[name]["changed_paths"]["file"]).read_text().splitlines()
            self.assertEqual(len(paths), lock[name]["changed_paths_count"])
        for entry in entries:
            self.assertEqual(hashlib.sha256((ROOT / entry["file"]).read_bytes()).hexdigest(), entry["sha256"], entry["file"])
        for name, digest in lock["native_build_inputs"].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, name)
        for name, digest in lock["launchers"].items():
            self.assertEqual(hashlib.sha256((ROOT / "launchers" / name).read_bytes()).hexdigest(), digest, name)
        old = json.loads((ROOT / lock["base_lock"]["file"]).read_text())
        self.assertEqual(old["vllm"]["native_inputs"], lock["vllm"]["native_inputs"])
        self.assertEqual(old["flashkda"]["spark_extension_sha256"], lock["flashkda"]["spark_extension_sha256"])
        self.assertEqual(old["native_reuse"]["stable_sha256"], lock["stable_native_sha256"])

    def shell(self, text):
        return subprocess.run(["bash", "-c", f"source ./build_safety.sh\n{text}"], cwd=ROOT, capture_output=True, text=True)

    def test_failed_idle_probes_refuse(self):
        self.assertEqual(self.shell("podman() { return 127; }; require_idle_pair").returncode, 78)
        self.assertEqual(self.shell("podman() { :; }; ssh() { return 255; }; require_idle_pair").returncode, 78)

    def test_busy_and_idle(self):
        self.assertEqual(self.shell("podman() { echo busy; }; require_idle_pair").returncode, 78)
        self.assertEqual(self.shell("podman() { :; }; ssh() { :; }; require_idle_pair").returncode, 0)

    def test_failed_gate_never_tags(self):
        result = self.shell("podman() { echo UNEXPECTED_TAG; }; publish_after_gates candidate final false")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("UNEXPECTED_TAG", result.stdout)

    def test_success_tags_after_gate(self):
        result = self.shell("podman() { echo TAG; }; publish_after_gates candidate final echo GATE")
        self.assertEqual(result.stdout.splitlines(), ["GATE", "TAG"])

    def test_arch_specific(self):
        require_sm121a(".sm_121a.cubin")
        with self.assertRaises(RuntimeError):
            require_sm121a(".sm_121.cubin .sm_120a.cubin")

    def test_optimized_python_refused(self):
        result = subprocess.run([sys.executable, "-O", "-c", "import contracts"], cwd=ROOT, capture_output=True)
        self.assertNotEqual(result.returncode, 0)

    def test_launchers(self):
        for path in (ROOT / "launchers").glob("*.sh"):
            result = subprocess.run(["bash", str(path)], env={**os.environ, "DRY_RUN": "1"}, text=True, capture_output=True, check=True)
            for argument in ("--load-format instanttensor", "--recurrent-checkpoint-policy aligned", "--gpu-memory-utilization 0.85"):
                self.assertIn(argument, result.stdout)
            result = subprocess.run(["bash", str(path)], env={**os.environ, "DRY_RUN": "1", "LMCACHE_ENABLED": "1"}, capture_output=True)
            self.assertEqual(result.returncode, 78)
            result = subprocess.run(["bash", str(path)], env={**os.environ, "DRY_RUN": "yes"}, capture_output=True)
            self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
