#!/usr/bin/env python3
"""CPU-only tests of native preservation and ARM wheel metadata."""

if not __debug__:
    raise RuntimeError("R32 verification requires Python assertions enabled")

import base64
import ast
import csv
import hashlib
import io
import sys
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from native_reuse import TAG, lmcache_preinstall_inventory, platform_wheel, verify_manifest
from native_reuse import classify_installed_files, verify_wheel_native_payload
from run_r32_regressions import CASES, RequiredCases, run_all, run_case


class NativeReuse(unittest.TestCase):
    def setUp(self):
        self.dist = "lmcache-0.5.5.dist-info/"
        self.files = {
            "lmcache/__init__.py": b"# R32 Python\n",
            self.dist + "WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            self.dist + "RECORD": b"old-record",
        }
        self.native = {"lmcache/cuda_ops.so": b"preserved-native-bytes"}

    def test_native_change_addition_and_removal_fail(self):
        expected = {"a.so": "first", "b.so": "second"}
        verify_manifest(expected, dict(expected))
        for actual in ({"a.so": "changed", "b.so": "second"}, {"a.so": "first"},
                       {**expected, "c.so": "third"}):
            with self.subTest(actual=actual), self.assertRaises(RuntimeError):
                verify_manifest(expected, actual)

    def test_native_bytes_and_python_payload_retained(self):
        result = platform_wheel(self.files, self.native, TAG)
        for name, data in self.native.items():
            self.assertEqual(result[name], data)
        self.assertEqual(result["lmcache/__init__.py"], self.files["lmcache/__init__.py"])
        self.assertIn(f"Tag: {TAG}\n".encode(), result[self.dist + "WHEEL"])
        self.assertIn(b"Root-Is-Purelib: false", result[self.dist + "WHEEL"])
        self.assertNotIn(b"py3-none-any", result[self.dist + "WHEEL"])

    def test_record_contains_exact_digest_and_size_of_every_file(self):
        result = platform_wheel(self.files, self.native, TAG)
        record = self.dist + "RECORD"
        rows = list(csv.reader(io.StringIO(result[record].decode())))
        self.assertEqual({r[0] for r in rows}, set(result))
        for name, digest, size in rows:
            if name == record:
                self.assertEqual((digest, size), ("", ""))
            else:
                expected = base64.urlsafe_b64encode(hashlib.sha256(result[name]).digest()).rstrip(b"=").decode()
                self.assertEqual(digest, "sha256=" + expected)
                self.assertEqual(int(size), len(result[name]))

    def test_wrong_arch_missing_or_foreign_native_payload_fails(self):
        for tag, native in [("cp312-cp312-linux_x86_64", self.native), (TAG, {}),
                            (TAG, {"outside.so": b"no"}), (TAG, {"lmcache/code.py": b"no"})]:
            with self.subTest(tag=tag, native=native), self.assertRaises(ValueError):
                platform_wheel(self.files, native, tag)

    def test_compiled_or_malformed_input_wheel_fails(self):
        for files in ({**self.files, **self.native}, {"lmcache/code.py": b"missing wheel"},
                      {**self.files, "second.dist-info/WHEEL": b"ambiguous"}):
            with self.subTest(files=files), self.assertRaises(ValueError):
                platform_wheel(files, self.native, TAG)

    def test_upstream_matrix_requires_all_cases_without_skips(self):
        for collected, passed, invalid in ((0, 0, []), (234, 234, []), (235, 234, ["skipped"])):
            gate = RequiredCases(235)
            gate.collected, gate.passed, gate.invalid = collected, passed, invalid
            with self.assertRaises(RuntimeError):
                gate.verify(0)
        gate = RequiredCases(235)
        gate.collected = gate.passed = 235
        gate.verify(0)
        with self.assertRaises(RuntimeError):
            gate.verify(1)

    def test_inventory_classifies_source_generated_and_native(self):
        package = Path("/venv/site/lmcache")
        metadata = Path("/venv/site/lmcache-0.5.dist-info")
        source = {package / "code.py", package / "removed_in_r32.py"}
        native = {package / "cuda_ops.so"}
        generated = {package / "_version.py", Path("/venv/bin/lmcache")}
        paths = source | native | generated | {metadata / "RECORD", package / "__pycache__/code.cpython-312.pyc"}
        result = classify_installed_files(paths, source, native, generated, metadata)
        self.assertEqual(result[str(package / "cuda_ops.so")], "preserved-native")
        self.assertEqual(result[str(package / "_version.py")], "generated")
        self.assertEqual(result[str(package / "__pycache__/code.cpython-312.pyc")], "generated-bytecode")
        self.assertEqual(result[str(metadata / "RECORD")], "distribution-metadata")
        self.assertEqual(result[str(package / "removed_in_r32.py")], "tracked-source")

    def test_unknown_payloads_inside_and_outside_package_fail(self):
        for path in ("/venv/site/lmcache/helper", "/venv/site/lmcache/data.bin",
                     "/venv/site/lmcache.libs/libhelper.so.1", "/venv/bin/native_helper"):
            with self.subTest(path=path), self.assertRaisesRegex(RuntimeError, "Unclassified"):
                classify_installed_files({Path(path)}, set(), set(), set(), Path("/venv/site/lmcache.dist-info"))

    def test_known_native_requires_exact_wheel_replacement(self):
        site = Path("/venv/site")
        payload = b"native-library"
        expected = {site / "lmcache/native.so": hashlib.sha256(payload).hexdigest()}
        verify_wheel_native_payload(expected, site, {"lmcache/native.so": payload})
        for files in ({}, {"lmcache/native.so": b"changed"}):
            with self.assertRaises(RuntimeError):
                verify_wheel_native_payload(expected, site, files)
        with self.assertRaisesRegex(RuntimeError, "relocation"):
            verify_wheel_native_payload({Path("/venv/bin/helper"): "digest"}, site, {})

    def test_distribution_record_siblings_are_inventoried_before_reinstall(self):
        fixtures = ROOT / ".compose"
        fixtures.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="inventory-test-", dir=fixtures) as temporary:
            venv = Path(temporary) / "venv"
            root = venv / "lib/python3.12/site-packages"
            root.mkdir(parents=True)
            scripts = venv / "bin"
            scripts.mkdir()
            console = scripts / "lmcache"
            console.write_text("#!/usr/bin/python3\n# generated console entrypoint\n")
            package = root / "lmcache"
            package.mkdir()
            for name, data in (("__init__.py", b"# source"), ("_version.py", b"# generated"),
                               ("cuda_ops.so", b"native")):
                (package / name).write_bytes(data)
            metadata = root / "lmcache-0.5.dist-info"
            metadata.mkdir()
            (metadata / "WHEEL").write_text("Wheel-Version: 1.0\n")
            wheel = root / "replacement.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("lmcache/cuda_ops.so", b"native")
            owned = [Path("lmcache/__init__.py"), Path("lmcache/_version.py"),
                     Path("lmcache/cuda_ops.so"), Path("lmcache-0.5.dist-info/WHEEL"),
                     Path("../../../bin/lmcache")]
            dist = SimpleNamespace(files=owned, locate_file=lambda p: root / p,
                                   entry_points=(SimpleNamespace(name="lmcache", group="console_scripts"),), version="base")
            native = {str(package / "cuda_ops.so"): hashlib.sha256(b"native").hexdigest()}
            with patch("native_reuse.importlib.metadata.distribution", return_value=dist), \
                 patch("native_reuse.subprocess.check_output", return_value=b"lmcache/__init__.py\0"), \
                 patch("native_reuse.sysconfig.get_path", return_value=str(scripts)):
                result = lmcache_preinstall_inventory(root, "base", "target", native, wheel)
                self.assertEqual(result["files"][str(package / "_version.py")], "generated")
                self.assertEqual(result["files"][str(console)], "generated")
                sibling = root / "lmcache.libs/libhelper.so.1"
                sibling.parent.mkdir()
                sibling.write_bytes(b"must not silently disappear")
                owned.append(Path("lmcache.libs/libhelper.so.1"))
                with self.assertRaisesRegex(RuntimeError, "libhelper.so.1"):
                    lmcache_preinstall_inventory(root, "base", "target", native, wheel)
                self.assertTrue(sibling.exists())
                # Also inspect package files absent from the old RECORD.
                owned.pop()
                (package / "unrecorded-helper").write_bytes(b"unclassified")
                with self.assertRaisesRegex(RuntimeError, "unrecorded-helper"):
                    lmcache_preinstall_inventory(root, "base", "target", native, wheel)

    def test_inventory_precedes_every_reinstall(self):
        tree = ast.parse((ROOT / "install_runtime.py").read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        inventory = [n.lineno for n in calls if n.func.id == "lmcache_preinstall_inventory"]
        installs = [n.lineno for n in calls if n.func.id == "run" and n.args
                    and isinstance(n.args[0], ast.Constant) and n.args[0].value == "uv"]
        self.assertEqual(len(inventory), 1)
        self.assertEqual(len(installs), 2)
        self.assertLess(inventory[0], min(installs))

    def test_all_groups_run_in_separate_processes(self):
        with patch("run_r32_regressions.subprocess.run") as run:
            run_all()
        self.assertEqual(run.call_count, 5)
        self.assertEqual([call.args[0][-1] for call in run.call_args_list], list(CASES))
        for call in run.call_args_list:
            self.assertEqual(call.args[0][0], sys.executable)
            self.assertIn("--case", call.args[0])
            self.assertTrue(call.kwargs["check"])
            self.assertEqual(call.kwargs["env"]["B12X_GLM53_GPU_TEST"], "1")
        with patch("run_r32_regressions.subprocess.run", side_effect=subprocess.CalledProcessError(1, "test")) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                run_all()
            self.assertEqual(run.call_count, 1)

    def test_pooled_child_sets_gpu_env_and_rejects_skips(self):
        def skipped(args, plugins):
            import os
            self.assertEqual(os.environ["B12X_GLM53_GPU_TEST"], "1")
            plugins[0].collected = 1
            plugins[0].invalid = ["skipped"]
            return 0

        pytest = SimpleNamespace(main=skipped)
        with patch.dict(sys.modules, {"pytest": pytest}), patch("run_r32_regressions.os.chdir"), \
             patch.dict("os.environ", {"B12X_GLM53_GPU_TEST": "0"}):
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                run_case("pooled_indexer")

    def test_new_selection_counts_are_fixed(self):
        self.assertEqual(CASES["pooled_indexer"][1], 1)
        path, count, selection = CASES["mamba_retirement"]
        self.assertEqual(count, 7)
        self.assertTrue(path.endswith("test_single_type_kv_cache_manager.py"))
        self.assertEqual(selection, "test_mamba_retirement_crosses_null_gaps or test_mamba_retirement_bounds_prefill_states")


if __name__ == "__main__":
    unittest.main(verbosity=2)
