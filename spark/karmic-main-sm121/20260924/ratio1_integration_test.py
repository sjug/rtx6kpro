"""Proposal tests for ds41-ratio1-integration.patch (80022 pin for the ratio-1 control/variant/return).

Applies the patch to a temporary copy of the kit; the real kit is never patched or written.

    .venv-snapshot-cpu/bin/python -m unittest ratio1_integration_test
"""
import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

KIT = Path(__file__).resolve().parent
PATCH = KIT / 'ds41-ratio1-integration.patch'
BASES = {'distribute_diagnostic.py': '3f2ed03f5ef0d18f6490a69dc8c58de15f0a1988eb9b1f2e174d6358e7c30c8f',
         'launch_contract.py': '4e9f961696266bf859ee26472ac4d1e22a64539727bf6ad2784502db2e33507e',
         'run_node.py': 'df767ed6254285ced9690104509675246721e3004f12ff3204b00c12c24ce22b',
         'runtime.py': '72aaa0205bac39b50c6af5bac882a93289ecb0d678bfb155c97230da33751b96',
         'start_moe_repaired.py': '9986ae747e065a92e09156bd1a4df72f1f59fce03223ed10c2a0d795121f71f5'}
EXTRAS = ('router-fence-20260926T045016Z', 'needle-sensitivity-20260926T050535Z/construction.json',
          'router-release-build-20260925T220934Z/BUILD-OK', 'router-release-build-20260925T220934Z/image-inspect.json')
SUITES = ('test_launch_contract', 'test_precision_orchestration', 'test_window_orchestration', 'test_indexer_orchestration',
          'test_start_diagnostic_profile', 'test_router_fixed_control', 'test_chunking_contract', 'test_distribute_diagnostic',
          'claude_test_run_decision_capture', 'test_qualify_engram_repair', 'test_ds41_ratio1',
          'test_ds41_precision_release', 'claude_test_gate_frozen_mixed_needle')
_tmp = BASE = PATCHED = None


def sha(data):
    return hashlib.sha256(data).hexdigest()


def copy_kit(target):
    target.mkdir()
    for path in KIT.iterdir():
        if path.is_file() and path.name != PATCH.name:
            shutil.copy2(path, target / path.name)
    (target / 'receipts').mkdir()
    for path in (KIT / 'receipts').iterdir():
        if path.is_file() and path.suffix == '.json':
            shutil.copy2(path, target / 'receipts' / path.name)
    receipt = json.loads((KIT / 'receipts/precision-release-build-receipt.json').read_text())
    for extra in (*EXTRAS, Path(receipt['directory']).name,
                  'precision-release-full-qualification-20260927-r2/initial-selections'):
        source = KIT / 'receipts' / extra
        if source.is_dir():
            shutil.copytree(source, target / 'receipts' / extra)
        elif source.is_file():
            (target / 'receipts' / extra).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target / 'receipts' / extra)


def setUpModule():
    global _tmp, BASE, PATCHED
    for name, expected in BASES.items():
        if sha((KIT / name).read_bytes()) != expected:
            raise unittest.SkipTest(f'{name} changed since the proposal; regenerate the patch against it')
    _tmp = tempfile.TemporaryDirectory(prefix='ratio1-integration-')
    BASE, PATCHED = Path(_tmp.name) / 'base', Path(_tmp.name) / 'patched'
    copy_kit(BASE)
    copy_kit(PATCHED)
    subprocess.run(['patch', '-p1', '--batch', '--forward', '-s', '-d', str(PATCHED), '-i', str(PATCH)], check=True)
    sys.path.insert(0, str(PATCHED))


def tearDownModule():
    sys.path.remove(str(PATCHED))
    _tmp.cleanup()


def contract():
    import launch_contract
    assert Path(launch_contract.__file__).resolve().parent == PATCHED.resolve()
    return launch_contract


def pins():
    c = contract()
    return {kind: {'image_id': 'a' * 64, 'diagnostic': {'kind': kind, 'lock_sha256': lock}}
            for kind, lock in (('precision-release-candidate', c.PRECISION_RELEASE_LOCK),
                               ('ratio1-bf16-diagnostic', c.RATIO1_LOCK),
                               ('decision-row-capture', c.MATCHED_CAPTURE_LOCK),
                               ('precision-capture', c.PRECISION_CAPTURE_LOCK))}


class Scope(unittest.TestCase):
    def test_only_declared_files_and_current_locks(self):
        changed = {p.name for p in PATCHED.iterdir() if p.is_file() and (
            not (BASE / p.name).is_file() or sha(p.read_bytes()) != sha((BASE / p.name).read_bytes()))}
        self.assertEqual(changed, set(BASES))
        c = contract()
        self.assertEqual(c.RATIO1_LOCK, sha((KIT / 'ds41-ratio1.lock.json').read_bytes()))
        self.assertEqual(c.PRECISION_RELEASE_LOCK, sha((KIT / 'ds41-precision-release.lock.json').read_bytes()))
        self.assertEqual(c.RATIO1_BLOCKS, str(json.loads((KIT / 'ds41-ratio1.lock.json').read_text())['kv_blocks']))


class Contract(unittest.TestCase):
    def render(self, env, node='toby'):
        with mock.patch.dict(os.environ, env, clear=True):
            return contract().render(node)

    def test_render_80022_is_normal_profile_plus_pin(self):
        pinned, normal = self.render({'DS41_DECISION_ROW_BLOCKS': '80022'}), self.render({})
        self.assertEqual(pinned['env'].pop('DS41_DECISION_ROW_BLOCKS'), '80022')
        self.assertEqual([a for a in pinned['model'] if a not in normal['model'] or a == '--num-gpu-blocks-override'],
                         ['--num-gpu-blocks-override', '80022'])
        self.assertEqual(pinned['env'], normal['env'])
        self.assertEqual(pinned['unset'], normal['unset'])
        for extra in ({'DS41_PREFILL_THRESHOLD': '8192'}, {'DS41_NCCL_GEOMETRY': 'tree-simple-1ch'},
                      {'DS41_NCCL_ARM': 'standard-upstream'}, {'DS41_CHUNKING_BLOCKS': '81389'}):
            with self.assertRaises(ValueError):
                self.render(dict(extra, DS41_DECISION_ROW_BLOCKS='80022'))

    def test_image_gate(self):
        c, p = contract(), pins()
        env = {'DS41_DECISION_ROW_BLOCKS': '80022'}
        c.validate_chunking_image(p['precision-release-candidate'], env)
        c.validate_chunking_image(p['ratio1-bf16-diagnostic'], env)
        c.validate_chunking_image(p['precision-release-candidate'], {})
        refused = [(p['ratio1-bf16-diagnostic'], {}), (p['decision-row-capture'], env), (p['precision-capture'], env),
                   (p['ratio1-bf16-diagnostic'], dict(env, DS41_PREFILL_THRESHOLD='8192')),
                   (p['ratio1-bf16-diagnostic'], dict(env, DS41_NCCL_ARM='standard-upstream')),
                   (dict(p['ratio1-bf16-diagnostic'], diagnostic_overlay={}), env),
                   ({'image_id': 'a' * 64, 'diagnostic': {'kind': 'ratio1-bf16-diagnostic', 'lock_sha256': '0' * 64}}, env),
                   (p['precision-release-candidate'], {'DS41_DECISION_ROW_BLOCKS': '81389', 'DS41_PREFILL_THRESHOLD': '8192'})]
        for pin, e in refused:
            with self.assertRaises(ValueError, msg=(pin['diagnostic'], e)):
                c.validate_chunking_image(pin, e)

    def test_node_command(self):
        import run_node
        for kind in ('precision-release-candidate', 'ratio1-bf16-diagnostic'):
            for node in ('dusty', 'toby', 'rusty', 'kirby'):
                with mock.patch.dict(os.environ, {'DS41_DECISION_ROW_BLOCKS': '80022'}, clear=True):
                    argv, _, _ = run_node.command(node, pins()[kind], 'b' * 64)
                self.assertIn('DS41_DECISION_ROW_BLOCKS=80022', argv)
        self.assertIn("'precision-release-candidate', 'ratio1-bf16-diagnostic'", (PATCHED / 'runtime.py').read_text())

    def test_distribution(self):
        import distribute_diagnostic
        self.assertEqual(distribute_diagnostic.KINDS['ratio1'], 'ds41-ratio1')


class Start(unittest.TestCase):
    source = property(lambda self: (PATCHED / 'start_moe_repaired.py').read_text())

    def parse(self, argv):
        tree = ast.parse(self.source)
        cut = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
                   and getattr(node.targets[0], 'id', None) == 'stamp')
        body = [n for n in tree.body[:cut] if not (isinstance(n, ast.ImportFrom) and n.module == 'runtime')]
        ns = {'__file__': str(PATCHED / 'start_moe_repaired.py')}
        with mock.patch.object(sys, 'argv', ['start_moe_repaired.py', *argv]), contextlib.redirect_stderr(io.StringIO()):
            exec(compile(ast.Module(body=body, type_ignores=[]), 'start', 'exec'), ns)
        return ns['args']

    def test_arm_rules(self):
        self.parse(['--arm', 'ratio1-diag', '--decision-row-blocks', '80022'])
        self.parse(['--arm', 'precision-release', '--decision-row-blocks', '80022'])
        self.parse(['--arm', 'precision-release'])
        for argv in (['--arm', 'ratio1-diag'], ['--arm', 'precision-release', '--decision-row-blocks', '81389'],
                     ['--arm', 'decision-row', '--decision-row-blocks', '80022'],
                     ['--arm', 'ratio1-diag', '--decision-row-blocks', '80022', '--prefill-threshold', '4096'],
                     ['--arm', 'ratio1-diag', '--decision-row-blocks', '80022', '--nccl-arm', 'standard-upstream'],
                     ['--arm', 'ratio1-diag', '--decision-row-blocks', '80022', '--ops-trace']):
            with self.assertRaises(SystemExit, msg=argv):
                self.parse(argv)

    def test_launch_and_markers(self):
        tree = ast.parse(self.source)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'launch_command')
        texts = {}
        for arm in ('ratio1-diag', 'precision-release'):
            ns = {'REMOTE': '/r', 'args': SimpleNamespace(arm=arm, decision_row_blocks='80022', cuda_module_loading=None,
                                                          prefill_threshold=None, nccl_geometry=None, nccl_arm=None)}
            exec(compile(ast.Module(body=[fn], type_ignores=[]), 'start', 'exec'), ns)
            texts[arm] = ns['launch_command']('dusty')
        self.assertEqual(texts['ratio1-diag'], texts['precision-release'])
        for item in ('DS41_DECISION_ROW_BLOCKS=80022', 'B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1', 'B12X_DENSE_SPLITK_TURBO=0'):
            self.assertIn(item, texts['ratio1-diag'])
        self.assertIn("'ratio1-diag': 'ratio1-bf16-diagnostic',", self.source)
        watch, check, repro = (self.source.index("'watch-startup.py'"), self.source.index('pin_marker_problems(marker'),
                               self.source.index('repro = subprocess.run'))
        self.assertLess(watch, check)
        self.assertLess(check, repro)


class Regression(unittest.TestCase):
    def outcome(self, root, suite):
        result = subprocess.run([sys.executable, '-m', 'unittest', suite], cwd=root, capture_output=True,
                                text=True, timeout=900, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        return result.returncode, [l.split(' in ')[0] if l.startswith('Ran ') else l for l in result.stderr.splitlines()
                                   if l.startswith(('Ran ', 'OK', 'FAILED', 'ERROR:', 'FAIL:'))]

    def test_existing_suites_same_outcome(self):
        for suite in SUITES:
            with self.subTest(suite=suite):
                self.assertEqual(self.outcome(BASE, suite), self.outcome(PATCHED, suite))


if __name__ == '__main__':
    unittest.main()
