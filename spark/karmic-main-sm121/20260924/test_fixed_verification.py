"""Fixed-verification diagnostic: unapplied patch, recorder gates and comparison.

The patch is applied only to a temporary copy of the kit; the real kit is never patched. If the live
files already carry the patch, the base copy is reverse-patched and both sides are checked against
the exact reviewed hashes; any other state fails loudly.

    .venv-snapshot-cpu/bin/python -m unittest test_fixed_verification
"""
import ast
import contextlib
import copy
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
sys.path.insert(0, str(KIT))
import run_fixed_verification as fixed  # noqa: E402

PATCH = KIT / 'ds41-fixed-verification.patch'
BASES = {'launch_contract.py': 'e3d87f62bd833885ea9feb3c38857428ebd66ee4df6bc75f9fbafca730a19839',
         'start_moe_repaired.py': 'fe0e86aedae6cb78d2bda194f3bf68f7f5496362de0cc23f5aca8534e3ce8b7a'}
PATCHED_HASHES = {'launch_contract.py': 'e3a1f9cf5cc5aee07a785a12391c83e75d9f6bb3f2046d568aeedb8e560dd7ed',
                  'start_moe_repaired.py': '6f0935e33e2c7d08c77d02f7e6622b37b6241f090657175ed8c9f5e72ffeb32b'}
CONTROL = KIT / 'receipts/ratio1-control-20260927T0122Z'
RETURN = KIT / 'receipts/ratio1-return-20260927T0201Z'
CONTROL_LOG = KIT / 'receipts/ratio1-control-observed-20260927T0122Z/dusty-container.log'
SNAPSHOT = KIT / 'receipts/ratio1-namespace-snapshot-20260927T010531Z'
SUITES = ('test_launch_contract', 'test_precision_orchestration', 'test_start_diagnostic_profile',
          'test_chunking_contract', 'test_router_fixed_control', 'test_window_orchestration',
          'test_indexer_orchestration', 'test_distribute_diagnostic')
_tmp = BASE = PATCHED = None


def sha(data):
    return hashlib.sha256(data).hexdigest()


def setUpModule():
    global _tmp, BASE, PATCHED
    live = {name: sha((KIT / name).read_bytes()) for name in BASES}
    if live == BASES:
        applied = False
    elif live == PATCHED_HASHES:
        applied = True
    else:
        raise RuntimeError(f'Live files are neither the reviewed base nor the exact patched tree: {live}')
    _tmp = tempfile.TemporaryDirectory(prefix='fixed-verification-')
    BASE, PATCHED = Path(_tmp.name) / 'base', Path(_tmp.name) / 'patched'
    for target in (BASE, PATCHED):
        target.mkdir()
        for path in KIT.iterdir():
            if path.is_file() and path.name != PATCH.name:
                shutil.copy2(path, target / path.name)
        (target / 'receipts').mkdir()
        for path in (KIT / 'receipts').iterdir():
            if path.is_file() and path.suffix == '.json':
                shutil.copy2(path, target / 'receipts' / path.name)
        for extra in ('router-fence-20260926T045016Z', 'router-release-build-20260925T220934Z'):
            if (KIT / 'receipts' / extra).is_dir():
                shutil.copytree(KIT / 'receipts' / extra, target / 'receipts' / extra)
    if applied:
        subprocess.run(['patch', '-p1', '--batch', '--reverse', '-s', '-d', str(BASE), '-i', str(PATCH)], check=True)
    else:
        subprocess.run(['patch', '-p1', '--batch', '--forward', '-s', '-d', str(PATCHED), '-i', str(PATCH)], check=True)
    for tree, expected in ((BASE, BASES), (PATCHED, PATCHED_HASHES)):
        for name, digest in expected.items():
            if sha((tree / name).read_bytes()) != digest:
                raise RuntimeError(f'{tree.name}/{name} does not match the reviewed hash')


def tearDownModule():
    _tmp.cleanup()


def patched_contract():
    import importlib.util
    spec = importlib.util.spec_from_file_location('patched_launch_contract', PATCHED / 'launch_contract.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Contract(unittest.TestCase):
    def render(self, env):
        c = patched_contract()
        with mock.patch.dict(os.environ, env, clear=True):
            return c, c.render('toby')

    def test_only_adaptive_flag_changes(self):
        c, row = self.render({'DS41_DECISION_ROW_BLOCKS': '80022', 'DS41_FIXED_VERIFICATION': 'depth7-adaptive-off'})
        _, control = self.render({'DS41_DECISION_ROW_BLOCKS': '80022'})
        self.assertEqual(row['env'].pop('DS41_FIXED_VERIFICATION'), c.FIXED_VERIFICATION)
        self.assertEqual((row['env'], row['unset']), (control['env'], control['unset']))
        i = row['model'].index('--speculative-config') + 1
        self.assertEqual([a for j, a in enumerate(row['model']) if j != i], [a for j, a in enumerate(control['model']) if j != i])
        new, old = json.loads(row['model'][i]), json.loads(control['model'][i])
        self.assertEqual(list(new), list(old))
        self.assertEqual({k: v for k, v in new.items() if new[k] != old[k]}, {'enable_adaptive_verification': False})
        self.assertEqual((new['num_speculative_tokens'], new['adaptive_verification_cost_scale']), (7, 1.0))
        self.assertEqual(row['model'][i], control['model'][i].replace('"enable_adaptive_verification": true',
                                                                      '"enable_adaptive_verification": false'))

    def test_refusals(self):
        c = patched_contract()
        for env in ({'DS41_FIXED_VERIFICATION': 'depth7-adaptive-off'},
                    {'DS41_DECISION_ROW_BLOCKS': '81389', 'DS41_PREFILL_THRESHOLD': '8192', 'DS41_FIXED_VERIFICATION': 'depth7-adaptive-off'},
                    {'DS41_DECISION_ROW_BLOCKS': '80022', 'DS41_FIXED_VERIFICATION': 'off'}):
            with mock.patch.dict(os.environ, env, clear=True), self.assertRaises(ValueError):
                c.render('toby')
        release = {'image_id': 'a' * 64, 'diagnostic': {'kind': 'precision-release-candidate', 'lock_sha256': c.PRECISION_RELEASE_LOCK}}
        ratio1 = {'image_id': 'a' * 64, 'diagnostic': {'kind': 'ratio1-bf16-diagnostic', 'lock_sha256': c.RATIO1_LOCK}}
        env = {'DS41_DECISION_ROW_BLOCKS': '80022', 'DS41_FIXED_VERIFICATION': 'depth7-adaptive-off'}
        c.validate_chunking_image(release, env)
        c.validate_chunking_image(release, {'DS41_DECISION_ROW_BLOCKS': '80022'})
        for pin, bad in ((ratio1, env), (release, dict(env, DS41_FIXED_VERIFICATION='other')),
                         (release, {'DS41_FIXED_VERIFICATION': 'depth7-adaptive-off'}),
                         (release, dict(env, DS41_PREFILL_THRESHOLD='8192'))):
            with self.assertRaises(ValueError):
                c.validate_chunking_image(pin, bad)

    def test_changed_launch_contract_refused(self):
        c = patched_contract()
        reference = json.loads((PATCHED / 'upstream-launch.json').read_text())
        model = reference['reference']['model']
        i = model.index('--speculative-config') + 1
        for change in ({'num_speculative_tokens': 5}, {'enable_adaptive_verification': False}, {'adaptive_verification_cost_scale': 2.0}):
            altered = copy.deepcopy(reference)
            altered['reference']['model'][i] = json.dumps(dict(json.loads(model[i]), **change))
            (PATCHED / 'upstream-launch.json').write_text(json.dumps(altered))
            try:
                with mock.patch.dict(os.environ, {'DS41_DECISION_ROW_BLOCKS': '80022',
                                                  'DS41_FIXED_VERIFICATION': 'depth7-adaptive-off'}, clear=True), \
                        self.assertRaisesRegex(ValueError, 'Speculative launch contract changed'):
                    c.render('toby')
            finally:
                shutil.copy2(BASE / 'upstream-launch.json', PATCHED / 'upstream-launch.json')


class Start(unittest.TestCase):
    source = property(lambda self: (PATCHED / 'start_moe_repaired.py').read_text())

    def parse(self, argv):
        tree = ast.parse(self.source)
        cut = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
                   and getattr(node.targets[0], 'id', None) == 'stamp')
        body = [n for n in tree.body[:cut] if not (isinstance(n, ast.ImportFrom) and n.module == 'runtime')]
        with mock.patch.object(sys, 'argv', ['start_moe_repaired.py', *argv]), contextlib.redirect_stderr(io.StringIO()):
            ns = {'__file__': str(PATCHED / 'start_moe_repaired.py')}
            exec(compile(ast.Module(body=body, type_ignores=[]), 'start', 'exec'), ns)
        return ns['args']

    def test_rules_and_launch(self):
        ok = ['--arm', 'fixed-verification', '--decision-row-blocks', '80022']
        self.parse(ok)
        for argv in (ok[:2], ['--arm', 'fixed-verification', '--decision-row-blocks', '81389'], ok + ['--prefill-threshold', '4096'],
                     ok + ['--nccl-arm', 'standard-upstream'], ok + ['--ops-trace']):
            with self.assertRaises(SystemExit, msg=argv):
                self.parse(argv)
        fn = next(n for n in ast.parse(self.source).body if isinstance(n, ast.FunctionDef) and n.name == 'launch_command')
        texts = {}
        for arm in ('fixed-verification', 'precision-release'):
            ns = {'REMOTE': '/r', 'args': SimpleNamespace(arm=arm, decision_row_blocks='80022', cuda_module_loading=None,
                                                          prefill_threshold=None, nccl_geometry=None, nccl_arm=None)}
            exec(compile(ast.Module(body=[fn], type_ignores=[]), 'start', 'exec'), ns)
            texts[arm] = ns['launch_command']('dusty')
        self.assertEqual(texts['fixed-verification'].replace(' DS41_FIXED_VERIFICATION=depth7-adaptive-off', ''),
                         texts['precision-release'])
        self.assertIn('-u DS41_FIXED_VERIFICATION', texts['precision-release'])
        self.assertIn("'fixed-verification': 'precision-release-candidate',", self.source)

    def test_scope_and_regression(self):
        changed = {p.name for p in PATCHED.iterdir() if p.is_file() and sha(p.read_bytes()) != sha((BASE / p.name).read_bytes())}
        self.assertEqual(changed, set(BASES))
        runtime = json.loads((KIT / 'runtime-files.json').read_text())
        self.assertEqual(changed & set(runtime), {'launch_contract.py'})

        def outcome(root, suite):
            result = subprocess.run([sys.executable, '-m', 'unittest', suite], cwd=root, capture_output=True, text=True,
                                    timeout=900, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
            return result.returncode, [l.split(' in ')[0] if l.startswith('Ran ') else l for l in result.stderr.splitlines()
                                       if l.startswith(('Ran ', 'OK', 'FAILED', 'ERROR:', 'FAIL:'))]
        for suite in SUITES:
            with self.subTest(suite=suite):
                base = outcome(BASE, suite)
                self.assertEqual(base[0], 0, base)
                self.assertEqual(base, outcome(PATCHED, suite))


class FakeNodes:
    def setUp(self):
        info = json.loads((CONTROL / 'dusty-identity-before.json').read_text())
        self.base = info[0] if isinstance(info, list) else info
        self.env = dict(item.split('=', 1) for item in self.base['Config']['Env'])
        self.env['DS41_FIXED_VERIFICATION'] = 'depth7-adaptive-off'
        self.kit = self.base['Config']['Labels']['local-inference.ds41.kit.sha256']
        self.pin = {'image_id': self.base['Image'].removeprefix('sha256:')}
        spec = next(l for l in CONTROL_LOG.read_text().splitlines() if "'speculative_config'" in l)
        self.spec_off = spec.replace("'enable_adaptive_verification': True", "'enable_adaptive_verification': False")
        self.state = {}
        for rank, node in enumerate(fixed.NODES):
            log = (f'DS41-PRECISION-APPLIED rank={rank} allow_bf16_reduced_precision_reduction=False before=True t\n'
                   'b12x autotuning uses 32 temporary KV blocks before allocating 80022 serving blocks\n'
                   'b12x ready gemm.block_fp8_linear: 503/503 ready, 0 measured, 168 cached, 0 compilations, 0:02\n')
            if rank == 0:
                log += self.spec_off + '\n'
            self.state[node] = {'env': dict(self.env), 'log': log, 'selection': (SNAPSHOT / f'{node}.json').read_text()}
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def ssh(self, node, command):
        s = self.state[node]
        if command.startswith('podman inspect'):
            info = copy.deepcopy(self.base)
            info['Config']['Env'] = [f'{k}={v}' for k, v in s['env'].items()]
            info['Id'] = s.get('id', info['Id'])
            info['State']['StartedAt'] = s.get('started', info['State']['StartedAt'])
            info['Mounts'] = [m if m['Destination'] != '/cache' else dict(m, Source=s.get('mount', fixed.HOST_CACHE))
                              for m in info['Mounts']]
            return json.dumps([info])
        if command.startswith('podman logs'):
            return s['log']
        if command.startswith('cat '):
            return s['selection']
        raise AssertionError(command)

    def identity(self, out=None, phase='before'):
        return fixed.identity(self.ssh, self.pin, Path(out or self.tmp.name), phase, SNAPSHOT, self.kit)


class Recorder(FakeNodes, unittest.TestCase):
    def test_mount_source_is_verified(self):
        self.state['toby']['mount'] = '/home/jugs/git/ds41-r38/karmic-main-20260924/selection-replay-cache-standard8192-233036Z'
        with self.assertRaisesRegex(RuntimeError, 'toby: /cache is mounted from'):
            self.identity()

    def test_selection_mutation_during_requests_rejected(self):
        out = Path(self.tmp.name)
        self.identity(out, 'before')
        records = json.loads(self.state['rusty']['selection'])
        records['records']['added-during-request'] = {'assignment': {}, 'config': {}, 'coverage': {}, 'programs': []}
        self.state['rusty']['selection'] = json.dumps(records)
        self.identity(out, 'after')
        with self.assertRaisesRegex(RuntimeError, 'rusty: selections changed during requests: 1 added'):
            fixed.stable_selections(out)

    def test_pass_and_real_adaptive_log_refused(self):
        self.assertEqual(set(self.identity()), set(fixed.NODES))
        self.assertTrue(fixed.speculative_problems(CONTROL_LOG.read_text()))
        self.assertEqual(fixed.speculative_problems(self.spec_off), [])

    def test_faults(self):
        faults = [('dusty', 'log', self.state['dusty']['log'].replace(self.spec_off, '')),
                  ('toby', 'log', self.state['toby']['log'].replace('80022 serving', '81486 serving')),
                  ('rusty', 'log', self.state['rusty']['log'].replace('0 measured', '2 measured')),
                  ('kirby', 'env', {k: v for k, v in self.env.items() if k != 'DS41_FIXED_VERIFICATION'}),
                  ('toby', 'env', dict(self.env, DS41_PREFILL_THRESHOLD='8192')),
                  ('rusty', 'log', self.state['rusty']['log'] + 'DS41-RATIO1-EXTEND-PIN layer=x mode=extend v41_compute_mode=bf16\n')]
        for node, field, value in faults:
            with self.subTest(node=node, field=field):
                saved = self.state[node][field]
                self.state[node][field] = value
                shutil.rmtree(self.tmp.name)
                os.mkdir(self.tmp.name)
                with self.assertRaises(RuntimeError):
                    self.identity()
                self.state[node][field] = saved
        records = json.loads(self.state['kirby']['selection'])
        key = next(iter(records['records']))
        records['records'][key] = dict(records['records'][key], config={'changed': True})
        self.state['kirby']['selection'] = json.dumps(records)
        with self.assertRaises(RuntimeError):
            self.identity()


class Compare(FakeNodes, unittest.TestCase):
    def fake_boot(self, root, name, source, container_id, mutate=None):
        """A boot directory produced by the real recorder path against fake nodes."""
        target = root / name
        target.mkdir()
        for node in fixed.NODES:
            self.state[node]['id'] = container_id + node
            self.state[node]['started'] = container_id
        if mutate:
            mutate(self.state)
        before = self.identity(target, 'before')
        after = self.identity(target, 'after')
        (target / 'identity.json').write_text(json.dumps({'before': before, 'after': after}))
        shutil.copytree(source / 'original-524k', target / 'original-524k')
        return target

    def test_real_control_return_and_fabricated_fixed_boots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self.fake_boot(root, 'a', CONTROL, 'id-a')
            b = self.fake_boot(root, 'b', CONTROL, 'id-b')
            result = fixed.compare(a, b, CONTROL, RETURN)
            self.assertTrue(result['fixed_boots_equal'])
            self.assertFalse(result['adaptive_boots_equal'])
            self.assertEqual(result['first_divergence_control_return'], 4)
            self.assertEqual(result['token0_equal_to_control'], {'a': True, 'b': True, 'return': True})
            c = self.fake_boot(root, 'c', RETURN, 'id-c')
            self.assertFalse(fixed.compare(a, c, CONTROL, RETURN)['fixed_boots_equal'])
            self.assertEqual({n: v['boot_a_added'] for n, v in result['fixed_pair_selections'].items()},
                             {n: [] for n in fixed.NODES})
            with self.assertRaisesRegex(RuntimeError, 'not independent containers'):
                fixed.compare(a, self.fake_boot(root, 'd', CONTROL, 'id-a'), CONTROL, RETURN)

    def test_pair_mismatches_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self.fake_boot(root, 'a', CONTROL, 'id-a')
            snapshot = {n: (SNAPSHOT / f'{n}.json').read_text() for n in fixed.NODES}

            def added(state):
                records = json.loads(snapshot['kirby'])
                records['records']['boot-b-only'] = {'assignment': {}, 'config': {}, 'coverage': {}, 'programs': []}
                state['kirby']['selection'] = json.dumps(records)

            def profile(state):
                state['dusty']['env'] = dict(state['dusty']['env'], B12X_W4A16_TC_DECODE='0')

            for name, mutate, message in (('b1', added, 'kirby: selection records differ between fixed boots'),
                                          ('b2', profile, 'dusty: profile environment differs')):
                with self.subTest(name=name):
                    for node in fixed.NODES:
                        self.state[node]['selection'] = snapshot[node]
                        self.state[node]['env'] = dict(self.env)
                    b = self.fake_boot(root, name, CONTROL, 'id-' + name, mutate)
                    with self.assertRaisesRegex(RuntimeError, message):
                        fixed.compare(a, b, CONTROL, RETURN)


if __name__ == '__main__':
    unittest.main()
