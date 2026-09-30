"""Proposal tests for precision-default-nccl.patch: the precision image with the ordinary NCCL settings.

Runs on a temporary copy of the kit (top-level files plus top-level receipt JSON). The real kit is
never patched or written. Apply checks: exact base hashes, only the six declared files change.
Behaviour checks: explicit arm selection, only the diagnostic tree/Simple/channel overrides are
absent, the ordinary unset list and cluster NCCL settings remain, lock/image checks unchanged,
tree and standard receipts never interchange, and no weakening for the indexer or window arms.
Regression: every related existing suite has the same outcome on the base and patched copies.

    .venv-snapshot-cpu/bin/python -m unittest precision_default_nccl_test
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
PATCH = KIT / 'precision-default-nccl.patch'
BASES = {
    'launch_contract.py': '83d2fb18e56c5b4a89c39caec8c10cf081110dec984b116841012a86556e4474',
    'run_node.py': 'db67b620af2cde9f57df90342b70a2cd21de9ba3a8073189768ab5b68bdab6a9',
    'start_moe_repaired.py': '5e51fe3f4039c974c0cac108c1e7f9a72e193eccde768b97ee47d55e2720432c',
    'claude_run_decision_capture.py': '20596f25a7ae86d2e0e8c37af25dcb294a6717ededf862551678c7d1a3f2643e',
    'run_indexer_comparison.py': '7f30cb2f5ec519cabe65e37d9f555b71b2e817c78ef2b8b64ec65b79d4ad8a07',
    'run_geometry_decision_cross.py': '50ebac898a56aa11e347b353caf66f885c6d73ecbf43f2428d9c48796212023b',
}
SUITES = ('test_audit_window_geometry', 'test_indexer_orchestration', 'test_router_fixed_control',
          'test_launch_contract', 'test_precision_orchestration', 'test_window_orchestration',
          'test_select_router_candidate', 'test_start_diagnostic_profile', 'claude_test_run_decision_capture',
          'test_window_geometry', 'test_chunking_contract', 'claude_test_engram_failclosed',
          'claude_test_probe_chunking', 'claude_test_probe_needle_sensitivity')
EXTRAS = ('router-fence-20260926T045016Z', 'needle-sensitivity-20260926T050535Z/construction.json')
ARM = 'standard-upstream'
TREE_ENV = ('NCCL_ALGO', 'NCCL_PROTO', 'NCCL_MIN_NCHANNELS', 'NCCL_MAX_NCHANNELS')
_tmp = None
BASE = PATCHED = None


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_kit(target):
    target.mkdir()
    for path in KIT.iterdir():
        if path.is_file() and path.name != PATCH.name:
            shutil.copy2(path, target / path.name)
    (target / 'receipts').mkdir()
    for path in (KIT / 'receipts').iterdir():
        if path.is_file() and path.suffix == '.json':
            shutil.copy2(path, target / 'receipts' / path.name)
    # Small retained evidence that existing suites read; missing items fail identically on both copies.
    for extra in EXTRAS:
        source = KIT / 'receipts' / extra
        if source.is_dir():
            shutil.copytree(source, target / 'receipts' / extra)
        elif source.is_file():
            (target / 'receipts' / extra).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target / 'receipts' / extra)


def setUpModule():
    global _tmp, BASE, PATCHED
    for name, expected in BASES.items():
        if sha(KIT / name) != expected:
            raise unittest.SkipTest(f'{name} changed since the proposal; regenerate the patch against it')
    _tmp = tempfile.TemporaryDirectory(prefix='precision-default-')
    BASE, PATCHED = Path(_tmp.name) / 'base', Path(_tmp.name) / 'patched'
    copy_kit(BASE)
    copy_kit(PATCHED)
    subprocess.run(['patch', '-p1', '--batch', '--forward', '-s', '-d', str(PATCHED), '-i', str(PATCH)], check=True)
    sys.path.insert(0, str(PATCHED))


def tearDownModule():
    sys.path.remove(str(PATCHED))
    _tmp.cleanup()


def modules():
    import launch_contract, run_node, claude_run_decision_capture, precision_default_nccl
    for module in (launch_contract, run_node, claude_run_decision_capture, precision_default_nccl):
        assert Path(module.__file__).resolve().parent == PATCHED.resolve(), module.__file__
    return launch_contract, run_node, claude_run_decision_capture, precision_default_nccl


def env_for(threshold, geometry=None, arm=None):
    env = {'DS41_DECISION_ROW_BLOCKS': '81389', 'DS41_PREFILL_THRESHOLD': threshold}
    if geometry:
        env['DS41_NCCL_GEOMETRY'] = geometry
    if arm:
        env['DS41_NCCL_ARM'] = arm
    return env


class PatchScope(unittest.TestCase):
    def test_only_declared_files_change(self):
        changed = {p.name for p in PATCHED.iterdir() if p.is_file() and (
            not (BASE / p.name).is_file() or sha(p) != sha(BASE / p.name))}
        self.assertEqual(changed, set(BASES))
        for name in BASES:
            compile((PATCHED / name).read_text(), name, 'exec')

    def test_patch_leaves_existing_gate_lines(self):
        """Every tree-arm and window/indexer refusal line of the base survives unchanged."""
        for name, needles in {
                'launch_contract.py': ["raise ValueError('NCCL geometry diagnostic requires tree-simple-1ch on the matched 81389 capture profile')",
                                       "raise ValueError('NCCL geometry diagnostic requires the reviewed window-capture lock without overlay')",
                                       "or diagnostic.get('lock_sha256') != expected_lock"],
                'run_node.py': ["'NCCL geometry diagnostic is restricted to window-capture')"],
                'claude_run_decision_capture.py': ["problems.append('NCCL warned about the requested geometry; unsupported settings are not accepted')",
                                                   "problems += check_precision_marker(marker, NODES.index(node))"]}.items():
            for needle in needles:
                self.assertIn(needle, (BASE / name).read_text())
                self.assertIn(needle, (PATCHED / name).read_text())


class Contract(unittest.TestCase):
    def render(self, node, env):
        contract = modules()[0]
        with mock.patch.dict(os.environ, env, clear=True):
            return contract.render(node)

    def test_standard_render_is_the_ordinary_launch_plus_selector(self):
        contract = modules()[0]
        for threshold in ('8192', '4096'):
            for node in contract.NODES:
                standard = self.render(node, env_for(threshold, arm=ARM))
                ordinary = self.render(node, env_for(threshold))
                tree = self.render(node, env_for(threshold, geometry=contract.NCCL_GEOMETRY))
                self.assertEqual(standard['env'].pop('DS41_NCCL_ARM'), ARM)
                self.assertEqual(standard, ordinary)
                self.assertEqual(standard['unset'], contract.NCCL_TUNING_UNSET)
                self.assertEqual(tree['unset'], [])
                for key in TREE_ENV + ('DS41_NCCL_GEOMETRY',):
                    self.assertNotIn(key, standard['env'])
                upstream = {k: v for k, v in tree['env'].items() if k.startswith('NCCL_') and k not in TREE_ENV}
                self.assertTrue(upstream and all(standard['env'].get(k) == v for k, v in upstream.items()))
                self.assertEqual(standard['model'], tree['model'])

    def test_standard_render_refusals(self):
        contract = modules()[0]
        for env in (env_for('8192', geometry=contract.NCCL_GEOMETRY, arm=ARM), env_for('8192', arm='standard'),
                    dict(env_for('8192'), DS41_NCCL_ARM=''), {'DS41_DECISION_ROW_BLOCKS': '81389', 'DS41_NCCL_ARM': ARM},
                    {'DS41_DECISION_ROW_BLOCKS': '80927', 'DS41_PREFILL_THRESHOLD': '8192', 'DS41_NCCL_ARM': ARM},
                    {'DS41_NCCL_ARM': ARM}):
            with self.assertRaises(ValueError, msg=env):
                self.render('dusty', env)

    def test_image_gate(self):
        contract = modules()[0]
        pins = {kind: {'image_id': 'a' * 64, 'diagnostic': {'kind': kind, 'lock_sha256': lock}}
                for kind, lock in (('precision-capture', contract.PRECISION_CAPTURE_LOCK),
                                   ('indexer-capture', contract.INDEXER_CAPTURE_LOCK),
                                   ('window-capture', contract.WINDOW_CAPTURE_LOCK),
                                   ('decision-row-capture', contract.MATCHED_CAPTURE_LOCK))}
        tree = dict(env_for('4096', geometry=contract.NCCL_GEOMETRY), **contract.NCCL_GEOMETRY_ENV)
        standard = env_for('4096', arm=ARM)
        contract.validate_chunking_image(pins['precision-capture'], standard)
        contract.validate_chunking_image(pins['precision-capture'], tree)
        contract.validate_chunking_image(pins['indexer-capture'], tree)
        contract.validate_chunking_image(pins['window-capture'], tree)
        contract.validate_chunking_image(pins['window-capture'], env_for('4096'))
        refused = [(pins['precision-capture'], env_for('4096')),
                   (pins['precision-capture'], dict(tree, DS41_NCCL_ARM=ARM)),
                   (pins['precision-capture'], env_for('4096', arm='standard')),
                   (dict(pins['precision-capture'], diagnostic_overlay={}), standard),
                   ({'image_id': 'a' * 64, 'diagnostic': {'kind': 'precision-capture',
                                                          'lock_sha256': contract.INDEXER_CAPTURE_LOCK}}, standard),
                   (pins['indexer-capture'], standard), (pins['indexer-capture'], env_for('4096')),
                   (pins['window-capture'], standard), (pins['decision-row-capture'], standard),
                   ({'image_id': contract.CHUNKING_IMAGE,
                     'diagnostic': {'kind': 'router-stage-release-candidate'}}, {'DS41_NCCL_ARM': ARM})]
        for pin, env in refused:
            with self.assertRaises(ValueError, msg=(pin['diagnostic'], env)):
                contract.validate_chunking_image(pin, env)

    def test_node_command(self):
        contract, run_node = modules()[:2]
        pin = {'image_id': 'a' * 64, 'diagnostic': {'kind': 'precision-capture',
                                                    'lock_sha256': contract.PRECISION_CAPTURE_LOCK}}
        for threshold in ('8192', '4096'):
            for node in contract.NODES:
                with mock.patch.dict(os.environ, env_for(threshold, arm=ARM), clear=True):
                    argv, _, _ = run_node.command(node, pin, 'b' * 64)
                with mock.patch.dict(os.environ, env_for(threshold, geometry=contract.NCCL_GEOMETRY), clear=True):
                    tree, _, _ = run_node.command(node, pin, 'b' * 64)
                self.assertIn('DS41_NCCL_ARM=' + ARM, argv)
                self.assertFalse([a for a in argv if a.split('=')[0] in TREE_ENV + ('DS41_NCCL_GEOMETRY',)])
                self.assertIn('unset ' + ' '.join(contract.NCCL_TUNING_UNSET) + '; ', argv[-1])
                self.assertEqual(argv[argv.index('--entrypoint'):-1], tree[tree.index('--entrypoint'):-1])
                self.assertEqual(self.split(argv, TREE_ENV + ('DS41_NCCL_ARM', 'DS41_NCCL_GEOMETRY')),
                                 self.split(tree, TREE_ENV + ('DS41_NCCL_ARM', 'DS41_NCCL_GEOMETRY')))

    @staticmethod
    def split(argv, drop):
        """(container env without the named keys, all other argv before the entrypoint body)."""
        env, rest, items = {}, [], iter(argv[:-1])
        for item in items:
            if item == '-e':
                key, value = next(items).split('=', 1)
                env[key] = value
            else:
                rest.append(item)
        return {k: v for k, v in env.items() if k not in drop}, rest


class Driver(unittest.TestCase):
    def test_profiles_are_distinct(self):
        contract, _, driver, _ = modules()
        standard = driver.profile_for('matched', 8192, None, ARM)
        tree = driver.profile_for('matched', 8192, contract.NCCL_GEOMETRY)
        plain = driver.profile_for('matched', 8192)
        self.assertEqual(standard['tag'], 'matched8192-nccl-standard-upstream-')
        self.assertEqual(tree['tag'], 'matched8192-geometry-tree-simple-1ch-')
        self.assertEqual(plain['tag'], 'matched8192-')
        self.assertEqual((standard['geometry'], standard['nccl_arm']), (None, ARM))
        self.assertIsNone(tree['nccl_arm'])
        self.assertIsNone(driver.profile_for('historical')['nccl_arm'])
        for args in (('historical', None, None, ARM), ('matched', 4096, contract.NCCL_GEOMETRY, ARM),
                     ('matched', 4096, None, 'standard')):
            with self.assertRaises(ValueError):
                driver.profile_for(*args)

    def test_environment_checks(self):
        contract, _, driver, _ = modules()
        standard = driver.profile_for('matched', 4096, None, ARM)
        tree = driver.profile_for('matched', 4096, contract.NCCL_GEOMETRY)
        plain = driver.profile_for('matched', 4096)
        tree_env = dict(contract.NCCL_GEOMETRY_ENV, DS41_NCCL_GEOMETRY=contract.NCCL_GEOMETRY)
        self.assertEqual(driver.check_geometry_env({'DS41_NCCL_ARM': ARM}, standard), [])
        self.assertEqual(driver.check_geometry_env(tree_env, tree), [])
        self.assertEqual(driver.check_geometry_env({}, plain), [])
        for env, profile in (({}, standard), ({'DS41_NCCL_ARM': ARM, 'NCCL_PROTO': 'Simple'}, standard),
                             (dict(tree_env, DS41_NCCL_ARM=ARM), standard), (dict(tree_env, DS41_NCCL_ARM=ARM), tree),
                             ({'DS41_NCCL_ARM': ARM}, plain), ({'DS41_NCCL_ARM': 'standard'}, standard)):
            self.assertTrue(driver.check_geometry_env(env, profile), (env, profile['tag']))

    def container_problems(self, kind, profile, env):
        driver = modules()[2]
        image_id, lock = 'c' * 64, 'd' * 64
        container = {'State': {'Running': True}, 'Image': image_id,
                     'Config': {'Env': [f'{k}={v}' for k, v in env.items()],
                                'Labels': {'local-inference.ds41.kit.sha256': 'k'}}}
        image = {'Id': image_id, 'Config': {'Labels': {'local-inference.ds41.diagnostic.kind': kind,
                                                       'local-inference.ds41.diagnostic.lock.sha256': lock}}}
        with mock.patch.object(driver, 'DIAGNOSTIC_KIND', kind):
            problems, _ = driver.check_container('toby', container, image,
                                                 {'image_id': image_id, 'lock_sha256': lock}, lock, profile)
        return problems

    def test_container_kind_gate(self):
        _, _, driver, _ = modules()
        standard = driver.profile_for('matched', 8192, None, ARM)
        env = {'DS41_DECISION_ROW_BLOCKS': '81389', 'DS41_PREFILL_THRESHOLD': '8192', 'DS41_NODE': 'toby',
               'B12X_DYNAMIC_DETERMINISTIC_OUTPUT': '1', 'B12X_DENSE_SPLITK_TURBO': '0',
               'DS41_KIT_SHA256': 'k', 'B12X_COMPILE_CACHE_DIR': '/cache/x', 'DS41_NCCL_ARM': ARM}
        self.assertEqual(self.container_problems('precision-capture', standard, env), [])
        for kind in ('indexer-capture', 'window-capture', 'decision-row-capture'):
            self.assertIn('the standard NCCL arm is approved for the precision-capture kind only',
                          self.container_problems(kind, standard, env))
        self.assertTrue(self.container_problems('precision-capture', standard,
                                                {k: v for k, v in env.items() if k != 'DS41_NCCL_ARM'}))

    def test_cli_requires_explicit_arm(self):
        contract, _, driver, _ = modules()
        with mock.patch.object(driver, 'BUILD'), mock.patch.object(driver, 'LOCK'), \
                mock.patch.object(driver, 'DIAGNOSTIC_KIND'), mock.patch.object(driver, 'phase_capture') as run:
            driver.main(['--kind', 'precision', '--mode', 'matched', '--chunk-rows', '4096', '--nccl-arm', ARM])
            self.assertEqual(driver.LOCK.name, 'ds41-precision.lock.json')
            self.assertEqual(driver.DIAGNOSTIC_KIND, 'precision-capture')
            self.assertEqual(run.call_args.args[0].nccl_arm, ARM)
        refused = (['--kind', 'precision', '--mode', 'matched', '--chunk-rows', '4096'],
                   ['--kind', 'precision', '--mode', 'matched', '--chunk-rows', '4096', '--nccl-arm', ARM,
                    '--nccl-geometry', contract.NCCL_GEOMETRY],
                   ['--kind', 'indexer', '--mode', 'matched', '--chunk-rows', '4096', '--nccl-arm', ARM],
                   ['--kind', 'window', '--mode', 'matched', '--chunk-rows', '4096', '--nccl-arm', ARM],
                   ['--kind', 'decision-row', '--mode', 'matched', '--chunk-rows', '4096', '--nccl-arm', ARM],
                   ['--kind', 'precision', '--nccl-arm', ARM])
        for argv in refused:
            with mock.patch.object(driver, 'phase_capture') as run, self.assertRaises(SystemExit), \
                    contextlib.redirect_stderr(io.StringIO()):
                driver.main(argv)
            run.assert_not_called()


class Start(unittest.TestCase):
    source = property(lambda self: (PATCHED / 'start_moe_repaired.py').read_text())

    def parse(self, argv):
        """Execute only the argument parser and its refusals (statements before the receipt stamp)."""
        tree = ast.parse(self.source)
        cut = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
                   and getattr(node.targets[0], 'id', None) == 'stamp')
        body = [n for n in tree.body[:cut] if not (isinstance(n, ast.ImportFrom) and n.module == 'runtime')]
        ns = {'__file__': str(PATCHED / 'start_moe_repaired.py')}
        with mock.patch.object(sys, 'argv', ['start_moe_repaired.py', *argv]), \
                contextlib.redirect_stderr(io.StringIO()):
            exec(compile(ast.Module(body=body, type_ignores=[]), 'start', 'exec'), ns)
        return ns['args']

    def test_arm_selection(self):
        base = ['--arm', 'precision', '--decision-row-blocks', '81389', '--prefill-threshold', '4096']
        self.assertEqual(self.parse(base + ['--nccl-arm', ARM]).nccl_arm, ARM)
        self.assertEqual(self.parse(base + ['--nccl-geometry', 'tree-simple-1ch']).nccl_geometry, 'tree-simple-1ch')
        for argv in (base, base + ['--nccl-arm', ARM, '--nccl-geometry', 'tree-simple-1ch'],
                     ['--arm', 'indexer', '--decision-row-blocks', '81389', '--prefill-threshold', '4096', '--nccl-arm', ARM],
                     ['--arm', 'window', '--decision-row-blocks', '81389', '--prefill-threshold', '4096', '--nccl-arm', ARM],
                     ['--arm', 'precision', '--decision-row-blocks', '81389', '--nccl-arm', ARM],
                     base + ['--nccl-arm', ARM, '--ops-trace']):
            with self.assertRaises(SystemExit, msg=argv):
                self.parse(argv)
        # Indexer and window keep their existing acceptance.
        self.parse(['--arm', 'indexer', '--decision-row-blocks', '81389', '--prefill-threshold', '4096',
                    '--nccl-geometry', 'tree-simple-1ch'])
        self.parse(['--arm', 'window', '--decision-row-blocks', '81389', '--prefill-threshold', '4096'])

    def test_launch_and_receipt_names(self):
        tree = ast.parse(self.source)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'launch_command')
        texts = {}
        for arm, geometry in ((ARM, None), (None, 'tree-simple-1ch')):
            ns = {'REMOTE': '/r', 'args': SimpleNamespace(arm='precision', decision_row_blocks='81389',
                  cuda_module_loading=None, prefill_threshold='8192', nccl_geometry=geometry, nccl_arm=arm)}
            exec(compile(ast.Module(body=[fn], type_ignores=[]), 'start', 'exec'), ns)
            texts[arm or geometry] = ns['launch_command']('dusty')
        self.assertIn(' DS41_NCCL_ARM=' + ARM + ' ', texts[ARM])
        self.assertNotIn('DS41_NCCL_GEOMETRY=', texts[ARM])
        self.assertNotIn('DS41_NCCL_ARM=', texts['tree-simple-1ch'])
        for text in texts.values():
            self.assertIn('-u DS41_NCCL_GEOMETRY -u DS41_NCCL_ARM', text)
        self.assertEqual(texts[ARM].replace(' DS41_NCCL_ARM=' + ARM, ''),
                         texts['tree-simple-1ch'].replace(' DS41_NCCL_GEOMETRY=tree-simple-1ch', ''))
        self.assertIn("f'{args.arm}-nccl-{args.nccl_arm}-chunk{args.prefill_threshold}-blocks81389-{stamp}'", self.source)


class Receipts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'receipts').mkdir()
        (self.root / 'receipts/precision-build-receipt.json').write_text(json.dumps({'image_id': 'p' * 64}))

    def tearDown(self):
        self.tmp.cleanup()

    def receipt(self, name, chunk, geometry=None, arm=None, image='p' * 64, verdict='within-conformance-envelope'):
        from audit_window_geometry import REMOTE, NODES
        path = self.root / 'receipts' / name
        path.mkdir()
        summary = {'remote_dir': REMOTE + '/receipts/' + name, 'nccl_geometry': geometry, 'chunk_rows': chunk,
                   'image_id': image, 'blocks': 81389, 'mode': 'matched', 'regime_source': 'pinned-045954Z',
                   'captures': {n: n + '.pt' for n in NODES}, 'capture_problems': {n: [] for n in NODES}}
        if arm is not None:
            summary['nccl_arm'] = arm
        (path / 'summary.json').write_text(json.dumps(summary))
        (path / 'window-captures.json').write_text(json.dumps({n: {'problems': [], 'container_file': n} for n in NODES}))
        (path / 'audit.json').write_text(json.dumps({'verdict': verdict}))
        return path

    def test_arms_never_interchange(self):
        pdn = modules()[3]
        from audit_window_geometry import load_receipt
        s8, s4 = self.receipt('s8', 8192, arm=ARM), self.receipt('s4', 4096, arm=ARM)
        t8, t4 = self.receipt('t8', 8192, geometry='tree-simple-1ch'), self.receipt('t4', 4096, geometry='tree-simple-1ch')
        l8 = self.receipt('l8', 8192)
        pair = pdn.load_grids(s8, s4, ARM, root=self.root)
        self.assertEqual([r['summary']['nccl_arm'] for r in pair], [ARM, ARM])
        self.assertEqual([r['path'].name for r in pdn.load_grids(t8, t4, 'tree-simple-1ch', root=self.root)], ['t8', 't4'])
        for left, right, arm in ((t8, t4, ARM), (l8, s4, ARM), (s8, s4, 'tree-simple-1ch'), (s8, t4, ARM),
                                 (t8, s4, 'tree-simple-1ch'), (s4, s8, ARM), (s8, s4, 'standard')):
            with self.assertRaises(ValueError, msg=(left.name, right.name, arm)):
                pdn.load_grids(left, right, arm, root=self.root)
        with self.assertRaises(ValueError):
            load_receipt(s8, geometry=True, root=self.root)

    def test_identity_and_verdict(self):
        pdn = modules()[3]
        for kwargs in ({'image': 'q' * 64}, {'verdict': 'outside-conformance-envelope'}):
            bad = self.receipt('bad-' + next(iter(kwargs)), 8192, arm=ARM, **kwargs)
            with self.assertRaises(ValueError):
                pdn.load_standard_receipt(bad, root=self.root)
        s8, s4 = self.receipt('s8', 8192, arm=ARM), self.receipt('s4', 4096, arm=ARM, image='q' * 64)
        with self.assertRaises(ValueError):
            pdn.load_grids(s8, s4, ARM, root=self.root)

    def test_runners_default_to_tree_and_name_outputs_by_arm(self):
        cross = (PATCHED / 'run_geometry_decision_cross.py').read_text()
        index = (PATCHED / 'run_indexer_comparison.py').read_text()
        for text in (cross, index):
            self.assertIn("p.add_argument('--nccl', choices=ARMS, default=GEOMETRY", text)
            self.assertIn('load_grids(a.left, a.right, a.nccl)', text)
        self.assertIn("stem = 'geometry-decision-cross' if a.nccl == GEOMETRY else 'standard-decision-cross'", cross)
        self.assertIn("if a.nccl == STANDARD_NCCL_ARM and a.kind != 'precision':", index)


class Regression(unittest.TestCase):
    def outcome(self, root, suite):
        result = subprocess.run([sys.executable, '-m', 'unittest', suite], cwd=root, capture_output=True,
                                text=True, timeout=600, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        lines = [l.split(' in ')[0] if l.startswith('Ran ') else l for l in result.stderr.splitlines()
                 if l.startswith(('Ran ', 'OK', 'FAILED', 'ERROR:', 'FAIL:'))]
        return result.returncode, lines

    def test_existing_suites_same_outcome(self):
        for suite in SUITES:
            with self.subTest(suite=suite):
                self.assertEqual(self.outcome(BASE, suite), self.outcome(PATCHED, suite))


if __name__ == '__main__':
    unittest.main()
