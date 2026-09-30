"""Proposal tests for precision-release-integration.patch (select, distribute, start, qualify, two boots).

Runs on a temporary copy of the kit with the patch applied; the real kit is never patched
or written. Release build receipts are fabricated inside the copy because nothing is built.

    .venv-snapshot-cpu/bin/python -m unittest precision_release_integration_test
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
PATCH = KIT / 'precision-release-integration.patch'
BASES = {'distribute_diagnostic.py': '0f9bfa8133c0b5119429c0f39e7a9661c4f53d496e014f474f7f7cd2b45f005a',
         'start_moe_repaired.py': 'a4749179e28a942ef34c9ca1d9ab02340fc7641d5b4c6fa30b8bebbf643c89d8',
         'qualify_engram_repair.py': '3c24d9c5e42049d4eb87e90806dfa52a09a8fbf0f1aa97343851502222fa683b',
         'qualify_dense_release.py': '28fbf4d083baa66140d7ca1b313922cb23d61b9bbf7542e024ab664444a2ef5b'}
NEW = {'precision_release.py', 'select_precision_release.py', 'precision_release_boots.py'}
PARENT_AMENDMENT = 'router-fence-20260926T005102Z-amendment.json'
EXTRAS = ('router-fence-20260926T045016Z', 'needle-sensitivity-20260926T050535Z/construction.json',
          'router-release-build-20260925T220934Z/BUILD-OK', 'router-release-build-20260925T220934Z/image-inspect.json')
SUITES = ('test_audit_window_geometry', 'test_indexer_orchestration', 'test_router_fixed_control',
          'test_launch_contract', 'test_precision_orchestration', 'test_window_orchestration',
          'test_select_router_candidate', 'test_start_diagnostic_profile', 'claude_test_run_decision_capture',
          'test_window_geometry', 'test_chunking_contract', 'claude_test_engram_failclosed',
          'claude_test_probe_chunking', 'claude_test_probe_needle_sensitivity', 'test_distribute_diagnostic',
          'test_qualify_engram_repair', 'test_ds41_precision', 'test_ds41_precision_release')
RELEASE_IMAGE = 'f' * 64
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
        if sha((KIT / name).read_bytes()) != expected:
            raise unittest.SkipTest(f'{name} changed since the proposal; regenerate the patch against it')
    _tmp = tempfile.TemporaryDirectory(prefix='precision-release-integration-')
    BASE, PATCHED = Path(_tmp.name) / 'base', Path(_tmp.name) / 'patched'
    copy_kit(BASE)
    copy_kit(PATCHED)
    subprocess.run(['patch', '-p1', '--batch', '--forward', '-s', '-d', str(PATCHED), '-i', str(PATCH)], check=True)
    sys.path.insert(0, str(PATCHED))


def tearDownModule():
    sys.path.remove(str(PATCHED))
    _tmp.cleanup()


def fabricate_build(root, image=RELEASE_IMAGE):
    """A release build receipt and gate result for the real release lock (nothing is built)."""
    digest = sha((root / 'ds41-precision-release.lock.json').read_bytes())
    directory = root / 'receipts/precision-release-build-test'
    directory.mkdir(exist_ok=True)
    (directory / 'BUILD-OK').write_text(image + '\n')
    (root / 'receipts/precision-release-build-receipt.json').write_text(json.dumps(
        {'directory': str(directory), 'image_id': image, 'lock_sha256': digest}))
    return digest


def release_module():
    import precision_release
    assert Path(precision_release.__file__).resolve().parent == PATCHED.resolve()
    return precision_release


def router_pin():
    return json.loads((PATCHED / PARENT_AMENDMENT).read_text())['candidate']


class Scope(unittest.TestCase):
    def test_only_declared_files(self):
        changed = {p.name for p in PATCHED.iterdir() if p.is_file() and (
            not (BASE / p.name).is_file() or sha(p.read_bytes()) != sha((BASE / p.name).read_bytes()))}
        self.assertEqual(changed, set(BASES) | NEW)
        runtime = json.loads((KIT / 'runtime-files.json').read_text())
        self.assertFalse(changed & set(runtime), 'no runtime-mounted file changes')
        for name in changed:
            compile((PATCHED / name).read_text(), name, 'exec')

    def test_existing_gate_calls_kept(self):
        text = (PATCHED / 'qualify_engram_repair.py').read_text()
        for needle in ("'probe_length_transition.py'", "'probe_interleaved_repeatability.py'", "'--cycles', '50'",
                       "len(records) != 1100", "'qualify_dense_release.py'", "'claude_gate_conversations.py'",
                       "'--lengths', '385', '--repeats', '3'"):
            self.assertEqual(text.count(needle), (BASE / 'qualify_engram_repair.py').read_text().count(needle), needle)
        dense = (PATCHED / 'qualify_dense_release.py').read_text()
        for needle in ("'262000,500000,599936'", "'--repeats', '3'], 'ORIGINAL-NEEDLE-PASS'", "(131072, 8), (524288, 4)"):
            self.assertIn(needle, dense)
        base_gates = (BASE / 'qualify_engram_repair.py').read_text().split('def run_gates')[1]
        self.assertEqual(text.split('def run_gates')[1], base_gates)


class Identity(unittest.TestCase):
    def setUp(self):
        self.digest = fabricate_build(PATCHED)
        self.rel = release_module()

    def test_candidate_from_exact_router_parent(self):
        parent = router_pin()
        self.assertTrue(self.rel.router_pin_matches(parent, PATCHED))
        pin = self.rel.candidate(parent, PATCHED)
        lock = json.loads((PATCHED / 'ds41-precision-release.lock.json').read_text())
        self.assertEqual(pin['image_id'], RELEASE_IMAGE)
        self.assertEqual(pin['diagnostic'], {'kind': 'precision-release-candidate', 'lock_sha256': self.digest,
                                             'vllm_tree': lock['trees']['vllm'], 'b12x_tree': lock['trees']['b12x']})
        self.assertEqual({k: v for k, v in pin.items() if k not in ('image_id', 'diagnostic')},
                         {k: v for k, v in parent.items() if k not in ('image_id', 'diagnostic')})

    def test_candidate_refusals(self):
        parent = router_pin()
        capture = json.loads((PATCHED / 'candidate.json').read_text())
        bad = [capture, dict(parent, diagnostic_overlay={}), dict(parent, image_id='0' * 64),
               dict(parent, diagnostic=dict(parent['diagnostic'], lock_sha256='0' * 64)),
               dict(parent, diagnostic=dict(parent['diagnostic'], vllm_tree='0' * 40))]
        for pin in bad:
            with self.assertRaises(RuntimeError):
                self.rel.candidate(pin, PATCHED)
        fabricate_build(PATCHED)
        (PATCHED / 'receipts/precision-release-build-test/BUILD-OK').write_text('e' * 64 + '\n')
        with self.assertRaisesRegex(RuntimeError, 'gated build'):
            self.rel.candidate(parent, PATCHED)

    def test_select_and_restore_round_trip(self):
        saved = (PATCHED / 'candidate.json').read_bytes()
        self.addCleanup((PATCHED / 'candidate.json').write_bytes, saved)
        import select_precision_release as select
        manifest = json.loads((PATCHED / 'runtime-files.json').read_text())
        parent = router_pin()
        payload = json.dumps(parent, sort_keys=True, indent=2) + '\n'
        (PATCHED / 'candidate.json').write_text(payload)
        manifest = dict(manifest, **{'candidate.json': sha(payload.encode())})
        pin, pin_payload, updated, labels = select.plan(parent, manifest, root=PATCHED)
        self.assertEqual(labels['local-inference.ds41.diagnostic.kind'], 'precision-release-candidate')
        self.assertEqual(labels['local-inference.ds41.release.base-image'], parent['image_id'])
        self.assertEqual({k for k in manifest if manifest[k] != updated[k]}, {'candidate.json'})
        amendment = {'prior_candidate': parent, 'prior_manifest': manifest, 'candidate': pin, 'manifest': updated}
        (PATCHED / 'candidate.json').write_text(pin_payload)
        restored, restored_payload, restored_manifest, restore_labels = select.plan(pin, updated, amendment, root=PATCHED)
        self.assertEqual((restored, restored_payload, restored_manifest), (parent, payload, manifest))
        self.assertEqual(restore_labels['local-inference.ds41.diagnostic.kind'], 'router-stage-release-candidate')
        for mutate in (lambda a: a.update(candidate=dict(a['candidate'], extra=1)),
                       lambda a: a.update(prior_candidate=json.loads((PATCHED / 'candidate.json').read_text())),
                       lambda a: a['prior_manifest'].update({'run_node.py': '0' * 64}),
                       lambda a: a.update(prior_candidate=dict(parent, recipe_sha256='0' * 64))):
            bad = copy.deepcopy(amendment)
            mutate(bad)
            with self.assertRaises(RuntimeError):
                select.plan(pin, updated, bad, root=PATCHED)
        (PATCHED / 'launch_contract.py').write_text((PATCHED / 'launch_contract.py').read_text() + '\n')
        try:
            with self.assertRaisesRegex(RuntimeError, 'Runtime file drift'):
                select.plan(parent, manifest, root=PATCHED)
        finally:
            shutil.copy2(BASE / 'launch_contract.py', PATCHED / 'launch_contract.py')

    def test_environment_and_markers(self):
        from launch_contract import render, NCCL_GEOMETRY_ENV
        with mock.patch.dict(os.environ, {}, clear=True):
            env = render('toby')['env'] | dict(self.rel.REQUIRED_ENV)
        self.assertEqual(self.rel.environment_problems(env), [])
        for key in self.rel.DIAGNOSTIC_ENV:
            self.assertTrue(self.rel.environment_problems(dict(env, **{key: '1'})), key)
        for key in NCCL_GEOMETRY_ENV:
            self.assertIn(key, self.rel.DIAGNOSTIC_ENV)
        for key in self.rel.REQUIRED_ENV:
            self.assertTrue(self.rel.environment_problems({k: v for k, v in env.items() if k != key}))
        self.assertTrue(self.rel.environment_problems(dict(env, B12X_DENSE_SPLITK_TURBO='1')))
        line = '(Worker pid=9) DS41-PRECISION-APPLIED rank={} allow_bf16_reduced_precision_reduction=False before=True torch=t pid=9\n'
        for rank, node in enumerate(self.rel.NODES):
            self.assertEqual(self.rel.marker_problems(line.format(rank), node), [])
            self.assertTrue(self.rel.marker_problems(line.format((rank + 1) % 4), node))
            self.assertTrue(self.rel.marker_problems(line.format(rank) * 2, node))
            self.assertTrue(self.rel.marker_problems('', node))

    def test_distribution_resolves_release_receipt(self):
        import distribute_diagnostic as distribute
        self.assertEqual(distribute.KINDS['precision-release'], 'ds41-precision-release')
        build, receipt = distribute.resolve('precision-release', PATCHED)
        self.assertEqual((build['image_id'], receipt.name), (RELEASE_IMAGE, 'precision-release-build-test'))


class Qualification(unittest.TestCase):
    """The qualification snapshot for the release kind, with node responses mocked."""

    def setUp(self):
        fabricate_build(PATCHED)
        rel = release_module()
        self.pin = rel.candidate(router_pin(), PATCHED)
        self.saved = {n: (PATCHED / n).read_bytes() for n in ('candidate.json', 'runtime-files.json')}
        payload = json.dumps(self.pin, sort_keys=True, indent=2) + '\n'
        (PATCHED / 'candidate.json').write_text(payload)
        manifest = json.loads((PATCHED / 'runtime-files.json').read_text())
        manifest['candidate.json'] = sha(payload.encode())
        (PATCHED / 'runtime-files.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
        from runtime import kit_digest
        from launch_contract import render
        self.kit = kit_digest()
        with mock.patch.dict(os.environ, {}, clear=True):
            self.envs = {n: render(n)['env'] | dict(rel.REQUIRED_ENV) for n in rel.NODES}
        self.markers = {n: f'DS41-PRECISION-APPLIED rank={i} allow_bf16_reduced_precision_reduction=False before=True x\n'
                        for i, n in enumerate(rel.NODES)}
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        for name, data in self.saved.items():
            (PATCHED / name).write_bytes(data)
        self.tmp.cleanup()

    def respond(self, argv, text=True):
        node, command = argv[3], argv[4]
        if command.startswith('podman inspect'):
            env = self.envs[node]
            return json.dumps([{'Id': node * 8, 'Image': 'sha256:' + self.pin['image_id'], 'RestartCount': 0,
                                'State': {'Running': True, 'StartedAt': 't0'}, 'Cmd': ['-c', 'x'],
                                'Config': {'Env': [f'{k}={v}' for k, v in env.items()], 'Cmd': ['-c', 'x'],
                                           'Labels': {'local-inference.ds41.kit.sha256': self.kit}}}])
        if 'DS41-PRECISION-APPLIED' in command:
            return self.markers[node]
        raise AssertionError(command)

    def snapshot(self):
        import qualify_engram_repair as qualify
        with mock.patch.object(qualify.subprocess, 'check_output', side_effect=self.respond), \
                mock.patch.dict(os.environ, {}, clear=True):
            return qualify.snapshot(Path(self.tmp.name), 'before', 'precision-release-candidate')

    def test_normal_release_boot_passes(self):
        result = self.snapshot()
        self.assertEqual(set(result), {'dusty', 'toby', 'rusty', 'kirby'})
        self.assertTrue((Path(self.tmp.name) / 'kirby-precision-marker-before.log').is_file())

    def test_diagnostic_or_marker_faults_fail(self):
        faults = [('env', 'rusty', {'DS41_NCCL_ARM': 'standard-upstream'}),
                  ('env', 'toby', {'DS41_DECISION_ROW_BLOCKS': '81389'}),
                  ('env', 'dusty', {'NCCL_PROTO': 'Simple'}),
                  ('env', 'kirby', {'DS41_PREFILL_THRESHOLD': '4096'}),
                  ('marker', 'toby', ''), ('marker', 'rusty', 'DS41-PRECISION-APPLIED rank=0 allow_bf16_reduced_precision_reduction=False x\n'),
                  ('marker', 'dusty', 2)]
        for kind, node, value in faults:
            with self.subTest(kind=kind, node=node, value=value):
                saved_env, saved_marker = dict(self.envs[node]), self.markers[node]
                if kind == 'env':
                    self.envs[node].update(value)
                else:
                    self.markers[node] = saved_marker * value if isinstance(value, int) else value
                with self.assertRaises(RuntimeError):
                    self.snapshot()
                self.envs[node], self.markers[node] = saved_env, saved_marker
                shutil.rmtree(self.tmp.name)
                os.mkdir(self.tmp.name)

    def test_wrong_build_fails(self):
        fabricate_build(PATCHED, image='e' * 64)
        with self.assertRaisesRegex(RuntimeError, 'gated build'):
            self.snapshot()

    def test_cli_and_dense_kind(self):
        text = (PATCHED / 'qualify_engram_repair.py').read_text()
        self.assertIn("'precision-release-candidate' if args.precision_release", text)
        self.assertIn("parser.error('Choose one candidate kind')", text)
        self.assertIn("'router-stage-release-candidate', 'precision-release-candidate')",
                      (PATCHED / 'qualify_dense_release.py').read_text())


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

    def test_normal_profile_only(self):
        self.assertEqual(self.parse(['--arm', 'precision-release']).arm, 'precision-release')
        for extra in (['--decision-row-blocks', '81389'], ['--prefill-threshold', '4096'],
                      ['--nccl-geometry', 'tree-simple-1ch'], ['--nccl-arm', 'standard-upstream'],
                      ['--ops-trace'], ['--cuda-module-loading', 'EAGER']):
            with self.assertRaises(SystemExit, msg=extra):
                self.parse(['--arm', 'precision-release', *extra])

    def test_launch_command_and_kind(self):
        tree = ast.parse(self.source)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'launch_command')
        texts = {}
        for arm in ('precision-release', 'router-fence'):
            ns = {'REMOTE': '/r', 'args': SimpleNamespace(arm=arm, decision_row_blocks=None, cuda_module_loading=None,
                                                          prefill_threshold=None, nccl_geometry=None, nccl_arm=None)}
            exec(compile(ast.Module(body=[fn], type_ignores=[]), 'start', 'exec'), ns)
            texts[arm] = ns['launch_command']('dusty')
        self.assertEqual(texts['precision-release'], texts['router-fence'])
        for item in ('B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1', 'B12X_DENSE_SPLITK_TURBO=0', '-u DS41_NCCL_ARM',
                     '-u DS41_DECISION_ROW_BLOCKS'):
            self.assertIn(item, texts['precision-release'])
        for item in ('DS41_PREFILL_THRESHOLD=', 'DS41_DECISION_ROW_BLOCKS=', 'DS41_NCCL_GEOMETRY=', 'DS41_NCCL_ARM='):
            self.assertNotIn(item, texts['precision-release'])
        self.assertIn("'precision-release': 'precision-release-candidate',", self.source)
        watch, marker, repro = (self.source.index("'watch-startup.py'"), self.source.index('PRECISION-RELEASE-MARKERS-PASS'),
                                self.source.index("repro = subprocess.run"))
        self.assertLess(watch, marker)
        self.assertLess(marker, repro)


class Boots(unittest.TestCase):
    def setUp(self):
        import precision_release_boots as boots
        self.boots = boots
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, name, *, prefix='', ids='a', started='t1', image='i' * 64, correct=True, contents=('739184, 482617',) * 3,
             stress=None):
        path = self.root / name
        (path / prefix / 'original-524k').mkdir(parents=True)
        for node in ('dusty', 'toby', 'rusty', 'kirby'):
            (path / f'{node}-identity-before.json').write_text(json.dumps([{
                'Id': ids + node, 'Image': 'sha256:' + image, 'State': {'StartedAt': started},
                'Config': {'Labels': {'local-inference.ds41.kit.sha256': 'k'}}}]))
        report = []
        for repeat, text in enumerate(contents):
            (path / prefix / 'original-524k' / f'{repeat}.json').write_text(json.dumps({'response': {'choices': [{
                'message': {'content': text}, 'finish_reason': 'stop', 'logprobs': {'content': [text]}}]}}))
            report.append({'repeat': repeat, 'elapsed_s': 180.0 + repeat, 'correct': correct,
                           'identical': text == contents[0], 'cached_tokens': 0})
        (path / prefix / 'original-524k/report.json').write_text(json.dumps(report))
        if stress is not None:
            (path / 'residual-stress').mkdir()
            (path / 'residual-stress/records.json').write_text(json.dumps(
                [{'length': n, 'elapsed_s': stress, 'server_seconds': stress / 2} for n in (128, 256) for _ in range(3)]))
        return path

    def test_two_independent_passing_boots(self):
        parent = self.make('parent', prefix='correctness', correct=False, stress=1.0)
        a = self.make('a', prefix='correctness', stress=1.1)
        b = self.make('b', ids='b', started='t2')
        result = self.boots.compare(a, b, parent)
        self.assertTrue(result['cross_boot_signature_equal'])
        self.assertEqual(result['parent_original_524k_correct'], [False] * 3)
        timing = self.boots.stress_timing(a, parent)
        self.assertAlmostEqual(timing['per_length'][128]['release_over_parent_elapsed'], 1.1)

    def test_refusals(self):
        parent = self.make('parent', prefix='correctness', correct=False)
        a = self.make('a', prefix='correctness')
        cases = {'same-container': dict(ids='a', started='t2'), 'same-start': dict(ids='b', started='t1'),
                 'wrong': dict(ids='b', started='t2', correct=False), 'image': dict(ids='b', started='t2', image='j' * 64),
                 'unstable': dict(ids='b', started='t2', contents=('739184, 482617', '739184, 482617', 'x'))}
        for name, kwargs in cases.items():
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                self.boots.compare(a, self.make(name, **kwargs), parent)

    def test_real_parent_receipt_shape(self):
        result = self.boots.boot(KIT / 'receipts/router-b2-full-qualification-20260926/gates')
        self.assertEqual(len(result['signatures']), 3)
        self.assertEqual(len(set(result['signatures'])), 1)
        self.assertFalse(any(r['correct'] for r in result['report']))


class Regression(unittest.TestCase):
    def outcome(self, root, suite):
        result = subprocess.run([sys.executable, '-m', 'unittest', suite], cwd=root, capture_output=True,
                                text=True, timeout=600, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        return result.returncode, [l.split(' in ')[0] if l.startswith('Ran ') else l for l in result.stderr.splitlines()
                                   if l.startswith(('Ran ', 'OK', 'FAILED', 'ERROR:', 'FAIL:'))]

    def test_existing_suites_same_outcome(self):
        for suite in SUITES:
            with self.subTest(suite=suite):
                self.assertEqual(self.outcome(BASE, suite), self.outcome(PATCHED, suite))


if __name__ == '__main__':
    unittest.main()
