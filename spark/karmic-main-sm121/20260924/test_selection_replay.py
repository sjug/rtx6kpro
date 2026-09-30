"""Selection replay kit: source checks, seeding, recorder gates, verdicts, and the unapplied patch.

The patch is applied only to a temporary copy of the kit; the real kit is never patched.

    .venv-snapshot-cpu/bin/python -m unittest test_selection_replay
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
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT))
import seed_selection_replay as seeding  # noqa: E402
import run_selection_replay as replay  # noqa: E402

PATCH = KIT / 'ds41-selection-replay.patch'
BASES = {'launch_contract.py': '87cd8a830a989eba8126549efbf55059288c74bf364f5c1301e588790180753d',
         'run_node.py': '838af0cd5ac8b3cf718bc892ba055542b01c785866d83b1ba4e313eedf79eea9',
         'start_moe_repaired.py': '9fe9c54abd8a44ee54003ba37abdf7d200b580be073a8c74889e456245eef0e1'}
RELEASE_IDENTITY = KIT / 'receipts/ratio1-control-20260927T0122Z/dusty-identity-before.json'
# Suites that import the patched files and run inside a kit copy; each must pass on the base copy,
# so equal outcomes cannot hide a suite that never ran.
SUITES = ('test_launch_contract', 'test_precision_orchestration', 'test_start_diagnostic_profile',
          'test_chunking_contract', 'test_router_fixed_control', 'test_window_orchestration',
          'test_indexer_orchestration', 'test_distribute_diagnostic')


def sha(data):
    return hashlib.sha256(data).hexdigest()


class Source(unittest.TestCase):
    def test_real_source_is_the_passing_arm(self):
        namespace, files = seeding.check()
        self.assertEqual(namespace, json.loads((KIT / 'ds41-precision-release.lock.json').read_text())['cache_fingerprint'])
        self.assertEqual(set(files), set(seeding.NODES))
        self.assertEqual({f['records'] for f in files.values()}, {1162})
        for node, f in files.items():
            self.assertEqual(f['sha256'], sha((KIT / seeding.SOURCE / f'{node}-selection.json').read_bytes()))

    def test_source_mutations_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / seeding.SOURCE).mkdir(parents=True)
            for name in ('summary.json', 'response-row.json', *[f'{n}-selection.json' for n in seeding.NODES]):
                shutil.copy2(KIT / seeding.SOURCE / name, root / seeding.SOURCE / name)
            for subdir in ('unarmed-1', 'unarmed-2', '.'):
                target = root / seeding.SOURCE / subdir
                target.mkdir(exist_ok=True)
                for name in ('trial.json', 'trial-cache.json'):
                    shutil.copy2(KIT / seeding.SOURCE / subdir / name, target / name)
            (root / Path(seeding.RELEASE_IDENTITY).parent).mkdir(parents=True)
            shutil.copy2(KIT / seeding.RELEASE_IDENTITY, root / seeding.RELEASE_IDENTITY)
            shutil.copy2(KIT / 'ds41-precision-release.lock.json', root / 'ds41-precision-release.lock.json')
            seeding.check(root)
            row = json.loads((root / seeding.SOURCE / 'response-row.json').read_text())
            (root / seeding.SOURCE / 'response-row.json').write_text(json.dumps(dict(row, response_signature='0' * 64)))
            with self.assertRaisesRegex(RuntimeError, 'not the recorded passing'):
                seeding.check(root)
            (root / seeding.SOURCE / 'response-row.json').write_text(json.dumps(row))
            payload = json.loads((root / seeding.SOURCE / 'kirby-selection.json').read_text())
            payload['identity'] = dict(payload['identity'], sm_count=1)
            (root / seeding.SOURCE / 'kirby-selection.json').write_text(json.dumps(payload))
            with self.assertRaisesRegex(RuntimeError, 'identity differs'):
                seeding.check(root)

    def test_anchor_response_and_cache_are_recomputed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for subdir in ('unarmed-1', 'unarmed-2', '.'):
                target = root / subdir
                target.mkdir(exist_ok=True)
                for name in ('trial.json', 'trial-cache.json'):
                    shutil.copy2(KIT / seeding.SOURCE / subdir / name, target / name)
            seeding.verify_anchor(root)
            path = root / 'trial.json'
            original = path.read_bytes()
            trial = json.loads(original)
            trial['response']['choices'][0]['logprobs']['content'][0]['logprob'] -= 0.5
            path.write_text(json.dumps(trial))
            with self.assertRaisesRegex(RuntimeError, 'anchor response'):
                seeding.verify_anchor(root)
            path.write_bytes(original)
            path = root / 'trial-cache.json'
            data = json.loads(path.read_text())
            data['after_samples'][-1]['hits'] += 1
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(RuntimeError, 'cold accounting'):
                seeding.verify_anchor(root)


class Seeding(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = str(Path(self.tmp.name) / 'replay-root')
        self.data = (KIT / seeding.SOURCE / 'toby-selection.json').read_bytes()

    def tearDown(self):
        self.tmp.cleanup()

    def run_seed(self, data=None, expected=None):
        command = seeding.seed_command(self.root, 'ns', expected or sha(self.data))
        return subprocess.run(['bash', '-c', command], input=data if data is not None else self.data,
                              capture_output=True)

    def test_exact_bytes_marker_and_nothing_else(self):
        result = self.run_seed()
        self.assertEqual(result.returncode, 0, result.stderr)
        target = Path(self.root) / 'jit/ns' / seeding.SELECTION
        self.assertEqual(target.read_bytes(), self.data)
        marker = json.loads((Path(self.root) / seeding.MARKER).read_text())
        self.assertEqual(marker['selection_sha256'], sha(self.data))
        files = sorted(str(p.relative_to(self.root)) for p in Path(self.root).rglob('*') if p.is_file())
        self.assertEqual(files, sorted([seeding.MARKER, 'jit/ns/' + seeding.SELECTION]))
        self.assertFalse(Path(self.root + '.seeding').exists())

    def test_refusals(self):
        self.assertNotEqual(self.run_seed(data=self.data + b' ').returncode, 0)
        self.assertFalse(Path(self.root).exists())
        shutil.rmtree(self.root + '.seeding')
        Path(self.root).mkdir()
        self.assertNotEqual(self.run_seed().returncode, 0)

    def test_seed_driver_sends_each_rank_its_own_file(self):
        sent = {}

        def run(node, command, data):
            sent[node] = (command, data)
            return f'SEEDED {seeding.HOST_ROOT}\n'
        out = Path(self.tmp.name) / 'receipt'
        seeding.seed(out, run)
        for node in seeding.NODES:
            self.assertEqual(sent[node][1], (KIT / seeding.SOURCE / f'{node}-selection.json').read_bytes())
            self.assertIn(sha(sent[node][1]), sent[node][0])
        self.assertTrue((out / 'seeded.json').is_file())

    def test_roce_digest_refuses_missing_and_empty_directory(self):
        cache = Path(self.tmp.name) / 'roce'
        script = shlex.split(seeding.roce_command('image'))[-1].replace('/opt/b12x-roce-cache', str(cache))
        def run():
            return subprocess.run(['bash', '-c', script], capture_output=True, text=True)
        self.assertNotEqual(run().returncode, 0)
        cache.mkdir()
        self.assertNotEqual(run().returncode, 0)
        (cache / 'object.so').write_bytes(b'fixture')
        result = run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(result.stdout.strip(), r'^[0-9a-f]{64}$')


class Recorder(unittest.TestCase):
    def setUp(self):
        info = json.loads(RELEASE_IDENTITY.read_text())
        info = info[0] if isinstance(info, list) else info
        env = dict(item.split('=', 1) for item in info['Config']['Env'])
        env.update(replay.REPLAY_ENV)
        self.base = info
        self.env = env
        self.pin = {'image_id': info['Image'].removeprefix('sha256:')}
        self.kit = info['Config']['Labels']['local-inference.ds41.kit.sha256']
        _, files = seeding.check()
        self.seeded = {n: f['bytes'].decode() for n, f in files.items()}
        self.state = {n: {'env': dict(env), 'mount': seeding.HOST_ROOT, 'selection': self.seeded[n],
                          'log': f'(Worker pid=1) DS41-PRECISION-APPLIED rank={r} allow_bf16_reduced_precision_reduction=False before=True t\n'
                                 'b12x ready gemm.block_fp8_linear: 503/503 ready, 0 measured, 168 cached, 0 compilations, 0:02\n'
                                 'b12x autotuning uses 32 temporary KV blocks before allocating 81389 serving blocks\n'}
                      for r, n in enumerate(seeding.NODES)}
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def ssh(self, node, command):
        s = self.state[node]
        if command.startswith('podman inspect'):
            info = copy.deepcopy(self.base)
            info['Config']['Env'] = [f'{k}={v}' for k, v in s['env'].items()]
            info['Mounts'] = [{'Destination': '/cache', 'Source': s['mount']}]
            return json.dumps([info])
        if command.startswith('podman logs'):
            return s['log']
        if command.startswith('cat '):
            self.assertTrue(command.startswith('cat ' + seeding.HOST_ROOT + '/jit/'))
            return s['selection']
        raise AssertionError(command)

    def identity(self):
        return replay.identity(self.ssh, self.pin, Path(self.tmp.name), 'before', self.seeded, self.kit)

    def test_pass(self):
        self.assertEqual(set(self.identity()), set(seeding.NODES))

    def test_wrong_kit_refused(self):
        self.kit = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'container kit'):
            self.identity()

    def test_faults(self):
        added = json.loads(self.seeded['rusty'])
        added['records']['new'] = {'assignment': {}, 'config': {}, 'coverage': {}, 'programs': []}
        faults = [('toby', 'mount', '/home/jugs/.cache/vllm-jj-ds41-tp4'),
                  ('rusty', 'selection', json.dumps(added)),
                  ('kirby', 'log', self.state['kirby']['log'].replace('0 measured', '3 measured')),
                  ('toby', 'log', self.state['toby']['log'].replace('81389 serving', '80022 serving')),
                  ('dusty', 'log', self.state['dusty']['log'].replace('b12x ready', 'other ready')),
                  ('kirby', 'log', self.state['kirby']['log'] + 'b12x measuring x: 1/1 ready, 2 measured, 0 cached\n'),
                  ('dusty', 'log', self.state['dusty']['log'] + 'DS41-RATIO1-EXTEND-PIN layer=x mode=extend rows=8192 v41_compute_mode=bf16\n'),
                  ('toby', 'log', 'b12x ready x: 1/1 ready, 0 measured, 1 cached\n'),
                  ('rusty', 'env', dict(self.env, DS41_PREFILL_THRESHOLD='4096')),
                  ('kirby', 'env', dict(self.env, DS41_NCCL_ARM='standard-upstream')),
                  ('dusty', 'env', {k: v for k, v in self.env.items() if k != 'DS41_SELECTION_REPLAY'})]
        for node, field, value in faults:
            with self.subTest(node=node, field=field):
                saved = copy.deepcopy(self.state[node][field])
                self.state[node][field] = value
                shutil.rmtree(self.tmp.name)
                os.mkdir(self.tmp.name)
                with self.assertRaises(RuntimeError):
                    self.identity()
                self.state[node][field] = saved

    def test_classify(self):
        passing = json.loads((KIT / seeding.SOURCE / 'unarmed-1/trial.json').read_text())
        failing = json.loads((KIT / 'receipts/ratio1-control-20260927T0122Z/original-524k/0.json').read_text())
        row = {'correct': True, 'cached_tokens': 0, 'first_token': {'margin_nats': 0.125}}
        self.assertEqual(replay.classify([row] * 3, [passing] * 3)['verdict'], 'identical-to-passing')
        first = passing['response']['choices'][0]['logprobs']['content'][0]
        self.assertEqual(replay.classify([row] * 3, [passing] * 3, anchor_first=first)['first_token_matches_anchor'], [True] * 3)
        other = copy.deepcopy(passing)
        other['response']['choices'][0]['logprobs']['content'][0]['logprob'] -= 0.5
        self.assertEqual(replay.classify([row] * 3, [other] * 3)['verdict'], 'correct-different')
        self.assertEqual(replay.classify([row] * 3, [other] * 3, anchor_first=first)['first_token_matches_anchor'], [False] * 3)
        self.assertEqual(replay.classify([row] * 3, [other, passing, passing])['verdict'], 'correct-nonrepeatable')
        with self.assertRaises(RuntimeError):
            replay.classify([row] * 3, [passing])
        self.assertEqual(replay.classify([dict(row, correct=False)] * 3, [failing] * 3)['verdict'], 'wrong')
        with self.assertRaises(RuntimeError):
            replay.classify([dict(row, cached_tokens=1024)] * 3, [passing] * 3)


_tmp = BASE = PATCHED = None


def setUpModule():
    global _tmp, BASE, PATCHED
    already_applied = any(sha((KIT / name).read_bytes()) != expected for name, expected in BASES.items())
    _tmp = tempfile.TemporaryDirectory(prefix='selection-replay-')
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
        for extra in ('router-fence-20260926T045016Z', 'precision-release-full-qualification-20260927-r2/initial-selections',
                      Path(json.loads((KIT / 'receipts/precision-release-build-receipt.json').read_text())['directory']).name,
                      'router-release-build-20260925T220934Z'):
            source = KIT / 'receipts' / extra
            if source.is_dir():
                shutil.copytree(source, target / 'receipts' / extra)
    if already_applied:
        subprocess.run(['patch', '-p1', '--batch', '--reverse', '-s', '-d', str(BASE), '-i', str(PATCH)], check=True)
    else:
        subprocess.run(['patch', '-p1', '--batch', '--forward', '-s', '-d', str(PATCHED), '-i', str(PATCH)], check=True)
    for name, expected in BASES.items():
        if sha((BASE / name).read_bytes()) != expected:
            raise RuntimeError(f'{name} is neither the reviewed base nor its exact replay patch')


def tearDownModule():
    _tmp.cleanup()


def patched(module):
    """Import a module from the patched copy under a private name."""
    import importlib.util
    sys.path.insert(0, str(PATCHED))
    try:
        for name in ('launch_contract', 'runtime', 'diagnostic_overlay', 'run_node'):
            sys.modules.pop(name, None)
        spec = importlib.util.spec_from_file_location(module, PATCHED / f'{module}.py')
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        return loaded
    finally:
        sys.path.remove(str(PATCHED))


class Patch(unittest.TestCase):
    def test_scope(self):
        changed = {p.name for p in PATCHED.iterdir() if p.is_file() and sha(p.read_bytes()) != sha((BASE / p.name).read_bytes())}
        self.assertEqual(changed, set(BASES))

    def test_contract(self):
        c = patched('launch_contract')
        release = {'image_id': 'a' * 64, 'diagnostic': {'kind': 'precision-release-candidate', 'lock_sha256': c.PRECISION_RELEASE_LOCK}}
        env = {'DS41_DECISION_ROW_BLOCKS': '81389', 'DS41_PREFILL_THRESHOLD': '8192', 'DS41_SELECTION_REPLAY': c.REPLAY_SELECTOR}
        with mock.patch.dict(os.environ, env, clear=True):
            row = c.render('toby')
        with mock.patch.dict(os.environ, {'DS41_DECISION_ROW_BLOCKS': '81389', 'DS41_PREFILL_THRESHOLD': '8192'}, clear=True):
            matched = c.render('toby')
        self.assertEqual(row['env'].pop('DS41_SELECTION_REPLAY'), c.REPLAY_SELECTOR)
        self.assertEqual((row['env'], row['model'], row['unset']), (matched['env'], matched['model'], matched['unset']))
        self.assertIn('--long-prefill-token-threshold', row['model'])
        c.validate_chunking_image(release, env)
        c.validate_chunking_image(release, {})
        for bad_pin, bad_env in ((release, {k: v for k, v in env.items() if k != 'DS41_SELECTION_REPLAY'}),
                                 (release, dict(env, DS41_PREFILL_THRESHOLD='4096')),
                                 (release, dict(env, DS41_NCCL_ARM='standard-upstream')),
                                 (release, dict(env, DS41_SELECTION_REPLAY='other')),
                                 (dict(release, diagnostic_overlay={}), env),
                                 ({'image_id': 'a' * 64, 'diagnostic': {'kind': 'ratio1-bf16-diagnostic', 'lock_sha256': c.RATIO1_LOCK}}, env)):
            with self.assertRaises(ValueError):
                c.validate_chunking_image(bad_pin, bad_env)
        for bad in (dict(env, DS41_DECISION_ROW_BLOCKS='80022'), dict(env, DS41_PREFILL_THRESHOLD='4096'),
                    {'DS41_SELECTION_REPLAY': c.REPLAY_SELECTOR}):
            with mock.patch.dict(os.environ, bad, clear=True), self.assertRaises(ValueError):
                c.render('toby')

    def test_node_mount(self):
        run_node = patched('run_node')
        c = sys.modules['launch_contract']
        release = {'image_id': 'a' * 64, 'diagnostic': {'kind': 'precision-release-candidate', 'lock_sha256': c.PRECISION_RELEASE_LOCK}}
        env = {'DS41_DECISION_ROW_BLOCKS': '81389', 'DS41_PREFILL_THRESHOLD': '8192', 'DS41_SELECTION_REPLAY': c.REPLAY_SELECTOR}
        with mock.patch.dict(os.environ, env, clear=True):
            argv, _, cache = run_node.command('dusty', release, 'b' * 64)
        self.assertEqual(cache, Path.home() / c.REPLAY_CACHE_ROOT)
        self.assertIn(f'{cache}:/cache:rw', argv)
        self.assertEqual(str(Path.home() / c.REPLAY_CACHE_ROOT), seeding.HOST_ROOT.replace('/home/jugs', str(Path.home())))
        with mock.patch.dict(os.environ, {}, clear=True):
            _, _, normal = run_node.command('dusty', release, 'b' * 64)
        self.assertEqual(normal, Path.home() / '.cache/vllm-jj-ds41-tp4')
        self.assertIn("require((cache / 'SELECTION-REPLAY-SEEDED.json').is_file()", (PATCHED / 'run_node.py').read_text())

    def parse(self, argv):
        source = (PATCHED / 'start_moe_repaired.py').read_text()
        tree = ast.parse(source)
        cut = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
                   and getattr(node.targets[0], 'id', None) == 'stamp')
        body = [n for n in tree.body[:cut] if not (isinstance(n, ast.ImportFrom) and n.module == 'runtime')]
        with mock.patch.object(sys, 'argv', ['start_moe_repaired.py', *argv]), contextlib.redirect_stderr(io.StringIO()):
            ns = {'__file__': str(PATCHED / 'start_moe_repaired.py')}
            exec(compile(ast.Module(body=body, type_ignores=[]), 'start', 'exec'), ns)
        return ns['args']

    def test_start(self):
        ok = ['--arm', 'selection-replay', '--decision-row-blocks', '81389', '--prefill-threshold', '8192']
        self.parse(ok)
        self.parse(['--arm', 'precision-release'])
        for argv in (['--arm', 'selection-replay'], ok[:4], ok[:2] + ['--decision-row-blocks', '80022', '--prefill-threshold', '8192'],
                     ok + ['--nccl-arm', 'standard-upstream'], ok[:5] + ['4096'], ok + ['--ops-trace'],
                     ['--arm', 'precision-release', '--decision-row-blocks', '81389', '--prefill-threshold', '8192']):
            with self.assertRaises(SystemExit, msg=argv):
                self.parse(argv)
        source = (PATCHED / 'start_moe_repaired.py').read_text()
        fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'launch_command')
        ns = {'REMOTE': '/r', 'args': SimpleNamespace(arm='selection-replay', decision_row_blocks='81389', cuda_module_loading=None,
                                                      prefill_threshold='8192', nccl_geometry=None, nccl_arm=None)}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), 'start', 'exec'), ns)
        text = ns['launch_command']('dusty')
        for item in ('DS41_SELECTION_REPLAY=standard8192-233036Z', 'DS41_PREFILL_THRESHOLD=8192', 'DS41_DECISION_ROW_BLOCKS=81389',
                     'B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1', 'B12X_DENSE_SPLITK_TURBO=0', '-u DS41_SELECTION_REPLAY'):
            self.assertIn(item, text)
        self.assertNotIn('DS41_NCCL_ARM=', text)
        self.assertIn("'selection-replay': 'precision-release-candidate',", source)

    def test_existing_suites_same_outcome(self):
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


if __name__ == '__main__':
    unittest.main()
