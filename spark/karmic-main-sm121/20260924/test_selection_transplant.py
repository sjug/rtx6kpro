"""Selection transplant: frozen derivation, mutation refusals, seeding, recorder gates, first-token
classification, and the unapplied patch (base or exactly-applied tree; anything else fails loudly).

    .venv-snapshot-cpu/bin/python -m unittest test_selection_transplant
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
import seed_selection_transplant as tp  # noqa: E402
import run_selection_transplant as rec  # noqa: E402

PATCH = KIT / 'ds41-selection-transplant.patch'
BASES = {'launch_contract.py': 'e3d87f62bd833885ea9feb3c38857428ebd66ee4df6bc75f9fbafca730a19839',
         'run_node.py': '2ec4d8b2cd6d8b292854d06c3733466b27cdb69c16f5d65d79d2662867998d99',
         'start_moe_repaired.py': 'fe0e86aedae6cb78d2bda194f3bf68f7f5496362de0cc23f5aca8534e3ce8b7a'}
PATCHED_HASHES = {'launch_contract.py': 'fc069df0b11b01796a69d0c98dc5603973d966a022007525e08ac291267ec447',
                  'run_node.py': '42a6005fe16f87434705d578550c303c25d036016ca36f17f2832b2ffbb58682',
                  'start_moe_repaired.py': 'e11733b30a5610cdac8b5511e26b8bf15ea9bab356ada2ab5e3a682018ff5b5c'}
CONTROL = KIT / 'receipts/ratio1-control-20260927T0122Z'
SUITES = ('test_launch_contract', 'test_precision_orchestration', 'test_start_diagnostic_profile',
          'test_chunking_contract', 'test_router_fixed_control', 'test_window_orchestration',
          'test_indexer_orchestration', 'test_distribute_diagnostic')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def sources(node='dusty'):
    passing = json.loads((KIT / tp.replay.SOURCE / f'{node}-selection.json').read_text())
    release = json.loads((KIT / tp.RELEASE_SNAPSHOT / f'{node}.json').read_text())
    return passing, release


class Derivation(unittest.TestCase):
    def test_real_receipts(self):
        namespace, files = tp.check()
        self.assertEqual(namespace, json.loads((KIT / 'ds41-precision-release.lock.json').read_text())['cache_fingerprint'])
        for node in tp.NODES:
            passing, release = sources(node)
            payload = json.loads(files[node]['bytes'])
            changed = {k for k in payload['records'] if payload['records'][k] != passing['records'][k]}
            self.assertEqual(len(changed), 275)
            self.assertEqual(tp.keys_digest(changed), tp.TRANSPLANT_KEYS_SHA256)
            self.assertTrue(all(payload['records'][k] == release['records'][k] for k in changed))
            self.assertEqual(set(payload['records']), set(passing['records']))
            self.assertEqual(payload['identity'], passing['identity'])

    def test_mutations_refused(self):
        passing, release = sources()
        payload, changed, _ = tp.transplant(passing, release)
        shared_same = sorted(k for k in set(passing['records']) & set(release['records'])
                             if passing['records'][k]['config'] == release['records'][k]['config'])

        def extra_config(p, r):
            r['records'][shared_same[0]] = dict(r['records'][shared_same[0]], config={'x': 1}, assignment={'x': 1})

        def assignment_only(p, r):
            r['records'][shared_same[0]] = dict(r['records'][shared_same[0]], assignment={'x': 1})

        def metadata(p, r):
            key = next(k for k in shared_same if p['records'][k] == r['records'][k])
            r['records'][key] = dict(r['records'][key], programs=[['x', 'y']])

        def identity(p, r):
            r['identity'] = dict(r['identity'], sm_count=1)

        def baseline(p, r):
            key = sorted(changed)[0]
            p['records'][key] = copy.deepcopy(r['records'][key])

        for mutate, message in ((extra_config, 'frozen 275'), (assignment_only, 'same key set'),
                                (metadata, 'metadata-only'), (identity, 'identities differ'), (baseline, 'frozen 275')):
            with self.subTest(mutate=mutate.__name__):
                p, r = copy.deepcopy(passing), copy.deepcopy(release)
                mutate(p, r)
                with self.assertRaisesRegex(RuntimeError, message):
                    tp.transplant(p, r)
        for key in (sorted(changed)[0], sorted(set(passing['records']) - changed)[0]):
            bad = copy.deepcopy(payload)
            bad['records'][key] = dict(bad['records'][key], coverage={'tampered': 1})
            with self.assertRaisesRegex(RuntimeError, 'differs from its source'):
                tp.verify(bad, passing, release, changed)
        bad = copy.deepcopy(payload)
        bad['records']['extra'] = {}
        with self.assertRaisesRegex(RuntimeError, 'key set'):
            tp.verify(bad, passing, release, changed)

    def test_references(self):
        refs = tp.references()
        tops = {n: sorted(r['top_logprobs'], key=lambda x: -x['logprob'])[:2] for n, r in refs.items()}
        self.assertEqual([t['token'] for t in tops['passing-anchor']], ['739', '510'])
        self.assertEqual([t['token'] for t in tops['bf16-ratio1-variant']], ['510', '739'])
        self.assertEqual([t['token'] for t in tops['release-control']], ['510', '739'])
        margin = lambda t: t[0]['logprob'] - t[1]['logprob']
        self.assertAlmostEqual(margin(tops['passing-anchor']), 0.125, places=5)
        self.assertAlmostEqual(margin(tops['bf16-ratio1-variant']), 4.125, places=5)
        self.assertAlmostEqual(margin(tops['release-control']), 3.375, places=5)

    def test_seed_writes_exact_bytes_and_refuses_existing_root(self):
        _, files = tp.check()
        with tempfile.TemporaryDirectory() as tmp:
            root = str(Path(tmp) / 'root')
            command = tp.replay.seed_command(root, 'ns', files['kirby']['sha256'])
            ok = subprocess.run(['bash', '-c', command], input=files['kirby']['bytes'], capture_output=True)
            self.assertEqual(ok.returncode, 0, ok.stderr)
            self.assertEqual((Path(root) / 'jit/ns' / tp.replay.SELECTION).read_bytes(), files['kirby']['bytes'])
            again = subprocess.run(['bash', '-c', command], input=files['kirby']['bytes'], capture_output=True)
            self.assertNotEqual(again.returncode, 0)


class Recorder(unittest.TestCase):
    def setUp(self):
        info = json.loads((CONTROL / 'dusty-identity-before.json').read_text())
        self.base = info[0] if isinstance(info, list) else info
        env = dict(item.split('=', 1) for item in self.base['Config']['Env'])
        env.update(rec.ARM_ENV)
        self.env = env
        self.kit = self.base['Config']['Labels']['local-inference.ds41.kit.sha256']
        self.pin = {'image_id': self.base['Image'].removeprefix('sha256:')}
        _, files = tp.check()
        self.seeded = {n: f['bytes'].decode() for n, f in files.items()}
        self.state = {n: {'env': dict(env), 'mount': tp.HOST_ROOT, 'selection': self.seeded[n],
                          'log': f'DS41-PRECISION-APPLIED rank={r} allow_bf16_reduced_precision_reduction=False before=True t\n'
                                 'b12x autotuning uses 32 temporary KV blocks before allocating 81389 serving blocks\n'
                                 'b12x ready gemm.block_fp8_linear: 503/503 ready, 0 measured, 168 cached, 0 compilations\n'}
                      for r, n in enumerate(tp.NODES)}
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
        if command.startswith('cat ' + tp.HOST_ROOT + '/jit/'):
            return s['selection']
        raise AssertionError(command)

    def identity(self):
        return rec.identity(self.ssh, self.pin, Path(self.tmp.name), 'before', self.seeded, self.kit)

    def test_pass_and_faults(self):
        self.assertEqual(set(self.identity()), set(tp.NODES))
        passing_only = json.loads((KIT / tp.replay.SOURCE / 'rusty-selection.json').read_text())
        faults = [('toby', 'mount', tp.replay.HOST_ROOT),
                  ('dusty', 'env', dict(self.env, DS41_SELECTION_REPLAY='standard8192-233036Z')),
                  ('kirby', 'env', dict(self.env, DS41_NCCL_ARM='standard-upstream')),
                  ('rusty', 'selection', json.dumps(passing_only)),
                  ('kirby', 'log', self.state['kirby']['log'].replace('0 measured', '1 measured')),
                  ('toby', 'log', self.state['toby']['log'].replace('81389 serving', '80022 serving'))]
        for node, field, value in faults:
            with self.subTest(node=node, field=field):
                saved = self.state[node][field]
                self.state[node][field] = value
                shutil.rmtree(self.tmp.name)
                os.mkdir(self.tmp.name)
                with self.assertRaises(RuntimeError):
                    self.identity()
                self.state[node][field] = saved


class Classify(unittest.TestCase):
    def trial(self, path):
        return json.loads((KIT / path).read_text())

    def test_each_reference_and_none(self):
        refs = tp.references()
        row = {'correct': False, 'cached_tokens': 0, 'first_token': {'margin_nats': 1.0}}
        for name, path in tp.REFERENCES.items():
            t = self.trial(path)
            t = {'response': t['response']}
            with self.subTest(name=name):
                self.assertEqual(rec.classify([row] * 3, [t] * 3, refs)['verdict'], 'token0-equals-' + name)
        other = copy.deepcopy({'response': self.trial(tp.REFERENCES['release-control'])['response']})
        other['response']['choices'][0]['logprobs']['content'][0]['logprob'] -= 0.0625
        self.assertEqual(rec.classify([row] * 3, [other] * 3, refs)['verdict'], 'token0-matches-none')
        mixed = [{'response': self.trial(tp.REFERENCES['passing-anchor'])['response']}, other, other]
        self.assertEqual(rec.classify([row] * 3, mixed, refs)['verdict'], 'token0-nonrepeatable')
        with self.assertRaises(RuntimeError):
            rec.classify([row] * 2, [other] * 2, refs)
        with self.assertRaises(RuntimeError):
            rec.classify([dict(row, cached_tokens=8)] * 3, [other] * 3, refs)


_tmp = BASE = PATCHED = None


def setUpModule():
    global _tmp, BASE, PATCHED
    live = {n: sha((KIT / n).read_bytes()) for n in BASES}
    if live == BASES:
        applied = False
    elif live == PATCHED_HASHES:
        applied = True
    else:
        raise RuntimeError(f'Live files are neither the reviewed base nor the exact transplant-patched tree: {live}')
    _tmp = tempfile.TemporaryDirectory(prefix='selection-transplant-')
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


def load(tree, module):
    import importlib.util
    sys.path.insert(0, str(tree))
    try:
        for name in ('launch_contract', 'runtime', 'diagnostic_overlay', 'run_node'):
            sys.modules.pop(name, None)
        spec = importlib.util.spec_from_file_location(module, tree / f'{module}.py')
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        return loaded
    finally:
        sys.path.remove(str(tree))


class Patch(unittest.TestCase):
    def test_renders_and_gates(self):
        c = load(PATCHED, 'launch_contract')
        base = load(BASE, 'launch_contract')
        replay_env = {'DS41_DECISION_ROW_BLOCKS': '81389', 'DS41_PREFILL_THRESHOLD': '8192',
                      'DS41_SELECTION_REPLAY': 'standard8192-233036Z'}
        with mock.patch.dict(os.environ, replay_env, clear=True):
            self.assertEqual(c.render('toby'), base.render('toby'))
        with mock.patch.dict(os.environ, dict(replay_env, DS41_SELECTION_REPLAY=tp.SELECTOR), clear=True):
            row = c.render('toby')
        with mock.patch.dict(os.environ, replay_env, clear=True):
            reference = base.render('toby')
        self.assertEqual(row['env'].pop('DS41_SELECTION_REPLAY'), tp.SELECTOR)
        reference['env'].pop('DS41_SELECTION_REPLAY')
        self.assertEqual(row, reference)
        self.assertEqual(c.SELECTION_ROOTS, {'standard8192-233036Z': c.REPLAY_CACHE_ROOT, tp.SELECTOR: c.TRANSPLANT_CACHE_ROOT})
        self.assertEqual('/home/jugs/' + c.TRANSPLANT_CACHE_ROOT, tp.HOST_ROOT)
        release = {'image_id': 'a' * 64, 'diagnostic': {'kind': 'precision-release-candidate', 'lock_sha256': c.PRECISION_RELEASE_LOCK}}
        env = dict(replay_env, DS41_SELECTION_REPLAY=tp.SELECTOR)
        c.validate_chunking_image(release, env)
        for bad_pin, bad_env in ((release, dict(env, DS41_SELECTION_REPLAY='transplant')),
                                 (release, dict(env, DS41_PREFILL_THRESHOLD='4096')),
                                 ({'image_id': 'a' * 64, 'diagnostic': {'kind': 'ratio1-bf16-diagnostic', 'lock_sha256': c.RATIO1_LOCK}}, env)):
            with self.assertRaises(ValueError):
                c.validate_chunking_image(bad_pin, bad_env)
        with mock.patch.dict(os.environ, dict(env, DS41_DECISION_ROW_BLOCKS='80022'), clear=True), self.assertRaises(ValueError):
            c.render('toby')

    def test_node_mounts(self):
        run_node = load(PATCHED, 'run_node')
        c = sys.modules['launch_contract']
        release = {'image_id': 'a' * 64, 'diagnostic': {'kind': 'precision-release-candidate', 'lock_sha256': c.PRECISION_RELEASE_LOCK}}
        for selector, root in c.SELECTION_ROOTS.items():
            env = {'DS41_DECISION_ROW_BLOCKS': '81389', 'DS41_PREFILL_THRESHOLD': '8192', 'DS41_SELECTION_REPLAY': selector}
            with mock.patch.dict(os.environ, env, clear=True):
                argv, _, cache = run_node.command('dusty', release, 'b' * 64)
            self.assertEqual(cache, Path.home() / root)
            self.assertIn(f'{cache}:/cache:rw', argv)
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(run_node.command('dusty', release, 'b' * 64)[2], Path.home() / '.cache/vllm-jj-ds41-tp4')

    def test_start(self):
        source = (PATCHED / 'start_moe_repaired.py').read_text()

        def parse(argv):
            tree = ast.parse(source)
            cut = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
                       and getattr(node.targets[0], 'id', None) == 'stamp')
            body = [n for n in tree.body[:cut] if not (isinstance(n, ast.ImportFrom) and n.module == 'runtime')]
            with mock.patch.object(sys, 'argv', ['s', *argv]), contextlib.redirect_stderr(io.StringIO()):
                ns = {'__file__': str(PATCHED / 'start_moe_repaired.py')}
                exec(compile(ast.Module(body=body, type_ignores=[]), 'start', 'exec'), ns)
        ok = ['--arm', 'selection-transplant', '--decision-row-blocks', '81389', '--prefill-threshold', '8192']
        parse(ok)
        for argv in (ok[:2], ok[:4], ok + ['--nccl-arm', 'standard-upstream'], ok[:5] + ['4096'],
                     ['--arm', 'selection-transplant', '--decision-row-blocks', '80022']):
            with self.assertRaises(SystemExit, msg=argv):
                parse(argv)
        fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'launch_command')
        texts = {}
        for arm in ('selection-transplant', 'selection-replay'):
            ns = {'REMOTE': '/r', 'args': SimpleNamespace(arm=arm, decision_row_blocks='81389', cuda_module_loading=None,
                                                          prefill_threshold='8192', nccl_geometry=None, nccl_arm=None)}
            exec(compile(ast.Module(body=[fn], type_ignores=[]), 'start', 'exec'), ns)
            texts[arm] = ns['launch_command']('dusty')
        self.assertEqual(texts['selection-transplant'].replace(tp.SELECTOR, 'standard8192-233036Z'), texts['selection-replay'])

    def test_scope_and_regression(self):
        changed = {p.name for p in PATCHED.iterdir() if p.is_file() and sha(p.read_bytes()) != sha((BASE / p.name).read_bytes())}
        self.assertEqual(changed, set(BASES))

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
