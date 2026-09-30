"""Contingent two-mHC-8192 transplant: unapplied patch over the live 275 kit (runtime + helpers).

Proves on a temporary copy: the 275 set's seed bytes, renders and launch line are unchanged; the new
set differs from the passing file in exactly the two rebuilt mHC keys, each equal to the release
record, with every other record (capacity records included) byte-equal to passing; the selector is
reachable only through the selection-transplant arm; and existing suites keep their outcomes.
Live files must be the reviewed base or the exact patched tree; anything else fails loudly.

    .venv-snapshot-cpu/bin/python -m unittest test_selection_mhc8192
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
PATCH = KIT / 'ds41-selection-mhc8192.patch'
BASES = {'launch_contract.py': 'fc069df0b11b01796a69d0c98dc5603973d966a022007525e08ac291267ec447',
         'start_moe_repaired.py': 'e11733b30a5610cdac8b5511e26b8bf15ea9bab356ada2ab5e3a682018ff5b5c',
         'seed_selection_transplant.py': '7a17e964539d3eec0967745065cf3f717ad9d93f7cc8396d2879218eceebfc4f',
         'run_selection_transplant.py': 'e9b0933caacedf99a72c3df64c25ba113d8d1e7b05259be7f8abb5c8c5279855'}
PATCHED_HASHES = {'launch_contract.py': '0b432bda7d26336fc5cb7abc3310bf4abcbbbf41dc0938f99e91e36817336887',
                  'start_moe_repaired.py': '6956fb95c6e3779d21d5211f76a20a777350b9133a85f89e429354f2ace6c93b',
                  'seed_selection_transplant.py': '5d1e002efc26629718b9c1530ef79c75fcbb917e63b40c7b807b298ca1fae554',
                  'run_selection_transplant.py': '3c8610d0d1ca887e98fc93cfbcbc85b92af3b206821390a07f115a99bf6a1c03'}
MHC_KEYS = {'2394c70075c7fe2b87a1bd135e1ba48faaf75dfbb048b7f683b8df73d6de75c3',
            '1b6c36a14eaadabaa5b558f21ee4755eb16d6f5c5bc7c1f8ff4daae6b64f10a6'}
SUITES = ('test_launch_contract', 'test_precision_orchestration', 'test_start_diagnostic_profile',
          'test_chunking_contract', 'test_router_fixed_control', 'test_window_orchestration',
          'test_indexer_orchestration', 'test_distribute_diagnostic')
_tmp = BASE = PATCHED = None


def sha(data):
    return hashlib.sha256(data).hexdigest()


def setUpModule():
    global _tmp, BASE, PATCHED
    live = {n: sha((KIT / n).read_bytes()) for n in BASES}
    if live == BASES:
        applied = False
    elif live == PATCHED_HASHES:
        applied = True
    else:
        raise RuntimeError(f'Live files are neither the reviewed base nor the exact mHC-patched tree: {live}')
    _tmp = tempfile.TemporaryDirectory(prefix='selection-mhc8192-')
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
        for extra in ('router-fence-20260926T045016Z', 'router-release-build-20260925T220934Z',
                      'decision-row-matched8192-nccl-standard-upstream-capture-20260926T233036Z',
                      'ratio1-namespace-snapshot-20260927T010531Z', 'ratio1-control-20260927T0122Z',
                      'ratio1-variant-20260927T0141Z', 'precision-release-full-qualification-20260927-r2/initial-selections',
                      Path(json.loads((KIT / 'receipts/precision-release-build-receipt.json').read_text())['directory']).name):
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


def run_in(tree, code):
    """Evaluate code in a fresh interpreter rooted at one tree; returns its JSON output."""
    result = subprocess.run([sys.executable, '-c', code], cwd=tree, capture_output=True, text=True, timeout=600,
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    if result.returncode:
        raise AssertionError(result.stderr[-2000:])
    return json.loads(result.stdout.strip().splitlines()[-1])


SEED_DIGESTS = ("import json, seed_selection_transplant as t\n"
                "ns, files = t.check({args})\n"
                "print(json.dumps({{n: f['sha256'] for n, f in files.items()}}))")


class Sets(unittest.TestCase):
    def test_275_seed_bytes_unchanged(self):
        self.assertEqual(run_in(BASE, SEED_DIGESTS.format(args='')), run_in(PATCHED, SEED_DIGESTS.format(args='')))

    def test_mhc_set_exact(self):
        code = ("import json, seed_selection_transplant as t\n"
                "ns, files = t.check(name='mhc8192')\n"
                "out = {}\n"
                "for n, f in files.items():\n"
                "    payload = json.loads(f['bytes'])\n"
                "    p = json.load(open(t.replay.SOURCE + '/' + n + '-selection.json'))\n"
                "    r = json.load(open(t.RELEASE_SNAPSHOT + '/' + n + '.json'))\n"
                "    diff = sorted(k for k in p['records'] if payload['records'][k] != p['records'][k])\n"
                "    out[n] = {'diff': diff, 'release_equal': all(payload['records'][k] == r['records'][k] for k in diff),\n"
                "              'keys_equal': set(payload['records']) == set(p['records']), 'identity': payload['identity'] == p['identity']}\n"
                "print(json.dumps(out))")
        result = run_in(PATCHED, code)
        for node, row in result.items():
            self.assertEqual(set(row['diff']), MHC_KEYS, node)
            self.assertTrue(row['release_equal'] and row['keys_equal'] and row['identity'], node)

    def test_refusals(self):
        code = ("import json, copy, seed_selection_transplant as t\n"
                "p = json.load(open(t.replay.SOURCE + '/dusty-selection.json'))\n"
                "r = json.load(open(t.RELEASE_SNAPSHOT + '/dusty.json'))\n"
                "out = {}\n"
                "def attempt(label, fn):\n"
                "    try:\n"
                "        fn(); out[label] = 'accepted'\n"
                "    except Exception as e:\n"
                "        out[label] = type(e).__name__ + ': ' + str(e)[:80]\n"
                "attempt('unknown', lambda: t.select(p, r, 'mhc'))\n"
                "key = t.MHC8192_KEYS['mhc.pre.8192']\n"
                "p2 = copy.deepcopy(p); p2['records'][key] = copy.deepcopy(r['records'][key])\n"
                "attempt('baseline', lambda: t.select(p2, r, 'mhc8192'))\n"
                "t.MHC8192_KEYS = dict(t.MHC8192_KEYS, **{'mhc.pre.8192': '0' * 64})\n"
                "attempt('rebuild', lambda: t.select(p, r, 'mhc8192'))\n"
                "print(json.dumps(out))")
        result = run_in(PATCHED, code)
        self.assertTrue(result['unknown'].startswith('ValueError'))
        self.assertIn('frozen 275', result['baseline'])
        self.assertIn('no longer rebuild', result['rebuild'])


def load(tree, module):
    import importlib.util
    sys.path.insert(0, str(tree))
    try:
        for name in ('launch_contract', 'runtime', 'diagnostic_overlay', 'run_node', 'seed_selection_replay',
                     'seed_selection_transplant', 'run_selection_transplant', 'run_selection_replay'):
            sys.modules.pop(name, None)
        spec = importlib.util.spec_from_file_location(module, tree / f'{module}.py')
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        return loaded
    finally:
        sys.path.remove(str(tree))


class Runtime(unittest.TestCase):
    def test_contract_and_mounts(self):
        c, base = load(PATCHED, 'launch_contract'), load(BASE, 'launch_contract')
        geometry = {'DS41_DECISION_ROW_BLOCKS': '81389', 'DS41_PREFILL_THRESHOLD': '8192'}
        for selector in base.SELECTION_ROOTS:
            with mock.patch.dict(os.environ, dict(geometry, DS41_SELECTION_REPLAY=selector), clear=True):
                self.assertEqual(c.render('toby'), base.render('toby'))
        with mock.patch.dict(os.environ, dict(geometry, DS41_SELECTION_REPLAY=c.MHC8192_SELECTOR), clear=True):
            row = c.render('toby')
        with mock.patch.dict(os.environ, dict(geometry, DS41_SELECTION_REPLAY=c.TRANSPLANT_SELECTOR), clear=True):
            reference = c.render('toby')
        self.assertEqual(row['env'].pop('DS41_SELECTION_REPLAY'), c.MHC8192_SELECTOR)
        reference['env'].pop('DS41_SELECTION_REPLAY')
        self.assertEqual(row, reference)
        self.assertEqual({k: v for k, v in c.SELECTION_ROOTS.items() if k != c.MHC8192_SELECTOR}, base.SELECTION_ROOTS)
        release = {'image_id': 'a' * 64, 'diagnostic': {'kind': 'precision-release-candidate', 'lock_sha256': c.PRECISION_RELEASE_LOCK}}
        c.validate_chunking_image(release, dict(geometry, DS41_SELECTION_REPLAY=c.MHC8192_SELECTOR))
        with self.assertRaises(ValueError):
            c.validate_chunking_image(release, dict(geometry, DS41_SELECTION_REPLAY='transplant2-mhc'))
        run_node = load(PATCHED, 'run_node')
        with mock.patch.dict(os.environ, dict(geometry, DS41_SELECTION_REPLAY=c.MHC8192_SELECTOR), clear=True):
            argv, _, cache = run_node.command('dusty', release, 'b' * 64)
        self.assertEqual(cache, Path.home() / c.MHC8192_CACHE_ROOT)
        seed = load(PATCHED, 'seed_selection_transplant')
        self.assertEqual('/home/jugs/' + c.MHC8192_CACHE_ROOT, seed.SETS['mhc8192']['host_root'])
        self.assertEqual(c.MHC8192_SELECTOR, seed.SETS['mhc8192']['selector'])

    def test_start(self):
        def parse(tree, argv):
            source = (tree / 'start_moe_repaired.py').read_text()
            nodes = ast.parse(source).body
            cut = next(i for i, node in enumerate(nodes) if isinstance(node, ast.Assign)
                       and getattr(node.targets[0], 'id', None) == 'stamp')
            body = [n for n in nodes[:cut] if not (isinstance(n, ast.ImportFrom) and n.module == 'runtime')]
            with mock.patch.object(sys, 'argv', ['s', *argv]), contextlib.redirect_stderr(io.StringIO()):
                ns = {'__file__': str(tree / 'start_moe_repaired.py')}
                exec(compile(ast.Module(body=body, type_ignores=[]), 'start', 'exec'), ns)
            return ns['args']

        def line(tree, args):
            fn = next(n for n in ast.parse((tree / 'start_moe_repaired.py').read_text()).body
                      if isinstance(n, ast.FunctionDef) and n.name == 'launch_command')
            ns = {'REMOTE': '/r', 'args': args}
            exec(compile(ast.Module(body=[fn], type_ignores=[]), 'start', 'exec'), ns)
            return ns['launch_command']('dusty')
        ok = ['--arm', 'selection-transplant', '--decision-row-blocks', '81389', '--prefill-threshold', '8192']
        default, mhc = parse(PATCHED, ok), parse(PATCHED, ok + ['--transplant-set', 'mhc8192'])
        base_args = parse(BASE, ok)
        self.assertEqual(line(PATCHED, default), line(BASE, base_args))
        self.assertEqual(line(PATCHED, mhc), line(BASE, base_args).replace('transplant275-release-into-233036Z',
                                                                          'transplant2-mhc8192-release-into-233036Z'))
        for argv in (['--arm', 'selection-replay', '--decision-row-blocks', '81389', '--prefill-threshold', '8192',
                      '--transplant-set', 'mhc8192'], ['--arm', 'precision-release', '--transplant-set', 'mhc8192'],
                     ok + ['--transplant-set', 'other']):
            with self.assertRaises(SystemExit, msg=argv):
                parse(PATCHED, argv)
        replay = ['--arm', 'selection-replay', '--decision-row-blocks', '81389', '--prefill-threshold', '8192']
        self.assertEqual(line(PATCHED, parse(PATCHED, replay)), line(BASE, parse(BASE, replay)))

    def test_recorder_set_gates(self):
        rec = load(PATCHED, 'run_selection_transplant')
        info = json.loads((KIT / 'receipts/ratio1-control-20260927T0122Z/dusty-identity-before.json').read_text())
        base = info[0] if isinstance(info, list) else info
        env = dict(item.split('=', 1) for item in base['Config']['Env'])
        env.update(rec.arm_env('mhc8192'))
        self.assertEqual(rec.environment_problems(env, 'mhc8192'), [])
        self.assertTrue(rec.environment_problems(env, 'transplant275'))
        self.assertTrue(rec.environment_problems(dict(env, **rec.arm_env()), 'mhc8192'))
        self.assertEqual(rec.ARM_ENV, rec.arm_env())

    def test_scope_manifest_and_regression(self):
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


if __name__ == '__main__':
    unittest.main()
