"""One-key split of the two-mHC 8192 transplant: unapplied patch over the live (two-mHC applied) kit.

Proves on temporary copies: the 275 and two-record sets' seed bytes, renders and launch lines are
unchanged; each one-key set differs from the passing file in exactly its one rebuilt mHC key, equal to
the release record, with every other record byte-equal to passing; the two one-key sets are disjoint
and together are the two-record set; each has its own fixed selector and task-tree cache root and is
reachable only through the selection-transplant arm; the recorder gates each set; existing suites
keep their outcomes. Live files must be the reviewed base or the exact patched tree.

    .venv-snapshot-cpu/bin/python -m unittest test_selection_mhc_onekey
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
PATCH = KIT / 'ds41-selection-mhc-onekey.patch'
# Base: the live tree with the two-mHC patch applied (test_selection_mhc8192 PATCHED_HASHES).
BASES = {'launch_contract.py': '0b432bda7d26336fc5cb7abc3310bf4abcbbbf41dc0938f99e91e36817336887',
         'start_moe_repaired.py': '6956fb95c6e3779d21d5211f76a20a777350b9133a85f89e429354f2ace6c93b',
         'seed_selection_transplant.py': '5d1e002efc26629718b9c1530ef79c75fcbb917e63b40c7b807b298ca1fae554'}
PATCHED_HASHES = {'launch_contract.py': '9c1d8615e8e07f851e3e8735878bb319189810fcee36e19b8d73a483238f62f2',
                  'start_moe_repaired.py': '6c700bdaac753d5d295d12a999a8f56da364e539cf05c8ff9548ecc2989e02da',
                  'seed_selection_transplant.py': '83410604372887b722c25eed245b3b89b1e53f17a4e95bf93ebf6b9bf6e5519e'}
UNCHANGED = {'run_selection_transplant.py': '3c8610d0d1ca887e98fc93cfbcbc85b92af3b206821390a07f115a99bf6a1c03'}
PRE = '2394c70075c7fe2b87a1bd135e1ba48faaf75dfbb048b7f683b8df73d6de75c3'
EXPANDED = '1b6c36a14eaadabaa5b558f21ee4755eb16d6f5c5bc7c1f8ff4daae6b64f10a6'
ONE_KEY = {'mhc-pre8192': ('transplant1-mhc-pre8192-release-into-233036Z',
                           'git/ds41-r38/karmic-main-20260924/selection-transplant-cache-mhc-pre8192-233036Z', PRE),
           'mhc-expanded8192': ('transplant1-mhc-expanded8192-release-into-233036Z',
                                'git/ds41-r38/karmic-main-20260924/selection-transplant-cache-mhc-expanded8192-233036Z',
                                EXPANDED)}
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
        raise RuntimeError(f'Live files are neither the reviewed two-mHC base nor the exact one-key tree: {live}')
    for name, digest in UNCHANGED.items():
        if sha((KIT / name).read_bytes()) != digest:
            raise RuntimeError(f'{name} changed; this patch was reviewed against {digest[:8]}')
    _tmp = tempfile.TemporaryDirectory(prefix='selection-mhc-onekey-')
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
    def test_existing_seed_bytes_unchanged(self):
        for args in ('', "name='mhc8192'"):
            with self.subTest(args=args):
                self.assertEqual(run_in(BASE, SEED_DIGESTS.format(args=args)), run_in(PATCHED, SEED_DIGESTS.format(args=args)))

    def test_one_key_sets_exact(self):
        code = ("import json, seed_selection_transplant as t\n"
                "out = {}\n"
                "for name in ('mhc-pre8192', 'mhc-expanded8192', 'mhc8192'):\n"
                "    ns, files = t.check(name=name)\n"
                "    for n, f in files.items():\n"
                "        payload = json.loads(f['bytes'])\n"
                "        p = json.load(open(t.replay.SOURCE + '/' + n + '-selection.json'))\n"
                "        r = json.load(open(t.RELEASE_SNAPSHOT + '/' + n + '.json'))\n"
                "        diff = sorted(k for k in p['records'] if payload['records'][k] != p['records'][k])\n"
                "        out[name + '/' + n] = {'diff': diff, 'sha': f['sha256'],\n"
                "            'release_equal': all(payload['records'][k] == r['records'][k] for k in diff),\n"
                "            'keys_equal': set(payload['records']) == set(p['records']), 'identity': payload['identity'] == p['identity'],\n"
                "            'others_equal': all(payload['records'][k] == p['records'][k] for k in p['records'] if k not in diff)}\n"
                "print(json.dumps(out))")
        result = run_in(PATCHED, code)
        for node in ('dusty', 'toby', 'rusty', 'kirby'):
            pre, exp, pair = (result[f'{s}/{node}'] for s in ('mhc-pre8192', 'mhc-expanded8192', 'mhc8192'))
            self.assertEqual(pre['diff'], [PRE], node)
            self.assertEqual(exp['diff'], [EXPANDED], node)
            self.assertEqual(set(pre['diff']) | set(exp['diff']), set(pair['diff']))
            for row in (pre, exp):
                self.assertTrue(row['release_equal'] and row['keys_equal'] and row['identity'] and row['others_equal'], node)
            self.assertEqual(len({pre['sha'], exp['sha'], pair['sha']}), 3, node)

    def test_set_table(self):
        seed = load(PATCHED, 'seed_selection_transplant')
        base = load(BASE, 'seed_selection_transplant')
        self.assertEqual({k: v for k, v in seed.SETS.items() if k in base.SETS}, base.SETS)
        self.assertEqual(set(seed.SETS) - set(base.SETS), set(ONE_KEY))
        roots = [v['host_root'] for v in seed.SETS.values()]
        selectors = [v['selector'] for v in seed.SETS.values()]
        self.assertEqual(len(set(roots)), len(roots))
        self.assertEqual(len(set(selectors)), len(selectors))
        for name, (selector, root, key) in ONE_KEY.items():
            row = seed.SETS[name]
            self.assertEqual((row['selector'], row['host_root'], row['count']), (selector, '/home/jugs/' + root, 1))
            self.assertEqual(row['keys_sha256'], seed.keys_digest({key}))
            self.assertTrue(row['host_root'].startswith('/home/jugs/git/ds41-r38/karmic-main-20260924/selection-transplant-cache-'))

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
                "attempt('unknown', lambda: t.select(p, r, 'mhc-pre'))\n"
                "for name, label in (('mhc-pre8192', 'mhc.pre.8192'), ('mhc-expanded8192', 'mhc.pre.expanded.8192')):\n"
                "    key = t.MHC8192_KEYS[label]\n"
                "    p2 = copy.deepcopy(p); p2['records'][key] = copy.deepcopy(r['records'][key])\n"
                "    attempt('baseline-' + name, lambda: t.select(p2, r, name))\n"
                "saved = dict(t.SETS['mhc-pre8192'])\n"
                "t.SETS['mhc-pre8192']['keys_sha256'] = t.keys_digest({t.MHC8192_KEYS['mhc.pre.expanded.8192']})\n"
                "attempt('digest', lambda: t.select(p, r, 'mhc-pre8192'))\n"
                "t.SETS['mhc-pre8192'] = saved\n"
                "t.SETS['mhc8192'] = dict(t.SETS['mhc8192'], keys_sha256='0' * 64)\n"
                "attempt('pair', lambda: t.select(p, r, 'mhc-expanded8192'))\n"
                "print(json.dumps(out))")
        result = run_in(PATCHED, code)
        self.assertTrue(result['unknown'].startswith('ValueError'))
        for name in ONE_KEY:
            self.assertIn('frozen 275', result['baseline-' + name])
        self.assertIn('frozen 275', result['digest'])
        self.assertIn('pair digest', result['pair'])


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
        self.assertEqual({k: v for k, v in c.SELECTION_ROOTS.items() if k in base.SELECTION_ROOTS}, base.SELECTION_ROOTS)
        self.assertEqual({k: v for k, v in c.SELECTION_ROOTS.items() if k not in base.SELECTION_ROOTS},
                         {s: r for s, r, _ in ONE_KEY.values()})
        with mock.patch.dict(os.environ, dict(geometry, DS41_SELECTION_REPLAY=c.MHC8192_SELECTOR), clear=True):
            reference = c.render('toby')
        reference['env'].pop('DS41_SELECTION_REPLAY')
        release = {'image_id': 'a' * 64, 'diagnostic': {'kind': 'precision-release-candidate', 'lock_sha256': c.PRECISION_RELEASE_LOCK}}
        run_node = load(PATCHED, 'run_node')
        seed = load(PATCHED, 'seed_selection_transplant')
        for name, (selector, root, _) in ONE_KEY.items():
            with self.subTest(name=name):
                with mock.patch.dict(os.environ, dict(geometry, DS41_SELECTION_REPLAY=selector), clear=True):
                    row = c.render('toby')
                    argv, _, cache = run_node.command('dusty', release, 'b' * 64)
                self.assertEqual(row['env'].pop('DS41_SELECTION_REPLAY'), selector)
                self.assertEqual(row, reference)
                self.assertEqual(cache, Path.home() / root)
                c.validate_chunking_image(release, dict(geometry, DS41_SELECTION_REPLAY=selector))
                self.assertEqual((seed.SETS[name]['selector'], seed.SETS[name]['host_root']), (selector, '/home/jugs/' + root))
                with mock.patch.dict(os.environ, dict(geometry, DS41_SELECTION_REPLAY=c.MHC8192_SELECTOR), clear=True):
                    pair_argv, _, _ = run_node.command('dusty', release, 'b' * 64)
                self.assertEqual([a.replace(root, 'ROOT').replace(selector, 'SEL') for a in argv],
                                 [a.replace(c.MHC8192_CACHE_ROOT, 'ROOT').replace(c.MHC8192_SELECTOR, 'SEL') for a in pair_argv])
        for bad in ('transplant1-mhc-pre8192', 'transplant1-mhc-pre8192-release-into-233036Z '):
            with self.assertRaises(ValueError):
                c.validate_chunking_image(release, dict(geometry, DS41_SELECTION_REPLAY=bad))
        for name, (selector, _, _) in ONE_KEY.items():
            for blocks, threshold in (('80022', '8192'), ('81389', '4096')):
                with mock.patch.dict(os.environ, {'DS41_DECISION_ROW_BLOCKS': blocks, 'DS41_PREFILL_THRESHOLD': threshold,
                                                  'DS41_SELECTION_REPLAY': selector}, clear=True):
                    with self.assertRaises(Exception, msg=(name, blocks, threshold)):
                        c.render('toby')

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
        for extra in ([], ['--transplant-set', 'mhc8192']):
            self.assertEqual(line(PATCHED, parse(PATCHED, ok + extra)), line(BASE, parse(BASE, ok + extra)))
        pair = line(BASE, parse(BASE, ok + ['--transplant-set', 'mhc8192']))
        for name, (selector, _, _) in ONE_KEY.items():
            self.assertEqual(line(PATCHED, parse(PATCHED, ok + ['--transplant-set', name])),
                             pair.replace('transplant2-mhc8192-release-into-233036Z', selector))
            for argv in (['--arm', 'selection-replay', '--decision-row-blocks', '81389', '--prefill-threshold', '8192',
                          '--transplant-set', name], ['--arm', 'precision-release', '--transplant-set', name],
                         ['--arm', 'selection-transplant', '--decision-row-blocks', '81389', '--prefill-threshold', '4096',
                          '--transplant-set', name]):
                with self.assertRaises(SystemExit, msg=argv):
                    parse(PATCHED, argv)
        with self.assertRaises(SystemExit):
            parse(PATCHED, ok + ['--transplant-set', 'mhc-pre'])
        source = (PATCHED / 'start_moe_repaired.py').read_text()
        self.assertIn("out = ROOT / 'receipts' / f'{args.arm}-{args.transplant_set}-chunk8192-blocks81389-{stamp}'", source)

    def test_recorder_set_gates(self):
        rec = load(PATCHED, 'run_selection_transplant')
        info = json.loads((KIT / 'receipts/ratio1-control-20260927T0122Z/dusty-identity-before.json').read_text())
        base = info[0] if isinstance(info, list) else info
        env = dict(item.split('=', 1) for item in base['Config']['Env'])
        names = ('transplant275', 'mhc8192', *ONE_KEY)
        for name in names:
            arm = dict(env, **rec.arm_env(name))
            for other in names:
                with self.subTest(name=name, other=other):
                    problems = rec.environment_problems(arm, other)
                    self.assertEqual(problems == [], name == other, problems)
        self.assertEqual(rec.ARM_ENV, rec.arm_env())

    def test_scope_manifest_and_regression(self):
        changed = {p.name for p in PATCHED.iterdir() if p.is_file() and sha(p.read_bytes()) != sha((BASE / p.name).read_bytes())}
        self.assertEqual(changed, set(BASES))
        runtime = json.loads((KIT / 'runtime-files.json').read_text())
        self.assertEqual(changed & set(runtime), {'launch_contract.py'})
        for name, digest in UNCHANGED.items():
            self.assertEqual(sha((PATCHED / name).read_bytes()), digest)

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
