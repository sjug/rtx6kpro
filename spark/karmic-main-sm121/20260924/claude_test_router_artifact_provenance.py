"""CPU tests for claude_router_artifact_provenance.py (synthetic caches, retained isolation SASS/objects; read-only).
  python3 -m unittest claude_test_router_artifact_provenance
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('claude_router_provenance_under_test',
                                              HERE / 'claude_router_artifact_provenance.py')
prov = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prov)
BASE = HERE / 'receipts/router-isolation-baseline-20260925T212450Z'
FENCED = HERE / 'receipts/router-isolation-fenced-20260925T212904Z'
CUOBJDUMP = shutil.which('cuobjdump') or ('/opt/cuda/bin/cuobjdump' if Path('/opt/cuda/bin/cuobjdump').exists() else None)


def tensor(name, dtype, dims):
    return ['tensor', name, dtype, 2, [['dim', kind, value] for kind, value in dims],
            [['dim', 'exact', 5120], ['dim', 'exact', 1]], ['cuda', 0], None, None]


def spec_json(weight=384, output='torch.float32', kernel='gemm.bf16_prefill'):
    fields = [['field', 'source', tensor('source', 'torch.bfloat16', [('dynamic', None), ('exact', 5120)])],
              ['field', 'weight', tensor('weight', 'torch.bfloat16', [('exact', weight), ('exact', 5120)])],
              ['field', 'output', tensor('output', output, [('dynamic', None), ('exact', weight)])]]
    return json.dumps({'facts': ['legacy', kernel, 3, fields]})


def manifest(key, obj_bytes, fingerprint, *, uuid='GPU-1', env=None, **spec_args):
    kernel = spec_args.get('kernel', 'gemm.bf16_prefill')
    return {'schema': 'b12x._lib.compile_manifest.v3', 'cache_key': key, 'kernel_id': kernel,
            'compile_spec_json': spec_json(**spec_args), 'compile_spec_hash': 'spec',
            'cache_payload': ['fmt', ['target'], fingerprint], 'package_fingerprint': fingerprint,
            'object_sha256': hashlib.sha256(obj_bytes).hexdigest(), 'object_bytes': len(obj_bytes),
            'semantic_payload': {'device_uuid': ['device_uuid', uuid]}, 'target': 'b12x.gemm.bf16_gemv._prefill.Bf16PrefillKernel',
            'toolchain': [['cutlass_dsl', '4.6.2']], 'compile_options': ['opt-level=3'],
            'compile_environment': env if env is not None else [['B12X_COMPILE_CACHE_DIR', '/cache/jit/x/b12x'],
                                                                ['B12X_DENSE_SPLITK_TURBO', '0']],
            'launch_metadata': {}}


class Cache:
    def __init__(self, root, fingerprint, *, uuid='GPU-1', env=None):
        self.root, self.fingerprint, self.uuid, self.env = Path(root), fingerprint, uuid, env

    def add(self, obj_bytes, *, key=None, write_object=True, **spec_args):
        key = key or hashlib.sha256(self.fingerprint.encode() + obj_bytes + json.dumps(spec_args).encode()).hexdigest()
        folder = self.root / key[:2]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f'{key}.json').write_text(json.dumps(manifest(key, obj_bytes, self.fingerprint, uuid=self.uuid,
                                                                env=self.env, **spec_args)))
        if write_object:
            (folder / f'{key}.o').write_bytes(obj_bytes)
        return folder / f'{key}.json'


class Selection(unittest.TestCase):
    def test_router_manifest_is_selected_and_others_are_not(self):
        self.assertEqual(prov.router_match(manifest('k', b'x', 'fp')), (True, 'router'))
        for args, reason in (({'weight': 512}, 'weight dims'), ({'output': 'torch.bfloat16'}, 'dtypes'),
                             ({'kernel': 'gemm.dense'}, 'kernel_id')):
            ok, why = prov.router_match(manifest('k', b'x', 'fp', **args))
            self.assertFalse(ok)
            self.assertTrue(why.startswith(reason), why)
        broken = manifest('k', b'x', 'fp')
        broken['compile_spec_json'] = json.dumps({'facts': ['legacy', 'gemm.bf16_prefill', 3, []]})
        self.assertEqual(prov.router_match(broken), (False, 'tensor facts absent'))


class Collect(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def cache(self, name, fingerprint, **kwargs):
        return Cache(self.root / name, fingerprint, **kwargs)

    def test_single_router_artifact_verifies_and_nothing_is_written(self):
        c = self.cache('a', 'fp-a')
        c.add(b'router-object')
        c.add(b'other', weight=512)
        (self.root / 'a/preparation').mkdir()
        (self.root / 'a/preparation' / ('0' * 64 + '.json')).write_text('{"identity": {}, "records": {}}')
        before = sorted((str(p), p.stat().st_mtime_ns) for p in (self.root / 'a').rglob('*'))
        report = prov.collect(self.root / 'a')
        self.assertEqual(before, sorted((str(p), p.stat().st_mtime_ns) for p in (self.root / 'a').rglob('*')))
        (entry,) = report['router_artifacts']
        self.assertEqual((entry['problems'], entry['package_fingerprint'], report['manifests']), ([], 'fp-a', 2))
        self.assertEqual(entry['object_sha256'], hashlib.sha256(b'router-object').hexdigest())
        self.assertEqual([o['reason'].split()[0] for o in report['other_prefill_manifests']], ['weight'])

    def test_object_and_manifest_tampering_is_reported(self):
        c = self.cache('a', 'fp-a')
        path = c.add(b'router-object')
        path.with_suffix('.o').write_bytes(b'router-objecT')
        self.assertIn('object sha256 differs from manifest', prov.collect(self.root / 'a')['router_artifacts'][0]['problems'])
        path.with_suffix('.o').unlink()
        self.assertEqual(prov.collect(self.root / 'a')['router_artifacts'][0]['problems'], ['object missing'])
        data = json.loads(path.read_text())
        data['cache_payload'][2] = 'other'
        data['cache_key'] = 'f' * 64
        path.write_text(json.dumps(data))
        problems = prov.collect(self.root / 'a')['router_artifacts'][0]['problems']
        self.assertIn('manifest name differs from cache_key', problems)
        self.assertIn('package_fingerprint differs from cache_payload[2]', problems)

    def test_cli_refuses_output_inside_the_cache_and_flags_ambiguity(self):
        c = self.cache('a', 'fp-a')
        c.add(b'one')
        with self.assertRaises(SystemExit):
            prov.main(['collect', '--cache-dir', str(self.root / 'a'), '--out', str(self.root / 'a/report.json')])
        self.assertEqual(prov.main(['collect', '--cache-dir', str(self.root / 'a'), '--out', str(self.root / 'r.json')]), 0)
        c.add(b'two', key='ab' + 'c' * 62)
        self.assertEqual(prov.main(['collect', '--cache-dir', str(self.root / 'a'), '--out', str(self.root / 'r2.json')]), 1)


class Compare(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def pair(self, parent_kwargs=None, candidate_kwargs=None, candidate_fp='fp-b', parent_obj=b'p', candidate_obj=b'c'):
        a = Cache(self.root / 'a', 'fp-a', **(parent_kwargs or {}))
        b = Cache(self.root / 'b', candidate_fp, **(candidate_kwargs or {}))
        a.add(parent_obj)
        b.add(candidate_obj)
        return prov.collect(self.root / 'a'), prov.collect(self.root / 'b')

    def test_matched_pair_passes_with_path_only_environment_differences(self):
        env_b = [['B12X_COMPILE_CACHE_DIR', '/cache/jit/y/b12x'], ['B12X_DENSE_SPLITK_TURBO', '0']]
        result = prov.compare(*self.pair(candidate_kwargs={'env': env_b}))
        self.assertEqual((result['verdict'], result['problems']), ('pass', []))

    def test_mismatches_fail(self):
        cases = [({}, {'uuid': 'GPU-2'}, 'fp-b', b'p', b'c', 'device_uuid differs'),
                 ({}, {}, 'fp-a', b'p', b'c', 'share a package fingerprint'),
                 ({}, {}, 'fp-b', b'same', b'same', 'byte-identical'),
                 ({}, {'env': [['B12X_COMPILE_CACHE_DIR', '/x'], ['B12X_DENSE_SPLITK_TURBO', '1']]}, 'fp-b', b'p', b'c',
                  'non-path compile environment differs')]
        for parent_kwargs, candidate_kwargs, fp, po, co, text in cases:
            shutil.rmtree(self.root / 'a', ignore_errors=True)
            shutil.rmtree(self.root / 'b', ignore_errors=True)
            result = prov.compare(*self.pair(parent_kwargs, candidate_kwargs, fp, po, co))
            self.assertEqual(result['verdict'], 'fail', text)
            self.assertTrue(any(text in p for p in result['problems']), (text, result['problems']))

    def test_mixed_candidate_fingerprints_fail(self):
        parent, candidate = self.pair()
        candidate = copy.deepcopy(candidate)
        candidate['package_fingerprints']['fp-old'] = 3
        self.assertIn('candidate cache holds more than one package fingerprint', prov.compare(parent, candidate)['problems'])


@unittest.skipUnless((BASE / 'router.sass').exists() and (FENCED / 'router.sass').exists(), 'retained router SASS absent')
class Sass(unittest.TestCase):
    def test_retained_isolation_dumps(self):
        base = prov.analyze_sass((BASE / 'router.sass').read_text())
        fenced = prov.analyze_sass((FENCED / 'router.sass').read_text())
        self.assertEqual((base['consumer_release_sites'], base['consumer_releases_fenced'], base['cta_fences']), (1, 0, 0))
        self.assertEqual((fenced['consumer_release_sites'], fenced['consumer_releases_fenced'], fenced['cta_fences']), (1, 1, 1))
        self.assertEqual(len(fenced['arrive_sites']), 9)
        self.assertTrue(prov.sass_verdict(base, 'baseline') and prov.sass_verdict(fenced, 'fenced'))
        self.assertFalse(prov.sass_verdict(base, 'fenced') or prov.sass_verdict(fenced, 'baseline'))

    def test_removed_or_misplaced_fence_fails(self):
        text = (FENCED / 'router.sass').read_text()
        self.assertFalse(prov.sass_verdict(prov.analyze_sass(text.replace('MEMBAR.ALL.CTA', 'NOP')), 'fenced'))
        lines = text.splitlines()
        index = next(i for i, l in enumerate(lines) if 'MEMBAR.ALL.CTA' in l)
        moved = lines[:index] + lines[index + 1:]
        moved.insert(index - 4, lines[index])
        self.assertFalse(prov.sass_verdict(prov.analyze_sass('\n'.join(moved)), 'fenced'))

    @unittest.skipUnless(CUOBJDUMP and (FENCED / 'router.o').exists(), 'cuobjdump or retained object absent')
    def test_extraction_and_dump_reproduce_the_retained_cubin(self):
        with tempfile.TemporaryDirectory() as tmp:
            for arm, folder in (('baseline', BASE), ('fenced', FENCED)):
                report = prov.sass(folder / 'router.o', Path(tmp) / arm, CUOBJDUMP, arm)
                self.assertTrue(report['pass'], arm)
                self.assertEqual(report['cubin_sha256'], hashlib.sha256((folder / 'router.cubin').read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()
