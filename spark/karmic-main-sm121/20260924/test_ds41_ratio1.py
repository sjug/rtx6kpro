"""Ratio-1 extend BF16 pin diagnostic: source delta, lock, installer, seeding, arm checks, selector.

    .venv-snapshot-cpu/bin/python -m unittest test_ds41_ratio1
"""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import ds41_ratio1_prepare as prep  # noqa: E402
import ds41_ratio1_install as installer  # noqa: E402
import seed_ratio1_namespace as seeding  # noqa: E402
import run_ratio1_arm as arm  # noqa: E402
import select_ratio1 as select  # noqa: E402

LOCK_RAW = (ROOT / 'ds41-ratio1.lock.json').read_bytes()
LOCK = json.loads(LOCK_RAW)
RELEASE = json.loads((ROOT / 'ds41-precision-release.lock.json').read_text())
SNAPSHOT = ROOT / 'receipts/precision-release-full-qualification-20260927-r2/initial-selections'


def sha(data):
    return hashlib.sha256(data).hexdigest()


class SourceDelta(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lock, _, _ = prep.release_identity()
        cls.base = prep.release_attention(lock)
        cls.composed = prep.compose(cls.base)

    def test_output_and_lock_reproduce(self):
        self.assertEqual((ROOT / prep.OUTPUT).read_text(), self.composed)
        entry = LOCK['targets'][prep.TARGET]
        self.assertEqual(entry['input_sha256'], RELEASE['after']['vllm'][prep.TARGET]['sha256'])
        self.assertEqual(entry['output_sha256'], sha(self.composed.encode()))
        import difflib
        patch = ''.join(difflib.unified_diff(self.base.splitlines(True), self.composed.splitlines(True),
                                             'a/' + prep.TARGET, 'b/' + prep.TARGET))
        self.assertEqual((ROOT / 'ds41-ratio1.patch').read_text(), patch)
        added = [l for l in patch.splitlines() if l.startswith('+') and not l.startswith('+++')]
        removed = [l for l in patch.splitlines() if l.startswith('-') and not l.startswith('---')]
        self.assertEqual((len(added), len(removed)), (16, 0))
        self.assertEqual(sum(l[1:].strip() == 'override=ds41_pin,' for l in added), 1)

    def test_mutations_refused(self):
        condition = 'if mode == "extend" and self.compress_ratio == 1 and not self.is_draft'
        for old, new in ((condition, condition.replace('"extend"', '"decode"')),
                         (condition, condition.replace('== 1', '== 2')),
                         (condition, condition.replace(' and not self.is_draft', '')),
                         ('v41_compute_mode="bf16", v41_heads_per_block=16', 'v41_compute_mode="fp8", v41_heads_per_block=16'),
                         ('v41_heads_per_block=16,\n)', 'v41_heads_per_block=8,\n)'),
                         ('                override=ds41_pin,\n', '                override=None,\n'),
                         ('flush=True)\n            declaration', 'flush=True)\n                caps = caps\n            declaration')):
            with self.subTest(old=old):
                self.assertEqual(self.composed.count(old), 1)
                with self.assertRaises(ValueError):
                    prep.verify_delta(self.base, self.composed.replace(old, new))
        with self.assertRaises(ValueError):
            prep.verify_delta(self.base, self.composed + '\nEXTRA = 1\n')
        with self.assertRaises(ValueError):
            prep.compose(self.base.replace(prep.DECLARE_ANCHOR, prep.DECLARE_ANCHOR + '\n', 1) + prep.DECLARE_ANCHOR)

    def test_pin_is_the_release_record_with_only_the_mode_changed(self):
        from claude_decode_sparse_mla import decode, regimes
        for node in ('dusty', 'toby', 'rusty', 'kirby'):
            records = json.loads((SNAPSHOT / f'{node}.json').read_text())['records']
            found, _ = decode(records, [LOCK['kv_blocks']], None)
            tuned = regimes(records, found)[LOCK['kv_blocks']]['ratio1.extend']
            self.assertEqual(tuned['v41_compute_mode'], 'fp8')
            self.assertEqual(dict(tuned, v41_compute_mode='bf16'), LOCK['pin'])


class LockAndImage(unittest.TestCase):
    def test_inputs_allowlist_base_and_scope(self):
        import build_ratio1
        lock, digest = build_ratio1.frozen()
        self.assertEqual(digest, sha(LOCK_RAW))
        self.assertEqual(set(lock['inputs']), set(prep.PACKAGED))
        self.assertEqual(lock['before'], RELEASE['after'])
        self.assertEqual(lock['after']['b12x'], RELEASE['after']['b12x'])
        before, after = lock['before']['vllm'], lock['after']['vllm']
        self.assertEqual(set(before), set(after))
        self.assertEqual([p for p in before if before[p] != after[p]], [prep.TARGET])
        self.assertEqual(lock['base_cache_fingerprint'], RELEASE['cache_fingerprint'])
        self.assertTrue(lock['cache_fingerprint'].startswith('ds41-ratio1-diag-'))
        self.assertNotEqual(lock['cache_fingerprint'], RELEASE['cache_fingerprint'])
        self.assertEqual((lock['kind'], lock['kv_blocks']), ('ratio1-bf16-diagnostic', 80022))

    def test_tree_reproduces(self):
        import ds41_precision_release_prepare as release_prep
        from contracts import git_tree
        files = release_prep.router_vllm_files()
        _, after = release_prep.release(files)
        after[prep.TARGET] = (after[prep.TARGET][0], (ROOT / prep.OUTPUT).read_bytes())
        self.assertEqual(git_tree(after), LOCK['trees']['vllm'])

    def test_dockerfile(self):
        import build_ratio1
        text = (ROOT / 'Dockerfile.ds41-ratio1').read_text()
        self.assertEqual([l for l in text.splitlines() if l.startswith('FROM ')], ['FROM ' + LOCK['base_image_id']])
        self.assertEqual(text.count('\nRUN '), 1)
        for key, value in build_ratio1.expected_labels(LOCK, '${RATIO1_LOCK}').items():
            value = '${RATIO1_CACHE}' if key == 'local-inference.cache.fingerprint' else value
            self.assertIn(f'{key}="{value}"', text)


class Installer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.here, self.root, self.parent = base / 'here', base / 'jj', base / 'release'
        self.here.mkdir()
        self.parent.mkdir()
        preimage = prep.release_attention(RELEASE).encode()
        self.files = {('vllm', prep.TARGET): preimage, ('vllm', 'vllm/other.py'): b'x = 1\n',
                      ('b12x', 'b12x/native.so'): b'\x7fELF'}
        for (component, name), data in self.files.items():
            path = self.root / component / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (self.parent / 'ds41-precision-release.lock.json').write_bytes(b'{"release": 1}\n')
        (self.parent / 'preserved-files.json').write_text(json.dumps(installer.inventory(self.root)))
        for name in (prep.OUTPUT, 'ds41_ratio1_install.py'):
            (self.here / name).write_bytes((ROOT / name).read_bytes())
        manifest = {'vllm': {n: {'mode': '100644', 'sha256': sha(d)} for (c, n), d in self.files.items() if c == 'vllm'},
                    'b12x': {}}
        after = copy.deepcopy(manifest)
        after['vllm'][prep.TARGET]['sha256'] = LOCK['targets'][prep.TARGET]['output_sha256']
        self.lock = dict(LOCK, base_lock_sha256=sha(b'{"release": 1}\n'), before=manifest, after=after,
                         inputs={n: sha((self.here / n).read_bytes()) for n in (prep.OUTPUT, 'ds41_ratio1_install.py')})
        (self.here / 'ds41-ratio1.lock.json').write_text(json.dumps(self.lock))

    def tearDown(self):
        self.tmp.cleanup()

    def run_install(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            installer.install(self.here, self.root, self.parent)
        return out.getvalue().strip()

    def test_success_is_confined(self):
        before = installer.inventory(self.root)
        self.assertEqual(self.run_install(), LOCK['install_pass'])
        after = installer.inventory(self.root)
        self.assertEqual({k for k in after if after[k] != before.get(k)}, {str(self.root / 'vllm' / prep.TARGET)})

    def test_refusals(self):
        cases = {'Recipe drift': lambda: (self.here / prep.OUTPUT).write_text('x'),
                 'Parent release lock differs': lambda: (self.parent / 'ds41-precision-release.lock.json').write_text('{}'),
                 'Base source drift': lambda: (self.root / 'vllm/vllm/other.py').write_text('x = 2\n'),
                 'inventory differs': lambda: (self.root / 'b12x/b12x/extra.py').write_text('')}
        for message, mutate in cases.items():
            with self.subTest(message=message):
                self.tearDown()
                self.setUp()
                mutate()
                with self.assertRaisesRegex(RuntimeError, message):
                    self.run_install()


class Seeding(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.jit = Path(self.tmp.name)
        self.source = self.jit / 'release-ns'
        selection = self.source / seeding.SELECTION
        selection.parent.mkdir(parents=True)
        selection.write_text('{"records": {}}')
        (self.source / 'inductor').mkdir()
        (self.source / 'inductor/artifact').write_bytes(b'compiled')
        self.expected = sha(selection.read_bytes())

    def tearDown(self):
        self.tmp.cleanup()

    def run_seed(self, expected=None):
        command = seeding.seed_command(str(self.source), str(self.jit / 'diag-ns'), expected or self.expected)
        return subprocess.run(['bash', '-c', command], capture_output=True, text=True)

    def listing(self, path):
        return subprocess.run(['bash', '-c', seeding.listing_command(str(path))], capture_output=True, text=True).stdout

    def test_copy_is_exact_and_source_untouched(self):
        before = self.listing(self.source)
        result = self.run_seed()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f'SEEDED {self.jit / "diag-ns"}', result.stdout.splitlines())
        self.assertEqual(self.listing(self.jit / 'diag-ns'), before)
        self.assertEqual(self.listing(self.source), before)
        self.assertFalse((self.jit / 'diag-ns.seeding').exists())

    def test_refuses_existing_destination_or_wrong_snapshot(self):
        (self.jit / 'diag-ns').mkdir()
        self.assertNotEqual(self.run_seed().returncode, 0)
        shutil.rmtree(self.jit / 'diag-ns')
        self.assertNotEqual(self.run_seed(expected='0' * 64).returncode, 0)
        self.assertFalse((self.jit / 'diag-ns').exists())

    def test_snapshot_from_real_release_selections(self):
        texts = {n: (SNAPSHOT / f'{n}.json').read_text() for n in seeding.NODES}
        with tempfile.TemporaryDirectory() as tmp:
            result = seeding.snapshot(Path(tmp) / 'snap', lambda node, cmd: texts[node])
        self.assertEqual(result['source_namespace'], RELEASE['cache_fingerprint'])
        for node in seeding.NODES:
            self.assertEqual(result['nodes'][node]['sha256'], sha(texts[node].encode()))
            self.assertEqual(result['nodes'][node]['regime']['ratio1.extend']['v41_compute_mode'], 'fp8')
        bad = copy.deepcopy(result)
        bad['nodes']['kirby']['regime']['ratio1.extend']['v41_compute_mode'] = 'bf16'
        with self.assertRaises(RuntimeError):
            seeding.check_snapshot(bad, result['source_namespace'], LOCK['kv_blocks'])


class ArmChecks(unittest.TestCase):
    def test_markers_and_environment(self):
        line = 'DS41-RATIO1-EXTEND-PIN layer=model.layers.3.attn mode=extend rows=8192 v41_compute_mode=bf16\n'
        self.assertEqual(arm.pin_marker_problems(line * 2, True), [])
        self.assertTrue(arm.pin_marker_problems('', True))
        self.assertTrue(arm.pin_marker_problems(line.replace('bf16', 'fp8'), True))
        self.assertTrue(arm.pin_marker_problems(line, False))
        env = {'B12X_DYNAMIC_DETERMINISTIC_OUTPUT': '1', 'B12X_DENSE_SPLITK_TURBO': '0', 'CUDA_MODULE_LOADING': 'LAZY',
               'DS41_DECISION_ROW_BLOCKS': '80022'}
        self.assertEqual(arm.environment_problems(env, 80022), [])
        self.assertTrue(arm.environment_problems(dict(env, DS41_DECISION_ROW_BLOCKS='81389'), 80022))
        self.assertTrue(arm.environment_problems(dict(env, DS41_PREFILL_THRESHOLD='8192'), 80022))
        self.assertTrue(arm.environment_problems(dict(env, NCCL_PROTO='Simple'), 80022))

    def test_selection_check(self):
        records = json.loads((SNAPSHOT / 'dusty.json').read_text())['records']
        result = arm.selection_check(dict(records, extra={'config': {}}), records, 80022)
        self.assertEqual(result['added'], ['extra'])
        self.assertEqual(result['regime']['ratio1.extend']['v41_compute_mode'], 'fp8')
        key = next(iter(records))
        with self.assertRaises(RuntimeError):
            arm.selection_check(dict(records, **{key: {'config': {'changed': True}}}), records, 80022)

    def make_arm(self, root, name, image, signature_text, regime):
        d = root / name
        (d / 'original-524k').mkdir(parents=True)
        (d / 'arm.json').write_text(json.dumps({'arm': name, 'image_id': image}))
        ident = {n: {'regime': regime, 'kit': 'k', 'added': []} for n in arm.NODES}
        (d / 'identity.json').write_text(json.dumps({'before': ident, 'after': ident}))
        report = []
        for i in range(3):
            (d / f'original-524k/{i}.json').write_text(json.dumps({'response': {'choices': [{
                'message': {'content': signature_text}, 'finish_reason': 'stop', 'logprobs': {'content': []}}]}}))
            report.append({'repeat': i, 'correct': signature_text.startswith('739'), 'cached_tokens': 0,
                           'elapsed_s': 180.0, 'first_token': {'margin_nats': 1.0}})
        (d / 'original-524k/report.json').write_text(json.dumps(report))
        return d

    def test_compare(self):
        regime = {'ratio1.extend': {'v41_compute_mode': 'fp8'}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = self.make_arm(root, 'control', 'r', '510c, 482617', regime)
            v = self.make_arm(root, 'variant', 'd', '739184, 482617', regime)
            r = self.make_arm(root, 'return', 'r', '510c, 482617', regime)
            result = arm.compare(c, v, r)
            self.assertTrue(result['valid'])
            self.assertTrue(result['variant_differs_from_control'])
            self.assertEqual(result['correctness']['variant']['correct'], [True] * 3)
            r2 = self.make_arm(root / 'x', 'return', 'r', 'other', regime)
            self.assertFalse(arm.compare(c, v, r2)['valid'])
            with self.assertRaises(RuntimeError):
                arm.compare(c, self.make_arm(root / 'y', 'variant', 'r', '739', regime), r)
            with self.assertRaises(RuntimeError):
                arm.compare(c, self.make_arm(root / 'z', 'variant', 'd', '739', {'ratio1.extend': {'v41_compute_mode': 'bf16'}}), r)


class Selector(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'receipts').mkdir()
        for name in ('ds41-precision-release.lock.json', 'ds41-ratio1.lock.json'):
            shutil.copy2(ROOT / name, self.root / name)
        release_receipt = json.loads((ROOT / 'receipts/precision-release-build-receipt.json').read_text())
        rdir = self.root / 'receipts' / Path(release_receipt['directory']).name
        rdir.mkdir()
        shutil.copy2(Path(release_receipt['directory']) / 'BUILD-OK', rdir / 'BUILD-OK')
        (self.root / 'receipts/precision-release-build-receipt.json').write_text(json.dumps(
            dict(release_receipt, directory=str(rdir))))
        ddir = self.root / 'receipts/ratio1-build-test'
        ddir.mkdir()
        (ddir / 'BUILD-OK').write_text('d' * 64 + '\n')
        (self.root / 'receipts/ratio1-build-receipt.json').write_text(json.dumps(
            {'directory': str(ddir), 'image_id': 'd' * 64, 'lock_sha256': sha(LOCK_RAW)}))
        self.release_pin = json.loads((ROOT / 'candidate.json').read_text())
        (self.root / 'candidate.json').write_text(json.dumps(self.release_pin, sort_keys=True, indent=2) + '\n')
        self.manifest = {'candidate.json': sha((self.root / 'candidate.json').read_bytes())}

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip_and_refusals(self):
        if self.release_pin.get('diagnostic', {}).get('kind') != 'precision-release-candidate':
            self.skipTest('current kit pin is not the release; round trip uses the recorded release pin only')
        pin, payload, updated, labels = select.plan(self.release_pin, self.manifest, root=self.root)
        self.assertEqual(pin['diagnostic']['kind'], 'ratio1-bf16-diagnostic')
        self.assertEqual(labels['local-inference.ds41.diagnostic.base-image'], self.release_pin['image_id'])
        amendment = {'prior_candidate': self.release_pin, 'prior_manifest': self.manifest,
                     'candidate': pin, 'manifest': updated}
        (self.root / 'candidate.json').write_text(payload)
        restored, _, restored_manifest, _ = select.plan(pin, updated, amendment, root=self.root)
        self.assertEqual((restored, restored_manifest), (self.release_pin, self.manifest))
        with self.assertRaises(RuntimeError):
            select.plan(pin, self.manifest, root=self.root)
        bad = dict(amendment, prior_candidate=dict(self.release_pin, recipe_sha256='0' * 64))
        with self.assertRaises(RuntimeError):
            select.plan(pin, updated, bad, root=self.root)


if __name__ == '__main__':
    unittest.main()
