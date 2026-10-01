"""Regressions for the functional review; no Spark, image or GPU access."""
import json
import hashlib
import io
import tarfile
from types import SimpleNamespace
from unittest.mock import patch
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent


class ReviewRegression(unittest.TestCase):
    def test_aot_guard_rejects_missing_and_external_artifacts_and_compilation(self):
        from aot_guard import enforce_aot
        class Spec:
            name = 'sampling'
            aot_path = Path('/opt/venv/cache/sampling.so')
            is_aot = True
            def build_and_load(self):
                return 'loaded'
            def build(self):
                return 'compiled'
        used = enforce_aot(SimpleNamespace(JitSpecNvcc=Spec), Path('/opt/venv/cache'))
        self.assertEqual(Spec().build_and_load(), 'loaded')
        self.assertEqual(len(used), 1)
        with self.assertRaisesRegex(RuntimeError, 'compilation is forbidden'):
            Spec().build()
        Spec.is_aot = False
        with self.assertRaisesRegex(RuntimeError, 'Missing rebuilt AOT'):
            Spec().build_and_load()
        Spec.is_aot = True
        Spec.aot_path = Path('/cache/old/sampling.so')
        with self.assertRaisesRegex(RuntimeError, 'Missing rebuilt AOT'):
            Spec().build_and_load()

    def test_source_identity_survives_different_gzip_compression(self):
        import gzip
        from unpack_sources import artifact_sha
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = b'canonical tar contents' * 1000
            paths = [root / 'fast.gz', root / 'small.gz']
            paths[0].write_bytes(gzip.compress(data, compresslevel=1, mtime=1))
            paths[1].write_bytes(gzip.compress(data, compresslevel=9, mtime=2))
            self.assertNotEqual(paths[0].read_bytes(), paths[1].read_bytes())
            row = {'digest_kind': 'uncompressed-tar-sha256'}
            self.assertEqual(artifact_sha(paths[0], row), artifact_sha(paths[1], row))

    def test_production_comparison_rejects_wrong_image_and_checkpoint(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('production_compare', ROOT / 'compare-production.py')
        compare = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compare)
        for model in ('qwen', 'glm'):
            name, revision = compare.PROFILES[model]
            data = {'metadata': {'model': name}, 'run_metadata': {
                'image_id': compare.PRODUCTION_IMAGE, 'checkpoint_revision': revision,
                'recurrent_checkpoint_policy': 'aligned', 'harness_sha256': compare.HARNESS, 'hc_tp': '0'}}
            compare.validate_identity(data, model, compare.PRODUCTION_IMAGE)
            data['run_metadata']['image_id'] = 'old-r38'
            with self.assertRaisesRegex(RuntimeError, 'image_id'):
                compare.validate_identity(data, model, compare.PRODUCTION_IMAGE)
            data['run_metadata']['image_id'] = compare.PRODUCTION_IMAGE
            data['run_metadata']['checkpoint_revision'] = 'old-checkpoint'
            with self.assertRaisesRegex(RuntimeError, 'checkpoint_revision'):
                compare.validate_identity(data, model, compare.PRODUCTION_IMAGE)

    def test_complete_production_comparison_and_protocol_rejection(self):
        import copy
        levels, contexts = [1, 2, 4], [0, 16384, 32768, 65536, 131072]
        metadata = {'model': 'Qwen3.8-Flash-Next', 'concurrency_levels': levels,
                    'version': 1, 'decode_mode': 'stream', 'duration_per_test': 30,
                    'decode_warmup_seconds': 5, 'context_lengths': contexts,
                    'ignore_eos': True, 'max_tokens': 4096, 'chat_template_kwargs': {},
                    'temperature': 0.8, 'prefill_mode': 'scout', 'max_total_tokens': 6412288}
        data = {'metadata': metadata, 'run_metadata': {
            'image_id': '500ae05b98da0658c1a5e1820387f96f2121c5659bd7954ad1c5861f20934f05',
            'checkpoint_revision': '7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd',
            'recurrent_checkpoint_policy': 'aligned', 'hc_tp': '0',
            'harness_sha256': '2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3'},
            'results': [{'concurrency': c, 'context_tokens': context,
                         'num_errors': 0, 'warmup_timed_out': False, 'capacity_limited': False,
                         'underfilled': False, 'avg_running_reqs': c, 'max_running_reqs': c,
                         'aggregate_tps': 100, 'server_steps_per_s': 20, 'server_accept_len_effective': 3}
                        for c in levels for context in contexts],
            'prefill': {str(c): {'tok_per_sec': 1000} for c in contexts}}
        with tempfile.TemporaryDirectory() as directory:
            baseline, candidate = Path(directory) / 'baseline.json', Path(directory) / 'candidate.json'
            baseline.write_text(json.dumps(data))
            changed = copy.deepcopy(data)
            changed['run_metadata']['image_id'] = 'a' * 64
            for row in changed['results']:
                row['aggregate_tps'] = 110
            candidate.write_text(json.dumps(changed))
            command = ['python3', str(ROOT / 'compare-production.py'), '--model', 'qwen']
            subprocess.run([*command, '--validate-baseline', str(baseline)], check=True, capture_output=True)
            env = dict(os.environ, EXPECTED_IMAGE_ID='a' * 64)
            result = subprocess.run([*command, str(baseline), str(candidate)], check=True, env=env, capture_output=True, text=True)
            report = json.loads(result.stdout)
            self.assertAlmostEqual(report['summary'][0]['aggregate_tps']['change_pct'], 10)
            changed['metadata']['max_total_tokens'] = 1
            candidate.write_text(json.dumps(changed))
            result = subprocess.run([*command, str(baseline), str(candidate)], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('token budget mismatch', result.stderr)

    def test_cutlass_gate_checks_actual_import_path_and_version(self):
        from verify_compiler import validate_cutlass
        with self.assertRaises(RuntimeError):
            validate_cutlass(SimpleNamespace(__file__='/usr/local/lib/python3.12/site-packages/cutlass/__init__.py', __version__='4.7.1'))
        with self.assertRaises(RuntimeError):
            validate_cutlass(SimpleNamespace(__file__='/opt/venv/lib/cutlass/__init__.py', __version__='4.6.2'))
        self.assertEqual(validate_cutlass(SimpleNamespace(__file__='/opt/venv/lib/cutlass/__init__.py', __version__='4.7.1'))['version'], '4.7.1')

    def test_different_transport_headers_produce_identical_locked_archives(self):
        from unpack_sources import canonicalize_archive
        data = b'compiler input\n'
        oid = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tree = root / 'tree.json'
            tree.write_text(json.dumps({'tree': [{'path': 'source.py', 'type': 'blob', 'mode': '100644', 'sha': oid}]}))
            row = {'tree_file': tree.name, 'tree_sha256': hashlib.sha256(tree.read_bytes()).hexdigest()}
            outputs = []
            for i in (1, 2):
                archive = root / f'transport-{i}.tar.gz'
                with tarfile.open(archive, 'w:gz') as bundle:
                    member = tarfile.TarInfo(f'changed-prefix-{i}/source.py')
                    member.mtime, member.uid, member.uname = i, i, f'user{i}'
                    member.size = len(data)
                    bundle.addfile(member, io.BytesIO(data))
                output = root / f'canonical-{i}.tar.gz'
                canonicalize_archive(archive, output, row)
                outputs.append(output.read_bytes())
            self.assertEqual(outputs[0], outputs[1])

    def test_input_restore_never_rewrites_frozen_lock(self):
        import prepare_inputs
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'inputs').mkdir()
            payload = b'locked wheel'
            (root / 'inputs/pkg.whl').write_bytes(payload)
            lock = root / 'inputs.lock.json'
            lock.write_text(json.dumps({'wheels': {'pkg': {'file': 'pkg.whl', 'url': 'unused', 'sha256': hashlib.sha256(payload).hexdigest()}}, 'sources': {}}))
            before = lock.read_bytes()
            with patch.object(prepare_inputs.urllib.request, 'urlopen', side_effect=AssertionError('Should reuse locked input')):
                prepare_inputs.restore_inputs(root)
            self.assertEqual(lock.read_bytes(), before)

    def test_ds4_manifest_is_shipped_and_covers_48_weight_shards(self):
        manifest = json.loads((ROOT / 'ds4-vision/model-manifest.json').read_text())
        self.assertEqual(manifest['revision'], '6821d6ad3681a4b137b066b76094fa82ebd0a380')
        self.assertEqual(sum(row['path'].endswith('.safetensors') for row in manifest['files']), 48)

    def test_ds4_runner_is_in_its_integrity_manifest(self):
        files = (ROOT / 'ds4-vision/runtime-files.sha256').read_text()
        self.assertIn('  run-node.sh\n', files)

    def test_glm_stop_removes_its_candidate_container(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            state = path / 'container'
            state.touch()
            podman = path / 'podman'
            podman.write_text('#!/bin/bash\ncase "$1" in\ncontainer) test -f "$TEST_CONTAINER";;\nstop) :;;\nrm) rm "$TEST_CONTAINER";;\n*) exit 90;;\nesac\n')
            podman.chmod(0o755)
            env = dict(os.environ, PATH=str(path) + ':' + os.environ['PATH'],
                       TEST_CONTAINER=str(state), ROLE='stop', DRY_RUN='0', NODE_RANK='0', HOST_IP='10.11.11.1')
            subprocess.run(['bash', str(ROOT / 'glm/run-glm-tp4-node.sh')], env=env, check=True,
                           capture_output=True, text=True)
            self.assertFalse(state.exists(), 'Stop leaves a name collision that prevents the next start')

    def test_candidate_tools_do_not_pin_the_previous_image_or_deployment(self):
        for name in ('distribute.sh', 'execute.sh', 'benchmark.sh', 'ds4-vision/execute.sh',
                     'ds4-vision/benchmark.sh', 'ds4-vision/probe-prefill.py', 'ds4-vision/compare.py',
                     'glm/execute-glm.sh', 'glm/qualify-glm.sh', 'glm/confirm-transport.sh'):
            with self.subTest(name=name):
                text = (ROOT / name).read_text()
                self.assertNotIn('vllm:karmic-beta-20260929', text)
                self.assertNotIn('karmic-beta-sm121/20260929', text)
                self.assertNotIn('500ae05b98da0658c1a5e1820387f96f2121c5659bd7954ad1c5861f20934f05', text)

    def test_required_candidate_tools_exist(self):
        for name in ('distribute.sh', 'execute.sh', 'benchmark.sh', 'ds4-vision/execute.sh',
                     'ds4-vision/benchmark.sh', 'glm/execute-glm.sh', 'glm/qualify-glm.sh',
                     'glm/confirm-transport.sh', 'glm/probe-fresh-prefill.py'):
            with self.subTest(name=name):
                self.assertTrue((ROOT / name).is_file())


    def test_sampler_gate_covers_served_vocabularies_beyond_one_pass(self):
        import ast
        tree = ast.parse((ROOT / 'gate_flashinfer.py').read_text())
        cases = next(ast.literal_eval(node.iter) for node in ast.walk(tree)
                     if isinstance(node, ast.For) and isinstance(node.target, ast.Tuple)
                     and len(node.target.elts) == 5)
        vocabularies = {case[1] for case in cases}
        # DS4 Vision, GLM-5.3-Flash and Qwen3.8-Flash-Next text vocabularies.
        self.assertLessEqual({129280, 154880, 248320}, vocabularies)
        for batch, vocab, high, second, tolerance in cases:
            self.assertTrue(0 <= high < vocab and 0 <= second < vocab and high != second)
            # At least four standard deviations of the binomial frequency estimate.
            self.assertGreaterEqual(tolerance, 4 * ((0.67 * 0.33 / batch) ** 0.5))


if __name__ == '__main__':
    unittest.main(verbosity=2)
