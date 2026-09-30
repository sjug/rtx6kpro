import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from verify_sources import verify, tree_id, object_id

ROOT = Path(__file__).resolve().parent


class CandidateTests(unittest.TestCase):
    def test_normalizer_flags_match_pinned_source(self):
        source = json.loads((ROOT / 'source-selection.json').read_text())
        normalizer = subprocess.check_output(['git', '-C', str(Path.home() / 'git/vllm'),
            'show', source['sources']['vllm']['commit'] + ':tools/jovian_wheel_release/normalize_wheel.py'], text=True)
        allowed = set(re.findall(r'parser.add_argument\("(--[a-z-]+)"', normalizer))
        recipe = (ROOT / 'Dockerfile.vllm').read_text().split('python tools/jovian_wheel_release/normalize_wheel.py', 1)[1].split('    && test', 1)[0]
        supplied = set(re.findall(r'^\s+(--[a-z-]+)', recipe, re.M))
        self.assertTrue(supplied <= allowed, supplied - allowed)

    def test_flashkda_replay_poisoning(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source'
            destination = Path(directory) / 'adapted'
            source.mkdir()
            before = '        saved.fill_(float("nan"))\n        graph.replay()\n'
            (source / 'test_vsplit.py').write_text(before * 2)
            subprocess.run(['python3', str(ROOT / 'prepare_flashkda_tests.py'),
                            str(source), str(destination)], check=True)
            text = (destination / 'test_vsplit.py').read_text()
            self.assertEqual(text.count('output.fill_'), 2)
            self.assertEqual(text.count('final.fill_'), 2)
            self.assertEqual((source / 'test_vsplit.py').read_text(), before * 2)
            receipt = json.loads((destination / 'spark-test-adaptation.json').read_text())
            self.assertNotEqual(receipt['original_sha256'], receipt['adapted_sha256'])

    def test_dry_runs_do_not_call_node_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ('podman', 'ssh', 'hostname', 'flock'):
                path = Path(directory) / name
                path.write_text('#!/bin/sh\necho UNEXPECTED_NODE_CALL >&2\nexit 99\n')
                path.chmod(0o755)
            env = {**os.environ, 'DRY_RUN': '1', 'PATH': directory + ':' + os.environ['PATH']}
            for script, args in [('build-component.sh', ['vllm']),
                                 ('build-serving.sh', ['0' * 64])]:
                result = subprocess.run(['bash', str(ROOT / script), *args],
                                        env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertNotIn('UNEXPECTED_NODE_CALL', result.stderr)

    def test_runtime_dependency_inputs_unchanged(self):
        repo = str(Path.home() / 'git/blackwell-llm-docker')
        paths = ['tools/jovian_wheel_runtime/qwen38-runtime.in',
                 'tools/jovian_wheel_runtime/ngc-python-patches.json',
                 'tools/jovian_wheel_runtime/ngc-venv-python-patches.json',
                 'recipes/glm53/install_dependency_python_patches.py',
                 'recipes/glm53/torch-schema-enumeration.patch',
                 'recipes/glm53/cutlass-sentinel-identity.patch']
        result = subprocess.check_output(['git', '-C', repo, 'diff', '--name-only',
            '8c5aa7f828689e6385fa309acd0cce168fb38ec7',
            '23d674e8f658dae2db75399c48430693c196b258', '--', *paths])
        self.assertEqual(result, b'')
        for name in ('runtime-arm64.in', 'runtime-arm64.lock'):
            self.assertEqual((ROOT / name).read_bytes(), (ROOT.parent / 'karmic-beta-sm121' / name).read_bytes())

    def test_shared_component_ids_in_runtime(self):
        lock = json.loads((ROOT / 'build.lock.json').read_text())
        recipe = (ROOT / 'Dockerfile.runtime').read_text()
        for name, image in lock['components'].items():
            if name != 'vllm':
                self.assertIn('FROM ' + image, recipe)
        self.assertIn(lock['b12x_tree'], recipe)
        self.assertIn(lock['flashkda_commit'], recipe)

    def test_pinned_overlay(self):
        lock = json.loads((ROOT / 'build.lock.json').read_text())
        for key, value in verify().items():
            self.assertEqual(lock[key], value)

    def test_empty_git_tree(self):
        self.assertEqual(tree_id([]), '4b825dc642cb6eb9a060e54bf8d69288fbee4904')

    def test_tree_preserves_mode(self):
        blob = object_id('blob', b'hello\n')
        self.assertNotEqual(tree_id([(b'a', b'100644', blob)]), tree_id([(b'a', b'100755', blob)]))

    def test_native_build_identity(self):
        lock = json.loads((ROOT / 'build.lock.json').read_text())
        source = json.loads((ROOT / 'source-selection.json').read_text())
        recipe = (ROOT / 'Dockerfile.vllm').read_text()
        for value in (source['sources']['vllm']['commit'], source['sources']['vllm']['tree'],
                      lock['vllm_overlay_tree'], lock['overlay_sha256']):
            self.assertIn(value, recipe)
        self.assertIn('MAX_JOBS=20 NVCC_THREADS=1', recipe)
        self.assertIsNone(lock['components']['vllm'])
        self.assertIsNone(lock['serving_image_id'])

    def test_build_tool_versions_match_upstream(self):
        source = json.loads((ROOT / 'source-selection.json').read_text())
        upstream = subprocess.check_output(['git', '-C', str(Path.home() / 'git/vllm'),
            'show', source['sources']['vllm']['commit'] + ':tools/jovian_wheel_release/build-requirements.lock'], text=True)
        pattern = r'^([a-zA-Z0-9_-]+)==([^\s]+)'
        self.assertEqual(re.findall(pattern, upstream, re.M),
                         re.findall(pattern, (ROOT / 'vllm-build-requirements.lock').read_text(), re.M))

    def test_launch_contracts(self):
        for model in ('qwen38-flash-next', 'glm53-flash'):
            with self.subTest(model=model):
                launcher = ROOT / 'launchers' / f'serve-{model}-jj-main-spark.sh'
                text = launcher.read_text()
                self.assertIn('export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True', text)
                self.assertIn('if not __debug__:', text)
                result = subprocess.run(['bash', str(launcher)], env={**os.environ, 'DRY_RUN': '1'},
                                        capture_output=True, text=True, check=True)
                self.assertIn('--recurrent-checkpoint-policy aligned', result.stdout)
                self.assertNotIn('NCCL_MIN_NCHANNELS=', text)
                self.assertNotIn('NCCL_MAX_NCHANNELS=', text)


if __name__ == '__main__':
    unittest.main()
