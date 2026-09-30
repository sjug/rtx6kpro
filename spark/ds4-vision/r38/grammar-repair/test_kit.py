"""Local fail-closed recipe and unchanged-serving-contract checks."""
import hashlib
import json
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parent


class KitTests(unittest.TestCase):
    def test_patch_identity(self):
        lock = json.loads((ROOT / 'source.lock.json').read_text())
        self.assertEqual(hashlib.sha256((ROOT / 'upstream.patch').read_bytes()).hexdigest(), lock['patch_sha256'])
        self.assertEqual(len(lock['files']), 7)
        self.assertEqual(sum(name.startswith('vllm/') for name in lock['files']), 5)

    def test_adapter_only_drops_unsupported_argument(self):
        original = subprocess.check_output(['git', '-C', '/home/jugs/git/vllm', 'show',
            '8e1f1e587f8d24faf606f334a1c4bdaaa6bd4368:tests/v1/spec_decode/test_mtp_structured_output.py'])
        removed = b'        grammar_output.num_invalid_spec_tokens,\n'
        self.assertEqual(original.count(removed), 1)
        self.assertEqual(original.replace(removed, b''), (ROOT / 'test_r38_baseline_grammar.py').read_bytes())

    def test_build_format_and_gate_order(self):
        build = (ROOT / 'build.sh').read_text()
        self.assertIn('podman build --format docker --pull=never --network=none', build)
        self.assertLess(build.index('bash gate.sh'), build.index('podman tag'))
        self.assertIn('application/vnd.docker.distribution.manifest.v2+json', build)

    def test_base_is_immutable(self):
        lock = json.loads((ROOT / 'source.lock.json').read_text())
        self.assertEqual((ROOT / 'Dockerfile').read_text().splitlines()[0], 'FROM ' + lock['base_image_id'])

    def test_fixture_weights_are_not_downloaded(self):
        stage = (ROOT / 'stage-tokenizer.py').read_text()
        self.assertNotIn('*.safetensors', stage)
        self.assertNotIn('pytorch_model', stage)
        self.assertEqual(stage.count('allow_patterns='), 3)

    def test_runtime_only_identity_changes(self):
        base = (ROOT.parent / 'run-node.sh').read_text().splitlines()
        candidate = (ROOT / 'runtime/run-node.sh').read_text().splitlines()
        self.assertEqual(len(base), len(candidate))
        changed = [(a, b) for a, b in zip(base, candidate) if a != b]
        self.assertEqual([a.split('=')[0] for a, _ in changed], ['root', 'image', 'expected', 'name'])
        for name in ('launch-in-container.sh', 'runtime-preflight.py', 'verify-model.py', 'parse-cli.py'):
            self.assertEqual((ROOT.parent / name).read_bytes(), (ROOT / 'runtime' / name).read_bytes())


if __name__ == '__main__':
    unittest.main()
