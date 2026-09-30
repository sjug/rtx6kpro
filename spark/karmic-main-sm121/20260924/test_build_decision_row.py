import json
from pathlib import Path
import tempfile
import unittest

from build_decision_row import ROOT, STEM, SMOKE, frozen_inputs


class DecisionBuildInputs(unittest.TestCase):
    def test_smoke_checks_loaded_attention_and_helper(self):
        import ast
        tree = ast.parse(SMOKE)
        self.assertFalse(any(isinstance(n, ast.Assert) for n in ast.walk(tree)))
        for required in ('attention._claude_row is not helper', 'module.__file__',
                         'hashlib.sha256(path.read_bytes())', "['output_sha256']",
                         'torch.cuda.get_device_capability()', 'helper.SCHEMA'):
            self.assertIn(required, SMOKE)
        source = (ROOT / 'build_decision_row.py').read_text()
        self.assertIn("'--device', 'nvidia.com/gpu=all'", source)

    def copy_inputs(self, target):
        raw = (ROOT / (STEM + '.lock.json')).read_bytes()
        lock = json.loads(raw)
        for name in [*lock['inputs'], STEM + '.lock.json', STEM + '.ignore', lock['base_lock']]:
            (target / name).write_bytes((ROOT / name).read_bytes())
        (target / 'receipts').mkdir()
        receipt = 'receipts/router-release-build-receipt.json'
        (target / receipt).write_bytes((ROOT / receipt).read_bytes())
        return lock

    def test_frozen_kit(self):
        lock, digest = frozen_inputs()
        self.assertEqual(len(digest), 64)
        self.assertEqual(lock['base_kind'], 'router-stage-release-candidate')

    def test_tampered_input_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            raw = (ROOT / (STEM + '.lock.json')).read_bytes()
            (target / (STEM + '.lock.json')).write_bytes(raw)
            lock = json.loads(raw)
            for name in lock['inputs']:
                (target / name).write_bytes((ROOT / name).read_bytes())
            name = next(iter(lock['inputs']))
            (target / name).write_bytes((target / name).read_bytes() + b'\n')
            with self.assertRaisesRegex(RuntimeError, 'input drift'):
                frozen_inputs(target)

    def test_no_serving_tag_or_implicit_pull(self):
        source = (ROOT / 'build_decision_row.py').read_text()
        self.assertIn("'--pull=never'", source)
        self.assertIn("'--network=none'", source)
        self.assertNotIn('localhost/voipmonitor/vllm:', source)
        self.assertIn('idle()', source)

    def test_ignore_drift_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            self.copy_inputs(target)
            (target / (STEM + '.ignore')).write_text('**\n')
            with self.assertRaisesRegex(RuntimeError, 'context differs'):
                frozen_inputs(target)

    def test_base_receipt_drift_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            self.copy_inputs(target)
            path = target / 'receipts/router-release-build-receipt.json'
            receipt = json.loads(path.read_text())
            receipt['image_id'] = 'wrong'
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(RuntimeError, 'not based on'):
                frozen_inputs(target)

    def test_base_lock_drift_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            lock = self.copy_inputs(target)
            path = target / lock['base_lock']
            path.write_bytes(path.read_bytes() + b'\n')
            with self.assertRaisesRegex(RuntimeError, 'Local router lock'):
                frozen_inputs(target)


if __name__ == '__main__':
    unittest.main()
