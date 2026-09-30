import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from distribute_diagnostic import resolve, KINDS, image_tag


class Receipt(unittest.TestCase):
    def test_ratio1_uses_actual_builder_tag(self):
        from build_ratio1 import TAG
        self.assertEqual(image_tag('ratio1'), TAG)
        self.assertEqual(image_tag('precision-release'),
                         'localhost/voipmonitor/build-components:ds41-precision-release')

    def test_window_has_separate_lock(self):
        self.assertEqual(KINDS['window'], 'claude-window')
        self.assertNotEqual(KINDS['window'], KINDS['decision-row'])

    def test_lock_gate_and_scope_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = root / 'receipts' / 'build'
            receipt.mkdir(parents=True)
            image = 'a' * 64
            lock = root / 'claude-decision-row.lock.json'
            lock.write_text('{}')
            data = {'directory': str(receipt), 'image_id': image,
                    'lock_sha256': hashlib.sha256(lock.read_bytes()).hexdigest()}
            path = root / 'receipts/decision-row-build-receipt.json'
            path.write_text(json.dumps(data))
            (receipt / 'BUILD-OK').write_text(image + '\n')
            self.assertEqual(resolve('decision-row', root)[0], data)
            lock.write_text('{"changed": true}')
            with self.assertRaisesRegex(RuntimeError, 'lock changed'):
                resolve('decision-row', root)
            (receipt / 'BUILD-OK').write_text('b' * 64)
            with self.assertRaisesRegex(RuntimeError, 'gate identity'):
                resolve('decision-row', root)
            data['directory'] = str(root)
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(RuntimeError, 'outside'):
                resolve('decision-row', root)


if __name__ == '__main__':
    unittest.main()
