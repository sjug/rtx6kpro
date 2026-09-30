import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from build_mhc_expanded import frozen, STEM
from distribute_diagnostic import KINDS, image_tag


class FrozenBuild(unittest.TestCase):
    def test_distribution_identity(self):
        self.assertEqual(KINDS['mhc-expanded'], STEM)
        self.assertEqual(image_tag('mhc-expanded'),
                         'localhost/voipmonitor/build-components:ds41-mhc-expanded')

    def test_preflight_rejects_drift_before_remote_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'receipts').mkdir()
            source = root / 'helper.py'
            source.write_bytes(b'pass\n')
            base = root / 'base.json'
            base.write_bytes(b'{}\n')
            lock = {'inputs': {'helper.py': hashlib.sha256(source.read_bytes()).hexdigest()},
                    'base_image_id': 'a' * 64, 'base_lock': 'base.json',
                    'base_lock_sha256': hashlib.sha256(base.read_bytes()).hexdigest()}
            (root / (STEM + '.lock.json')).write_text(json.dumps(lock))
            receipt = root / 'receipts/precision-release-build-receipt.json'
            receipt.write_text(json.dumps({'image_id': lock['base_image_id'],
                                           'lock_sha256': lock['base_lock_sha256']}))
            self.assertEqual(frozen(root)[0], lock)
            source.write_bytes(b'changed\n')
            with self.assertRaisesRegex(RuntimeError, 'input drift'):
                frozen(root)
            source.write_bytes(b'pass\n')
            base.write_bytes(b'changed\n')
            with self.assertRaisesRegex(RuntimeError, 'Base lock differs'):
                frozen(root)
            base.write_bytes(b'{}\n')
            receipt.write_text(json.dumps({'image_id': 'b' * 64,
                                           'lock_sha256': lock['base_lock_sha256']}))
            with self.assertRaisesRegex(RuntimeError, 'Base build receipt differs'):
                frozen(root)


if __name__ == '__main__':
    unittest.main()
