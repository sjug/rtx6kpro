import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch
from diagnostic_overlay import IMAGE, SOURCE, TARGET, validate


class OverlayTests(unittest.TestCase):
    def pin(self):
        return {'image_id': IMAGE, 'diagnostic': {'kind': 'attention-trace'},
                'diagnostic_overlay': {'source': SOURCE, 'target': TARGET,
                                       'sha256': hashlib.sha256(b'helper').hexdigest()}}

    def test_optional(self):
        self.assertIsNone(validate({}, Path('.')))

    def test_identity(self):
        pin = self.pin()
        with patch.object(Path, 'read_bytes', return_value=b'helper'):
            self.assertEqual(validate(pin, Path('.')), pin['diagnostic_overlay'])
            self.assertEqual(validate(pin, Path('.'), installed=True), pin['diagnostic_overlay'])
        with patch.object(Path, 'read_bytes', return_value=b'wrong'):
            with self.assertRaises(RuntimeError):
                validate(pin, Path('.'))

    def test_reject_other_mounts(self):
        for field, value in [('source', '../other.py'), ('target', '/etc/passwd')]:
            pin = self.pin()
            pin['diagnostic_overlay'][field] = value
            with self.assertRaises(RuntimeError):
                validate(pin, Path('.'))
        pin = self.pin()
        pin['image_id'] = 'wrong'
        with self.assertRaises(RuntimeError):
            validate(pin, Path('.'))


if __name__ == '__main__':
    unittest.main()
