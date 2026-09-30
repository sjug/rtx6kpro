import hashlib
import json
import unittest
from unittest.mock import patch

from router_tuning_transfer import validate


class SelectionIdentityTests(unittest.TestCase):
    def setUp(self):
        identity = {'compute_capability': [12, 1], 'schema_version': 6}
        self.data = {'identity': identity, 'records': {'query': {'config': {'backend': 'prefill'}}}}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        self.mock = patch('router_tuning_transfer.RELATIVE', 'preparation/' + digest + '.json')
        self.mock.start()
        self.addCleanup(self.mock.stop)

    def test_identity_matches_filename(self):
        self.assertEqual(validate(json.dumps(self.data)), self.data)

    def test_identity_drift_rejected(self):
        self.data['identity']['schema_version'] = 7
        with self.assertRaises(RuntimeError):
            validate(json.dumps(self.data))

    def test_empty_cache_rejected(self):
        self.data['records'] = {}
        with self.assertRaises(RuntimeError):
            validate(json.dumps(self.data))

    def test_missing_identity_rejected(self):
        del self.data['identity']
        with self.assertRaises(RuntimeError):
            validate(json.dumps(self.data))


if __name__ == '__main__':
    unittest.main()
