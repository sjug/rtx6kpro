"""Stdlib-only fail-closed coverage tests for the dense replay inventory."""
import json
from pathlib import Path
import tempfile
import unittest

from claude_dense_regression import enumerate_winners

ROOT = Path(__file__).resolve().parent


class Coverage(unittest.TestCase):
    def test_all_original_dense_records_including_draft(self):
        _, rows = enumerate_winners(ROOT / 'receipts/combined-tuning.json', turbo=True)
        self.assertEqual(len(rows), 152)
        self.assertEqual(sum(k == 15360 and n == 5120 for k, n, _, _ in rows), 19)

    def test_complete_nonatomic_control(self):
        _, rows = enumerate_winners(ROOT / 'receipts/non-atomic-complete-control-tuning.json', turbo=False)
        self.assertEqual(len(rows), 64)
        self.assertEqual(sum(k == 15360 for k, _, _, _ in rows), 8)
        self.assertTrue(all(config['split_k_slices'] in (1, 2) for *_, config in rows))

    def test_wrong_codegen_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'coverage mismatch'):
            enumerate_winners(ROOT / 'receipts/combined-tuning.json', turbo=False)

    def test_unknown_dense_record_rejected(self):
        data = json.loads((ROOT / 'receipts/combined-tuning.json').read_text())
        record = next(value for value in data['records'].values()
                      if 'split_k_slices' in value['assignment'])
        data['records']['unknown-dense-key'] = record
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json') as stream:
            json.dump(data, stream)
            stream.flush()
            with self.assertRaisesRegex(RuntimeError, '1 unmatched'):
                enumerate_winners(Path(stream.name), turbo=True)


if __name__ == '__main__':
    unittest.main()
