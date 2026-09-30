import tempfile
import unittest
from pathlib import Path

import torch

from replay_indexer_projection import load_pair, place_rows, reduction_mode
from reference_indexer_projection import references


class ReplayTests(unittest.TestCase):
    def test_tail_placement(self):
        x = torch.tensor([[1., -2.], [3., 4.]], dtype=torch.bfloat16)
        y = place_rows(x, 5, 'cpu')
        self.assertTrue(torch.equal(y[-2:], x))
        self.assertEqual(int(torch.count_nonzero(y[:-2])), 0)
        self.assertTrue(y.is_contiguous())
        with self.assertRaises(ValueError):
            place_rows(x, 1, 'cpu')

    def test_flag_restored_on_failure(self):
        old = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        with self.assertRaisesRegex(RuntimeError, 'sentinel'):
            with reduction_mode(not old):
                self.assertEqual(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction, not old)
                raise RuntimeError('sentinel')
        self.assertEqual(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction, old)

    def test_reference_known_arithmetic(self):
        x = torch.tensor([[1., 2., -1., .5]], dtype=torch.bfloat16)
        w = torch.tensor([[2., 1., -3., 2.]], dtype=torch.bfloat16)
        precise, variants = references(x, w)
        self.assertEqual(precise.item(), 8)
        self.assertTrue(all(v.item() == 8 for v in variants.values()))

    def test_hash_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'invalid.pt'
            p.write_bytes(b'not a torch file')
            with self.assertRaisesRegex(ValueError, 'digest mismatch'):
                load_pair((p, p), ('0' * 64, '0' * 64))


if __name__ == '__main__':
    unittest.main()
