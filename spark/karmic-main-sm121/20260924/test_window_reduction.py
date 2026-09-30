import unittest
import torch
from analyze_window_reduction import references, analyze


class ReductionArithmetic(unittest.TestCase):
    def test_rounding_association_changes_sum(self):
        parts = [torch.tensor([[v]], dtype=torch.bfloat16) for v in (256, 1, -256, 1)]
        refs = dict(references(parts))
        self.assertEqual(refs['fp64_sum_round_once'].item(), 2)
        self.assertEqual(refs['bf16_sequential_0123'].item(), 1)
        self.assertEqual(refs['bf16_sequential_0213'].item(), 2)
        result = analyze(parts, {'observed': refs['bf16_sequential_0123']})['observed']
        self.assertEqual(result['references']['bf16_sequential_0123']['exact_rows'], [0])
        self.assertEqual(result['references']['fp64_sum_round_once']['exact_rows'], [])

    def test_requires_four_bf16_parts(self):
        with self.assertRaises(ValueError):
            list(references([torch.ones(1, 1)] * 4))
        with self.assertRaises(ValueError):
            list(references([torch.ones(1, 1, dtype=torch.bfloat16)] * 3))


if __name__ == '__main__':
    unittest.main()
