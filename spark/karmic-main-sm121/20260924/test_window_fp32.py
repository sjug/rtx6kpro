import unittest
import json

import torch

from analyze_window_fp32 import analyze


def parts(values):
    return [torch.tensor([[v]], dtype=torch.bfloat16) for v in values]


class FP32OrderStudy(unittest.TestCase):
    def test_narrow_exponents_match_all_orders(self):
        result = analyze(parts((256, 1, -256, 1)))
        self.assertEqual(result['orders_tested'], 27)
        self.assertTrue(result['all_orders_bit_equal_after_bf16_round'])
        self.assertTrue(result['all_orders_match_fp64_round_once'])
        self.assertTrue(result['fp64_exactness_sufficient_bound_pass'])

    def test_fp32_is_not_universally_order_invariant(self):
        result = analyze(parts((2**30, 1, -(2**30), 1)))
        self.assertFalse(result['all_orders_bit_equal_after_bf16_round'])
        self.assertFalse(result['all_orders_match_fp64_round_once'])
        self.assertTrue(result['fp64_exactness_sufficient_bound_pass'])

    def test_do_not_call_fp64_universally_exact(self):
        result = analyze(parts((2**100, 1, -(2**100), 1)))
        self.assertFalse(result['fp64_exactness_sufficient_bound_pass'])

    def test_zero_inputs(self):
        result = analyze(parts((0, 0, 0, 0)))
        self.assertEqual(result['max_nonzero_exponent_span'], 0)
        self.assertTrue(result['all_orders_match_fp64_round_once'])

    def test_overflow_is_recorded_without_invalid_json(self):
        value = torch.finfo(torch.bfloat16).max
        result = analyze(parts((value, value, -value, -value)))
        self.assertTrue(any(r['nonfinite_output_elements'] for r in result['orders'].values()))
        json.dumps(result, allow_nan=False)

    def test_reject_nonfinite_wrong_shape_or_dtype(self):
        cases = [parts((1, 2, 3, float('inf'))), parts((1, 2, 3, float('nan'))),
                 [torch.ones(1, 1)] * 4, parts((1, 2, 3))]
        bad_shape = parts((1, 2, 3, 4))
        bad_shape[0] = torch.ones(2, 1, dtype=torch.bfloat16)
        cases.append(bad_shape)
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                analyze(case)


if __name__ == '__main__':
    unittest.main()
