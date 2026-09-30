import unittest
import torch
from compare_window_baseline import compare
from test_compare_decision_grids import capture


def complete():
    result = capture()
    for row in result['layers'].values():
        row.update(swa_len=128, swa_records=torch.zeros(128, 528, dtype=torch.uint8))
    return result


class WindowBaseline(unittest.TestCase):
    def test_new_identity_and_addresses_are_not_numerical_differences(self):
        old, new = complete(), complete()
        new['meta']['source_trees'] = {'vllm': 'new-capture', 'b12x': 'b'}
        new['meta']['kit_sha256'] = 'new-kit'
        old['layers'][0]['swa_slots'] = torch.arange(128)
        new['layers'][0]['swa_slots'] = torch.arange(128) + 1280
        self.assertTrue(compare(old, new)['equal_observations'])

    def test_actual_bytes_and_signed_zero_are_detected(self):
        old, new = complete(), complete()
        new['layers'][1]['swa_records'][25, 7] = 1
        result = compare(old, new)
        self.assertFalse(result['equal_observations'])
        self.assertFalse(result['layers'][1]['swa_records'])
        new = complete()
        new['layers'][0]['attn_sink'][0] = -0.0
        self.assertFalse(compare(old, new)['equal_observations'])

    def test_wrong_chunk_or_plan_is_not_an_arithmetic_difference(self):
        old, new = complete(), complete()
        new['layers'][0]['plan'] = {'config': 'changed'}
        with self.assertRaises(ValueError):
            compare(old, new)


if __name__ == '__main__':
    unittest.main()
