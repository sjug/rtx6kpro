import unittest
import torch

from compare_indexer_inputs import compare_tensors, tensor_difference, compare_captures


def capture():
    return {'positions': torch.arange(524160, 524288, dtype=torch.int64),
            'hidden_input': torch.ones(128, 5120, dtype=torch.bfloat16),
            'kv_norm': torch.ones(128, 512, dtype=torch.bfloat16),
            'q_rotated': torch.ones(128, 2, 512, dtype=torch.bfloat16),
            'index_query_rotated': torch.ones(128, 32, 128, dtype=torch.bfloat16),
            'raw_weights': torch.ones(128, 32, dtype=torch.bfloat16),
            'scaled_weights': torch.full((128, 32), 1 / 64, dtype=torch.bfloat16),
            'projection_weight': torch.ones(32, 5120, dtype=torch.bfloat16)}


class IndexerInputs(unittest.TestCase):
    def test_capture_identity_and_plan_reporting(self):
        def wrapped(rows):
            layer = capture()
            layer.update(layer_id=2, chunk_rows=rows, chunk_row_start=rows - 128,
                plan={'lookup': 'rows', 'rows': rows, 'dtype': 'bfloat16', 'plan_handle': 42,
                      'query': {'max_rows': rows, 'in_features': 5120, 'out_features': 32},
                      'selection': {'config': {'backend': 'torch'}, 'source': 'fixture'}})
            return {'schema': 'ds41-indexer-capture-v1', 'layer': layer,
                    'meta': {'chunk_rows': rows, 'prompt_tokens': 524288, 'row_position': 524287,
                             'layer': 2, 'window_rows': 128, 'batch_requests': 1, 'problems': [],
                             'rank': 0, 'node': 'dusty', 'kit_sha256': 'kit',
                             'source_trees': {'vllm': 'v', 'b12x': 'b'}, 'tp_world_size': 4,
                             'hidden': 5120, 'heads': 2, 'index_heads': 32}}
        left, right = wrapped(8192), wrapped(4096)
        self.assertEqual(compare_captures(left, right)['plans']['right']['rows'], 4096)
        right['meta']['kit_sha256'] = 'different'
        with self.assertRaises(ValueError):
            compare_captures(left, right)

    def test_equal(self):
        report = compare_tensors(capture(), capture())
        self.assertTrue(all(v['bit_equal'] for v in report['fields'].values()))

    def test_hidden_difference_not_exonerated_by_equal_rounded_products(self):
        a, b = capture(), capture()
        b['hidden_input'][127, 1] += 0.125
        result = compare_tensors(a, b)
        self.assertTrue(result['finding'].startswith('hidden inputs differ'))
        self.assertEqual(result['fields']['hidden_input']['changed_rows'], [127])

    def test_projection_difference(self):
        a, b = capture(), capture()
        b['raw_weights'][127, 2] += 0.125
        self.assertIn('bit-identical captured inputs', compare_tensors(a, b)['finding'])

    def test_scale_difference(self):
        a, b = capture(), capture()
        b['scaled_weights'][1, 1] += 0.125
        self.assertTrue(compare_tensors(a, b)['finding'].startswith('scaled outputs differ'))

    def test_invalid_weights_positions_and_nonfinite(self):
        for key, mutate in (('projection_weight', lambda t: t.add_(1)),
                            ('positions', lambda t: t.add_(1)),
                            ('raw_weights', lambda t: t.fill_(float('nan')))):
            with self.subTest(key=key):
                a, b = capture(), capture()
                mutate(b[key])
                with self.assertRaises(ValueError):
                    compare_tensors(a, b)

    def test_bit_equality_distinguishes_signed_zero(self):
        r = tensor_difference(torch.tensor([0.]), torch.tensor([-0.]))
        self.assertFalse(r['bit_equal'])
        self.assertEqual(r['changed_values'], 0)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
