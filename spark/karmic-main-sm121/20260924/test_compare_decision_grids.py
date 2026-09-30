"""CPU-only descriptive comparison tests; no model or node operations."""
import unittest
import torch
from compare_decision_grids import compare_rank, tensor_delta, selection_delta, swa_delta


def capture(chunk=8192):
    return {'schema': 'claude-decision-row-v3', 'meta': {
        'rank': 0, 'node': 'dusty', 'kit_sha256': 'kit',
        'source_trees': {'vllm': 'v', 'b12x': 'b'}, 'problems': [],
        'prompt_tokens': 524288, 'row_position': 524287,
        'chunk_rows': chunk, 'row_index': chunk - 1, 'batch_requests': 1},
        'layers': {i: {'position': 524287, 'query_rows': chunk if i < 20 else 128,
                      'row': (chunk if i < 20 else 128) - 1, 'ced_decoder': i >= 20,
                      'q': torch.ones(16, 512, dtype=torch.bfloat16),
                      'out': torch.ones(16, 512, dtype=torch.bfloat16),
                      'attn_sink': torch.zeros(16), 'plan': {'config': 'same'}}
                   for i in range(40)}}


class GridComparisonTests(unittest.TestCase):
    def test_swa_compares_only_active_records_and_decoded_values(self):
        a = {'swa_len': 2, 'swa_records': torch.zeros(3, 528, dtype=torch.uint8)}
        a['swa_records'][:, 512:] = 127
        b = {'swa_len': 2, 'swa_records': a['swa_records'].clone()}
        b['swa_records'][2] = 42
        self.assertTrue(swa_delta(a, b)['decoded_values']['equal_values'])
        b['swa_records'][1, 0] = 56  # E4M3 1.0, UE8M0 scale 1.0
        d = swa_delta(a, b)
        self.assertEqual(d['changed_row_offsets'], [1])
        self.assertEqual(d['decoded_values']['max_abs'], 1.0)
        self.assertFalse(d['raw_row_multiset_equal'])
        c = {'swa_len': 2, 'swa_records': b['swa_records'].clone()}
        c['swa_records'][:2] = b['swa_records'][:2].flip(0)
        self.assertTrue(swa_delta(b, c)['raw_row_multiset_equal'])
        self.assertFalse(swa_delta(b, c)['decoded_values']['equal_values'])
        b['swa_len'] = 1
        with self.assertRaises(ValueError):
            swa_delta(a, b)

    def test_self_and_chunk_geometry_alone_do_not_diverge(self):
        report = compare_rank(capture(), capture(4096))
        self.assertIsNone(report['first_differing_captured_layer'])
        self.assertEqual(report['scope'], 'descriptive-only; experiment comparability not established')
        self.assertNotIn('pass', report)

    def test_reports_first_observed_not_first_causal_layer(self):
        a, b = capture(), capture(4096)
        b['layers'][7]['q'][0, 0] = 2
        b['layers'][8]['out'][0, 0] = 3
        report = compare_rank(a, b)
        self.assertEqual(report['first_differing_captured_layer'], 7)
        self.assertEqual(report['first_differing_query_layer'], 7)
        self.assertEqual(report['first_differing_output_layer'], 8)
        self.assertEqual(report['layers'][7]['q']['changed_elements'], 1)
        self.assertEqual(report['layers'][7]['q']['max_abs'], 1)

    def test_refuses_incomplete_or_mismatched_evidence(self):
        for mutate in [lambda c: c['layers'].pop(39),
                       lambda c: c['meta'].update(rank=1),
                       lambda c: c['meta'].update(source_trees={'b12x': 'other'}),
                       lambda c: c['meta'].update(problems=['capture failed']),
                       lambda c: c['layers'][20].update(row=4095),
                       lambda c: c['layers'][1].update(plan={'config': 'different'})]:
            a, b = capture(), capture(4096)
            mutate(b)
            with self.assertRaises(ValueError):
                compare_rank(a, b)

    def test_norms_and_zero_denominator(self):
        d = tensor_delta(torch.tensor([3., 4.]), torch.tensor([0., 4.]))
        self.assertEqual(d['left_l2'], 5)
        self.assertAlmostEqual(d['relative_l2_to_left'], .6)
        self.assertIsNone(tensor_delta(torch.zeros(2), torch.ones(2))['relative_l2_to_left'])

    def test_nonfinite_and_shape_mismatch_rejected(self):
        for b in [torch.tensor([float('nan')]), torch.tensor([float('inf')]), torch.ones(2)]:
            with self.assertRaises(ValueError):
                tensor_delta(torch.ones(1), b)

    def test_selection_is_logical_set_and_order(self):
        result = selection_delta(torch.tensor([1, 2, -1]), torch.tensor([2, 3, -1]))
        self.assertEqual(result['intersection'], 1)
        self.assertEqual(result['left_only'], [1])
        self.assertEqual(result['right_only'], [3])
        self.assertAlmostEqual(result['jaccard'], 1 / 3)
        with self.assertRaises(ValueError):
            selection_delta(torch.tensor([1, 1]), torch.tensor([1, 2]))

    def test_indexer_presence_must_agree(self):
        a, b = capture(), capture(4096)
        b['layers'][2]['indexer'] = {}
        with self.assertRaises(ValueError):
            compare_rank(a, b)

    def test_indexer_query_selection_and_key_differences(self):
        a, b = capture(), capture(4096)
        for c in (a, b):
            c['layers'][2]['indexer'] = {
                'cache_length': 512, 'page_size': 256,
                'q_data': torch.zeros(1, 64, dtype=torch.uint8),
                'q_scales': torch.full((1, 4), 127, dtype=torch.uint8),
                'weights': torch.ones(1), 'topk': torch.tensor([0, 1]),
                'candidates': torch.tensor([0, 1, 2, 999, 999]), 'candidate_len': 3,
                'key_pages_sha256': 'same'}
        self.assertIsNone(compare_rank(a, b)['first_differing_captured_layer'])
        b['layers'][2]['indexer']['candidates'][3:] = -999
        self.assertIsNone(compare_rank(a, b)['first_differing_captured_layer'])
        x = b['layers'][2]['indexer']
        x['q_data'][0, 0] = 1
        x['topk'] = torch.tensor([1, 2])
        x['candidates'] = torch.tensor([1, 2, 3])
        x['key_pages_sha256'] = 'different'
        report = compare_rank(a, b)
        self.assertEqual(report['first_differing_captured_layer'], 2)
        delta = report['layers'][2]['indexer']
        self.assertEqual(delta['query']['changed_elements'], 1)
        self.assertEqual(delta['topk']['right_only'], [2])
        self.assertEqual(delta['candidates']['left_only'], [0])
        self.assertFalse(delta['key_pages_hash_equal'])
        x['page_size'] = 128
        with self.assertRaises(ValueError):
            compare_rank(a, b)


if __name__ == '__main__':
    unittest.main()
