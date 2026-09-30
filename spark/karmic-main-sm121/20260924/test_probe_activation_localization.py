import copy
import unittest

from probe_activation_localization import capture_downloads, check_coverage, complete_records, settle_capture


class Coverage(unittest.TestCase):
    def records(self):
        names = [f'layers.{layer}:{member}' for layer in (0, 1, 2, 8, 14, 19, 20, 39)
                 for member in range(5)] + ['layers.1.engram', 'layers.14.engram', '<output>:0']
        return [{'rank': rank, 'names': names[:]} for rank in range(4)]

    def report(self):
        return {'counts': {'covered': 3}, 'verdict': 'no-divergence-observed'}

    def test_full_coverage(self):
        check_coverage(self.records(), self.report())

    def test_tensor_return_without_auxiliary_states(self):
        records = self.records()
        for record in records:
            record['names'][-1] = '<output>'
        check_coverage(records, self.report())

    def test_missing_rank(self):
        with self.assertRaises(RuntimeError):
            check_coverage(self.records()[:-1], self.report())

    def test_missing_layer_or_output(self):
        for index in (0, -1):
            records = copy.deepcopy(self.records())
            records[2]['names'].pop(index)
            with self.assertRaises(RuntimeError):
                check_coverage(records, self.report())

    def test_no_aligned_coverage(self):
        for report in ({'counts': {'covered': 0}, 'verdict': 'no-coverage'},
                       {'counts': {'covered': 2}, 'verdict': 'no-divergence-observed'}):
            with self.assertRaises(RuntimeError):
                check_coverage(self.records(), report)

    def test_dropped_and_overflow(self):
        for key in ('dropped_before', 'overflow'):
            records = self.records()
            records[0][key] = 1
            with self.assertRaises(RuntimeError):
                check_coverage(records, self.report())

    def test_complete_record_counts(self):
        self.assertTrue(complete_records(self.records() * 3, 3))
        self.assertFalse(complete_records(self.records() * 2, 3))
        self.assertFalse(complete_records(self.records() * 4, 3))
        self.assertFalse(complete_records(self.records() * 3 + [self.records()[0]], 3))

    def test_all_ffn_branches_require_every_layer_and_rank(self):
        names = ['<output>:0'] + [name for layer in range(40) for name in (
            f'>layers.{layer}.ffn:0', f'layers.{layer}.ffn',
            f'layers.{layer}.ffn.gate:0', f'layers.{layer}.ffn.shared_experts')]
        records = [{'rank': rank, 'names': names[:]} for rank in range(4)]
        check_coverage(records, self.report(), 'all-ffn-branches')
        for rank in range(4):
            for layer in (0, 13, 20, 39):
                incomplete = copy.deepcopy(records)
                incomplete[rank]['names'].remove(f'layers.{layer}.ffn.shared_experts')
                with self.assertRaises(RuntimeError):
                    check_coverage(incomplete, self.report(), 'all-ffn-branches')

    def test_moe_seams_require_all_four_seams_on_every_rank(self):
        names = ['<output>:0'] + [f'layers.{layer}.ffn.experts#{seam}'
                 for layer in range(40)
                 for seam in ('shared', 'routed', 'pre_reduce', 'post_reduce')]
        records = [{'rank': rank, 'names': names[:], 'seam_runners': 40} for rank in range(4)]
        check_coverage(records, self.report(), 'moe-seams')
        for rank in range(4):
            incomplete = copy.deepcopy(records)
            incomplete[rank]['names'].remove('layers.13.ffn.experts#routed')
            with self.assertRaises(RuntimeError):
                check_coverage(incomplete, self.report(), 'moe-seams')
        records[0]['seam_runners'] = 39
        with self.assertRaises(RuntimeError):
            check_coverage(records, self.report(), 'moe-seams')

    def test_moe_inputs_require_every_input_router_and_seam(self):
        names = ['<output>:0'] + [f'layers.{layer}.ffn.experts#{seam}'
                 for layer in range(40)
                 for seam in ('shared', 'routed', 'pre_reduce', 'post_reduce')]
        names += [name for layer in range(40) for name in (
            f'>layers.{layer}.ffn:0', f'layers.{layer}.ffn.gate:0')]
        records = [{'rank': rank, 'names': names[:], 'seam_runners': 40} for rank in range(4)]
        check_coverage(records, self.report(), 'moe-inputs')
        for rank in range(4):
            for name in ('>layers.12.ffn:0', 'layers.12.ffn.gate:0',
                         'layers.12.ffn.experts#routed'):
                incomplete = copy.deepcopy(records)
                incomplete[rank]['names'].remove(name)
                with self.assertRaises(RuntimeError):
                    check_coverage(incomplete, self.report(), 'moe-inputs')

    def test_capture_requires_real_route_inputs_on_all_ranks(self):
        names = ['<output>:0'] + [f'layers.{layer}.ffn.experts#{seam}'
                 for layer in range(40) for seam in ('input', 'logits', 'topk_weights',
                 'topk_ids', 'input_after', 'shared', 'routed', 'pre_reduce', 'post_reduce')]
        records = [{'rank': rank, 'names': names[:], 'seam_runners': 40,
                    'capture': {'generation': 1, 'state': 'allocated'}} for rank in range(4)]
        check_coverage(records, self.report(), 'moe-capture')
        for rank in range(4):
            incomplete = copy.deepcopy(records)
            incomplete[rank]['names'].remove('layers.12.ffn.experts#topk_ids')
            with self.assertRaises(RuntimeError):
                check_coverage(incomplete, self.report(), 'moe-capture')
        for capture in (None, {'generation': True}, {'generation': 0},
                        {'generation': 1, 'state': 'error'}):
            incomplete = copy.deepcopy(records)
            incomplete[1]['capture'] = capture
            with self.assertRaises(RuntimeError):
                check_coverage(incomplete, self.report(), 'moe-capture')

    def test_capture_download_paths_are_scoped_and_rank_checked(self):
        row = {'arm': 'test-arm', 'rank': 1, 'node': 'toby', 'state': 'saved',
               'path': '/cache/ds41-act-trace/toby-441-g1-capture.pt'}
        self.assertEqual(capture_downloads([row], 'test-arm'), [('toby', 'toby-441-g1-capture.pt')])
        self.assertEqual(capture_downloads([row], 'another-arm'), [])
        for override in ({'rank': 0}, {'node': 'rusty'}, {'path': '/etc/passwd'},
                         {'path': '/cache/ds41-act-trace/../toby-441-g1-capture.pt'},
                         {'path': '/cache/ds41-act-trace/toby;bad-capture.pt'}):
            with self.assertRaises(RuntimeError):
                capture_downloads([dict(row, **override)], 'test-arm')

    def test_final_latch_flushes_and_waits_for_save(self):
        state = {'time': 0.0, 'flushed': 0}
        def pending():
            return [{'states': []}] if not state['flushed'] else (
                [{'states': ['fetching']}] if state['time'] < 1 else [])
        def flush(_):
            state['flushed'] += 1
        def sleep(seconds):
            state['time'] += seconds
        self.assertEqual(settle_capture(pending, flush, clock=lambda: state['time'], sleep=sleep), [])
        self.assertEqual(state['flushed'], 1)

    def test_capture_flush_is_bounded_and_errors_are_retained(self):
        state = {'time': 0.0, 'flushed': 0}
        def flush(_):
            state['flushed'] += 1
        def sleep(seconds):
            state['time'] += seconds
        error = [{'states': ['error']}]
        self.assertEqual(settle_capture(lambda: error, flush), error)
        self.assertEqual(state['flushed'], 0)
        pending = [{'states': []}]
        self.assertEqual(settle_capture(lambda: pending, flush, timeout=5,
            clock=lambda: state['time'], sleep=sleep), pending)
        self.assertEqual(state['flushed'], 3)
        self.assertLessEqual(state['time'], 5)


if __name__ == '__main__':
    unittest.main()
