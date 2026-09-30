import copy
import unittest
import compare_window_receipts as comparison

from compare_window_receipts import validate_pair, NODES


class MatchedWindows(unittest.TestCase):
    def setUp(self):
        self.left = dict(chunk_rows=8192, image_id='same', blocks=81389,
                         mode='matched', regime_source='pinned',
                         signature_equals_control=True, regime_equals_reference=True,
                         capture_problems={node: [] for node in NODES})
        self.right = copy.deepcopy(self.left)
        self.right['chunk_rows'] = 4096

    def test_matched(self):
        validate_pair(self.left, self.right)

    def test_reject_drift(self):
        for key, value in [('image_id', 'other'), ('blocks', 2), ('regime_source', 'other'),
                           ('mode', 'other'), ('chunk_rows', 8192),
                           ('signature_equals_control', False), ('regime_equals_reference', False)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_pair(self.left, {**self.right, key: value})

    def test_reject_mixed_geometry_arms(self):
        with self.assertRaises(ValueError):
            validate_pair(self.left, {**self.right, 'nccl_geometry': 'tree-simple-1ch'})
        validate_pair({**self.left, 'nccl_geometry': 'tree-simple-1ch'},
                      {**self.right, 'nccl_geometry': 'tree-simple-1ch'})

    def test_reject_capture_failure_or_missing_rank(self):
        for problems in ({}, {**self.right['capture_problems'], 'kirby': ['bad']}):
            with self.assertRaises(ValueError):
                validate_pair(self.left, {**self.right, 'capture_problems': problems})


class CompletedAudit(unittest.TestCase):
    def setUp(self):
        self.baselines = {n: {'equal_observations': True} for n in NODES}
        layer = {'packed': {'rows_bit_equal': 128, 'rows_not_in_gathered_records': []},
                 'slots': {'write_slots_equal_last_row_window': True,
                           'write_offsets_equal_position_offsets': True}}
        report = {f'within_{s}': {str(l): copy.deepcopy(layer) for l in (0, 1)}
                  for s in ('left', 'right')}
        report.update({f'decision_consistency_{s}': {str(l): {'all_equal': True} for l in (0, 1)}
                       for s in ('left', 'right')})
        self.reports = {n: copy.deepcopy(report) for n in NODES}
        self.ranks = {'replicated_all_equal': True}

    def validate(self):
        comparison.validate_completed_audits(self.baselines, self.reports, self.ranks)

    def test_completed_audit_passes(self):
        self.validate()

    def test_reject_failed_baseline(self):
        self.baselines['kirby']['equal_observations'] = False
        with self.assertRaises(ValueError):
            self.validate()

    def test_reject_failed_sibling(self):
        self.reports['dusty']['decision_consistency_right']['1']['all_equal'] = False
        with self.assertRaises(ValueError):
            self.validate()

    def test_reject_failed_pack(self):
        self.reports['rusty']['within_left']['0']['packed']['rows_bit_equal'] = 127
        with self.assertRaises(ValueError):
            self.validate()

    def test_reject_failed_slots(self):
        self.reports['toby']['within_left']['1']['slots']['write_offsets_equal_position_offsets'] = False
        with self.assertRaises(ValueError):
            self.validate()

    def test_reject_failed_replication(self):
        self.ranks['replicated_all_equal'] = False
        with self.assertRaises(ValueError):
            self.validate()


if __name__ == '__main__':
    unittest.main()
