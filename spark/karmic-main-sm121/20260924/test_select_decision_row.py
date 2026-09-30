import unittest
import hashlib
import json
from select_decision_row import candidate, restored


class DecisionSelection(unittest.TestCase):
    def setUp(self):
        self.parent = {'image_id': 'parent', 'diagnostic': {}, 'recipe_sha256': 'recipe'}
        self.lock = {'base_image_id': 'parent', 'trees': {'vllm': 'v', 'b12x': 'b'}}
        self.receipt = {'image_id': 'child', 'lock_sha256': 'lock'}

    def test_only_image_and_diagnostic_change(self):
        result = candidate(self.parent, self.lock, self.receipt, 'lock')
        self.assertEqual(result['image_id'], 'child')
        self.assertEqual(result['diagnostic']['kind'], 'decision-row-capture')
        self.assertEqual(result['diagnostic']['b12x_tree'], 'b')
        self.assertEqual(result['recipe_sha256'], self.parent['recipe_sha256'])
        self.assertEqual(self.parent['image_id'], 'parent')

    def test_wrong_parent_rejected(self):
        with self.assertRaises(RuntimeError):
            candidate(dict(self.parent, image_id='other'), self.lock, self.receipt, 'lock')

    def test_window_kind_changes_no_model_settings(self):
        parent = dict(self.parent, model_revision='frozen', other_setting=8192)
        result = candidate(parent, self.lock, self.receipt, 'lock', 'window-capture')
        self.assertEqual(result['diagnostic']['kind'], 'window-capture')
        self.assertEqual({k: v for k, v in result.items() if k not in ('image_id', 'diagnostic')},
                         {k: v for k, v in parent.items() if k not in ('image_id', 'diagnostic')})

    def test_wrong_lock_rejected(self):
        with self.assertRaises(RuntimeError):
            candidate(self.parent, self.lock, self.receipt, 'other')

    def test_fault_kind_is_explicit_and_other_kinds_rejected(self):
        self.assertEqual(candidate(self.parent, self.lock, self.receipt, 'lock',
                                   'engram-fault-inject')['diagnostic']['kind'], 'engram-fault-inject')
        with self.assertRaisesRegex(RuntimeError, 'Unsupported'):
            candidate(self.parent, self.lock, self.receipt, 'lock', 'other')

    def test_restore_requires_exact_amendment_and_only_identity_change(self):
        parent = dict(self.parent, diagnostic={'kind': 'router'})
        lock = dict(self.lock, base_kind='router')
        child = candidate(parent, lock, self.receipt, 'lock')
        digest = hashlib.sha256((json.dumps(parent, sort_keys=True, indent=2) + '\n').encode()).hexdigest()
        prior = {'candidate.json': digest, 'runtime.py': 'unchanged'}
        manifest = dict(prior, **{'candidate.json': 'child-digest'})
        amendment = {'candidate': child, 'manifest': manifest,
                     'prior_candidate': parent, 'prior_manifest': prior}
        self.assertEqual(restored(amendment, child, manifest, lock), (parent, prior))
        with self.assertRaisesRegex(RuntimeError, 'Current runtime'):
            restored(amendment, dict(child, image_id='other'), manifest, lock)
        with self.assertRaisesRegex(RuntimeError, 'declared router parent'):
            restored(amendment, child, manifest, dict(lock, base_image_id='other'))
        amendment['prior_manifest'] = dict(prior, **{'runtime.py': 'changed'})
        with self.assertRaisesRegex(RuntimeError, 'other runtime'):
            restored(amendment, child, manifest, lock)
