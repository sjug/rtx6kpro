import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from select_router_candidate import candidate
from select_router_parent import restored_parent


class RouterCandidateTests(unittest.TestCase):
    def setUp(self):
        self.parent = {'image_id': 'parent', 'diagnostic': {}, 'recipe_sha256': 'recipe',
                       'build_lock_sha256': 'build', 'source_lock_sha256': 'source'}
        self.lock = {'base_image_id': 'parent', 'trees': {'vllm': 'v', 'b12x': 'b'}}
        self.receipt = {'image_id': 'child', 'lock_sha256': 'lock'}

    def test_only_image_and_diagnostic_change(self):
        result = candidate(self.parent, self.lock, self.receipt, 'lock')
        self.assertEqual(result['image_id'], 'child')
        self.assertEqual(result['diagnostic']['kind'], 'router-stage-release-candidate')
        self.assertEqual(result['diagnostic']['b12x_tree'], 'b')
        for key in self.parent.keys() - {'image_id', 'diagnostic'}:
            self.assertEqual(result[key], self.parent[key])
        self.assertEqual(self.parent['image_id'], 'parent')

    def test_wrong_parent_rejected(self):
        self.parent['image_id'] = 'other'
        with self.assertRaises(RuntimeError):
            candidate(self.parent, self.lock, self.receipt, 'lock')

    def test_wrong_lock_rejected(self):
        with self.assertRaises(RuntimeError):
            candidate(self.parent, self.lock, self.receipt, 'different')

    def test_router_and_parent_launch_commands_match(self):
        # Extract the pure command renderer without importing the executable
        # orchestrator, which intentionally performs node operations at top level.
        source = ast.parse(Path(__file__).with_name('start_moe_repaired.py').read_text())
        function = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'launch_command')
        module = ast.Module(body=[function], type_ignores=[])
        for node in ('dusty', 'toby', 'rusty', 'kirby'):
            rendered = []
            for arm in ('engram-repair', 'router-fence'):
                namespace = {'args': SimpleNamespace(arm=arm, cuda_module_loading=None), 'REMOTE': '/task'}
                exec(compile(module, 'start_moe_repaired.py', 'exec'), namespace)
                rendered.append(namespace['launch_command'](node))
            self.assertEqual(rendered[0], rendered[1])
            self.assertIn('B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1', rendered[1])
            self.assertIn('B12X_DENSE_SPLITK_TURBO=0', rendered[1])

    def amendment(self):
        parent = dict(self.parent, diagnostic={'kind': 'engram-prequeued-native-io-fix'})
        prior = {'candidate.json': hashlib.sha256((json.dumps(parent, sort_keys=True, indent=2) + '\n').encode()).hexdigest(),
                 'runtime.py': 'unchanged'}
        current = candidate(parent, self.lock, self.receipt, 'lock')
        manifest = dict(prior, **{'candidate.json': 'current'})
        return {'prior_candidate': parent, 'prior_manifest': prior,
                'candidate': current, 'manifest': manifest}

    def test_symmetric_restore(self):
        a = self.amendment()
        self.assertEqual(restored_parent(a, a['candidate'], a['manifest']),
                         (a['prior_candidate'], a['prior_manifest']))

    def test_reject_changed_parent_digest(self):
        a = self.amendment()
        a['prior_candidate']['recipe_sha256'] = 'tampered'
        with self.assertRaises(RuntimeError):
            restored_parent(a, a['candidate'], a['manifest'])

    def test_reject_changed_current(self):
        a = self.amendment()
        with self.assertRaises(RuntimeError):
            restored_parent(a, dict(a['candidate'], image_id='other'), a['manifest'])

    def test_reject_other_runtime_change(self):
        a = self.amendment()
        a['prior_manifest']['runtime.py'] = 'other'
        with self.assertRaises(RuntimeError):
            restored_parent(a, a['candidate'], a['manifest'])


if __name__ == '__main__':
    unittest.main()
