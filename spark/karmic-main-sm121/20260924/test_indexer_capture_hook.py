import ast
from pathlib import Path
import unittest

from prepare_indexer_capture_hook import compose, verify_only_hook_changed

SOURCE = Path(__file__).with_name('claude-window-attention.py').read_text()


class IndexerCaptureSeam(unittest.TestCase):
    def test_current_source_compiles_with_only_observation_added(self):
        instrumented = compose(SOURCE)
        compile(instrumented, 'draft-attention.py', 'exec')
        verify_only_hook_changed(SOURCE, instrumented)

    def test_changed_source_fails_closed(self):
        with self.assertRaises(ValueError):
            compose(SOURCE + '\n')

    def test_numerical_change_is_rejected(self):
        instrumented = compose(SOURCE).replace('weights = self.indexer.weights_proj(hidden_states)',
                                               'weights = self.indexer.weights_proj(hidden_states.float())')
        with self.assertRaises(ValueError):
            verify_only_hook_changed(SOURCE, instrumented)

    def test_hook_observes_raw_and_scaled_values_after_scaling(self):
        tree = ast.parse(compose(SOURCE))
        hooks = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                 and n.func.value.id == '_indexer_capture']
        self.assertEqual(len(hooks), 1)
        observed = {k.arg: ast.unparse(k.value) for k in hooks[0].keywords}
        self.assertEqual(observed, dict(metadata='metadata', positions='positions',
            hidden_input='hidden_states', kv_norm='kv', q_rotated='q',
            index_query_rotated='iq', raw_weights='weights', scaled_weights='iw',
            projection_weight='self.indexer.weights_proj.weight'))


if __name__ == '__main__':
    unittest.main()
