"""Exercise the actual launch renderer without importing its operational driver."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import subprocess
import sys


class DiagnosticProfile(unittest.TestCase):
    def test_fixed_capture_rejects_other_arms_before_actions(self):
        script = Path(__file__).with_name('start_moe_repaired.py')
        result = subprocess.run([sys.executable, str(script), '--arm', 'router-fence',
                                 '--decision-row-blocks', '80927'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('restricted to the decision-row diagnostic', result.stderr)
        self.assertNotIn('RECEIPTS', result.stdout)

    def test_capture_pin_and_inherited_env_cleaning(self):
        source = ast.parse(Path(__file__).with_name('start_moe_repaired.py').read_text())
        function = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'launch_command')
        namespace = {'REMOTE': '/test/kit', 'args': SimpleNamespace(
            arm='decision-row', cuda_module_loading=None, decision_row_blocks='80927')}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'driver', 'exec'), namespace)
        command = namespace['launch_command']('dusty')
        self.assertIn('-u DS41_DECISION_ROW_BLOCKS', command)
        self.assertIn('DS41_DECISION_ROW_BLOCKS=80927', command)

    def test_new_diagnostics_keep_router_profile_exactly(self):
        source = ast.parse(Path(__file__).with_name('start_moe_repaired.py').read_text())
        function = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'launch_command')
        module = ast.Module(body=[function], type_ignores=[])
        namespace = {'REMOTE': '/test/kit'}
        exec(compile(module, 'start_moe_repaired.py', 'exec'), namespace)
        rendered = {}
        for arm in ('router-fence', 'decision-row', 'engram-fault'):
            namespace['args'] = SimpleNamespace(arm=arm, cuda_module_loading=None, decision_row_blocks=None)
            rendered[arm] = [namespace['launch_command'](node) for node in ('toby', 'rusty', 'kirby', 'dusty')]
        self.assertEqual(rendered['router-fence'], rendered['decision-row'])
        self.assertEqual(rendered['router-fence'], rendered['engram-fault'])
        self.assertIn('B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1 B12X_DENSE_SPLITK_TURBO=0', rendered['decision-row'][0])
        self.assertNotIn('KV', rendered['decision-row'][0])


if __name__ == '__main__':
    unittest.main()
