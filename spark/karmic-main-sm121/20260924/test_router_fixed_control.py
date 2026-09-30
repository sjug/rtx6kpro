import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from launch_contract import render, NODES
from prepare_router_fixed_control import transform_contract, transform_driver

ROOT = Path(__file__).resolve().parent


class FixedControlTests(unittest.TestCase):
    def renderer(self):
        namespace = {'__file__': str(ROOT / 'launch_contract.py'), '__name__': 'control_test'}
        exec(compile(transform_contract((ROOT / 'launch_contract.py').read_text()), 'control', 'exec'), namespace)
        return namespace['render']

    def test_default_unchanged(self):
        with patch.dict(os.environ, {}, clear=True):
            for node in NODES:
                self.assertEqual(render(node), self.renderer()(node))

    def test_only_declared_capacity_delta(self):
        for node in NODES:
            with patch.dict(os.environ, {}, clear=True):
                original = render(node)
            with patch.dict(os.environ, {'DS41_ROUTER_CONTROL_BLOCKS': '79000'}, clear=True):
                controlled = self.renderer()(node)
            self.assertEqual(controlled['env'].pop('DS41_ROUTER_CONTROL_BLOCKS'), '79000')
            index = controlled['model'].index('--num-gpu-blocks-override')
            self.assertEqual(controlled['model'][index + 1], '79000')
            del controlled['model'][index:index + 2]
            self.assertEqual(controlled, original)

    def test_unreviewed_capacity_rejected(self):
        for value in ('', '0', '80927', '79000 ', '-1', '79000;false'):
            with patch.dict(os.environ, {'DS41_ROUTER_CONTROL_BLOCKS': value}, clear=True):
                with self.assertRaises(ValueError):
                    self.renderer()('dusty')

    def test_both_arms_same_explicit_control(self):
        source = transform_driver((ROOT / 'start_moe_repaired.py').read_text())
        tree = ast.parse(source)
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'launch_command')
        module = ast.Module(body=[function], type_ignores=[])
        for enabled in (False, True):
            for node in NODES:
                commands = []
                for arm in ('engram-repair', 'router-fence'):
                    namespace = {'args': SimpleNamespace(arm=arm, cuda_module_loading=None, fixed_router_blocks=enabled, decision_row_blocks=None), 'REMOTE': '/task'}
                    exec(compile(module, 'driver', 'exec'), namespace)
                    commands.append(namespace['launch_command'](node))
                self.assertEqual(commands[0], commands[1])
                self.assertIn('-u DS41_ROUTER_CONTROL_BLOCKS', commands[0])
                self.assertEqual('DS41_ROUTER_CONTROL_BLOCKS=79000' in commands[0], enabled)

    def test_double_application_refused(self):
        source = (ROOT / 'launch_contract.py').read_text()
        with self.assertRaises(RuntimeError):
            transform_contract(transform_contract(source))

    def test_receipt_names_label_control(self):
        source = transform_driver((ROOT / 'start_moe_repaired.py').read_text())
        self.assertIn("profile_label = args.arm + ('-fixed79000' if args.fixed_router_blocks else '')", source)
        self.assertIn("f'{profile_label}-{stamp}'", source)


if __name__ == '__main__':
    unittest.main()
