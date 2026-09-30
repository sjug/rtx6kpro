"""Local-only checks for the approved window-capture plumbing."""
import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import claude_run_decision_capture as driver
from claude_test_run_decision_capture import container, image, BUILD, LOCK
from launch_contract import WINDOW_CAPTURE_LOCK
from run_node import command

ROOT = Path(__file__).resolve().parent


class WindowOrchestration(unittest.TestCase):
    def test_pinned_window_lock_and_render_all_ranks(self):
        self.assertEqual(hashlib.sha256((ROOT / 'claude-window.lock.json').read_bytes()).hexdigest(), WINDOW_CAPTURE_LOCK)
        self.assertEqual(json.loads((ROOT / 'window-capture-approval-20260926.json').read_text())['capture_lock_sha256'], WINDOW_CAPTURE_LOCK)
        pin = {'image_id': 'a' * 64, 'diagnostic': {'kind': 'window-capture', 'lock_sha256': WINDOW_CAPTURE_LOCK}}
        for threshold in ('8192', '4096'):
            with patch.dict(os.environ, {'DS41_DECISION_ROW_BLOCKS': '81389',
                                         'DS41_PREFILL_THRESHOLD': threshold}, clear=True):
                for node in ('dusty', 'toby', 'rusty', 'kirby'):
                    argv, _, _ = command(node, pin, 'b' * 64)
                    self.assertIn('DS41_DECISION_ROW_BLOCKS=81389', argv)
                    self.assertIn('DS41_PREFILL_THRESHOLD=' + threshold, argv)
                bad = dict(pin, diagnostic=dict(pin['diagnostic'], lock_sha256='other'))
                with self.assertRaises(ValueError):
                    command('dusty', bad, 'b' * 64)

    def test_window_cli_selects_separate_receipt_and_lock(self):
        with patch.object(driver, 'BUILD'), patch.object(driver, 'LOCK'), \
                patch.object(driver, 'DIAGNOSTIC_KIND'), patch.object(driver, 'phase_capture') as run:
            driver.main(['--kind', 'window', '--mode', 'matched', '--chunk-rows', '4096'])
            self.assertEqual(driver.BUILD, ROOT / 'receipts/window-build-receipt.json')
            self.assertEqual(driver.LOCK, ROOT / 'claude-window.lock.json')
            self.assertEqual(driver.DIAGNOSTIC_KIND, 'window-capture')
            self.assertEqual(run.call_args.args[0].chunk_rows, 4096)

    def test_window_cli_rejects_historical_mode(self):
        with patch.object(driver, 'phase_capture') as run, self.assertRaises(SystemExit):
            driver.main(['--kind', 'window'])
        run.assert_not_called()

    def test_window_identity_rejects_old_capture_image(self):
        profile = driver.profile_for('matched', 8192)
        c = container(env={'DS41_DECISION_ROW_BLOCKS': '81389', 'DS41_PREFILL_THRESHOLD': '8192'})
        with patch.object(driver, 'DIAGNOSTIC_KIND', 'window-capture'):
            errors, _ = driver.check_container('toby', c, image(), BUILD, LOCK, profile)
            self.assertTrue(any('window capture' in error for error in errors))
            errors, _ = driver.check_container('toby', c, image(**{
                'local-inference.ds41.diagnostic.kind': 'window-capture'}), BUILD, LOCK, profile)
            self.assertEqual(errors, [])

    def test_start_command_preserves_approved_profile(self):
        tree = ast.parse((ROOT / 'start_moe_repaired.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'launch_command')
        for threshold in ('8192', '4096'):
            ns = {'REMOTE': '/test', 'args': SimpleNamespace(arm='window',
                decision_row_blocks='81389', cuda_module_loading=None, prefill_threshold=threshold)}
            exec(compile(ast.Module(body=[fn], type_ignores=[]), 'driver', 'exec'), ns)
            command = ns['launch_command']('dusty')
            self.assertIn('DS41_PREFILL_THRESHOLD=' + threshold, command)
            self.assertIn('DS41_DECISION_ROW_BLOCKS=81389', command)
            self.assertIn('B12X_DENSE_SPLITK_TURBO=0', command)
            self.assertIn('B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1', command)
            self.assertNotIn('DS41_CHUNKING_BLOCKS=', command)


if __name__ == '__main__':
    unittest.main()
