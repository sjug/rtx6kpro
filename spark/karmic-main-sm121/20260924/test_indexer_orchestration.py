"""Fail-closed local checks for the separate approved layer-2 capture arm."""
import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import launch_contract as contract
import claude_run_decision_capture as driver
from run_node import command
from select_decision_row import candidate
from build_decision_row import frozen_inputs

ROOT = Path(__file__).resolve().parent


class IndexerOrchestration(unittest.TestCase):
    def test_lock_and_all_rank_commands(self):
        digest = hashlib.sha256((ROOT / 'ds41-indexer.lock.json').read_bytes()).hexdigest()
        self.assertEqual(contract.INDEXER_CAPTURE_LOCK, digest)
        pin = {'image_id': 'a' * 64, 'diagnostic': {'kind': 'indexer-capture', 'lock_sha256': digest}}
        for threshold in ('8192', '4096'):
            env = {'DS41_DECISION_ROW_BLOCKS': '81389', 'DS41_PREFILL_THRESHOLD': threshold,
                   'DS41_NCCL_GEOMETRY': contract.NCCL_GEOMETRY}
            with patch.dict(os.environ, env, clear=True):
                for node in contract.NODES:
                    argv, _, _ = command(node, pin, 'b' * 64)
                    for key, value in contract.NCCL_GEOMETRY_ENV.items():
                        self.assertIn(key + '=' + value, argv)
                with self.assertRaises(ValueError):
                    command('dusty', dict(pin, diagnostic={'kind': 'indexer-capture', 'lock_sha256': 'wrong'}), 'b' * 64)
            for missing in env:
                with patch.dict(os.environ, {k: v for k, v in env.items() if k != missing}, clear=True):
                    with self.assertRaises(ValueError):
                        command('dusty', pin, 'b' * 64)

    def test_driver_selects_separate_frozen_inputs(self):
        with patch.object(driver, 'BUILD'), patch.object(driver, 'LOCK'), patch.object(driver, 'DIAGNOSTIC_KIND'), \
                patch.object(driver, 'phase_capture') as run:
            driver.main(['--kind', 'indexer', '--mode', 'matched', '--chunk-rows', '4096',
                         '--nccl-geometry', contract.NCCL_GEOMETRY])
            self.assertEqual(driver.BUILD, ROOT / 'receipts/indexer-build-receipt.json')
            self.assertEqual(driver.LOCK, ROOT / 'ds41-indexer.lock.json')
            self.assertEqual(driver.DIAGNOSTIC_KIND, 'indexer-capture')
            run.assert_called_once()
        for args in (['--kind', 'indexer'], ['--kind', 'indexer', '--mode', 'matched', '--chunk-rows', '8192']):
            with patch.object(driver, 'phase_capture') as run, self.assertRaises(SystemExit):
                driver.main(args)
            run.assert_not_called()

    def test_build_parent_and_selection_preserve_settings(self):
        lock, digest = frozen_inputs(stem='ds41-indexer')
        parent = {'image_id': lock['base_image_id'], 'setting': 'unchanged'}
        result = candidate(parent, lock, {'image_id': 'child', 'lock_sha256': digest}, digest, 'indexer-capture')
        self.assertEqual(result['setting'], 'unchanged')
        self.assertEqual(result['diagnostic']['kind'], 'indexer-capture')

    def test_start_profile(self):
        tree = ast.parse((ROOT / 'start_moe_repaired.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'launch_command')
        for threshold in ('8192', '4096'):
            ns = {'REMOTE': '/test', 'args': SimpleNamespace(arm='indexer',
                decision_row_blocks='81389', cuda_module_loading=None, prefill_threshold=threshold,
                nccl_geometry=contract.NCCL_GEOMETRY)}
            exec(compile(ast.Module(body=[fn], type_ignores=[]), 'driver', 'exec'), ns)
            text = ns['launch_command']('dusty')
            for item in ('DS41_PREFILL_THRESHOLD=' + threshold, 'DS41_DECISION_ROW_BLOCKS=81389',
                         'DS41_NCCL_GEOMETRY=' + contract.NCCL_GEOMETRY,
                         'B12X_DENSE_SPLITK_TURBO=0', 'B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1'):
                self.assertIn(item, text)


if __name__ == '__main__':
    unittest.main()
