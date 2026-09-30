"""Precision arm retains the matched profile and a separate immutable identity."""
import ast
import hashlib
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


class PrecisionOrchestration(unittest.TestCase):
    def test_standard_arm_keeps_upstream_nccl_on_all_ranks(self):
        pin = {'image_id': 'a' * 64, 'diagnostic': {
            'kind': 'precision-capture', 'lock_sha256': contract.PRECISION_CAPTURE_LOCK}}
        for threshold in ('8192', '4096'):
            env = {'DS41_DECISION_ROW_BLOCKS': '81389', 'DS41_PREFILL_THRESHOLD': threshold,
                   'DS41_NCCL_ARM': contract.STANDARD_NCCL_ARM}
            with patch.dict(os.environ, env, clear=True):
                for node in contract.NODES:
                    argv, _, _ = command(node, pin, 'b' * 64)
                    self.assertIn('DS41_NCCL_ARM=standard-upstream', argv)
                    for key in contract.NCCL_GEOMETRY_ENV:
                        self.assertFalse(any(v.startswith(key + '=') for v in argv))
                    self.assertEqual(contract.render(node)['unset'], contract.NCCL_TUNING_UNSET)
            with patch.dict(os.environ, env | {'DS41_NCCL_GEOMETRY': contract.NCCL_GEOMETRY}, clear=True):
                with self.assertRaises(ValueError):
                    command('dusty', pin, 'b' * 64)
            for kind in ('indexer-capture', 'window-capture', 'router-stage-release-candidate'):
                other = dict(pin, diagnostic=dict(pin['diagnostic'], kind=kind))
                with patch.dict(os.environ, env, clear=True), self.assertRaises(ValueError):
                    command('dusty', other, 'b' * 64)

    def test_standard_driver_requires_explicit_arm(self):
        with patch.object(driver, 'BUILD'), patch.object(driver, 'LOCK'), patch.object(driver, 'DIAGNOSTIC_KIND'), \
                patch.object(driver, 'phase_capture') as run:
            driver.main(['--kind', 'precision', '--mode', 'matched', '--chunk-rows', '4096',
                         '--nccl-arm', contract.STANDARD_NCCL_ARM])
            self.assertEqual(run.call_args.args[0].nccl_arm, contract.STANDARD_NCCL_ARM)
            profile = driver.profile_for('matched', 4096, nccl_arm=contract.STANDARD_NCCL_ARM)
            self.assertIsNone(profile['geometry'])
            self.assertEqual(driver.check_geometry_env({'DS41_NCCL_ARM': contract.STANDARD_NCCL_ARM}, profile), [])
            self.assertTrue(driver.check_geometry_env({}, profile))
            self.assertTrue(driver.check_geometry_env({'DS41_NCCL_ARM': contract.STANDARD_NCCL_ARM,
                                                       'NCCL_ALGO': 'allreduce:tree'}, profile))

    def test_precision_markers_fail_closed(self):
        line = 'worker: DS41-PRECISION-APPLIED rank=2 allow_bf16_reduced_precision_reduction=False before=True torch=test pid=9\n'
        self.assertEqual(driver.check_precision_marker(line, 2), [])
        for bad in ('', line + line, line.replace('rank=2', 'rank=1'), line.replace('reduction=False', 'reduction=True')):
            self.assertTrue(driver.check_precision_marker(bad, 2))

    def test_frozen_lock_and_all_ranks(self):
        lock, digest = frozen_inputs(stem='ds41-precision')
        self.assertEqual(contract.PRECISION_CAPTURE_LOCK, digest)
        pin = {'image_id': 'a' * 64, 'diagnostic': {'kind': 'precision-capture', 'lock_sha256': digest}}
        for threshold in ('8192', '4096'):
            env = {'DS41_DECISION_ROW_BLOCKS': '81389', 'DS41_PREFILL_THRESHOLD': threshold,
                   'DS41_NCCL_GEOMETRY': contract.NCCL_GEOMETRY}
            with patch.dict(os.environ, env, clear=True):
                for node in contract.NODES:
                    argv, _, _ = command(node, pin, 'b' * 64)
                    for key, value in contract.NCCL_GEOMETRY_ENV.items():
                        self.assertIn(key + '=' + value, argv)
            for missing in env:
                with patch.dict(os.environ, {k: v for k, v in env.items() if k != missing}, clear=True):
                    with self.assertRaises(ValueError):
                        command('dusty', pin, 'b' * 64)
        parent = {'image_id': lock['base_image_id'], 'setting': 'unchanged'}
        self.assertEqual(candidate(parent, lock, {'image_id': 'child', 'lock_sha256': digest},
                                   digest, 'precision-capture')['setting'], 'unchanged')

    def test_driver_identity_and_profile_rejections(self):
        with patch.object(driver, 'BUILD'), patch.object(driver, 'LOCK'), patch.object(driver, 'DIAGNOSTIC_KIND'), \
                patch.object(driver, 'phase_capture') as run:
            driver.main(['--kind', 'precision', '--mode', 'matched', '--chunk-rows', '4096',
                         '--nccl-geometry', contract.NCCL_GEOMETRY])
            self.assertEqual(driver.BUILD, ROOT / 'receipts/precision-build-receipt.json')
            self.assertEqual(driver.LOCK, ROOT / 'ds41-precision.lock.json')
            self.assertEqual(driver.DIAGNOSTIC_KIND, 'precision-capture')
            run.assert_called_once()
        for args in (['--kind', 'precision'], ['--kind', 'precision', '--mode', 'matched', '--chunk-rows', '8192']):
            with patch.object(driver, 'phase_capture') as run, self.assertRaises(SystemExit):
                driver.main(args)
            run.assert_not_called()

    def test_start_profile(self):
        tree = ast.parse((ROOT / 'start_moe_repaired.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'launch_command')
        for threshold in ('8192', '4096'):
            ns = {'REMOTE': '/test', 'args': SimpleNamespace(arm='precision', decision_row_blocks='81389',
                cuda_module_loading=None, prefill_threshold=threshold, nccl_geometry=contract.NCCL_GEOMETRY)}
            exec(compile(ast.Module(body=[fn], type_ignores=[]), 'driver', 'exec'), ns)
            text = ns['launch_command']('dusty')
            for item in ('DS41_PREFILL_THRESHOLD=' + threshold, 'DS41_DECISION_ROW_BLOCKS=81389',
                         'DS41_NCCL_GEOMETRY=' + contract.NCCL_GEOMETRY,
                         'B12X_DENSE_SPLITK_TURBO=0', 'B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1'):
                self.assertIn(item, text)


if __name__ == '__main__':
    unittest.main()
