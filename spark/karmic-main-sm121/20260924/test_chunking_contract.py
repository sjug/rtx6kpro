"""CPU tests for the explicitly approved 7936/4096 diagnostic arms."""
import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from launch_contract import (NODES, render, MATCHED_CAPTURE_LOCK, WINDOW_CAPTURE_LOCK, NCCL_GEOMETRY,
                             NCCL_GEOMETRY_ENV, NCCL_TUNING_UNSET)
from run_node import command

ROOT = Path(__file__).resolve().parent
IMAGE = 'e06df11a8ca18fa514d9f28f67cc691aef296da2eeb22b113a734519853bccd7'


class ChunkingContract(unittest.TestCase):
    def test_matched_capture_lock_is_the_prepared_lock(self):
        self.assertEqual(hashlib.sha256((ROOT / 'claude-decision-row.lock.json').read_bytes()).hexdigest(),
                         MATCHED_CAPTURE_LOCK)
        approval = json.loads((ROOT / 'matched-capture-approval-20260926.json').read_text())
        self.assertEqual(approval['capture_lock_sha256'], MATCHED_CAPTURE_LOCK)

    def test_capture_start_uses_only_the_capture_block_pin(self):
        tree = ast.parse((ROOT / 'start_moe_repaired.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'launch_command')
        for threshold in ('8192', '4096'):
            ns = {'REMOTE': '/test', 'args': SimpleNamespace(arm='decision-row',
                decision_row_blocks='81389', cuda_module_loading=None, prefill_threshold=threshold)}
            exec(compile(ast.Module(body=[fn], type_ignores=[]), 'driver', 'exec'), ns)
            result = ns['launch_command']('dusty')
            self.assertIn('DS41_DECISION_ROW_BLOCKS=81389', result)
            self.assertIn('DS41_PREFILL_THRESHOLD=' + threshold, result)
            self.assertNotIn('DS41_CHUNKING_BLOCKS=', result)

    def test_matched_capture_exact_lock_and_geometry(self):
        pin = {'image_id': 'a' * 64, 'diagnostic': {
            'kind': 'decision-row-capture', 'lock_sha256': MATCHED_CAPTURE_LOCK}}
        with patch.dict(os.environ, {}, clear=True):
            baselines = {node: render(node) for node in NODES}
        for threshold in ('8192', '4096'):
            with patch.dict(os.environ, {'DS41_DECISION_ROW_BLOCKS': '81389',
                                        'DS41_PREFILL_THRESHOLD': threshold}, clear=True):
                for node in NODES:
                    argv, _, _ = command(node, pin, 'b' * 64)
                    row = render(node)
                    self.assertIn('DS41_DECISION_ROW_BLOCKS=81389', argv)
                    self.assertEqual(row['model'].count('--num-gpu-blocks-override'), 1)
                    self.assertEqual(row['model'][row['model'].index('--long-prefill-token-threshold') + 1], threshold)
                    self.assertNotIn('DS41_CHUNKING_BLOCKS', row['env'])
                    for flag in ('--long-prefill-token-threshold', '--num-gpu-blocks-override'):
                        index = row['model'].index(flag)
                        del row['model'][index:index + 2]
                    for key in ('DS41_PREFILL_THRESHOLD', 'DS41_DECISION_ROW_BLOCKS'):
                        row['env'].pop(key)
                    self.assertEqual(row, baselines[node])
                for bad in ({'image_id': IMAGE, 'diagnostic': {'kind': 'router-stage-release-candidate'}},
                            dict(pin, diagnostic=dict(pin['diagnostic'], lock_sha256='other')),
                            dict(pin, diagnostic_overlay={})):
                    with self.assertRaises(ValueError):
                        command('dusty', bad, 'b' * 64)

    def test_matched_capture_needs_both_controls(self):
        for env in ({'DS41_DECISION_ROW_BLOCKS': '81389'},
                    {'DS41_DECISION_ROW_BLOCKS': '81389', 'DS41_PREFILL_THRESHOLD': '7936'},
                    {'DS41_DECISION_ROW_BLOCKS': '81389', 'DS41_PREFILL_THRESHOLD': '4096',
                     'DS41_CHUNKING_BLOCKS': '81389'}):
            with patch.dict(os.environ, env, clear=True), self.assertRaises(ValueError):
                render('dusty')

    def test_only_approved_delta_all_ranks(self):
        for node in NODES:
            with patch.dict(os.environ, {}, clear=True):
                baseline = render(node)
            for threshold in ('7936', '4096'):
                with patch.dict(os.environ, {'DS41_PREFILL_THRESHOLD': threshold,
                                            'DS41_CHUNKING_BLOCKS': '81389'}, clear=True):
                    actual = render(node)
                for flag, value in (('--long-prefill-token-threshold', threshold),
                                    ('--num-gpu-blocks-override', '81389')):
                    self.assertEqual(actual['model'].count(flag), 1)
                    index = actual['model'].index(flag)
                    self.assertEqual(actual['model'][index + 1], value)
                    del actual['model'][index:index + 2]
                for key in ('DS41_PREFILL_THRESHOLD', 'DS41_CHUNKING_BLOCKS'):
                    actual['env'].pop(key)
                self.assertEqual(actual, baseline)

    def test_partial_unapproved_or_combined_controls_rejected(self):
        invalid = [
            {'DS41_PREFILL_THRESHOLD': '7936'}, {'DS41_CHUNKING_BLOCKS': '81389'},
            {'DS41_PREFILL_THRESHOLD': '8192', 'DS41_CHUNKING_BLOCKS': '81389'},
            {'DS41_PREFILL_THRESHOLD': '7936', 'DS41_CHUNKING_BLOCKS': '80927'},
            {'DS41_PREFILL_THRESHOLD': '7936', 'DS41_CHUNKING_BLOCKS': '81389',
             'DS41_DECISION_ROW_BLOCKS': '80927'},
        ]
        for env in invalid:
            with patch.dict(os.environ, env, clear=True), self.assertRaises(ValueError):
                render('dusty')

    def test_exact_clean_image_only(self):
        with patch.dict(os.environ, {'DS41_PREFILL_THRESHOLD': '7936',
                                    'DS41_CHUNKING_BLOCKS': '81389'}, clear=True):
            for image, kind in ((IMAGE, 'decision-row-capture'),
                                ('a' * 64, 'router-stage-release-candidate')):
                with self.assertRaises(ValueError):
                    command('dusty', {'image_id': image, 'diagnostic': {'kind': kind}}, 'b' * 64)
            argv, _, _ = command('dusty', {'image_id': IMAGE, 'diagnostic': {
                'kind': 'router-stage-release-candidate'}}, 'b' * 64)
            self.assertIn('DS41_CHUNKING_BLOCKS=81389', argv)

    def test_driver_explicit_and_clears_inherited_controls(self):
        tree = ast.parse((ROOT / 'start_moe_repaired.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'launch_command')
        for threshold in (None, '7936', '4096'):
            ns = {'REMOTE': '/test', 'args': SimpleNamespace(arm='router-fence',
                decision_row_blocks=None, cuda_module_loading=None, prefill_threshold=threshold)}
            exec(compile(ast.Module(body=[fn], type_ignores=[]), 'driver', 'exec'), ns)
            result = ns['launch_command']('dusty')
            self.assertIn('-u DS41_PREFILL_THRESHOLD -u DS41_CHUNKING_BLOCKS', result)
            self.assertEqual('DS41_CHUNKING_BLOCKS=81389' in result, threshold is not None)


class NcclGeometryContract(unittest.TestCase):
    """User-approved NCCL geometry diagnostic: four explicit overrides on the matched window arms only."""
    WINDOW = {'image_id': 'c' * 64, 'diagnostic': {'kind': 'window-capture', 'lock_sha256': WINDOW_CAPTURE_LOCK}}
    DECISION = {'image_id': 'a' * 64, 'diagnostic': {'kind': 'decision-row-capture', 'lock_sha256': MATCHED_CAPTURE_LOCK}}

    def test_geometry_is_exactly_the_four_overrides_on_all_ranks(self):
        for threshold in ('8192', '4096'):
            matched_env = {'DS41_DECISION_ROW_BLOCKS': '81389', 'DS41_PREFILL_THRESHOLD': threshold}
            for node in NODES:
                with patch.dict(os.environ, matched_env, clear=True):
                    matched = render(node)
                with patch.dict(os.environ, dict(matched_env, DS41_NCCL_GEOMETRY=NCCL_GEOMETRY), clear=True):
                    geometry = render(node)
                    argv, _, _ = command(node, self.WINDOW, 'b' * 64)
                self.assertEqual(geometry['unset'], [])
                self.assertEqual(matched['unset'], NCCL_TUNING_UNSET)
                for key, value in NCCL_GEOMETRY_ENV.items():
                    self.assertEqual(geometry['env'].pop(key), value)
                    self.assertIn(f'{key}={value}', argv)
                    self.assertNotIn(key, matched['env'])
                self.assertEqual(geometry['env'].pop('DS41_NCCL_GEOMETRY'), NCCL_GEOMETRY)
                geometry['unset'] = matched['unset']
                self.assertEqual(geometry, matched)
                body = argv[argv.index('-c') + 1]
                self.assertFalse(body.startswith('unset'))
                self.assertIn('exec /opt/venv/bin/python /opt/ds41-adapter/runtime.py', body)

    def test_normal_and_matched_launch_bodies_unchanged(self):
        with patch.dict(os.environ, {'DS41_DECISION_ROW_BLOCKS': '81389', 'DS41_PREFILL_THRESHOLD': '8192'}, clear=True):
            argv, _, _ = command('dusty', self.DECISION, 'b' * 64)
        body = argv[argv.index('-c') + 1]
        self.assertTrue(body.startswith('unset NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS; '))
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(render('dusty')['unset'], NCCL_TUNING_UNSET)

    def test_geometry_rejected_outside_the_approved_profile(self):
        for env in ({'DS41_NCCL_GEOMETRY': NCCL_GEOMETRY},
                    {'DS41_NCCL_GEOMETRY': NCCL_GEOMETRY, 'DS41_DECISION_ROW_BLOCKS': '81389'},
                    {'DS41_NCCL_GEOMETRY': NCCL_GEOMETRY, 'DS41_DECISION_ROW_BLOCKS': '80927'},
                    {'DS41_NCCL_GEOMETRY': NCCL_GEOMETRY, 'DS41_PREFILL_THRESHOLD': '7936',
                     'DS41_CHUNKING_BLOCKS': '81389'},
                    {'DS41_NCCL_GEOMETRY': 'ring', 'DS41_DECISION_ROW_BLOCKS': '81389',
                     'DS41_PREFILL_THRESHOLD': '8192'}):
            with patch.dict(os.environ, env, clear=True), self.assertRaises(ValueError):
                render('dusty')

    def test_geometry_requires_the_window_capture_lock(self):
        env = {'DS41_NCCL_GEOMETRY': NCCL_GEOMETRY, 'DS41_DECISION_ROW_BLOCKS': '81389',
               'DS41_PREFILL_THRESHOLD': '4096'}
        with patch.dict(os.environ, env, clear=True):
            for bad in (self.DECISION,
                        dict(self.WINDOW, diagnostic=dict(self.WINDOW['diagnostic'], lock_sha256=MATCHED_CAPTURE_LOCK)),
                        dict(self.WINDOW, diagnostic_overlay={}),
                        {'image_id': IMAGE, 'diagnostic': {'kind': 'router-stage-release-candidate'}}):
                with self.assertRaises(ValueError):
                    command('dusty', bad, 'b' * 64)
            argv, _, _ = command('dusty', self.WINDOW, 'b' * 64)
            self.assertIn('DS41_NCCL_GEOMETRY=' + NCCL_GEOMETRY, argv)

    def test_starter_passes_and_clears_the_geometry_switch(self):
        tree = ast.parse((ROOT / 'start_moe_repaired.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'launch_command')
        for geometry in (None, NCCL_GEOMETRY):
            ns = {'REMOTE': '/test', 'args': SimpleNamespace(arm='window', decision_row_blocks='81389',
                  cuda_module_loading=None, prefill_threshold='4096', nccl_geometry=geometry)}
            exec(compile(ast.Module(body=[fn], type_ignores=[]), 'driver', 'exec'), ns)
            result = ns['launch_command']('dusty')
            self.assertIn('-u DS41_NCCL_GEOMETRY', result)
            self.assertEqual('DS41_NCCL_GEOMETRY=' + NCCL_GEOMETRY in result, geometry is not None)
            self.assertIn('DS41_DECISION_ROW_BLOCKS=81389', result)
        source = (ROOT / 'start_moe_repaired.py').read_text()
        self.assertIn("args.arm not in ('window', 'indexer', 'precision') or not matched_capture", source)


if __name__ == '__main__':
    unittest.main()
