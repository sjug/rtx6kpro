"""Local-only tests for the approved NCCL geometry arm: driver profile, env checks, warning scan, comparator."""
import copy
import unittest
from unittest.mock import patch

import torch

import claude_run_decision_capture as driver
import compare_window_geometry as geometry
from claude_test_run_decision_capture import container, image, BUILD, LOCK
from launch_contract import NCCL_GEOMETRY, NCCL_GEOMETRY_ENV

GEOMETRY_ENV = dict(NCCL_GEOMETRY_ENV, DS41_NCCL_GEOMETRY=NCCL_GEOMETRY)


class DriverProfile(unittest.TestCase):
    def test_profile_only_on_matched_and_tags_receipts(self):
        profile = driver.profile_for('matched', 4096, NCCL_GEOMETRY)
        self.assertEqual((profile['geometry'], profile['tag'], profile['blocks'], profile['threshold']),
                         (NCCL_GEOMETRY, 'matched4096-geometry-tree-simple-1ch-', 81389, 4096))
        self.assertIsNone(driver.profile_for('matched', 8192)['geometry'])
        self.assertIsNone(driver.profile_for('historical')['geometry'])
        for mode, chunk, value in (('historical', None, NCCL_GEOMETRY), ('matched', 8192, 'ring')):
            with self.assertRaises(ValueError):
                driver.profile_for(mode, chunk, value)

    def test_cli_restricts_geometry_to_window_matched(self):
        with patch.object(driver, 'phase_capture') as run:
            for argv in (['--kind', 'decision-row', '--mode', 'matched', '--chunk-rows', '8192', '--nccl-geometry', NCCL_GEOMETRY],
                         ['--kind', 'window', '--mode', 'historical', '--nccl-geometry', NCCL_GEOMETRY]):
                with self.assertRaises(SystemExit):
                    driver.main(argv)
            run.assert_not_called()
        with patch.object(driver, 'BUILD'), patch.object(driver, 'LOCK'), patch.object(driver, 'DIAGNOSTIC_KIND'), \
                patch.object(driver, 'phase_capture') as run:
            driver.main(['--kind', 'window', '--mode', 'matched', '--chunk-rows', '4096', '--nccl-geometry', NCCL_GEOMETRY])
            self.assertEqual(run.call_args.args[0].nccl_geometry, NCCL_GEOMETRY)

    def test_geometry_env_exactly_present_or_absent(self):
        plain, armed = driver.profile_for('matched', 8192), driver.profile_for('matched', 8192, NCCL_GEOMETRY)
        self.assertEqual(driver.check_geometry_env({}, plain), [])
        self.assertEqual(driver.check_geometry_env(GEOMETRY_ENV, armed), [])
        self.assertTrue(driver.check_geometry_env({'NCCL_PROTO': 'Simple'}, plain))
        self.assertTrue(driver.check_geometry_env({}, armed))
        self.assertTrue(driver.check_geometry_env({**GEOMETRY_ENV, 'NCCL_ALGO': 'ring'}, armed))
        self.assertTrue(driver.check_geometry_env({**GEOMETRY_ENV, 'NCCL_MAX_NCHANNELS': '2'}, armed))

    def test_container_and_boot_profile_follow_the_profile(self):
        armed = driver.profile_for('matched', 8192, NCCL_GEOMETRY)
        base = {'DS41_DECISION_ROW_BLOCKS': '81389', 'DS41_PREFILL_THRESHOLD': '8192'}
        with patch.object(driver, 'DIAGNOSTIC_KIND', 'window-capture'):
            window = image(**{'local-inference.ds41.diagnostic.kind': 'window-capture'})
            errors, _ = driver.check_container('toby', container(env={**base, **GEOMETRY_ENV}), window, BUILD, LOCK, armed)
            self.assertEqual(errors, [])
            errors, _ = driver.check_container('toby', container(env=base), window, BUILD, LOCK, armed)
            self.assertTrue(any('NCCL geometry variable' in e for e in errors))
            plain = driver.profile_for('matched', 8192)
            errors, _ = driver.check_container('toby', container(env={**base, **GEOMETRY_ENV}), window, BUILD, LOCK, plain)
            self.assertTrue(any('unexpected NCCL tuning' in e for e in errors))
        with patch.object(driver, 'DIAGNOSTIC_KIND', 'decision-row-capture'):
            errors, _ = driver.check_container('toby', container(env={**base, **GEOMETRY_ENV}), image(), BUILD, LOCK, armed)
            self.assertTrue(any('window-capture kind only' in e for e in errors))
        rendered = {'node': 'toby', 'rank': 1, 'env': {**base, **GEOMETRY_ENV},
                    'model': ['/opt/venv/bin/vllm', 'serve', 'm', '--max-num-batched-tokens', '8192',
                              '--num-gpu-blocks-override', '81389', '--long-prefill-token-threshold', '8192']}
        self.assertEqual(driver.check_boot_profile(rendered, 'toby', armed), [])
        self.assertTrue(driver.check_boot_profile(rendered, 'toby', driver.profile_for('matched', 8192)))

    def test_nccl_warning_scan(self):
        text = ('[x] dusty:1:1 [0] NCCL WARN Invalid value for NCCL_ALGO\n'
                '[x] dusty:1:1 [0] misc/ibvwrap.cc:401 (wrap_ibv_query_port_speed) NCCL WARN Call to ibv_query_port_speed failed\n'
                '[x] dusty:1:1 [0] misc/ibvwrap.cc:401 (wrap_ibv_query_port_speed) NCCL WARN Call to ibv_query_port_speed failed with error Protocol not supported errno 93\n'
                '[x] dusty:1:1 [0] NCCL WARN NCCL_MAX_NCHANNELS ignored\n')
        found = driver.nccl_geometry_warnings(text)
        self.assertEqual(len(found), 2)
        self.assertTrue(all('NCCL_ALGO' in f or 'NCHANNELS' in f for f in found))
        self.assertEqual(driver.nccl_geometry_warnings(''), [])


def summary(chunk_rows, geometry=None):
    return {'chunk_rows': chunk_rows, 'image_id': 'img', 'blocks': 81389, 'mode': 'matched', 'regime_source': 'pinned',
            'signature_equals_control': True, 'regime_equals_reference': True,
            'capture_problems': {n: [] for n in geometry_nodes()}, 'nccl_geometry': geometry}


def geometry_nodes():
    return geometry.NODES


class GeometryComparison(unittest.TestCase):
    def test_pair_and_grid_validation(self):
        geometry.validate_geometry_pair(summary(8192, NCCL_GEOMETRY), summary(8192))
        geometry.validate_geometry_grids(summary(8192, NCCL_GEOMETRY), summary(4096, NCCL_GEOMETRY))
        for left, right, fn in ((summary(8192), summary(8192), geometry.validate_geometry_pair),
                                (summary(8192, NCCL_GEOMETRY), summary(4096), geometry.validate_geometry_pair),
                                (summary(8192, NCCL_GEOMETRY), summary(8192, NCCL_GEOMETRY), geometry.validate_geometry_pair),
                                (summary(8192, NCCL_GEOMETRY), summary(4096), geometry.validate_geometry_grids),
                                (summary(4096, NCCL_GEOMETRY), summary(8192, NCCL_GEOMETRY), geometry.validate_geometry_grids)):
            with self.assertRaises(ValueError):
                fn(left, right)

    def test_relaxed_identity_allows_kit_change_but_not_trees(self):
        import importlib.util
        from pathlib import Path
        tests = Path(__file__).resolve().parent / 'claude-window-tests.py'
        spec = importlib.util.spec_from_file_location('claude_window_tests', tests)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        compare = geometry.window_compare_module()
        left, right = module.synthetic(8192, 5), module.synthetic(8192, 5)
        right['meta']['kit_sha256'] = 'other-kit'
        report = geometry.relaxed_window_compare(left, right, compare)
        self.assertIsNone(report['first_differing_boundary'])
        self.assertFalse(report['kit_sha256']['equal'])
        self.assertIn('never a gate', report['scope'])
        right['layers'][0]['wo_reduced'][3] += 1
        self.assertEqual(geometry.relaxed_window_compare(left, right, compare)['first_differing_boundary'],
                         {'layer': 0, 'boundary': 'wo_reduced'})
        right['meta']['source_trees'] = {'vllm': 'x', 'b12x': 'b'}
        with self.assertRaises(ValueError):
            geometry.relaxed_window_compare(left, right, compare)

    def test_decision_row_fields_are_reported_not_required(self):
        from test_compare_decision_grids import capture
        def complete():
            c = capture()
            for row in c['layers'].values():
                row.update(swa_len=128, swa_records=torch.zeros(128, 528, dtype=torch.uint8))
            return c
        old, new = complete(), complete()
        new['layers'][7]['out'][0, 0] = 2
        report = geometry.decision_row_fields(old, new)
        self.assertFalse(report['equal_observations'])
        self.assertEqual((report['first_unequal_layer'], report['unequal_fields_by_layer'][7]), (7, ['out']))
        self.assertIn('not a requirement', report['scope'])


if __name__ == '__main__':
    unittest.main()
