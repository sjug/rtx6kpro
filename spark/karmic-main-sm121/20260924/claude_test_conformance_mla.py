"""CPU tests for the pure parts of claude_conformance_mla.py (the GPU execution runs only in-image).
  .venv-snapshot-cpu/bin/python -m unittest claude_test_conformance_mla
"""
import ast
import importlib.util
import json
from pathlib import Path
import unittest

import torch

HERE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


harness = load('claude_conformance_under_test', 'claude_conformance_mla.py')
decoder = load('claude_conformance_decoder_under_test', 'claude_decode_sparse_mla.py')
B1 = HERE / 'receipts/router-b1-tuning-20260925'


@unittest.skipUnless((B1 / 'dusty.json').exists(), 'retained B1 selection snapshot absent')
class Regime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        report = {}
        for node in ('dusty', 'toby'):
            records = json.loads((B1 / f'{node}.json').read_text())['records']
            found, _ = decoder.decode(records)
            report[str(B1 / node)] = {'regimes': {str(b): {'configs': r} for b, r in decoder.regimes(records, found).items()}}
        cls.decoded = report

    def test_regime_at_the_declared_capacity(self):
        configs = harness.regime_configs(self.decoded, 80927)
        self.assertEqual(sorted(configs), sorted(harness.SLOTS))
        self.assertEqual({s: c['v41_compute_mode'] for s, c in configs.items() if not s.startswith('draft')},
                         {'swa.extend': 'bf16', 'swa.decode': 'bf16', 'ratio1.extend': 'bf16', 'ratio1.decode': 'fp8',
                          'ratio2.extend': 'bf16', 'ratio2.decode': 'fp8'})       # R_B1, as decoded earlier
        with self.assertRaises(SystemExit):
            harness.regime_configs(self.decoded, 12345)

    def test_disagreeing_caches_are_refused(self):
        bad = json.loads(json.dumps(self.decoded))
        first = next(iter(bad))
        bad[first]['regimes']['80927']['configs']['swa.extend']['v41_compute_mode'] = 'fp8'
        with self.assertRaises(SystemExit):
            harness.regime_configs(bad, 80927)


class Geometry(unittest.TestCase):
    def test_role_facts_match_the_decoder(self):
        for role, (width, ratio, indexed_bytes) in decoder.ROLES.items():
            for mode, rows in decoder.MODES:
                facts = harness.role_facts(role, mode)
                self.assertEqual((facts['swa_width'], facts['ratio'], facts['rows']), (width, ratio, rows))
                self.assertEqual(facts['indexed_record_bytes'] or None, indexed_bytes)

    def test_views_have_the_production_descriptors(self):
        blocks = 80927
        pool = torch.empty((blocks * harness.BLOCK_STRIDE,), dtype=torch.uint8, device='meta')
        for role in ('swa', 'ratio1', 'ratio2'):
            facts = harness.role_facts(role, 'extend')
            layout = harness.pool_layout(facts, blocks)
            swa = pool.as_strided((blocks, facts['swa_page_bytes']), (harness.BLOCK_STRIDE, 1), layout['swa_offset'])
            query, invocation = decoder.query_and_invocation(role, 'extend', blocks)
            self.assertEqual([list(swa.shape), list(swa.stride())],
                             [invocation['swa_k_cache']['shape'], invocation['swa_k_cache']['stride']])
            if facts['ratio']:
                idx = pool.as_strided((blocks, facts['indexed_record_bytes']), (harness.BLOCK_STRIDE, 1),
                                      layout['indexed_offset'])
                self.assertEqual([list(idx.shape), list(idx.stride())],
                                 [invocation['indexed_k_cache']['shape'], invocation['indexed_k_cache']['stride']])
            self.assertEqual(query['query_rows'], 8192)

    def test_pages_cross_the_two_gib_offset_boundary(self):
        pages = harness.choose_pages(80927, 256, torch.Generator().manual_seed(1), torch)
        self.assertEqual(len(set(pages.tolist())), 256)
        self.assertTrue(bool((pages > 0).all()) and bool((pages < 80927).all()))
        offsets = pages * harness.BLOCK_STRIDE
        self.assertTrue(bool((offsets < 2 ** 31).any()) and bool((offsets >= 2 ** 31).any()))
        with self.assertRaises(ValueError):
            harness.choose_pages(5000, 256, torch.Generator().manual_seed(1), torch)


class Sampling(unittest.TestCase):
    def test_stratified_indices_are_unique_ascending_covering_and_fast(self):
        import time
        rows, visible = 8192, torch.full((8192,), 524288)
        g = torch.Generator().manual_seed(3)
        started = time.time()
        idx = harness.stratified_indices(visible, 512, g, torch)
        self.assertLess(time.time() - started, 5.0)                       # 8192 x 512 from 524288, vectorized
        self.assertEqual(tuple(idx.shape), (rows, 512))
        diffs = idx[:, 1:] - idx[:, :-1]
        self.assertTrue(bool((diffs > 0).all()))                          # unique and ascending
        self.assertTrue(bool((idx >= 0).all()) and bool((idx < 524288).all()))
        self.assertTrue(bool(((idx // 1024) == torch.arange(512)).all()))  # one entry per stratum: full coverage
        again = harness.stratified_indices(visible, 512, torch.Generator().manual_seed(3), torch)
        self.assertTrue(torch.equal(idx, again))                          # deterministic

    def test_clustered_and_small_rows(self):
        g = torch.Generator().manual_seed(4)
        visible = torch.tensor([524288, 524288, 300])
        clustered = torch.tensor([True, False, False])
        idx = harness.stratified_indices(visible, 512, g, torch, clustered=clustered)
        self.assertTrue(bool((idx[0, 1:] - idx[0, :-1] == 1).all()))      # contiguous run
        self.assertLess(int(idx[0, -1]), 524288)
        self.assertEqual(idx[2, :300].tolist(), list(range(300)))
        self.assertTrue(bool((idx[2, 300:] == -1).all()))

    def test_swa_windows_follow_production_chunk(self):
        table = torch.arange(4200) + 10
        positions = torch.tensor([524287, 5, 127])
        slots, lengths = harness.swa_windows(positions, 128, table, 128)
        self.assertEqual(lengths.tolist(), [128, 6, 128])
        logical = torch.arange(524160, 524288)
        self.assertEqual(slots[0].tolist(), (table[logical // 128] * 128 + logical % 128).tolist())
        self.assertEqual(slots[1, :6].tolist(), [10 * 128 + i for i in range(6)])   # valid entries first
        self.assertTrue(bool((slots[1, 6:] == -1).all()))
        draft, _ = harness.swa_windows(torch.tensor([524287]), 192, table, 128)
        self.assertEqual(int(draft[0, 0]), int(table[(524287 - 191) // 128] * 128 + (524287 - 191) % 128))

    def test_row_sample_and_errors(self):
        rows = harness.row_sample(8192, 48, torch.Generator().manual_seed(9), torch)
        self.assertIn(8191, rows)
        self.assertEqual(len(rows), 48)
        self.assertEqual(rows, harness.row_sample(8192, 48, torch.Generator().manual_seed(9), torch))
        good = torch.ones(16, 512)
        rel, zero = harness.relative_errors(good * 1.01, good, torch)
        self.assertEqual((len(rel), zero), (16, 0))
        self.assertAlmostEqual(rel[0], 0.01, places=5)
        expected = good.clone()
        expected[3] = 0
        self.assertEqual(harness.relative_errors(good, expected, torch)[1], 1)
        with self.assertRaises(FloatingPointError):
            harness.relative_errors(good * float('nan'), good, torch)

    def test_compile_workers_follow_production_policy(self):
        from unittest import mock
        with mock.patch.dict('os.environ', {'B12X_STATE_COMPILE_WORKERS': '16', 'B12X_COMPILE_WORKERS': '4'}):
            self.assertEqual(harness.compile_workers(), 16)
        with mock.patch.dict('os.environ', {'B12X_COMPILE_WORKERS': '4'}, clear=True):
            self.assertEqual(harness.compile_workers(), 4)
        with mock.patch.dict('os.environ', {}, clear=True):
            self.assertEqual(harness.compile_workers(), 8)

    def test_fill_pages_writes_decodable_records_into_the_strided_view(self):
        reference = load('claude_conformance_reference_under_test', 'claude-pinned-b12x-compressed_reference.py')
        blocks, stride = 6, 1024 + 1 * 288 * 2
        pool = torch.zeros(blocks * stride, dtype=torch.uint8)
        view = pool.as_strided((blocks, 2 * 288), (stride, 1), 1024)
        harness.fill_pages(view, torch.tensor([4, 2]), 2, 'indexed', torch.ones(512), 5, reference, torch, chunk=1)
        self.assertTrue(bool((pool.view(blocks, stride)[:, :1024] == 0).all()))         # neighbours untouched
        decoded = reference.unpack_deepseek_v41_cache_reference(view[[4, 2]].contiguous(), page_size=2,
                                                                cache_kind='indexed')
        self.assertTrue(bool(torch.isfinite(decoded).all()) and float(decoded.abs().sum()) > 0)


class Attribution(unittest.TestCase):
    def test_group_summaries_separate_distributions(self):
        pairs = [('gaussian', 0.01), ('outlier', 0.4), ('gaussian', 0.02), ('outlier', 0.2), ('wide-range', 0.1)]
        groups = harness.group_summaries(pairs)
        self.assertEqual(sorted(groups), ['gaussian', 'outlier', 'wide-range'])
        self.assertEqual((groups['gaussian']['count'], groups['gaussian']['max']), (2, 0.02))
        self.assertEqual(groups['outlier']['max'], 0.4)

    def test_every_sample_is_tagged_and_reported(self):
        text = (HERE / 'claude_conformance_mla.py').read_text()
        run_plan = ast.unparse(next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'run_plan'))
        self.assertIn('tagged.extend(((distribution, value) for value in rel))', run_plan)
        self.assertIn("'by_distribution': group_summaries(tagged)", run_plan)
        main = ast.unparse(next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'main'))
        self.assertIn("'per_mode_by_distribution'", main)
        self.assertIn("row.pop('raw_tagged')", main)                     # raw samples are not written out


class Envelope(unittest.TestCase):
    def test_envelope_rule(self):
        values = [i / 10000 for i in range(1000)]
        env = harness.envelope({'bf16': values, 'fp8': [0.05, 0.3]})
        self.assertAlmostEqual(env['bf16'], 2 * values[998])
        self.assertEqual(env['fp8'], 0.6)
        self.assertEqual(harness.summarize([3.0, 1.0, 2.0])['median'], 2.0)


class Source(unittest.TestCase):
    text = (HERE / 'claude_conformance_mla.py').read_text()

    def test_scope_never_claims_an_oracle(self):
        self.assertIn('not a model oracle', harness.SCOPE)
        self.assertIn('cannot reveal a bias shared by kernel and reference', harness.SCOPE)
        self.assertNotIn('certif', harness.SCOPE.replace('no output is certified', ''))

    def test_production_key_is_required_and_execution_is_override_only(self):
        self.assertIn("if key != decoder.key(role, mode, blocks):", self.text)
        self.assertIn('override=execution', self.text)
        self.assertIn('autotune=False', self.text)
        tree = ast.parse(self.text)
        run_plan = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_plan')
        body = ast.unparse(run_plan)
        self.assertLess(body.index('decoder.key(role, mode, blocks)'), body.index('override=execution'))
        self.assertIn('deterministic &= bool(torch.equal(first, output))', body)
        self.assertIn("output.fill_(float('nan'))", body)
        self.assertIn('torch.isfinite(first).all()', body)
        run_args = [l for l in body.splitlines() if 'run_args = dict(' in l or 'run_args.update(' in l]
        self.assertEqual(len(run_args), 2)
        self.assertFalse([l for l in run_args if 'page_size' in l or 'cache_format' in l], run_args)  # state supplies
        self.assertEqual(body.count('compressed_sparse_mla.run(binding=binding, out=output, **run_args)'), 2)
        self.assertIn('compile_workers=workers', body)

    def test_failures_withhold_the_envelope(self):
        main = ast.unparse(next(n for n in ast.parse(self.text).body if isinstance(n, ast.FunctionDef) and n.name == 'main'))
        self.assertIn("'envelope': None if failures else envelope(per_mode)", main)
        self.assertIn('if not failures:', main)


if __name__ == '__main__':
    unittest.main()
