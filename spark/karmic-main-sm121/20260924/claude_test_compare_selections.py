"""Tests for claude_compare_selections.py on retained real selection caches (read-only) and synthetic edits.
  python3 -m unittest claude_test_compare_selections
"""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('claude_compare_under_test', HERE / 'claude_compare_selections.py')
cmp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cmp)
CAPTURE = HERE / 'receipts/moe-capture-localization-20260925/probe/traces-00/dusty-tuning.json'
EARLIER = HERE / 'receipts/routed-isolation-dusty-20260925T201108Z/tuning-after.json'
REPLAY = HERE / 'claude_replay_gate_prefill.py'
LOG_A = ('(Worker_TP0 pid=416) INFO [b12x_prepare.py:631] b12x autotuning uses 32 temporary KV blocks before '
         'allocating {blocks} serving blocks\n(EngineCore pid=358) INFO GPU KV cache size: 11,367,869 tokens, x\n'
         'b12x ready attention.compressed_sparse_mla: 503/503 ready, 20 measured, 160 cached, 0 compilations, 0:09\n'
         'b12x ready comm.roce: 1051/1051 ready, candidates 0/1 prepared, 0 measured, 874 cached, 0 compilations\n')


@unittest.skipUnless(CAPTURE.exists() and EARLIER.exists(), 'retained selection caches absent')
class Real(unittest.TestCase):
    def setUp(self):
        self.base = cmp.load(CAPTURE)
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def log(self, name, blocks):
        path = Path(self.tmp.name) / name
        path.write_text(LOG_A.format(blocks=blocks))
        return path

    def edited(self, change):
        other = copy.deepcopy(self.base)
        change(other['records'])
        return other

    def test_router_and_mhc_decode_completely(self):
        decoded, undecoded = cmp.decode(self.base['records'])
        self.assertEqual(undecoded, [])
        self.assertEqual({k: len(v) for k, v in decoded.items() if k.startswith('mhc')},
                         {'mhc.pre': 18, 'mhc.pre.expanded': 18, 'mhc.post_pre': 18})
        post_pre = decoded['mhc.post_pre'][8192]['config']
        self.assertEqual((post_pre['backend'], post_pre['projection_tile_m'], post_pre['projection_tile_n'],
                          post_pre['projection_tile_k'], post_pre['projection_num_stages']),
                         ('tf32_tma', 192, 32, 64, 3))
        self.assertEqual(decoded['mhc.pre'][8192]['config']['backend'], 'native')
        self.assertEqual(decoded['mhc.pre.expanded'][8192]['config']['backend'], 'tf32_tma')
        self.assertEqual({r: decoded['router'][r]['config']['backend'] for r in (256, 8192)},
                         {256: 'prefill', 8192: 'prefill'})
        self.assertNotIn(254, decoded['router'])

    def test_every_config_family_is_known(self):
        self.assertFalse([r for r in self.base['records'].values() if cmp.family(r).startswith('unknown')])

    def test_identical_seed_is_identical(self):
        report = cmp.compare(self.base, copy.deepcopy(self.base), seed=copy.deepcopy(self.base))
        self.assertEqual((report['verdict'], report['issues']), ('identical-selections', []))
        self.assertEqual(report['router_plan_for_rows'], {'254': 8192, '256': 256, '258': 8192})

    def test_program_ids_alone_are_not_drift(self):
        other = self.edited(lambda r: [v.update(programs=[['cute', '0' * 64]]) for v in r.values()])
        report = cmp.compare(self.base, other)
        self.assertEqual(report['verdict'], 'identical-selections')
        self.assertEqual(report['common']['programs_changed'], len(self.base['records']))

    def test_router_and_mhc_winner_changes_are_drift(self):
        for label, rows in (('router', 256), ('mhc.post_pre', 8192)):
            key = next(k for k, v in cmp.known().items() if v == (label, rows))
            other = self.edited(lambda r: r[key]['config'].update(
                {'rows_per_tile': 4} if label == 'router' else {'projection_num_stages': 2}))
            report = cmp.compare(self.base, other)
            self.assertEqual(report['verdict'], 'drift', label)
            self.assertEqual(report['config_drift'][0]['decoded'], (label, rows))
            self.assertFalse(report['decoded'][label][str(rows)]['same'])

    def test_lost_and_new_capacity_independent_keys_are_drift(self):
        victim = next(iter(self.base['records']))
        report = cmp.compare(self.base, self.edited(lambda r: r.pop(victim)))
        self.assertEqual(report['verdict'], 'drift')
        gemv = next(r for r in self.base['records'].values() if cmp.family(r) == 'gemm.bf16_gemv')
        report = cmp.compare(self.base, self.edited(lambda r: r.update({'f' * 64: copy.deepcopy(gemv)})))
        self.assertEqual(report['new_classes'], {'drift: new capacity-independent query': 1})

    def test_new_sparse_mla_keys_depend_on_logged_capacity(self):
        earlier = cmp.load(EARLIER)
        new = set(self.base['records']) - set(earlier['records'])
        self.assertTrue(new)
        self.assertEqual({cmp.family(self.base['records'][k]) for k in new}, {'attention.compressed_sparse_mla'})
        self.assertEqual(cmp.compare(earlier, self.base)['verdict'], 'unresolved')
        differs = cmp.compare(earlier, self.base, parent_log=self.log('a.log', 81375),
                              candidate_log=self.log('b.log', 81360))
        self.assertEqual((differs['verdict'], differs['capacity']), ('capacity-only (conditional)', 'differs'))
        self.assertEqual(set(differs['new_measured_by_family']), {'attention.compressed_sparse_mla'})
        self.assertGreaterEqual(differs['new_measured_by_family']['attention.compressed_sparse_mla'], len(new))
        same = cmp.compare(earlier, self.base, parent_log=self.log('a.log', 81375),
                           candidate_log=self.log('b.log', 81375))
        self.assertEqual(same['verdict'], 'unresolved')
        self.assertIn('equal KV capacity', same['issues'][0])
        winners = cmp.compare(earlier, self.base, parent_before=earlier)['boot_new_winners']
        self.assertEqual(winners['parent_boot'], [])
        self.assertEqual(sum(n for _, n in winners['candidate_boot']), len(new))

    def test_identity_and_seed_mismatch_are_drift(self):
        other = copy.deepcopy(self.base)
        other['identity'] = dict(other['identity'], sm_count=47)
        self.assertEqual(cmp.compare(self.base, other)['verdict'], 'drift')
        bad_seed = self.edited(lambda r: next(iter(r.values()))['config'].update(backend='x'))
        self.assertEqual(cmp.compare(self.base, copy.deepcopy(self.base), seed=bad_seed)['verdict'], 'drift')


class Logs(unittest.TestCase):
    def test_log_lines(self):
        with tempfile.NamedTemporaryFile('w', suffix='.log') as f:
            f.write(LOG_A.format(blocks=81375))
            f.flush()
            parsed = cmp.parse_log(f.name)
        self.assertEqual((parsed['serving_blocks'], parsed['kv_tokens']), ([81375], [11367869]))
        self.assertEqual(parsed['ready'], {'attention.compressed_sparse_mla@503': {'measured': 20, 'cached': 160},
                                           'comm.roce@1051': {'measured': 0, 'cached': 874}})


class GateKey(unittest.TestCase):
    def test_matches_the_replay_helper(self):
        try:
            rspec = importlib.util.spec_from_file_location('claude_replay_for_key', REPLAY)
            replay = importlib.util.module_from_spec(rspec)
            rspec.loader.exec_module(replay)
        except ImportError:
            self.skipTest('replay helper needs torch; run with .venv-snapshot-cpu')
        for rows in (1, 254, 256, 8192):
            self.assertEqual(cmp.gate_key(rows), replay.choice_key(rows))


if __name__ == '__main__':
    unittest.main()
