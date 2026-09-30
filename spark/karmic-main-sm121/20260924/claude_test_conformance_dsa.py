"""CPU tests for the pure parts of claude_conformance_dsa.py (the GPU execution runs only in-image).
  .venv-snapshot-cpu/bin/python -m unittest claude_test_conformance_dsa
"""
import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dsa = load('claude_conformance_dsa_under_test', 'claude_conformance_dsa.py')
audit = load('claude_dsa_audit_reference', 'claude_decision_row_audit.py')
B1 = HERE / 'receipts/router-b1-tuning-20260925/dusty.json'


def pages_for(keys, page):
    data, scales = dsa.encode_mxfp4(keys, torch)
    n = keys.shape[0]
    count = -(-n // page)
    out = torch.zeros((count, page * 68), dtype=torch.uint8)
    for p in range(count):
        lo, hi = p * page, min((p + 1) * page, n)
        out[p, :(hi - lo) * 64] = data[lo:hi].reshape(-1)
        out[p, page * 64:page * 64 + (hi - lo) * 4] = scales[lo:hi].reshape(-1)
    return out


class Reference(unittest.TestCase):
    def setUp(self):
        self.g = torch.Generator().manual_seed(21)

    def test_future_mask_is_allowed_but_live_nan_is_rejected(self):
        scores = torch.tensor([[1., 2., float('-inf')], [1., 2., 3.]])
        lengths = torch.tensor([2, 3])
        dsa.valid_scores_finite(scores, lengths, torch)
        scores[0, 1] = float('nan')
        with self.assertRaises(FloatingPointError):
            dsa.valid_scores_finite(scores, lengths, torch)
        with self.assertRaises(ValueError):
            dsa.valid_scores_finite(scores, torch.tensor([2, 4]), torch)

    def test_codec_and_layout_match_the_audit(self):
        keys = torch.randn(300, 128, generator=self.g)
        data, scales = dsa.encode_mxfp4(keys, torch)
        self.assertTrue(torch.equal(dsa.decode_mxfp4(data, scales, torch), audit.decode_mxfp4(data, scales)))
        pages = pages_for(keys, 128)
        self.assertTrue(torch.equal(dsa.key_rows(pages, 128, 300, torch), audit.key_matrix(pages, 128, 300)))

    def test_published_scores_match_the_audit(self):
        keys = audit.decode_mxfp4(*dsa.encode_mxfp4(torch.randn(500, 128, generator=self.g), torch))
        q_data, q_scales = dsa.encode_mxfp4(torch.randn(32, 128, generator=self.g), torch)
        weights = (torch.randn(32, generator=self.g) * 0.05).bfloat16()
        ours = dsa.published_scores(q_data, q_scales, weights, keys, torch)
        theirs, _ = audit.mxfp4_scores(q_data, q_scales, weights, keys)
        self.assertTrue(torch.equal(ours, theirs))

    def test_deterministic_selection_matches_the_audit_and_tie_rules_differ(self):
        values = torch.tensor([3.0, 1.0, 2.0, 2.0, 2.0, 0.5, 2.0])
        pool = torch.arange(values.numel())
        ours = dsa.select_reference(values, pool, 3, torch).tolist()
        self.assertEqual(ours, audit.deterministic_topk(values, pool.tolist(), 3))
        self.assertEqual(ours, [0, 2, 3])
        self.assertEqual(sorted(pool[dsa.ranked(values, torch, lowest_index_first=False)[:3]].tolist()), [0, 4, 6])
        subset = torch.tensor([6, 4, 1])
        self.assertEqual(dsa.select_reference(values, subset, 2, torch).tolist(), [4, 6])

    def test_candidate_rule_matches_the_audit(self):
        count = 20000
        scores = (torch.randn(count, generator=self.g) * 4).bfloat16().float()
        positions, _, chosen = dsa.candidate_reference(scores, count, torch)
        self.assertEqual(positions.numel(), 16384)
        self.assertIn((count - 1) // 8, chosen.tolist())                   # last visible block forced
        ix = {'candidates': torch.cat((positions.int(), torch.full((16384 - positions.numel(),), -1, dtype=torch.int32))),
              'candidate_len': positions.numel()}
        result = audit.audit_candidates(ix, scores, count)
        self.assertTrue(result['exact_equal'] and not result['structure_problems'], result)

    def test_tie_exposure(self):
        values = torch.tensor([9.0, 5.0, 5.0, 5.0, 5.0, 1.0, 5.0, 7.0])
        pool = torch.arange(8)
        selected = dsa.select_reference(values, pool, 4, torch)
        ties = dsa.tie_exposure(values, pool, selected, 4, torch)
        self.assertEqual((ties['threshold'], ties['tie_population'], ties['strictly_above'], ties['tie_admitted']),
                         (5.0, 5, 2, 2))
        self.assertEqual(ties['opposite_rule_changes'], 2)                  # {1,2} vs {4,6}
        self.assertEqual(sum(ties['tie_admitted_position_quartiles']), 2)
        self.assertEqual(ties['tie_admitted_position_quartiles'], [1, 1, 0, 0])  # admitted 1 and 2 of 0..7: early

    def test_compare_row_structure_and_band(self):
        scores = torch.tensor([4.0, 3.0, 2.0, 2.0, 1.0])
        ok = dsa.compare_row([0, 1, 2, -1], [0, 1, 2], scores, 2.0)
        self.assertEqual((ok['problems'], ok['exact_equal'], ok['outside_band']), ([], True, 0))
        tie = dsa.compare_row([0, 1, 3, -1], [0, 1, 2], scores, 2.0)
        self.assertEqual((tie['exact_equal'], tie['outside_band']), (False, 0))
        wrong = dsa.compare_row([0, 1, 4, -1], [0, 1, 2], scores, 2.0)
        self.assertEqual(wrong['outside_band'], 1)       # 4 (score 1.0) replaced a threshold entry: a deviation
        bad = dsa.compare_row([0, 0, 1, -1], [0, 1, 2], scores, 2.0)
        self.assertTrue(bad['problems'])
        self.assertTrue(dsa.compare_row([-1, 0, 1, 2], [0, 1, 2], scores, 2.0)['problems'])

    def test_ulp_distance(self):
        a = torch.tensor([1.0, -1.0, 0.0, 2.0])
        step = torch.tensor([1.0078125, -1.0078125, 0.0, 2.0])             # next BF16 above 1 is 1 + 2^-7
        self.assertEqual(dsa.ulp_distance(a, step, torch).tolist(), [1, 1, 0, 0])
        self.assertEqual(int(dsa.ulp_distance(torch.tensor([1e-30]), torch.tensor([-1e-30]), torch)[0]) > 0, True)

    def test_selection_rejects_tied_entry_outside_candidate_pool(self):
        scores = torch.tensor([4., 3., 2., 2.])
        result = dsa.compare_row([0, 1, 3], [0, 1, 2], scores, 2., pool=[0, 1, 2])
        self.assertIn('entry outside allowed pool', result['problems'])

    def test_repeat_gate_checks_live_scores_and_all_outputs(self):
        left = {'scores': torch.tensor([[1., float('-inf')]]),
                'indices': torch.tensor([[0, -1]]),
                'candidates': torch.tensor([[0, -1]]), 'lengths': torch.tensor([1])}
        right = {k: v.clone() for k, v in left.items()}
        right['scores'][0, 1] = float('nan')
        dsa.assert_repeat_equal(left, right, torch.tensor([1]), torch)
        for key in left:
            changed = {k: v.clone() for k, v in left.items()}
            changed[key].reshape(-1)[0] += 1
            with self.assertRaises(FloatingPointError, msg=key):
                dsa.assert_repeat_equal(left, changed, torch.tensor([1]), torch)

    def test_distributions_and_pages(self):
        book = dsa.make_keys('codebook', 5000, 3, torch, torch.device('cpu'))
        self.assertLessEqual(len({tuple(r.tolist()) for r in book[:500]}), 64)   # exact duplicate keys
        low = dsa.make_keys('low-magnitude', 1000, 3, torch, torch.device('cpu')).float()
        self.assertLess(float(low.abs().max()), 0.01)
        pages = dsa.choose_pages(80927, 2048, torch.Generator().manual_seed(2), torch)
        offsets = pages * dsa.BLOCK_STRIDE
        self.assertEqual(len(set(pages.tolist())), 2048)
        self.assertTrue(bool((offsets < 2 ** 31).any()) and bool((offsets >= 2 ** 31).any()))

    @unittest.skipUnless(B1.exists(), 'retained router-namespace selection snapshot absent')
    def test_private_selection_file_uses_the_production_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            namespace, name = dsa.selection_file(B1, tmp)
            self.assertEqual(name, '6115b03c7a814701d5610b3e6aecf82e30fdbb46a122d70441228538b9bd1e5e.json')
            self.assertEqual(namespace['tensor_parallel'], 4)
            self.assertEqual((Path(tmp) / name).read_bytes(), B1.read_bytes())


class Source(unittest.TestCase):
    text = (HERE / 'claude_conformance_dsa.py').read_text()

    def test_session_owns_entire_role_execution(self):
        tree = ast.parse(self.text)
        owners = [node for node in ast.walk(tree) if isinstance(node, ast.With)
                  and any(ast.unparse(item.context_expr) == 'factory()' for item in node.items)]
        self.assertEqual(len(owners), 1)
        self.assertIn('run_role(', ast.unparse(owners[0]))
        self.assertIn('session=session', ast.unparse(owners[0]))

    def test_recorded_production_configs_fail_closed(self):
        self.assertIn('autotune=False', self.text)
        self.assertIn('override=config', self.text)
        self.assertIn("payload['identity']", self.text)
        self.assertIn("configuration.space.validate(assignment)", self.text)
        self.assertIn("no production selection record", self.text)
        self.assertIn("except LookupError as error:", self.text)
        self.assertIn("not a production plan", self.text)
        self.assertIn("'source': plan.selection.source", self.text)

    def test_production_paths_and_shapes(self):
        run_role = ast.unparse(next(n for n in ast.parse(self.text).body
                                    if isinstance(n, ast.FunctionDef) and n.name == 'run_role'))
        for needle in ('dsa_indexer.quantize_write_index_k_mxfp4', 'dsa_indexer.quantize_q_mxfp4',
                       'score_mxfp4(binding, launchers=state._launchers)', 'state.layout.scratch_specs()',
                       "mode='prefill'", "cache_format='mxfp4'", 'max_candidates=spec[', 'candidate_topk_blocks=spec['):
            self.assertIn(needle, run_role)
        self.assertEqual((dsa.HEADS, dsa.TOPK, dsa.WIDTH, dsa.INDEX_CHUNK), (32, 512, 2344, 256))
        self.assertEqual({r: (v['page'], v['max_candidates'], v['candidate_topk_blocks']) for r, v in dsa.ROLES.items()},
                         {'full': (128, 0, 0), 'produce': (256, 0, 2048), 'consume': (256, 16384, 0)})

    def test_fail_closed_conditions_and_scope(self):
        main = ast.unparse(next(n for n in ast.parse(self.text).body if isinstance(n, ast.FunctionDef) and n.name == 'main'))
        for needle in ('layout_equal', 'structure_problem_rows', 'outside_band_rows', 'candidate_exact_rows',
                       'FloatingPointError', 'return 1 if failures else 0'):
            self.assertIn(needle, main)
        self.assertIn('not a model oracle', dsa.SCOPE)
        self.assertIn('original 524K answer', dsa.SCOPE)


class SelectionContract(unittest.TestCase):
    def exercise(self, *, bad_identity=False, bad_config=False, absent=False):
        class Mapping(dict):
            def to_dict(self):
                return dict(self)

        config = Mapping(kind='scalar')
        configuration = SimpleNamespace(encoded_query=Mapping(rows=256), query='query', device='device',
                                        space=SimpleNamespace(validate=lambda a: None))
        contract = SimpleNamespace(component_id='dsa', query_schema_version=1, config_schema_version=1,
                                   semantic_version=1, candidate_contract_version=1,
                                   configure=lambda *a, **k: configuration,
                                   _lower=lambda *a: config, config_payload=lambda c: c,
                                   iterate=lambda c: SimpleNamespace(done=True))
        plan = SimpleNamespace(contract=contract, query='query', invocation=Mapping())
        record = dict(assignment={}, config={'kind': 'wrong' if bad_config else 'scalar'}, coverage={})
        payload = dict(identity={'namespace': {}, 'sm_count': 49 if bad_identity else 48},
                       records={} if absent else {'key': record})
        modules = {'b12x.preparation': SimpleNamespace(FrozenMapping=Mapping,
                     detect_device=lambda d: SimpleNamespace(identity='device')),
                   'b12x.preparation._cache': SimpleNamespace(
                     cache_identity=lambda ns, d: {'namespace': {}, 'sm_count': 48}, digest=lambda x: 'key')}
        with patch.dict('sys.modules', modules):
            return dsa.production_config(plan, payload, SimpleNamespace(index=0))

    def test_cached_assignment_is_preserved(self):
        config, receipt = self.exercise()
        self.assertEqual(config, {'kind': 'scalar'})
        self.assertEqual(receipt['source'], 'cached')

    def test_identity_assignment_and_missing_selection_fail_closed(self):
        for options in ({'bad_identity': True}, {'bad_config': True}, {'absent': True}):
            with self.subTest(options=options), self.assertRaises(LookupError):
                self.exercise(**options)


if __name__ == '__main__':
    unittest.main()
