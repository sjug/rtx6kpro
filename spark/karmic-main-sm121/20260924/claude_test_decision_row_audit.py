"""CPU tests for claude_decision_row_audit.py on synthetic four-rank captures built with the pinned B12X encoders.
  .venv-snapshot-cpu/bin/python -m unittest claude_test_decision_row_audit
"""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('claude_decision_row_audit_under_test', HERE / 'claude_decision_row_audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
REF = audit.load_reference()
EXPECTED = {'image_id': 'b2-image', 'kit_sha256': 'b2-kit', 'source_trees': {'b12x': 'tree', 'vllm': 'v-b2'},
            'regime': {'swa.extend': 'bf16/h16'}, 'response_signature': 'b2-signature'}
OBSERVED = dict(EXPECTED, image_id='diag-image', kit_sha256='diag-kit', source_trees={'b12x': 'tree', 'vllm': 'v-diag'})


def encode_mxfp4(x):
    """E2M1 x UE8M0 encoder with the rules of b12x tests/attention/test_dsa_indexer_mxfp4.py _oracle_quant."""
    groups = x.float().reshape(*x.shape[:-1], 4, 32)
    amax = groups.abs().amax(-1).clamp_min(6 * 2.0 ** -126)
    bits = (amax / 6).contiguous().view(torch.int32)
    scales = ((bits >> 23) + ((bits & 0x7FFFFF) != 0)).to(torch.uint8)
    scale = (scales.int() << 23).view(torch.float32)
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6])
    order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7])
    pick = (groups.div(scale[..., None]).abs()[..., None] - lut[order]).abs().argmin(-1)
    code = (order[pick].to(torch.uint8) | (torch.signbit(groups).to(torch.uint8) << 3)).reshape(*x.shape[:-1], 128)
    return code[..., ::2] | (code[..., 1::2] << 4), scales


def key_pages(keys, page_size):
    data, scales = encode_mxfp4(keys)
    pages = -(-keys.shape[0] // page_size)
    out = torch.zeros((pages, page_size * 68), dtype=torch.uint8)
    for p in range(pages):
        lo, hi = p * page_size, min((p + 1) * page_size, keys.shape[0])
        out[p, :(hi - lo) * 64] = data[lo:hi].reshape(-1)
        out[p, page_size * 64:page_size * 64 + (hi - lo) * 4] = scales[lo:hi].reshape(-1)
    return out


def records(kv, kind):
    return REF.pack_deepseek_v41_cache_reference(kv, page_size=1, cache_kind=kind).reshape(kv.shape[0], -1)


def reference_out(entry):
    kwargs = {}
    if entry['indexed_len'] is not None:
        n = entry['indexed_len']
        kwargs = dict(extra_k_cache=entry['indexed_records'], extra_indices=torch.arange(n, dtype=torch.int32)[None],
                      extra_topk_lengths=torch.tensor([n], dtype=torch.int32), extra_page_size=1)
    return REF.compressed_sparse_mla_reference(
        entry['q'].float()[None], entry['swa_records'], torch.arange(128, dtype=torch.int32)[None],
        torch.tensor([128], dtype=torch.int32), sm_scale=512 ** -0.5, attn_sink=entry['attn_sink'],
        swa_page_size=1, cache_format='deepseek_v41', **kwargs).reshape(16, 512)


class Synthetic:
    """Self-consistent four-rank capture: production equals the reference by construction."""

    COUNTS = {2: 640, 8: 640, 14: 640, 20: 20000}

    def __init__(self, seed=0):
        self.g = torch.Generator().manual_seed(seed)
        self.table = audit.layer_table()
        self.keys = {s: torch.randn(n, 128, generator=self.g) for s, n in self.COUNTS.items()}
        self.index_q = {l: torch.randn(32, 128, generator=self.g) for l in (2, 8, 14, 20, 24, 28, 32, 36)}
        self.weights = {l: (torch.rand(32, generator=self.g) / 8).to(torch.bfloat16) for l in self.index_q}
        self.layers = {}
        for l in self.table:                                                 # layer 20 before its consumers
            self.layers[l] = self.layer(l)

    def layer(self, layer):
        role, g = self.table[layer], self.g
        rows = 128 if role['ced_decoder'] else 8192
        entry = {'q': (torch.randn(16, 512, generator=g) / 4).to(torch.bfloat16), 'attn_sink': torch.randn(16, generator=g),
                 'ced_decoder': role['ced_decoder'], 'query_rows': rows, 'row': rows - 1, 'position': 524287,
                 'swa_len': 128, 'swa_slots': torch.arange(128, dtype=torch.int64) + 256,
                 'swa_records': records(torch.randn(128, 512, generator=g) / 4, 'swa'), 'indexed_len': None,
                 'plan': {'mode': 'extend', 'config': {'v41_compute_mode': 'bf16' if layer % 2 else 'fp8'}}}
        if role['ratio']:
            page = role['page']
            logical = torch.randperm(4 * page, generator=g)[:512].sort().values.int()
            table = torch.tensor([5, 9, 3, 7], dtype=torch.int32)
            slots = (table[logical.long() // page] * page + logical.long() % page)
            entry.update(indexed_len=512, indexed_logical=logical, indexed_page_table_row=table,
                         indexed_slots=slots, indexed_records=records(torch.randn(512, 512, generator=g) / 4, 'indexed'))
        entry['out'] = reference_out(entry).to(torch.bfloat16)
        if role['index_source']:
            entry['indexer'] = self.indexer(layer, role)
        return entry

    def indexer(self, layer, role):
        owner, page = role['kv_owner'], role['page']
        count = self.COUNTS[owner]
        pages = key_pages(self.keys[owner], page)
        q_data, q_scales = encode_mxfp4(self.index_q[layer])
        published, _ = audit.mxfp4_scores(q_data, q_scales, self.weights[layer], audit.key_matrix(pages, page, count))
        ix = {'q_data': q_data, 'q_scales': q_scales, 'weights': self.weights[layer], 'cache_length': count,
              'page_size': page, 'key_pages': pages, 'key_pages_sha256': audit.sha(pages), 'score_width': None}
        pool = list(range(count))
        if role['candidate_role'] == 'consume':
            source = self.layers[20]['indexer']
            pool = source['candidates'][:source['candidate_len']].tolist()
        k = min(512, len(pool))
        top = sorted(audit.deterministic_topk(published[torch.tensor(pool)], pool, k))
        ix['topk'] = torch.tensor(top + [-1] * (512 - k), dtype=torch.int32)
        if role['candidate_role'] == 'produce':
            blocks = (count + 7) // 8
            logits = torch.stack([published[b * 8:min(b * 8 + 8, count)].max() for b in range(blocks)])
            logits[(count - 1) // 8] = float('inf')
            chosen = audit.deterministic_topk(logits, list(range(blocks)), min(2048, blocks))
            cands = sorted(p for b in chosen for p in range(b * 8, min(b * 8 + 8, count)))
            ix['candidates'] = torch.tensor(cands + [-1] * (16384 - len(cands)), dtype=torch.int32)
            ix['candidate_len'] = len(cands)
        return ix

    def captures(self):
        out = []
        for rank in range(4):
            layers = copy.deepcopy(self.layers)
            if rank:
                for entry in layers.values():
                    entry.get('indexer', {}).pop('key_pages', None)
            out.append({'schema': audit.SCHEMA, 'layers': layers, 'meta': {
                'rank': rank, 'node': 'n', 'generation': 1, 'prompt_tokens': 524288, 'row_position': 524287,
                'chunk_rows': 8192, 'row_index': 8191, 'batch_requests': 1,
                'kit_sha256': OBSERVED['kit_sha256'], 'source_trees': OBSERVED['source_trees'], 'problems': []}})
        return out


SYNTHETIC = Synthetic()


def run(captures, expected=EXPECTED, observed=OBSERVED, **kw):
    return audit.audit(captures, expected, observed, **kw)


class LayerFacts(unittest.TestCase):
    def test_target_layer_roles(self):
        table = audit.layer_table()
        self.assertEqual(len(table), 40)                                     # draft/nextn layers 40-42 excluded
        self.assertEqual([l for l, r in table.items() if r['ratio'] == 0], [0, 1])
        self.assertEqual([l for l, r in table.items() if r['index_source']], [2, 8, 14, 20, 24, 28, 32, 36])
        self.assertEqual({table[l]['kv_owner'] for l in range(20, 40)}, {20})
        self.assertEqual((table[7]['kv_owner'], table[13]['index_owner'], table[39]['index_owner']), (2, 8, 36))
        self.assertEqual((table[5]['page'], table[30]['page']), (128, 256))
        self.assertEqual({l: table[l]['candidate_role'] for l in (14, 20, 24)}, {14: 'full', 20: 'produce', 24: 'consume'})
        self.assertEqual([l for l, r in table.items() if r['ced_decoder']], list(range(20, 40)))   # ced_decoder_start = 20

    def test_same_boot_control_separates_capacity_from_hooks(self):
        captures = SYNTHETIC.captures()
        control = dict(OBSERVED, response_signature='boot-signature')          # two unarmed cold requests, identical
        observed = dict(OBSERVED, response_signature='boot-signature')          # armed capture equals them
        report = run(captures, observed=observed, control=control)
        self.assertEqual(report['verdict'], 'within-conformance-envelope (provisional envelope)', report['problems'])
        b2, ctl = report['comparisons']['b2'], report['comparisons']['control']
        self.assertFalse(b2['response_signature_equal'])
        self.assertTrue(b2['regime_equal'] and b2['b12x_tree_equal'] and not b2['image_id_equal'])
        self.assertIn('HISTORICAL B2 FULL-RESPONSE MISMATCH', b2['response_note'])
        self.assertIn('not attributed', b2['response_note'])
        self.assertNotIn('effect', b2['response_note'])
        self.assertIn('an observation on this boot', ctl['note'])
        self.assertIn('remains a failed gate', b2['accuracy'])
        self.assertTrue(all(ctl[k + '_equal'] for k in audit.IDENTITY_KEYS))
        self.assertIn('no observed output perturbation', ctl['note'])
        same = run(captures, observed=observed, control=dict(control, response_signature='b2-signature'))
        self.assertEqual(same['verdict'], 'invalid-capture')                   # control must equal the capture
        for key, value in (('image_id', 'other-image'), ('kit_sha256', 'other-kit'),
                           ('source_trees', {'b12x': 'tree', 'vllm': 'v-other'}),
                           ('regime', {'swa.extend': 'fp8/h16'}), ('response_signature', 'other')):
            bad = run(captures, observed=observed, control=dict(control, **{key: value}))
            self.assertEqual(bad['verdict'], 'invalid-capture', key)
            self.assertTrue(any(f'observed {key} differs from the same-boot unarmed control' in p
                                for p in bad['problems']), (key, bad['problems']))
        lacking = run(captures, observed=observed, control={k: v for k, v in control.items() if k != 'regime'})
        self.assertTrue(any('control identity lacks regime' in p for p in lacking['problems']))
        # B2 regime and B12X tree stay validity requirements even with a control
        drift = run(captures, observed=dict(observed, regime={'swa.extend': 'fp8/h16'}),
                    control=dict(control, regime={'swa.extend': 'fp8/h16'}))
        self.assertEqual(drift['verdict'], 'invalid-capture')
        self.assertTrue(any('plans not equivalent' in p for p in drift['problems']))
        # without a control the legacy gate still requires the B2 full response
        legacy = run(captures, observed=observed)
        self.assertEqual(legacy['verdict'], 'invalid-capture')
        self.assertTrue(any('supply a same-boot unarmed control' in p for p in legacy['problems']))
        agree = run(captures, observed=OBSERVED, control=dict(OBSERVED))
        self.assertTrue(agree['comparisons']['b2']['response_signature_equal'])
        self.assertEqual(agree['verdict'], 'within-conformance-envelope (provisional envelope)')

    def test_row_geometry_and_position_are_validated(self):
        captures = SYNTHETIC.captures()
        self.assertEqual(audit.validate(captures[1], TABLE := audit.layer_table(), 1, OBSERVED), [])
        wrong = copy.deepcopy(captures[1])
        wrong['layers'][25].update(query_rows=8192, row=8191)          # a decoder layer indexed as if full-row
        wrong['layers'][3]['position'] = 524286
        problems = audit.validate(wrong, TABLE, 1, OBSERVED)
        self.assertTrue(any('layer 25: row geometry' in p for p in problems), problems)
        self.assertTrue(any('layer 3: captured position 524286' in p for p in problems), problems)

    def test_4096_geometry_through_full_audit(self):
        captures = SYNTHETIC.captures()
        for capture in captures:
            capture['meta'].update(chunk_rows=4096, row_index=4095)
            for entry in capture['layers'].values():
                if not entry['ced_decoder']:
                    entry.update(query_rows=4096, row=4095)
        report = run(captures)
        self.assertEqual(report['verdict'], 'within-conformance-envelope (provisional envelope)', report['problems'])


class Decode(unittest.TestCase):
    def test_mxfp4_round_trip_and_page_layout(self):
        x = torch.randn(70, 128, generator=torch.Generator().manual_seed(3))
        data, scales = encode_mxfp4(x)
        self.assertLess(float((audit.decode_mxfp4(data, scales).float() - x).norm() / x.norm()), 0.2)
        self.assertTrue(torch.equal(audit.key_matrix(key_pages(x, 64), 64, 70), audit.decode_mxfp4(data, scales)))
        with self.assertRaises(ValueError):
            audit.key_matrix(key_pages(x, 64), 64, 200)

    def test_reference_output_is_fp32(self):
        entry = SYNTHETIC.layers[25]
        row = audit.audit_attention(entry, REF, compute_mode='bf16', envelope=audit.PROVISIONAL_ENVELOPE)
        self.assertTrue(row['within'])
        self.assertEqual(reference_out(entry).dtype, torch.float32)
        self.assertGreater(row['rel_l2_max'], 0.0)                          # BF16 output rounding is visible


class Audit(unittest.TestCase):
    def setUp(self):
        self.captures = SYNTHETIC.captures()

    def test_consistent_capture(self):
        report = run(self.captures)
        self.assertEqual(report['verdict'], 'within-conformance-envelope (provisional envelope)', report.get('summary'))
        self.assertTrue(all(v['inputs_equal'] and v['selections_equal'] for v in report['cross_rank'].values()))
        self.assertEqual(report['ranks'][0][24]['indexer']['pool'], 16384)   # consumer scored the candidates only
        self.assertTrue(report['ranks'][0][20]['indexer']['candidates']['exact_equal'])
        self.assertIn('not an independent model oracle', report['limits'][0])
        calibrated = run(self.captures, envelope={'bf16': 0.02, 'fp8': 0.1})
        self.assertEqual(calibrated['verdict'], 'within-conformance-envelope')

    def test_incomplete_or_unidentified_captures_are_invalid(self):
        for c in (self.captures[:3], self.captures[:1]):
            self.assertEqual(run(c)['verdict'], 'invalid-capture')
        bad = copy.deepcopy(self.captures)
        bad[1]['meta'].pop('kit_sha256')
        self.assertEqual(run(bad)['verdict'], 'invalid-capture')
        bad = copy.deepcopy(self.captures)
        bad[2]['meta']['kit_sha256'] = 'another-kit'
        self.assertEqual(run(bad)['verdict'], 'invalid-capture')
        for key, value in (('response_signature', 'other'), ('regime', {'swa.extend': 'fp8/h16'}),
                           ('source_trees', {'b12x': 'other', 'vllm': 'v-diag'})):
            self.assertEqual(run(self.captures, observed=dict(OBSERVED, **{key: value}))['verdict'], 'invalid-capture')
        self.assertEqual(run(self.captures, expected={})['verdict'], 'invalid-capture')
        self.assertEqual(run(self.captures, observed={})['verdict'], 'invalid-capture')

    def test_structural_invalidity(self):
        for change in (lambda c: c['meta'].update(row_position=524286),
                       lambda c: c['meta'].update(batch_requests=2),
                       lambda c: c['meta'].update(problems=['layer 28: page table differs from key owner 20']),
                       lambda c: c['layers'].pop(39),
                       lambda c: c['layers'][5].pop('indexed_slots'),
                       lambda c: c['layers'][6].pop('out'),
                       lambda c: c['layers'][9].__setitem__('indexer', c['layers'][8]['indexer']),
                       lambda c: c['layers'][8]['indexer'].__setitem__('key_pages_sha256', '0' * 64),
                       lambda c: c['layers'][8]['indexer'].__setitem__('page_size', 256)):
            bad = copy.deepcopy(self.captures)
            change(bad[0])
            self.assertEqual(run(bad)['verdict'], 'invalid-capture')

    def test_attention_deviation(self):
        self.captures[3]['layers'][30]['out'][5] = (self.captures[3]['layers'][30]['out'][5].float() * 1.2).bfloat16()
        report = run(self.captures)
        self.assertEqual(report['verdict'], 'deviation-found')
        self.assertEqual(report['summary']['attention_outside_envelope'], [(3, 30)])

    def set_topk(self, layer, values):
        for c in self.captures:
            c['layers'][layer]['indexer']['topk'] = torch.tensor(values, dtype=torch.int32)

    def test_selection_structure_is_strict(self):
        top = SYNTHETIC.layers[8]['indexer']['topk'].tolist()
        for bad in (top[:-1] + [top[-2]],                                      # duplicate
                    top[:-1] + [-1],                                           # wrong cardinality
                    top[:-1] + [640],                                          # beyond cache_length
                    [-1] + top[:-1]):                                          # padding not a pure tail
            self.captures = SYNTHETIC.captures()
            self.set_topk(8, sorted(bad, key=lambda v: (v < 0, v)) if bad[0] != -1 else bad)
            report = run(self.captures)
            self.assertIn(8, report['summary']['indexer_structure'], bad[-3:])
            self.assertEqual(report['verdict'], 'deviation-found')

    def test_out_of_pool_consumer_selection(self):
        pool = set(SYNTHETIC.layers[20]['indexer']['candidates'].tolist())
        outside = next(i for i in range(20000) if i not in pool)
        top = SYNTHETIC.layers[24]['indexer']['topk'].tolist()
        self.set_topk(24, sorted(top[1:] + [outside]))
        self.assertIn(24, run(self.captures)['summary']['indexer_structure'])

    def scores(self, layer):
        ix = SYNTHETIC.layers[layer]['indexer']
        return audit.mxfp4_scores(ix['q_data'], ix['q_scales'], ix['weights'],
                                  audit.key_matrix(ix['key_pages'], ix['page_size'], ix['cache_length']))[0]

    def test_exact_tie_order_is_separate_from_band(self):
        published = self.scores(2)
        order = sorted(range(640), key=lambda i: (-float(published[i]), i))
        self.assertEqual(float(published[order[511]]), float(published[order[512]]))  # BF16 ties are common
        swapped = sorted(set(order[:512]) - {order[511]} | {order[512]})
        self.set_topk(2, swapped)
        report = run(self.captures)
        self.assertEqual(report['summary']['indexer_exact_order_differences'], [2])
        self.assertEqual(report['summary']['indexer_outside_band'], [])
        self.assertEqual(report['verdict'], 'unresolved-boundary-order')      # not a pass

    def test_outside_band_selection_is_a_deviation(self):
        published = self.scores(14)
        worst = int(torch.argmin(published))
        top = SYNTHETIC.layers[14]['indexer']['topk'].tolist()
        self.set_topk(14, sorted(top[1:] + [worst]))
        report = run(self.captures)
        self.assertEqual(report['summary']['indexer_outside_band'], [14])
        self.assertEqual(report['verdict'], 'deviation-found')

    def test_candidate_production_deviation(self):
        ix = self.captures[0]['layers'][20]['indexer']
        cands = ix['candidates'][:ix['candidate_len']].tolist()
        chosen = {p // 8 for p in cands}
        published = self.scores(20)
        missing = max((b for b in range(2500) if b not in chosen), key=lambda b: float(published[b * 8:b * 8 + 8].max()))
        weakest = min(chosen - {(20000 - 1) // 8}, key=lambda b: float(published[b * 8:b * 8 + 8].max()))
        new = sorted([p for p in cands if p // 8 != weakest] + list(range(missing * 8, missing * 8 + 8)))
        report = run([self.patch_candidates(c, new) for c in self.captures])
        row = report['ranks'][0][20]['indexer']['candidates']
        self.assertFalse(row['exact_equal'])
        self.assertIn(20, report['summary']['indexer_exact_order_differences'])

    @staticmethod
    def patch_candidates(capture, cands):
        ix = capture['layers'][20]['indexer']
        ix['candidates'] = torch.tensor(cands + [-1] * (16384 - len(cands)), dtype=torch.int32)
        ix['candidate_len'] = len(cands)
        return capture

    def test_replicated_input_divergence_versus_selection_mismatch(self):
        diverged = copy.deepcopy(self.captures)
        diverged[2]['layers'][28]['indexer']['weights'] = diverged[2]['layers'][28]['indexer']['weights'] * 2
        report = run(diverged)
        self.assertEqual(report['summary']['replicated_input_divergence'], [28])
        self.assertEqual(report['summary']['replicated_selection_mismatch'], [])
        mismatched = copy.deepcopy(self.captures)
        top = mismatched[1]['layers'][32]['indexer']['topk'].tolist()
        mismatched[1]['layers'][32]['indexer']['topk'] = torch.tensor(top[:-1] + [-1], dtype=torch.int32)
        report = run(mismatched)
        self.assertEqual(report['summary']['replicated_selection_mismatch'], [32])

    def test_mapper_mismatch(self):
        slots = self.captures[1]['layers'][22]['indexed_slots'].clone()
        slots[3] += 1
        self.captures[1]['layers'][22]['indexed_slots'] = slots
        self.assertEqual(run(self.captures)['summary']['mapper_mismatch'], [(1, 22)])

    def test_needle_coverage(self):
        self.assertEqual(audit.needle_coverage([1, 5, 9], 2, {'identity': (2, 6), 'code': (18, 20)}),
                         {'identity': [1], 'code': [9]})

    def test_tie_exposure_counts_and_opposite_rule(self):
        values = torch.tensor([9.0, 5.0, 5.0, 5.0, 5.0, 1.0, 5.0, 7.0])
        positions = list(range(100, 108))
        selected = audit.deterministic_topk(values, positions, 4)             # [100, 107, 101, 102]
        ties = audit.tie_exposure(values, positions, selected, 4, bound=108)
        self.assertEqual((ties['threshold'], ties['tie_population'], ties['strictly_above'], ties['tie_admitted']),
                         (5.0, 5, 2, 2))
        self.assertEqual(ties['opposite_selection'], [100, 104, 106, 107])
        self.assertEqual(ties['opposite_rule_changes'], 2)
        self.assertEqual(ties['tie_admitted_position_quartiles'], [0, 0, 0, 2])
        self.assertEqual(audit.opposite_topk(values, positions, 4), [100, 107, 106, 104])
        none = audit.tie_exposure(values[:0], [], [], 0, bound=1)
        self.assertEqual((none['threshold'], none['tie_population'], none['opposite_selection']), (None, 0, []))
        # ties reported against the reference rule even when production picked the other side
        swapped = audit.tie_exposure(values, positions, [100, 107, 104, 106], 4, bound=108)
        self.assertEqual((swapped['tie_admitted'], swapped['tie_population']), (2, 5))

    def test_report_carries_ties_and_needles_under_both_rules(self):
        entries = 640
        tokens = 524288 // entries
        spans = {'early': (0, tokens * 3), 'late': (tokens * 300, tokens * 302)}
        report = run(self.captures, spans=spans)
        self.assertEqual(report['verdict'], 'within-conformance-envelope (provisional envelope)')
        row = report['ranks'][0][2]['indexer']
        for key in ('threshold', 'tie_population', 'strictly_above', 'tie_admitted', 'opposite_rule_changes',
                    'tie_admitted_position_quartiles'):
            self.assertIn(key, row['ties'])
        self.assertNotIn('opposite_selection', row['ties'])
        self.assertNotIn('opposite_selected', row)                                 # popped like 'selected'
        self.assertEqual(set(row['needles']), {'lowest_index_rule', 'highest_index_rule'})
        self.assertEqual(set(row['needles']['lowest_index_rule']), {'early', 'late'})
        self.assertGreater(row['ties']['tie_population'], 0)                     # BF16 ties over 640 entries
        self.assertEqual(row['ties']['strictly_above'] + row['ties']['tie_admitted'], 512)
        self.assertIn('ties', report['ranks'][0][20]['indexer']['candidates'])
        summary = report['summary']
        self.assertIn(2, summary['tie_exposed_layers'])
        self.assertEqual(set(summary['tie_exposed_layers'][2]),
                         {'tie_population', 'tie_admitted', 'strictly_above', 'opposite_rule_changes'})
        self.assertIsInstance(summary['needle_coverage_rule_dependent'], list)
        self.assertNotIn('needles', run(self.captures)['ranks'][0][2]['indexer'])   # spans are optional

    def test_tie_context_never_changes_the_verdict(self):
        report = run(self.captures)
        self.assertTrue(report['summary']['tie_exposed_layers'])
        self.assertEqual(report['verdict'], 'within-conformance-envelope (provisional envelope)')
        published = self.scores(14)
        worst = int(torch.argmin(published))
        top = SYNTHETIC.layers[14]['indexer']['topk'].tolist()
        self.set_topk(14, sorted(top[1:] + [worst]))
        self.assertEqual(run(self.captures)['verdict'], 'deviation-found')

    def test_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = []
            for rank, capture in enumerate(self.captures):
                torch.save(capture, tmp / f'r{rank}.pt')
                paths.append(str(tmp / f'r{rank}.pt'))
            (tmp / 'expected.json').write_text(json.dumps(EXPECTED))
            (tmp / 'observed.json').write_text(json.dumps(OBSERVED))
            args = ['--expected', str(tmp / 'expected.json'), '--observed', str(tmp / 'observed.json')]
            with mock.patch('builtins.print'):
                self.assertEqual(audit.main([*paths, *args, '--json', str(tmp / 'o.json')]), 0)
                self.assertEqual(audit.main([*paths[:3], *args]), 3)
            (tmp / 'observed.json').write_text(json.dumps(dict(OBSERVED, response_signature='boot')))
            (tmp / 'control.json').write_text(json.dumps(dict(OBSERVED, response_signature='boot')))
            with mock.patch('builtins.print'):
                self.assertEqual(audit.main([*paths, *args]), 3)                                 # no control: invalid
                self.assertEqual(audit.main([*paths, *args, '--control', str(tmp / 'control.json'),
                                             '--json', str(tmp / 'c.json')]), 0)
            self.assertFalse(json.loads((tmp / 'c.json').read_text())['comparisons']['b2']['response_signature_equal'])


class MatchedRegime(unittest.TestCase):
    """--regime-reference: validity against the pinned matched regime, B2 reported and never silently dropped."""
    PINNED = {'swa.extend': 'bf16/h16', 'ratio2.extend': 'fp8/h16'}

    def setUp(self):
        self.captures = SYNTHETIC.captures()
        self.expected = dict(EXPECTED, regime={'swa.extend': 'bf16/h16', 'ratio2.extend': 'bf16/h16'})
        self.matched = {'mode': 'matched-pinned-regime', 'source': 'receipts/x/regime-pre.json', 'sha256': 'f' * 64,
                        'blocks': 81389, 'chunk_rows': 8192, 'regime': dict(self.PINNED),
                        'regime_difference_from_b2': {'ratio2.extend': {'b2': 'bf16/h16', 'pinned': 'fp8/h16'}}}
        self.observed = dict(OBSERVED, regime=dict(self.PINNED), response_signature='boot', chunk_rows=8192)
        self.control = dict(self.observed)

    def test_matched_regime_is_the_validity_reference_and_b2_is_reported(self):
        report = run(self.captures, expected=self.expected, observed=self.observed, control=self.control, matched=self.matched)
        self.assertEqual(report['verdict'], 'within-conformance-envelope (provisional envelope)', report['problems'])
        self.assertEqual(report['mode'], 'matched-pinned-regime')
        ref = report['comparisons']['regime_reference']
        self.assertEqual((ref['validity_regime'], ref['observed_equals_validity_regime'], ref['observed_equals_b2_regime']),
                         ('pinned', True, False))
        self.assertEqual((ref['sha256'], ref['blocks'], ref['chunk_rows']), ('f' * 64, 81389, 8192))
        b2 = report['comparisons']['b2']
        self.assertFalse(b2['regime_equal'])
        self.assertEqual(b2['regime_difference_from_pinned'], {'ratio2.extend': {'b2': 'bf16/h16', 'observed': 'fp8/h16'}})
        self.assertIn('B2 REGIME NOT REQUIRED IN MATCHED MODE', b2['regime_note'])
        self.assertIn('not claimed', b2['regime_note'])

    def test_without_the_reference_the_b2_regime_stays_required(self):
        report = run(self.captures, expected=self.expected, observed=self.observed, control=self.control)
        self.assertEqual(report['verdict'], 'invalid-capture')
        self.assertTrue(any('plans not equivalent' in p for p in report['problems']))
        self.assertEqual(report['mode'], 'historical-b2')

    def test_observed_regime_must_equal_the_pinned_regime(self):
        drift = dict(self.observed, regime={'swa.extend': 'bf16/h16', 'ratio2.extend': 'bf16/h16'})
        report = run(self.captures, expected=self.expected, observed=drift, control=dict(drift), matched=self.matched)
        self.assertEqual(report['verdict'], 'invalid-capture')
        self.assertTrue(any('pinned matched regime' in p for p in report['problems']))
        lacking = run(self.captures, expected=self.expected, observed=self.observed, control=self.control,
                      matched={k: v for k, v in self.matched.items() if k != 'sha256'})
        self.assertTrue(any('matched regime reference lacks sha256' in p for p in lacking['problems']))

    def test_chunk_grid_must_agree_between_capture_record_and_reference(self):
        wrong_ref = run(self.captures, expected=self.expected, observed=self.observed, control=self.control,
                        matched=dict(self.matched, chunk_rows=4096))
        self.assertTrue(any('observed chunk grid differs' in p for p in wrong_ref['problems']))
        observed_4096 = dict(self.observed, chunk_rows=4096)
        wrong_capture = run(self.captures, expected=self.expected, observed=observed_4096, control=dict(observed_4096),
                            matched=dict(self.matched, chunk_rows=4096))
        self.assertEqual(wrong_capture['verdict'], 'invalid-capture')
        self.assertTrue(any('capture chunk grid 8192 differs' in p for p in wrong_capture['problems']))
        captures = SYNTHETIC.captures()
        for capture in captures:
            capture['meta'].update(chunk_rows=4096, row_index=4095)
            for entry in capture['layers'].values():
                if not entry['ced_decoder']:
                    entry.update(query_rows=4096, row=4095)
        right = run(captures, expected=self.expected, observed=observed_4096, control=dict(observed_4096),
                    matched=dict(self.matched, chunk_rows=4096))
        self.assertEqual(right['verdict'], 'within-conformance-envelope (provisional envelope)', right['problems'])

    def test_cli_regime_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = []
            for rank, capture in enumerate(self.captures):
                torch.save(capture, tmp / f'r{rank}.pt')
                paths.append(str(tmp / f'r{rank}.pt'))
            for name, value in (('expected', self.expected), ('observed', self.observed), ('control', self.control),
                                ('regime-reference', self.matched)):
                (tmp / f'{name}.json').write_text(json.dumps(value))
            args = [*paths, '--expected', str(tmp / 'expected.json'), '--observed', str(tmp / 'observed.json'),
                    '--control', str(tmp / 'control.json')]
            with mock.patch('builtins.print'):
                self.assertEqual(audit.main(args), 3)                                              # B2 regime required
                self.assertEqual(audit.main([*args, '--regime-reference', str(tmp / 'regime-reference.json'),
                                             '--json', str(tmp / 'm.json')]), 0)
            self.assertEqual(json.loads((tmp / 'm.json').read_text())['mode'], 'matched-pinned-regime')


if __name__ == '__main__':
    unittest.main()
