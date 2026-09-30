"""CPU tests for claude_probe_needle_sensitivity.py (no nodes, no torch).
  .venv-snapshot-cpu/bin/python -m unittest claude_test_probe_needle_sensitivity
"""
import ast
import hashlib
import json
from pathlib import Path
import unittest
import tempfile

import claude_probe_needle_sensitivity as probe
from prepare_decision_reference import PROMPT_SHA, B2_IMAGE

HERE = Path(__file__).resolve().parent


class ComparisonScope(unittest.TestCase):
    def test_default_keeps_b2_gate(self):
        ref = {'regime': {'slot': {'mode': 'bf16'}}}
        selected, scope = probe.comparison_reference(ref, None)
        self.assertEqual(selected, ref)
        self.assertTrue(scope['regime_equals_b2'])

    def test_explicit_snapshot_records_difference_and_rejects_rank_drift(self):
        ref = {'regime': {'slot': {'mode': 'bf16'}}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'regime.json'
            ranks = {node: {'slot': {'mode': 'fp8'}} for node in probe.NODES}
            path.write_text(json.dumps(ranks))
            selected, scope = probe.comparison_reference(ref, path)
            self.assertEqual(selected['regime'], ranks[probe.NODES[0]])
            self.assertFalse(scope['regime_equals_b2'])
            self.assertIn('slot', scope['regime_difference_from_b2'])
            self.assertEqual(ref['regime']['slot']['mode'], 'bf16')
            ranks[probe.NODES[-1]] = ref['regime']
            path.write_text(json.dumps(ranks))
            with self.assertRaises(ValueError):
                probe.comparison_reference(ref, path)
FROZEN_HEX = '510c94b1bb4f42d2998093bd6b9c3d99'
PARTS = {'head': 'Archive identity ', 'hex': FROZEN_HEX,
         'code': '. Memorize this unique retrieval code: 739184.\nArchive:\n', 'n1': 20, 'late': '\nThe late retrieval code is 482617.\n',
         'n2': 9, 'tail': '\nArchive ends. Reply with only the initial retrieval code and the late retrieval code, in that order, separated by a comma and a space.'}
FRESH_A, FRESH_B = 'a' * 32, 'b' * 32


class FakeTokenizer:
    """Counts like the serving path for this structure: template 4 tokens (bos, user, assistant, /think),
    fixed text pieces by whitespace, one token per filler, and a per-identity token cost."""

    def __init__(self, hex_tokens):
        self.hex_tokens = hex_tokens
        self.calls = 0

    def __call__(self, messages):
        self.calls += 1
        text = messages[0]['content']
        parts = probe.CONTENT.fullmatch(text + ('' if text.endswith('.') else ''))
        head, hex_, rest = text[:17], text[17:49], text[49:]
        assert head == 'Archive identity ' and probe.HEX.match(hex_)
        pieces = len([w for w in rest.replace('\n', ' ').split(' ') if w])
        return 4 + 2 + self.hex_tokens[hex_] + pieces


class Structure(unittest.TestCase):
    def test_frozen_input_round_trips_byte_exact(self):
        content = json.loads(probe.INPUT.read_text())['messages'][0]['content']
        parts = probe.parse_content(content)
        self.assertEqual((parts['hex'], parts['n1'], parts['n2']), (FROZEN_HEX, 366944, 157263))
        rebuilt = probe.build_content(parts, parts['hex'], parts['n1'], parts['n2'])
        self.assertEqual(rebuilt, content)
        self.assertEqual(hashlib.sha256(rebuilt.encode()).hexdigest(), PROMPT_SHA)
        self.assertEqual(probe.before_late(parts, parts['hex'], parts['n1']), content.split('\nThe late retrieval code is')[0])

    def test_structure_is_required(self):
        with self.assertRaises(ValueError):
            probe.parse_content('Archive identity xyz. filler')
        with self.assertRaises(ValueError):
            probe.build_content(PARTS, 'not-hex', 1, 1)
        with self.assertRaises(ValueError):
            probe.build_content(PARTS, FROZEN_HEX, 0, 1)

    def test_plan_brackets_with_controls_and_uses_only_intended_changes(self):
        variants = probe.plan(FROZEN_HEX, [FRESH_A, FRESH_B])
        self.assertEqual([v[0] for v in variants], ['control-1', 'fresh-1', 'frozen-524285', 'fresh-2', 'frozen-524291', 'control-2'])
        self.assertEqual([v[2] for v in variants], [524288, 524288, 524285, 524288, 524291, 524288])
        self.assertEqual([v[1] for v in variants if v[0].startswith(('control', 'frozen'))], [FROZEN_HEX] * 4)
        self.assertEqual(len(probe.plan(FROZEN_HEX, [FRESH_A, FRESH_B, 'c' * 32])), 7)
        for bad in ([FRESH_A], [FRESH_A, FRESH_A], [FRESH_A, FROZEN_HEX], [FRESH_A, 'B' * 32]):
            with self.assertRaises(ValueError):
                probe.plan(FROZEN_HEX, bad)

    def test_fresh_hex_is_distinct_and_well_formed(self):
        value = probe.fresh_hex(FROZEN_HEX)
        self.assertTrue(probe.HEX.match(value))
        self.assertNotEqual(value, FROZEN_HEX)


class Fit(unittest.TestCase):
    """Small structure: prefix 4 template + 2 head words + hex tokens + 8 fixed words + n1 fillers.
    With the frozen hex costing 21 tokens (as measured live) the recorded late position is 4+2+21+8+20 = 55
    and the whole prompt is 55 + 6 + 9 + 24 tail words = 94 tokens."""

    def setUp(self):
        self.tok = FakeTokenizer({FROZEN_HEX: 21, FRESH_A: 18, FRESH_B: 23})
        probe.LATE_POSITION, probe.FROZEN_TOKENS = 55, 94
        self.addCleanup(setattr, probe, 'LATE_POSITION', 366986)
        self.addCleanup(setattr, probe, 'FROZEN_TOKENS', 524288)

    def test_fake_counts_match_the_frozen_structure(self):
        self.assertEqual(self.tok(probe.user(probe.before_late(PARTS, FROZEN_HEX, 20))), 55)
        self.assertEqual(self.tok(probe.user(probe.build_content(PARTS, FROZEN_HEX, 20, 9))), 94)

    def test_fresh_identity_moves_only_the_prefix_filler(self):
        for hex_, delta in ((FRESH_A, 3), (FRESH_B, -2)):
            record = probe.fit(PARTS, hex_, 94, self.tok)
            self.assertEqual(record['changed'], {'hex': True, 'n1_delta': delta, 'n2_delta': 0})
            self.assertEqual(record['rounds'][-1], {'n1': 20 + delta, 'n2': 9, 'prefix_tokens': 55, 'prompt_tokens': 94})
            self.assertEqual(record['content'], probe.build_content(PARTS, hex_, 20 + delta, 9))
            self.assertEqual(self.tok(probe.user(record['content'])), 94)

    def test_length_variants_move_only_the_filler_after_the_marker(self):
        for target, delta in ((91, -3), (97, 3)):
            record = probe.fit(PARTS, FROZEN_HEX, target, self.tok)
            self.assertEqual(record['changed'], {'hex': False, 'n1_delta': 0, 'n2_delta': delta})
            self.assertEqual(record['rounds'][-1]['prompt_tokens'], target)
            self.assertEqual(record['rounds'][-1]['prefix_tokens'], 55)
            frozen = probe.build_content(PARTS, FROZEN_HEX, 20, 9)
            self.assertTrue(record['content'].startswith(probe.before_late(PARTS, FROZEN_HEX, 20) + PARTS['late']))
            self.assertEqual(record['content'].split(PARTS['late'])[0], frozen.split(PARTS['late'])[0])
            self.assertTrue(record['content'].endswith(PARTS['tail']))

    def test_fit_fails_closed_when_counts_cannot_be_reproduced(self):
        drift = FakeTokenizer({FROZEN_HEX: 22, FRESH_A: 18, FRESH_B: 23})       # frozen prefix would be 56
        with self.assertRaises(RuntimeError):
            probe.fit(PARTS, FROZEN_HEX, 91, drift)
        with self.assertRaises(RuntimeError):
            probe.fit(PARTS, FRESH_A, 95, self.tok)                                # prefix fits, total cannot

    def test_construction_never_uses_estimates(self):
        source = Path(probe.__file__).read_text()
        self.assertIn("'/tokenize'", source)
        self.assertNotIn('len(content) //', source)
        self.assertNotIn('approx', source)


class Requests(unittest.TestCase):
    def test_tokenize_body_matches_qualify(self):
        body = probe.tokenize_body(probe.user('x'))
        self.assertEqual(body, {'model': 'DeepSeek-V4.1-Flash', 'messages': [{'role': 'user', 'content': 'x'}],
                                'chat_template_kwargs': {'thinking': False}, 'add_generation_prompt': True})

    def test_request_body_is_the_frozen_gate_request_with_content_and_salt(self):
        body = probe.request_body('content', 'salt')
        self.assertEqual(body, {'model': 'DeepSeek-V4.1-Flash', 'messages': [{'role': 'user', 'content': 'content'}],
                                'chat_template_kwargs': {'thinking': False}, 'temperature': 0, 'max_tokens': 64,
                                'logprobs': True, 'top_logprobs': 20, 'cache_salt': 'salt'})


def fake_response(tokens, tops):
    content = [{'token': t, 'logprob': tops[i][0][1], 'top_logprobs': [{'token': a, 'logprob': b} for a, b in tops[i]]}
               for i, t in enumerate(tokens)]
    return {'choices': [{'message': {'content': ''.join(tokens)}, 'finish_reason': 'stop', 'logprobs': {'content': content}}],
            'usage': {'prompt_tokens': 524288}}


class Margins(unittest.TestCase):
    def test_margins_for_identity_first_answer(self):
        result = fake_response(['510', 'c', ',', ' ', '739', '184'],
                               [[('510', -0.127), ('739', -2.127)], [('c', 0.0)], [(',', 0.0)], [(' ', 0.0)],
                                [('739', -0.474), ('482', -0.974)], [('184', 0.0)]])
        m = probe.margins(result, '739184, 482617')
        self.assertEqual({k: v for k, v in m['first_token'].items() if k != 'margin'},
                         {'expected_token': '739', 'expected_logprob': -2.127, 'competitor': '510', 'competitor_logprob': -0.127})
        self.assertAlmostEqual(m['first_token']['margin'], -2.0)
        self.assertAlmostEqual(m['late_code_after_separator']['margin'], -0.5)
        self.assertEqual(m['late_code_after_separator']['position'], 4)

    def test_margins_for_correct_answer_and_missing_expected_token(self):
        result = fake_response(['739', '184', ',', ' ', '482', '617'],
                               [[('739', -0.1), ('510', -3.1)], [('184', 0.0)], [(',', 0.0)], [(' ', 0.0)],
                                [('482', -0.2), ('739', -1.7)], [('617', 0.0)]])
        m = probe.margins(result, '739184, 482617')
        self.assertEqual(m['first_token']['margin'], 3.0)
        self.assertEqual(m['late_code_after_separator']['margin'], 1.5)
        absent = probe.choice_margin([{'token': '510', 'logprob': -0.1}], '739184, 482617')
        self.assertIsNone(absent['margin'])
        self.assertEqual(absent['competitor'], '510')

    def test_answer_shape(self):
        self.assertEqual(probe.answer_shape('510c94b1bb4f42d2998093bd6b9c3d99, 739184', '739184, 482617'), 'identity+early_code')
        self.assertEqual(probe.answer_shape('739184, 482617', '739184, 482617'), 'early_code+late_code')
        self.assertEqual(probe.answer_shape('x', '739184, 482617'), 'other')

    def test_response_row_and_summary(self):
        reference = {'response_signature': 'none'}
        result = fake_response(['739', '184', ',', ' ', '482', '617'],
                               [[('739', -0.1), ('510', -3.1)], [('184', 0.0)], [(',', 0.0)], [(' ', 0.0)],
                                [('482', -0.2), ('739', -1.7)], [('617', 0.0)]])
        row = probe.response_row(result, '739184, 482617', 524288, reference)
        self.assertTrue(row['correct'])
        self.assertFalse(row['signature_equals_b2'])
        rows = []
        for label in ('control-1', 'fresh-1', 'control-2'):
            r = dict(row, label=label, hex=FROZEN_HEX, elapsed_s=1.0, cached_tokens=0)
            rows.append(r)
        summary = probe.summarize(rows)
        self.assertTrue(summary['controls_identical'])
        self.assertFalse(summary['control_equals_b2'])
        self.assertEqual(summary['variants'][1]['first_token_margin'], 3.0)
        self.assertIn('not an acceptance gate', summary['scope'])
        rows[2]['response_signature'] = 'other'
        self.assertFalse(probe.summarize(rows)['controls_identical'])


def container(node='dusty', **overrides):
    env = {'DS41_NODE': node, 'DS41_KIT_SHA256': 'kit', 'B12X_DYNAMIC_DETERMINISTIC_OUTPUT': '1',
           'B12X_DENSE_SPLITK_TURBO': '0', 'B12X_COMPILE_CACHE_DIR': '/cache/jit/ns/b12x'}
    env.update(overrides.pop('env', {}))
    labels = {'local-inference.ds41.kit.sha256': 'kit'}
    image_labels = {'local-inference.ds41.diagnostic.kind': 'router-stage-release-candidate',
                    'vllm.source-tree': 'v', 'b12x.source-tree': 'b'}
    image_labels.update(overrides.pop('image_labels', {}))
    c = {'State': {'Running': True}, 'Image': 'sha256:' + B2_IMAGE, 'Id': 'abc',
         'Config': {'Env': [f'{k}={v}' for k, v in env.items()], 'Labels': labels}}
    c.update(overrides)
    image = {'Id': 'sha256:' + B2_IMAGE, 'Config': {'Labels': image_labels}}
    return c, image


REFERENCE = {'image_id': B2_IMAGE, 'kit_sha256': 'kit-b2', 'source_trees': {'vllm': 'v', 'b12x': 'b'}}


class Gate(unittest.TestCase):
    def test_authorized_contract_passes_and_reports_kit_relation(self):
        c, image = container()
        problems, identity = probe.check_container('dusty', c, image, REFERENCE, {'kit', 'kit-b2'})
        self.assertEqual(problems, [])
        self.assertEqual(identity['rank'], 0)
        self.assertFalse(identity['kit_equals_b2_reference'])
        self.assertEqual(identity['source_trees'], REFERENCE['source_trees'])

    def test_every_deviation_is_a_problem(self):
        cases = {
            'kv pin': container(env={'DS41_DECISION_ROW_BLOCKS': '80927'}),
            'prefetch': container(env={'VLLM_DS41_L2_PREFETCH': '0'}),
            'determinism': container(env={'B12X_DYNAMIC_DETERMINISTIC_OUTPUT': '0'}),
            'node': container(env={'DS41_NODE': 'toby'}),
            'image': container(Image='sha256:' + 'f' * 64),
            'kind': container(image_labels={'local-inference.ds41.diagnostic.kind': 'decision-row-capture'}),
            'tree': container(image_labels={'vllm.source-tree': 'other'}),
            'stopped': container(State={'Running': False}),
        }
        for name, (c, image) in cases.items():
            problems, _ = probe.check_container('dusty', c, image, REFERENCE, {'kit', 'kit-b2'})
            self.assertTrue(problems, name)
        problems, _ = probe.check_container('dusty', *container(), REFERENCE, {'newer-kit', 'kit-b2'})
        self.assertIn('container runs neither the current reviewed runtime kit nor the retained B2 kit', problems)
        c, image = container(env={'DS41_KIT_SHA256': 'kit-b2'})
        c['Config']['Labels']['local-inference.ds41.kit.sha256'] = 'kit-b2'
        problems, identity = probe.check_container('dusty', c, image, REFERENCE, {'newer-kit', 'kit-b2'})
        self.assertEqual(problems, [])
        self.assertTrue(identity['kit_equals_b2_reference'])

    def test_serving_blocks_from_boot_log(self):
        line = 'b12x autotuning uses 32 temporary KV blocks before allocating 81592 serving blocks'
        self.assertEqual(probe.serving_blocks('x\n' + line + '\n' + line + '\n'), 81592)
        with self.assertRaises(ValueError):
            probe.serving_blocks('nothing')
        with self.assertRaises(ValueError):
            probe.serving_blocks(line + '\n' + line.replace('81592', '80927'))


class Scope(unittest.TestCase):
    def test_probe_never_changes_node_state(self):
        source = Path(probe.__file__).read_text()
        for forbidden in ('podman start', 'podman stop', 'podman run', 'podman rm', 'podman build', 'rm -rf',
                          'launch_contract', 'start_moe_repaired', 'runtime-files.json', 'num-gpu-blocks'):
            self.assertNotIn(forbidden, source, forbidden)
        tree = ast.parse(source)
        calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)}
        self.assertNotIn('unlink', calls)
        self.assertNotIn('rmtree', calls)

    def test_ssh_targets_only_the_four_ranks(self):
        self.assertEqual(probe.NODES, ('dusty', 'toby', 'rusty', 'kirby'))
        self.assertNotIn('sparky', Path(probe.__file__).read_text())


class Alignment(unittest.TestCase):
    def test_plan_alignment_order_targets_and_limits(self):
        variants = probe.plan_alignment(FROZEN_HEX, probe.FRESH1['hex'])
        self.assertEqual([v[0] for v in variants], ['control-1', 'frozen-524287', 'frozen-524289', 'frozen-524032',
                                                    'frozen-516096', 'frozen-532480', 'fresh1-524285', 'fresh1-524291', 'control-2'])
        self.assertEqual([v[2] for v in variants], [524288, 524287, 524289, 524032, 516096, 532480, 524285, 524291, 524288])
        self.assertEqual({v[1] for v in variants if v[0].startswith(('control', 'frozen'))}, {FROZEN_HEX})
        self.assertEqual({v[1] for v in variants if v[0].startswith('fresh1')}, {probe.FRESH1['hex']})
        self.assertTrue(all(v[2] <= probe.MAX_PROMPT for v in variants))
        self.assertEqual(probe.MAX_PROMPT, 599936)
        self.assertEqual([n % 8192 == 0 for n in probe.ALIGNMENT_LENGTHS], [False, False, False, True, True])
        self.assertEqual([n % 256 == 0 for n in probe.ALIGNMENT_LENGTHS], [False, False, True, True, True])
        with self.assertRaises(ValueError):
            probe.plan_alignment(FROZEN_HEX, FROZEN_HEX)

    def test_fresh1_constant_matches_the_retained_receipt(self):
        hex_, delta = probe.fresh1_from_receipt(HERE / probe.FRESH1['receipt'])
        self.assertEqual((hex_, delta), ('94e8bd2ae83f4340bdcc38c52b3ad98a', 1))
        record = json.loads((HERE / probe.FRESH1['receipt']).read_text())['fresh-1']
        self.assertEqual(record['rounds'][-1], {'n1': 366945, 'n2': 157263, 'prefix_tokens': 366986, 'prompt_tokens': 524288})
        with tempfile.TemporaryDirectory() as tmp:
            bad = dict(record, hex='0' * 32)
            path = Path(tmp) / 'construction.json'
            path.write_text(json.dumps({'fresh-1': bad}))
            with self.assertRaises(RuntimeError):
                probe.fresh1_from_receipt(path)

    def test_default_series_plan_is_unchanged(self):
        self.assertEqual([v[0] for v in probe.plan(FROZEN_HEX, [FRESH_A, FRESH_B])],
                         ['control-1', 'fresh-1', 'frozen-524285', 'fresh-2', 'frozen-524291', 'control-2'])

    def test_caution_and_unresolved_wording_are_recorded(self):
        source = Path(probe.__file__).read_text()
        self.assertIn('suggestive, not proof', probe.ALIGNMENT_CAUTION)
        self.assertIn('separately approved launcher change', probe.ALIGNMENT_CAUTION)
        self.assertIn('stays unresolved', probe.SCOPE)
        self.assertNotIn('never passed', source)


class FixedPrefixFit(unittest.TestCase):
    def setUp(self):
        self.tok = FakeTokenizer({FROZEN_HEX: 21, FRESH_A: 18, FRESH_B: 23})
        probe.LATE_POSITION, probe.FROZEN_TOKENS = 55, 94
        self.addCleanup(setattr, probe, 'LATE_POSITION', 366986)
        self.addCleanup(setattr, probe, 'FROZEN_TOKENS', 524288)

    def test_fixed_compensation_varies_only_the_run_after_the_marker(self):
        for target, delta in ((91, -3), (97, 3), (86, -8)):
            record = probe.fit(PARTS, FRESH_A, target, self.tok, n1_fixed=23)
            self.assertEqual(record['changed'], {'hex': True, 'n1_delta': 3, 'n2_delta': delta})
            self.assertEqual(record['rounds'][-1], {'n1': 23, 'n2': 9 + delta, 'prefix_tokens': 55, 'prompt_tokens': target})
            self.assertEqual(record['content'], probe.build_content(PARTS, FRESH_A, 23, 9 + delta))
            self.assertEqual(len(record['rounds']), 1)

    def test_wrong_fixed_compensation_fails_closed(self):
        with self.assertRaises(RuntimeError):
            probe.fit(PARTS, FRESH_A, 91, self.tok, n1_fixed=22)
        with self.assertRaises(ValueError):
            probe.fit(PARTS, FROZEN_HEX, 91, self.tok, n1_fixed=20)

    def test_frozen_aligned_lengths_move_only_n2(self):
        for target in (86, 102):
            record = probe.fit(PARTS, FROZEN_HEX, target, self.tok)
            self.assertEqual(record['changed'], {'hex': False, 'n1_delta': 0, 'n2_delta': target - 94})


if __name__ == '__main__':
    unittest.main()
