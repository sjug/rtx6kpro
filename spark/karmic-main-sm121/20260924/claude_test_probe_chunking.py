"""CPU tests for claude_probe_chunking.py (no nodes, no torch).
  .venv-snapshot-cpu/bin/python -m unittest claude_test_probe_chunking
"""
import ast
import hashlib
import json
from pathlib import Path
import unittest

import claude_probe_chunking as probe
import claude_probe_needle_sensitivity as sens
from claude_decode_sparse_mla import EXTEND_ROWS
from prepare_decision_reference import PROMPT_SHA, B2_IMAGE, exact_regime

HERE = Path(__file__).resolve().parent
RETAINED = HERE / 'receipts/router-fence-20260926T045016Z'          # real 8192-chunking boot logs, read only
HEAD_LOG = (RETAINED / 'dusty-final.log').read_text()                # API server line with enum/object reprs
WORKER_LOG = (RETAINED / 'toby-startup.log').read_text()             # headless rank: profile line, no API line
BLOCKS_LINE = 'b12x autotuning uses 32 temporary KV blocks before allocating 81389 serving blocks\n'
HEAD_ARGS_LINE = next(l for l in HEAD_LOG.splitlines() if probe.ARGS_PREFIX in l) + '\n'


def approved_head_log():
    """The retained head line with the two approved settings spliced in, as vLLM would print them."""
    return HEAD_ARGS_LINE.replace("'max_num_batched_tokens': 8192,",
                            "'max_num_batched_tokens': 8192, 'long_prefill_token_threshold': 7936, "
                            "'num_gpu_blocks_override': 81389,") + BLOCKS_LINE


def approved_profile_log(node='toby', threshold='7936'):
    line = next(l for l in WORKER_LOG.splitlines() if l.startswith(probe.PROFILE_PREFIX))
    profile = json.loads(line)
    profile['node'], profile['rank'] = node, probe.NODES.index(node)
    profile['model'] += ['--long-prefill-token-threshold', threshold, '--num-gpu-blocks-override', '81389']
    profile['env'].update(DS41_PREFILL_THRESHOLD=threshold, DS41_CHUNKING_BLOCKS='81389')
    return 'boot noise\n' + json.dumps(profile, sort_keys=True) + '\n' + BLOCKS_LINE


class EffectiveArgs(unittest.TestCase):
    def test_retained_head_line_is_not_a_literal_and_is_read_by_key_pattern(self):
        self.assertIn('<CUDAGraphMode.FULL_AND_PIECEWISE', HEAD_LOG)
        with self.assertRaises((ValueError, SyntaxError)):
            ast.literal_eval(next(l for l in HEAD_LOG.splitlines() if probe.ARGS_PREFIX in l).split(probe.ARGS_PREFIX, 1)[1])
        args = probe.parse_non_default_args(HEAD_LOG)
        self.assertEqual(args, {'max_num_batched_tokens': 8192, 'max_num_seqs': 4, 'enable_chunked_prefill': True})
        problems = probe.check_args(args, 7936)
        self.assertEqual(len(problems), 2)                        # threshold and override absent on the 8192 boot
        self.assertEqual(probe.check_args(probe.parse_non_default_args(approved_head_log()), 7936), [])
        self.assertTrue(probe.check_args(probe.parse_non_default_args(approved_head_log()), 4096))
        with self.assertRaises(ValueError):
            probe.parse_non_default_args(WORKER_LOG)               # headless ranks print no API server line
        with self.assertRaises(ValueError):
            probe.parse_non_default_args(HEAD_LOG + approved_head_log())

    def test_check_args_requires_threshold_pin_and_unchanged_capacity(self):
        good = {'long_prefill_token_threshold': 7936, 'num_gpu_blocks_override': 81389, 'max_num_batched_tokens': 8192,
                'max_num_seqs': 4, 'enable_chunked_prefill': True}
        self.assertEqual(probe.check_args(good, 7936), [])
        self.assertTrue(probe.check_args(good, 4096))
        for key, value in (('long_prefill_token_threshold', 0), ('num_gpu_blocks_override', 80927),
                           ('max_num_batched_tokens', 7936), ('max_num_seqs', 1), ('enable_chunked_prefill', False)):
            self.assertTrue(probe.check_args(dict(good, **{key: value}), 7936), key)

    def test_retained_worker_profile_line_parses_and_carries_the_effective_argv(self):
        profile = probe.parse_profile(WORKER_LOG)
        self.assertEqual((profile['node'], profile['rank']), ('toby', 1))
        self.assertIn('--max-num-batched-tokens', profile['model'])
        problems = probe.check_profile(profile, 'toby', 7936)
        self.assertTrue(any('--long-prefill-token-threshold' in p for p in problems))
        self.assertTrue(any('--num-gpu-blocks-override' in p for p in problems))
        self.assertEqual(probe.check_profile(probe.parse_profile(approved_profile_log()), 'toby', 7936), [])
        self.assertTrue(probe.check_profile(probe.parse_profile(approved_profile_log()), 'rusty', 7936))
        self.assertTrue(probe.check_profile(probe.parse_profile(approved_profile_log(threshold='4096')), 'toby', 7936))
        with self.assertRaises(ValueError):
            probe.parse_profile('no profile here')

    def test_verify_boot_log_per_rank_requirements(self):
        self.assertEqual(probe.verify_boot_log('toby', approved_profile_log('toby'), 7936), [])
        self.assertTrue(probe.verify_boot_log('dusty', approved_profile_log('dusty'), 7936))      # head needs the API line
        head = approved_profile_log('dusty') + approved_head_log()
        self.assertEqual(probe.verify_boot_log('dusty', head, 7936), [])
        self.assertTrue(probe.verify_boot_log('dusty', head, 4096))
        self.assertTrue(probe.verify_boot_log('toby', approved_profile_log('toby').replace('81389 serving', '81592 serving'), 7936))
        self.assertTrue(probe.verify_boot_log('toby', WORKER_LOG + BLOCKS_LINE, 7936))              # 8192 boot argv
        pinned = approved_profile_log('toby').replace('"DS41_CHUNKING_BLOCKS": "81389"',
                                                      '"DS41_CHUNKING_BLOCKS": "81389", "DS41_DECISION_ROW_BLOCKS": "80927"')
        self.assertTrue(probe.verify_boot_log('toby', pinned, 7936))

    def test_thresholds_are_block_aligned_and_below_capacity(self):
        self.assertEqual(probe.THRESHOLDS, (7936, 4096))
        for value in probe.THRESHOLDS:
            self.assertEqual(value % 256, 0)
            self.assertLess(value, probe.CAPACITY)
        self.assertEqual((probe.CAPACITY, probe.BLOCKS), (8192, 81389))


def container(threshold='7936', **overrides):
    env = {'DS41_NODE': 'dusty', 'DS41_KIT_SHA256': 'kit', 'B12X_DYNAMIC_DETERMINISTIC_OUTPUT': '1',
           'B12X_DENSE_SPLITK_TURBO': '0', 'B12X_COMPILE_CACHE_DIR': '/cache/jit/ns/b12x',
           'DS41_PREFILL_THRESHOLD': threshold, 'DS41_CHUNKING_BLOCKS': '81389'}
    env.update(overrides.pop('env', {}))
    for key in overrides.pop('drop', ()):
        env.pop(key, None)
    labels = {'local-inference.ds41.kit.sha256': 'kit'}
    image_labels = {'local-inference.ds41.diagnostic.kind': 'router-stage-release-candidate',
                    'vllm.source-tree': 'v', 'b12x.source-tree': 'b'}
    c = {'State': {'Running': True}, 'Image': 'sha256:' + B2_IMAGE, 'Id': 'abc',
         'Config': {'Env': [f'{k}={v}' for k, v in env.items()], 'Labels': labels}}
    c.update(overrides)
    image = {'Id': 'sha256:' + B2_IMAGE, 'Config': {'Labels': image_labels}}
    return c, image


REFERENCE = {'image_id': B2_IMAGE, 'kit_sha256': 'kit-b2', 'source_trees': {'vllm': 'v', 'b12x': 'b'}}


class Gate(unittest.TestCase):
    def test_approved_chunking_boot_passes(self):
        problems, identity = probe.check_env('dusty', *container(), REFERENCE, {'kit'}, 7936)
        self.assertEqual(problems, [])
        self.assertEqual(identity['threshold'], 7936)
        self.assertEqual(identity['chunking_blocks'], 81389)

    def test_every_deviation_is_a_problem(self):
        cases = {
            'wrong threshold': container(threshold='4096'),
            'threshold absent': container(drop=('DS41_PREFILL_THRESHOLD',)),
            'blocks absent': container(drop=('DS41_CHUNKING_BLOCKS',)),
            'blocks wrong': container(env={'DS41_CHUNKING_BLOCKS': '81592'}),
            'capture pin': container(env={'DS41_DECISION_ROW_BLOCKS': '80927'}),
            'determinism': container(env={'B12X_DYNAMIC_DETERMINISTIC_OUTPUT': '0'}),
            'turbo': container(env={'B12X_DENSE_SPLITK_TURBO': '1'}),
            'prefetch': container(env={'VLLM_DS41_L2_PREFETCH': '1'}),
            'image': container(Image='sha256:' + 'f' * 64),
        }
        for name, (c, image) in cases.items():
            problems, _ = probe.check_env('dusty', c, image, REFERENCE, {'kit'}, 7936)
            self.assertTrue(problems, name)
        c, image = container()
        image['Config']['Labels']['local-inference.ds41.diagnostic.kind'] = 'decision-row-capture'
        self.assertTrue(probe.check_env('dusty', c, image, REFERENCE, {'kit'}, 7936)[0])

    def test_regime_snapshot_is_pinned_and_decodes_at_8192_rows(self):
        path = HERE / probe.REGIME_SNAPSHOT
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), probe.REGIME_SNAPSHOT_SHA256)
        regime = json.loads(path.read_text())
        selection = json.loads((path.parent / 'dusty-selection-pre.json').read_text())
        self.assertEqual(exact_regime(selection, probe.BLOCKS), regime['dusty'])
        self.assertEqual(EXTEND_ROWS, probe.CAPACITY)         # decoder keys use the prepared capacity, not the threshold
        selected, scope = sens.comparison_reference({'regime': regime['dusty'], 'x': 1}, path)
        self.assertEqual(selected['regime'], regime['dusty'])
        self.assertTrue(scope['regime_equals_b2'])


class Prompts(unittest.TestCase):
    def test_plan_order_and_sources(self):
        plan = probe.plan()
        self.assertEqual([p['label'] for p in plan], ['control-1', 'frozen-516096', 'fresh1-524288', 'frozen-532480',
                                                      'frozen-524287', 'control-2'])
        self.assertEqual([p['tokens'] for p in plan], [524288, 516096, 524288, 532480, 524287, 524288])
        self.assertEqual(plan[0]['source'], plan[-1]['source'])
        self.assertEqual(plan[2]['source'], ('receipts/needle-sensitivity-20260926T050535Z', 'fresh-1'))
        self.assertTrue(all(p['source'][0].startswith('receipts/needle-sensitivity-2026') for p in plan))

    def test_saved_prompts_load_and_verify_against_their_receipts(self):
        loaded = {p['label']: probe.load_prompt(HERE, p) for p in probe.plan()}
        self.assertEqual(loaded['control-1']['content_sha256'], PROMPT_SHA)
        self.assertEqual(loaded['control-1']['content'], loaded['control-2']['content'])
        self.assertEqual(loaded['fresh1-524288']['content'][17:49], '94e8bd2ae83f4340bdcc38c52b3ad98a')
        for label, item in loaded.items():
            self.assertEqual(hashlib.sha256(item['content'].encode()).hexdigest(), item['content_sha256'], label)
            self.assertEqual(item['historical']['prompt_tokens'], item['tokens'], label)
            self.assertIn('first_token_margin', item['historical'], label)
        self.assertFalse(loaded['frozen-516096']['historical']['correct'])
        self.assertTrue(loaded['frozen-524287']['historical']['correct'])
        self.assertEqual(loaded['control-1']['historical']['boot_receipt'], 'receipts/needle-sensitivity-20260926T053026Z')

    def test_tampered_prompt_fails_closed(self):
        item = dict(probe.plan()[0])
        with self.assertRaises(RuntimeError):
            probe.verify_prompt(item, 'not the saved content', {'content_sha256': 'abc', 'rounds': [{'prompt_tokens': 524288}]})


class Contrast(unittest.TestCase):
    def test_contrast_reports_margin_delta_and_answer_change(self):
        historical = {'first_token_margin': 0.75, 'late_code_margin': 12.875, 'answer_shape': 'early_code+late_code',
                      'correct': True, 'response_signature': 'h', 'boot_receipt': 'receipts/x'}
        row = {'margins': {'first_token': {'margin': 5.25}, 'late_code_after_separator': {'margin': 12.0}},
               'answer_shape': 'early_code+late_code', 'correct': True, 'response_signature': 'n'}
        c = probe.contrast(row, historical)
        self.assertAlmostEqual(c['first_token_margin_delta'], 4.5)
        self.assertAlmostEqual(c['late_code_margin_delta'], -0.875)
        self.assertFalse(c['answer_changed'])
        self.assertFalse(c['signature_equal'])
        row['answer_shape'], row['correct'] = 'identity+late_code', False
        self.assertTrue(probe.contrast(row, historical)['answer_changed'])
        row['margins']['first_token']['margin'] = None
        self.assertIsNone(probe.contrast(row, historical)['first_token_margin_delta'])

    def test_summary_requires_equal_controls_but_not_correct_answers(self):
        def row(label, sig, correct):
            return {'label': label, 'response_signature': sig, 'correct': correct, 'answer_shape': 'x', 'content': 'c',
                    'prompt_tokens': 1, 'target_tokens': 1, 'cached_tokens': 0, 'elapsed_s': 1.0, 'hex': 'h',
                    'margins': {'first_token': {'margin': 1.0}, 'late_code_after_separator': None},
                    'contrast': {'first_token_margin_delta': 0.0}}
        rows = [row('control-1', 's', False), row('frozen-516096', 't', True), row('control-2', 's', False)]
        summary = probe.summarize(rows, 7936)
        self.assertTrue(summary['controls_identical'])
        self.assertEqual(summary['threshold'], 7936)
        self.assertIn('not a causal precision claim', summary['scope'])
        self.assertIn('not an acceptance gate', summary['scope'])
        rows[2]['response_signature'] = 'u'
        self.assertFalse(probe.summarize(rows, 7936)['controls_identical'])


class Scope(unittest.TestCase):
    def test_probe_never_changes_node_state_or_launchers(self):
        source = Path(probe.__file__).read_text()
        for forbidden in ('podman start', 'podman stop', 'podman run', 'podman rm', 'podman build', 'rm -rf',
                          'import launch_contract', 'start_moe_repaired', 'run_node', 'vllm serve', 'uuid4().hex[:'):
            self.assertNotIn(forbidden, source, forbidden)
        tree = ast.parse(source)
        calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertNotIn('unlink', calls)
        self.assertNotIn('rmtree', calls)
        self.assertIn("'--threshold'", source)
        self.assertIn('parse_non_default_args', source)
        self.assertNotIn('literal_eval', source)
        self.assertNotIn('eval(', source)


if __name__ == '__main__':
    unittest.main()
