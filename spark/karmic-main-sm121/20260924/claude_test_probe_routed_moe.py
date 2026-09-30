"""CPU checks for claude_probe_routed_moe.py (no GPU, no B12X import).
  .venv-snapshot-cpu/bin/python -m unittest claude_test_probe_routed_moe
Pinned-source checks read B12X a7d7d29b and vLLM 1794dcf1 from the existing
~/git checkouts (read-only) and skip, saying so, without them.
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'claude_probe_routed_moe.py'
spec = importlib.util.spec_from_file_location('claude_probe_under_test', SOURCE)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
B12X, VLLM = Path.home() / 'git/b12x', Path.home() / 'git/vllm'


def pinned(repo, commit, path):
    return subprocess.run(['git', '-C', str(repo), 'show', f'{commit}:{path}'],
                          check=True, capture_output=True, text=True).stdout


def b12x_digest(value):          # b12x/preparation/_cache.py digest at a7d7d29b
    text = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


class Layout(unittest.TestCase):
    def test_pinned_activation_clamp_is_carried_into_weight_plan(self):
        args = probe.parse_args(['--tuning-receipt', 'unused'])
        self.assertEqual(args.swiglu_limit, 10.0)
        tree = ast.parse(SOURCE.read_text())
        factory = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                       and n.name == 'synthetic_experts')
        calls = [n for n in ast.walk(factory) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == 'ActivationSpec']
        self.assertEqual(len(calls), 1)
        self.assertEqual(ast.unparse(next(k.value for k in calls[0].keywords
                                         if k.arg == 'swiglu_limit')), 'a.swiglu_limit')

    def test_serving_lease_offsets(self):
        self.assertEqual(probe.arena_layout(254, 5120, 1001),
                         {'output_bytes': 2600960, 'workspace2_offset': 2600960,
                          'workspace2_bytes': 1002, 'total': 2601962})
        self.assertEqual(probe.arena_layout(3, 5120, 8)['workspace2_offset'], 30720)
        self.assertEqual(probe.arena_layout(1, 8, 0)['workspace2_offset'], 256)

    def test_prefill_rows_use_the_capacity_variant(self):
        counts = probe.parse_counts(probe.DEFAULT_COUNTS, 8192)
        self.assertEqual(max(counts), 8192)
        for rows in (33, 128, 254, 255, 256, 8191):
            self.assertTrue(probe.resolves_to_capacity(rows, counts), rows)
        for rows in (1, 8, 32, 8192, 8193, 0):
            self.assertFalse(probe.resolves_to_capacity(rows, counts), rows)


class SelectionFile(unittest.TestCase):
    def write(self, tmp, payload, name=None):
        path = Path(tmp) / ((name or b12x_digest(payload['identity'])) + '.json')
        path.write_text(json.dumps(payload))
        return path

    def test_name_must_be_identity_digest(self):
        payload = {'identity': {'schema_version': 6, 'namespace': {'tensor_parallel': 4}}, 'records': {}}
        with tempfile.TemporaryDirectory() as tmp:
            good = self.write(tmp, payload)
            self.assertEqual(probe.check_selection_file(good, b12x_digest), payload)
            with self.assertRaises(SystemExit) as refused:
                probe.check_selection_file(self.write(tmp, payload, name='0' * 64), b12x_digest)
            self.assertEqual(refused.exception.code, 2)
            with self.assertRaises(SystemExit):
                probe.check_selection_file(self.write(tmp, {'identity': {}, 'records': {}, 'x': 1}, 'y'),
                                           b12x_digest)


class Compare(unittest.TestCase):
    def test_identical_repeats_pass(self):
        c = probe.Comparator()
        base = torch.randn(6, 4).to(torch.bfloat16)
        for repeat in range(50):
            self.assertTrue(c.update(repeat, base.clone()))
        self.assertEqual((c.mismatches, c.first_failure, c.summary()['changed_rows']), (0, None, []))

    def test_first_failure_retained_and_rows_accumulated(self):
        c = probe.Comparator()
        base = torch.randn(6, 4).to(torch.bfloat16)
        for repeat in range(20):
            out = base.clone()
            if repeat in (5, 9):
                out[repeat % 6, 1] += 1
            c.update(repeat, out)
        s = c.summary()
        self.assertEqual((s['mismatches'], s['first_failure_repeat'], s['changed_rows']), (2, 5, [3, 5]))
        self.assertEqual(float(c.failure_output[5, 1]), float(base[5, 1] + 1))

    def test_nonfinite_is_a_failure(self):
        c = probe.Comparator()
        base = torch.zeros(2, 2)
        c.update(0, base)
        bad = base.clone()
        bad[0, 0] = float('nan')
        self.assertFalse(c.update(1, bad))
        self.assertEqual((c.mismatches, c.nonfinite), (1, 1))

    def test_state_is_bounded(self):
        c = probe.Comparator()
        base = torch.randn(4, 4)
        for repeat in range(5000):
            c.update(repeat, base + (repeat % 7 == 3))
        tensors = [v for v in vars(c).values() if isinstance(v, torch.Tensor)]
        self.assertEqual(len(tensors), 3)                                  # reference, failure, row mask
        self.assertFalse([v for v in vars(c).values() if isinstance(v, (list, dict))])


class Source(unittest.TestCase):
    tree = ast.parse(SOURCE.read_text())

    def function(self, name):
        return next(n for n in ast.walk(self.tree) if isinstance(n, ast.FunctionDef) and n.name == name)

    def test_side_operands_are_waited_before_first_side_use(self):
        body = ast.unparse(self.function('make_side_load'))
        self.assertLess(body.rindex('torch.randn'), body.index('side.wait_stream(torch.cuda.current_stream(device))'))
        main = ast.unparse(self.function('main'))
        self.assertLess(main.index('make_side_load('), main.index('for repeat in range(a.repeats)'))
        loop = main[main.index('for repeat in range(a.repeats)'):]
        self.assertLess(loop.index('side.wait_stream'), loop.index('with torch.cuda.stream(side)'))
        self.assertLess(loop.index('with torch.cuda.stream(side)'), loop.index('fused_moe.run(binding=binding)'))

    def test_strict_frozen_session_and_no_escape_hatches(self):
        main = self.function('main')
        call = next(c for c in ast.walk(main) if isinstance(c, ast.Call)
                    and ast.unparse(c.func) == 'PreparationSession')
        kw = {k.arg: ast.unparse(k.value) for k in call.keywords}
        self.assertEqual((kw['autotune'], kw['cache_only'], kw['cache_dir']), ('False', 'True', 'private'))
        text = SOURCE.read_text()
        self.assertNotIn('--allow-', text)
        self.assertIn('require(all(s == "cached" for s in sources.values())', text)

    def test_repeat_loop_retains_nothing(self):
        main = ast.unparse(self.function('main'))
        loop = main[main.index('for repeat in range(a.repeats)'):main.index('result = {')]
        self.assertNotIn('.append(', loop)
        self.assertNotIn('.clone(', loop)

    def test_args(self):
        a = probe.parse_args(['--tuning-receipt', '/cache/x.json'])
        self.assertEqual((a.rows, a.repeats, a.ids_dtype), ('128,254,255,256', 1000, 'int32'))
        with self.assertRaises(SystemExit):
            probe.parse_args(['--tuning-receipt', 'x', '--repeats', '1'])


@unittest.skipUnless((B12X / '.git').exists() and (VLLM / '.git').exists(), 'no pinned checkouts')
class PinnedSemantics(unittest.TestCase):
    def test_b12x_session_and_variant_rules(self):
        session = pinned(B12X, 'a7d7d29b', 'b12x/preparation/session.py')
        self.assertIn('return not self.session.cache_only and (not self.autotune or self.session._stop.is_set())',
                      session)                       # autotune=False alone: warm-up defaults, no lookup
        self.assertIn('raise LookupError(f"no completed selection for {request.name}")', session)
        self.assertIn('raise LookupError(f"missing required compiled artifact for {obligation.request.name}")',
                      session)
        self.assertIn('root = _cute_compile_cache_dir() / "preparation"', session)
        prep = pinned(B12X, 'a7d7d29b', 'b12x/moe/fused_moe/_preparation.py')
        self.assertIn('return variants[tokens] if tokens in variants else variants[capacity]', prep)
        cache = pinned(B12X, 'a7d7d29b', 'b12x/preparation/_cache.py')
        self.assertIn('self.path = Path(root) / f"{digest(self.identity)}.json"', cache)

    def test_vllm_binding_contract(self):
        b12x = pinned(VLLM, '1794dcf1', 'vllm/model_executor/layers/fused_moe/b12x.py')
        for line in ('required_nbytes = sum(spec.nbytes for spec in plan.scratch_specs())',
                     'return (0,), (max(1, (required_nbytes + itemsize - 1) // itemsize),), (M, K)',
                     'scratch = workspace2.view(-1).view(torch.uint8)',
                     'counts = tuple(sorted({workload.max_tokens, *workload.fixed_token_counts}))',
                     '"tuning_route_pattern": TUNING_WORKLOAD_VERSION'):
            self.assertIn(line, b12x)
        router = pinned(VLLM, '1794dcf1', 'vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py')
        self.assertIn('torch.int32 if indices_type is None else indices_type', router)


if __name__ == '__main__':
    unittest.main()
