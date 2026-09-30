"""CPU tests for claude_replay_gate_prefill.py (no GPU, no B12X import).
  .venv-snapshot-cpu/bin/python -m unittest claude_test_replay_gate_prefill
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('claude_replay_gate_under_test', HERE / 'claude_replay_gate_prefill.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)
PINNED = HERE / 'claude-pinned-bf16_gemv-_prefill.py'
CACHES = sorted((HERE / 'receipts').glob('routed-isolation-toby-*/tuning-before.json'))


class Keys(unittest.TestCase):
    @unittest.skipUnless(CACHES, 'no retained toby selection cache')
    def test_keys_resolve_production_plans_in_a_retained_cache(self):
        records = json.loads(CACHES[-1].read_text())['records']
        for rows in (128, 256, 8192):
            self.assertEqual(records[replay.choice_key(rows)]['config']['backend'], 'prefill', rows)
        self.assertEqual(replay.production_plan_rows(records, 256), 256)
        self.assertEqual(replay.production_plan_rows(records, 254), 8192)
        self.assertNotIn(replay.choice_key(254), records)

    def test_query_matches_the_production_layout(self):
        q = replay.query_dict(256)
        self.assertEqual((q['source_dtype'], q['weight_dtype'], q['output_dtype'], q['in_features'],
                          q['out_features'], q['bias_dtype']), ('bfloat16', 'bfloat16', 'float32', 5120, 384, None))


class FenceSource(unittest.TestCase):
    def setUp(self):
        self.pinned = PINNED.read_text()
        self.assertEqual(hashlib.sha256(self.pinned.encode()).hexdigest(), replay.PINNED_PREFILL_SHA256)
        self.lines = self.pinned.splitlines(keepends=True)
        self.index = next(i for i, l in enumerate(self.lines) if l.strip() == replay.RELEASE)
        self.indent = self.lines[self.index][:len(self.lines[self.index]) - len(self.lines[self.index].lstrip())]

    def fenced(self, at=None, indent=None, line=None):
        at = self.index if at is None else at
        lines = list(self.lines)
        lines.insert(at, (self.indent if indent is None else indent) + (line or replay.FENCE) + '\n')
        return ''.join(lines)

    def test_exactly_one_fence_before_the_release_is_accepted(self):
        self.assertTrue(replay.check_fence_source(self.fenced(), self.pinned))

    def test_everything_else_is_refused(self):
        self.assertFalse(replay.check_fence_source(self.pinned, self.pinned))
        self.assertFalse(replay.check_fence_source(self.fenced(at=self.index - 3), self.pinned))
        self.assertFalse(replay.check_fence_source(self.fenced(indent=self.indent + '    '), self.pinned))
        self.assertFalse(replay.check_fence_source(self.fenced(line='cute.arch.fence_proxy("async.shared")'),
                                                   self.pinned))
        extra = self.fenced().replace('num_stages = 2', 'num_stages = 3', 1)
        self.assertFalse(replay.check_fence_source(extra, self.pinned))
        double = self.fenced()
        double_lines = double.splitlines(keepends=True)
        double_lines.insert(self.index, self.indent + replay.FENCE + '\n')
        self.assertFalse(replay.check_fence_source(''.join(double_lines), self.pinned))


class Helpers(unittest.TestCase):
    def test_overlap_is_containment_in_a_measured_side_interval(self):
        self.assertTrue(replay.overlapped((0.2, 0.5), (0.1, 0.6)))
        self.assertFalse(replay.overlapped((0.2, 0.7), (0.1, 0.6)))
        self.assertFalse(replay.overlapped((0.05, 0.3), (0.1, 0.6)))

    def test_reservoir_is_bounded(self):
        r = replay.Reservoir(size=16)
        for i in range(10000):
            r.add(float(i))
        self.assertEqual(len(r.values), 16)
        s = r.summary()
        self.assertEqual((s['count'], s['min'] >= 0, s['max'] <= 9999), (10000, True, True))

    def test_tile_mismatches_and_confinement(self):
        ref = torch.zeros(256, 384)
        out = ref.clone()
        out[192:195, 320:384] = 1.0
        tiles = replay.tile_mismatches(ref, out)
        self.assertEqual([(t['m_tile'], t['n_tile'], t['elements'], t['rows']) for t in tiles],
                         [(3, 5, 192, [192, 193, 194])])
        self.assertTrue(replay.confined(tiles, {192, 193, 194}))
        self.assertFalse(replay.confined(tiles, {192, 193}))
        nan = ref.clone()
        nan[0, 0] = float('nan')
        self.assertEqual(replay.tile_mismatches(ref, nan)[0]['elements'], 1)

    def test_gate_weight_is_read_from_a_local_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            weight = torch.arange(384 * 5120, dtype=torch.float32).to(torch.bfloat16).view(384, 5120)
            data = bytes(weight.contiguous().view(-1).view(torch.uint8).tolist())
            other = b'\x00' * 64
            header = {'layers.8.ffn.gate.bias': {'dtype': 'F32', 'shape': [16], 'data_offsets': [0, 64]},
                      'layers.8.ffn.gate.weight': {'dtype': 'BF16', 'shape': [384, 5120],
                                                   'data_offsets': [64, 64 + len(data)]}}
            raw = json.dumps(header).encode()
            (tmp / 'model-00011-of-00048.safetensors').write_bytes(struct.pack('<Q', len(raw)) + raw + other + data)
            (tmp / 'model.safetensors.index.json').write_text(json.dumps(
                {'weight_map': {'layers.8.ffn.gate.weight': 'model-00011-of-00048.safetensors'}}))
            got, shard = replay.read_gate_weight(tmp, 8)
            self.assertEqual((got, shard), (data, 'model-00011-of-00048.safetensors'))


class Modes(unittest.TestCase):
    def test_mode_rules(self):
        base = ['--capture', 'c.pt', '--weight-bin', 'w.bin', '--weight-sha256', '0' * 64, '--out', 'o.json']
        a = replay.parse_args(base + ['--tuning-receipt', 'r.json'])
        self.assertEqual((a.fenced, a.rows, a.arms), (False, '256,254', 'idle,overlap'))
        for bad in (base, base + ['--fenced'], base + ['--fenced', '--compile-cache-dir', 'x', '--tuning-receipt', 'r'],
                    base + ['--tuning-receipt', 'r', '--compile-cache-dir', 'x'],
                    base + ['--tuning-receipt', 'r', '--repeats', '1']):
            with self.assertRaises(SystemExit):
                replay.parse_args(bad)


class Source(unittest.TestCase):
    tree = ast.parse((HERE / 'claude_replay_gate_prefill.py').read_text())

    def main_text(self):
        return ast.unparse(next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main'))

    def test_repeat_loop_is_bounded_and_measures_overlap(self):
        text = self.main_text()
        loop = text[text.index('for repeat in range(a.repeats)'):text.index('arm_report = {')]
        self.assertIn('if len(kept) < a.keep:', loop)
        self.assertEqual(loop.count('.append('), 1)
        self.assertIn("out.fill_(float('nan'))", loop)
        self.assertLess(loop.index("out.fill_(float('nan'))"), loop.index('release.record()'))
        self.assertIn('side.wait_event(release)', loop)
        self.assertIn('overlapped(gate, (release.elapsed_time(s0), release.elapsed_time(s1)))', loop)

    def test_side_operands_are_waited_before_use(self):
        text = self.main_text()
        self.assertLess(text.index('side_b = torch.randn'), text.index('side.wait_stream(torch.cuda.current_stream(device))'))
        self.assertLess(text.index('side.wait_stream(torch.cuda.current_stream(device))'),
                        text.index('for rows in rows_list'))

    def test_modes_never_mix_cached_and_changed_source(self):
        text = self.main_text()
        self.assertIn("require(prefill_sha == PINNED_PREFILL_SHA256, 'installed _prefill.py differs from a7d7d29b')", text)
        self.assertIn("require(selection.source == ('override' if a.fenced else 'cached')", text)
        self.assertIn("os.environ['B12X_COMPILE_CACHE_DIR'] = str(a.compile_cache_dir)", text)
        self.assertLess(text.index("os.environ['B12X_COMPILE_CACHE_DIR']"), text.index('from b12x.gemm import bf16_gemv'))
        self.assertIn('cache_only=True', text)
        loads = [c for c in ast.walk(self.tree) if isinstance(c, ast.Call) and ast.unparse(c.func) == 'torch.load']
        self.assertTrue(all({k.arg: ast.unparse(k.value) for k in c.keywords}.get('weights_only') == 'True'
                            for c in loads))


if __name__ == '__main__':
    unittest.main()
