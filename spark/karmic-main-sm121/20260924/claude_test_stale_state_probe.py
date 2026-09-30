"""Stdlib tests for the pure logic of claude_stale_state_probe.py (no torch, no GPU).

The torch paths (view, gather, transplant, poison) run only in the image; the
pinned-source tests below check that the replicated slot mapping and the hook
signatures still match vLLM 1794dcf1 through `git show` on ~/git/vllm.
"""
import importlib.util
import os
import re
import subprocess
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('claude_stale_state_probe', HERE / 'claude_stale_state_probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

VLLM_PIN = '1794dcf18454900263e0c66711af8ea4a1283ac1'
VLLM = Path(os.environ.get('VLLM_CHECKOUT', Path.home() / 'git' / 'vllm'))
SHA = 'a' * 64


def pinned(path):
    return subprocess.check_output(['git', '-C', str(VLLM), 'show', f'{VLLM_PIN}:{path}'], text=True)


class SlotMapping(unittest.TestCase):
    def test_paged_ratio_one(self):
        row = [7, 9, 0, 0]
        self.assertEqual(probe.position_slot(row, 0, page=128, ratio=1, circular=False), 7 * 128)
        self.assertEqual(probe.position_slot(row, 130, page=128, ratio=1, circular=False), 9 * 128 + 2)
        self.assertIsNone(probe.position_slot(row, 256, page=128, ratio=1, circular=False))   # null block
        self.assertIsNone(probe.position_slot(row, 600, page=128, ratio=1, circular=False))   # past width

    def test_ratio_two_shares_slots(self):
        row = [5]
        a = probe.position_slot(row, 10, page=64, ratio=2, circular=False)
        b = probe.position_slot(row, 11, page=64, ratio=2, circular=False)
        self.assertEqual(a, b)
        self.assertEqual(a, 5 * 64 + 5)

    def test_circular_ring_uses_first_column(self):
        row = [3, 99]
        self.assertEqual(probe.position_slot(row, 384, page=16, ratio=1, circular=True), 3 * 16 + 0)
        self.assertEqual(probe.position_slot(row, 385, page=16, ratio=1, circular=True), 3 * 16 + 1)

    def test_request_slots_tail(self):
        per_position, tail = probe.request_slots([4, 6, 8, 2], 393, page=128, ratio=1, circular=False)
        self.assertEqual(len(per_position), 393)
        self.assertEqual(per_position[392], 2 * 128 + 8)
        self.assertEqual(tail, [2 * 128 + o for o in range(9, 128)])
        _, ring_tail = probe.request_slots([3], 393, page=16, ratio=1, circular=True)
        self.assertEqual(ring_tail, [])

    def test_unbacked_prefix_is_none(self):
        per_position, _ = probe.request_slots([0, 0, 11, 12], 393, page=128, ratio=1, circular=False)
        self.assertTrue(all(s is None for s in per_position[:256]))
        self.assertEqual(per_position[256], 11 * 128)


class Control(unittest.TestCase):
    def base(self, actions=None, **extra):
        raw = {'prompt_token_sha256': SHA, 'scheduled_tokens': 8,
               'actions': actions or [{'run': 'orig-first', 'mode': 'observe'}]}
        raw.update(extra)
        return raw

    def test_valid_sequence_and_defaults(self):
        control = probe.parse_control(self.base([
            {'run': 'orig-first', 'mode': 'observe'},
            {'run': 'orig-repeat', 'mode': 'observe'},
            {'run': 'tx-1', 'mode': 'transplant', 'source': 'orig-repeat',
             'select': [{'layer_regex': r'layers\.(2\d|3\d)\.', 'positions': [0, 257]},
                        {'layer_regex': r'layers\.20\.', 'tail': [0, 16]}]},
            {'run': 'px-1', 'mode': 'poison', 'byte': 255,
             'select': [{'layer_regex': r'layers\.20\.', 'positions': [0, 128]}]}]))
        self.assertEqual(len(control['actions']), 4)
        self.assertEqual(control['coverage'], probe.DEFAULT_COVERAGE)
        self.assertEqual(control['allow_omitted'], '')

    def test_rejections(self):
        sel = [{'layer_regex': 'x', 'positions': [0, 1]}]
        bad = [
            {'prompt_token_sha256': 'x', 'actions': [{'run': 'a', 'mode': 'observe'}]},
            self.base([{'run': 'a', 'mode': 'observe'}, {'run': 'a', 'mode': 'observe'}]),
            self.base([{'run': 'a b', 'mode': 'observe'}]),
            self.base([{'run': 'a', 'mode': 'erase'}]),
            self.base([{'run': 'a', 'mode': 'observe', 'select': sel}]),          # observe stays narrow
            self.base([{'run': 'a', 'mode': 'poison', 'byte': 300, 'select': sel}]),
            self.base([{'run': 'a', 'mode': 'transplant', 'select': sel}]),
            self.base([{'run': 'a', 'mode': 'transplant', 'source': 'a', 'select': sel}]),
            self.base([{'run': 'a', 'mode': 'poison', 'byte': 1, 'source': 'b', 'select': sel}]),
            self.base([{'run': 'a', 'mode': 'poison', 'byte': 1,
                        'select': [{'layer_regex': 'x', 'positions': [5, 5]}]}]),
            self.base([{'run': 'a', 'mode': 'poison', 'byte': 1,
                        'select': [{'layer_regex': 'x', 'positions': [0, 1], 'tail': [0, 1]}]}]),
            self.base([{'run': 'a', 'mode': 'poison', 'byte': 1, 'select': [{'layer_regex': 'x'}]}]),
            self.base([{'run': 'a', 'mode': 'poison', 'byte': 1, 'select': []}]),
            self.base(coverage={}),
            self.base(coverage={'swa': {'suffix': '.swa_cache', 'layers': []}}),
            self.base(coverage={'swa': {'suffix': '.swa_cache', 'layers': [[3, 3]]}}),
            self.base(allow_omitted='('),
            dict(self.base(), extra=1),
        ]
        for raw in bad:
            with self.assertRaises((ValueError, KeyError, TypeError, re.error), msg=raw):
                probe.parse_control(raw)

    def test_matches(self):
        control = probe.parse_control(self.base())
        self.assertTrue(probe.matches(control, num_reqs=1, scheduled=8, prompt_sha=SHA))
        self.assertFalse(probe.matches(control, num_reqs=2, scheduled=8, prompt_sha=SHA))
        self.assertFalse(probe.matches(control, num_reqs=1, scheduled=385, prompt_sha=SHA))
        self.assertFalse(probe.matches(control, num_reqs=1, scheduled=8, prompt_sha='b' * 64))

    def test_token_sha_is_order_and_value_exact(self):
        self.assertNotEqual(probe.token_sha256([1, 2]), probe.token_sha256([2, 1]))
        self.assertEqual(probe.token_sha256([1, 2]), probe.token_sha256((1, 2)))


def ds41_names():
    names = [f'model.layers.{i}.attn.swa_cache' for i in range(40)]
    names += [f'drafter.model.layers.{i}.attn.swa_cache' for i in range(40, 43)]
    names += [f'model.layers.{i}.attn.compressor.state_cache' for i in range(2, 20)]
    names += [f'model.layers.{i}.attn' for i in (2, 8, 14, 20)]
    names += [f'model.layers.{i}.attn.indexer.k_cache' for i in (2, 8, 14, 20)]
    return names


class Classification(unittest.TestCase):
    def groups(self, statuses):
        return [{'index': i, 'spec_name': 'MLAAttentionSpec',
                 'layers': [{'name': n, 'status': st} for n, st in layer_list]}
                for i, layer_list in enumerate(statuses)]

    def test_all_captured(self):
        captured, omitted = probe.classify(self.groups([[('a', 'ok')], [('b', 'ok')]]), 2)
        self.assertEqual((captured, omitted), ([(0, 'a'), (1, 'b')], []))

    def test_group_table_count_mismatch_fails(self):
        with self.assertRaisesRegex(RuntimeError, '2 KV-cache groups but 1 block tables'):
            probe.classify(self.groups([[('a', 'ok')], [('b', 'ok')]]), 1)

    def test_omission_fails_closed_and_is_recorded_when_allowed(self):
        groups = self.groups([[('model.layers.3.attn.swa_cache', 'ok'), ('drafter.x', 'no_cache')]])
        with self.assertRaisesRegex(RuntimeError, r'drafter\.x \(no_cache\)'):
            probe.classify(groups, 1)
        captured, omitted = probe.classify(groups, 1, allow_omitted=r'^drafter\.')
        self.assertEqual(omitted, [(0, 'drafter.x', 'no_cache')])
        self.assertEqual(captured, [(0, 'model.layers.3.attn.swa_cache')])

    def test_empty_group_and_nothing_captured_fail(self):
        with self.assertRaisesRegex(RuntimeError, 'has no layers'):
            probe.classify(self.groups([[('a', 'ok')], []]), 2)
        with self.assertRaisesRegex(RuntimeError, 'no KV-cache layer captured'):
            probe.classify(self.groups([[('a', 'empty_cache')]]), 1, allow_omitted='a')

    def test_coverage_defaults(self):
        counts = probe.check_coverage(ds41_names(), probe.DEFAULT_COVERAGE)
        self.assertEqual(counts, {'swa': 43, 'ring': 18, 'main': 4, 'indexer': 4})
        for drop in ('drafter.model.layers.41.attn.swa_cache', 'model.layers.7.attn.compressor.state_cache',
                     'model.layers.20.attn', 'model.layers.14.attn.indexer.k_cache', 'model.layers.0.attn.swa_cache'):
            with self.assertRaisesRegex(RuntimeError, 'required coverage missing', msg=drop):
                probe.check_coverage([n for n in ds41_names() if n != drop], probe.DEFAULT_COVERAGE)

    def test_record_layout(self):
        self.assertEqual(probe.record_layout(spec_name='SlidingWindowMLASpec', page=64, ratio=1,
                                             page_bytes=64 * 528, block_bytes=64 * 528 + 512), 528)
        self.assertEqual(probe.record_layout(spec_name='CircularBufferSpec', page=16, ratio=1,
                                             page_bytes=16 * 4096, block_bytes=16 * 4096), 4096)
        for kwargs in (dict(spec_name='MambaSpec', page=1, ratio=1, page_bytes=8, block_bytes=8),
                       dict(spec_name='MLAAttentionSpec', page=64, ratio=2, page_bytes=1000, block_bytes=2000),
                       dict(spec_name='MLAAttentionSpec', page=64, ratio=2, page_bytes=64 * 288, block_bytes=100),
                       dict(spec_name='MLAAttentionSpec', page=0, ratio=2, page_bytes=0, block_bytes=0)):
            with self.assertRaises(ValueError, msg=kwargs):
                probe.record_layout(**kwargs)


class Writes(unittest.TestCase):
    def test_positions_and_tail_domains(self):
        per_position, tail = probe.request_slots([4, 6, 8, 2], 393, page=128, ratio=1, circular=False)
        plan = probe.plan_writes([{'layer_regex': r'layers\.20\.', 'positions': [390, 400]},
                                  {'layer_regex': r'layers\.20\.', 'tail': [0, 3]}],
                                 'model.layers.20.attn.swa_cache', per_position, tail)
        self.assertEqual([slot for slot, _ in plan],
                         [2 * 128 + 6, 2 * 128 + 7, 2 * 128 + 8, 2 * 128 + 9, 2 * 128 + 10, 2 * 128 + 11])
        self.assertEqual(plan[-1][1], [('tail', 2)])
        self.assertEqual(probe.plan_writes([{'layer_regex': 'nomatch', 'tail': [0, 3]}], 'x', per_position, tail), [])

    def test_tail_beyond_capture_and_unbacked_positions_write_nothing(self):
        per_position, tail = probe.request_slots([0, 0, 11], 300, page=128, ratio=1, circular=False)
        self.assertEqual(probe.plan_writes([{'layer_regex': '.', 'positions': [0, 256]}], 'l', per_position, tail), [])
        self.assertEqual(len(tail), 128 - (299 - 256) - 1)
        self.assertEqual(len(probe.plan_writes([{'layer_regex': '.', 'tail': [0, 10_000]}], 'l', per_position, tail)),
                         len(tail))

    def test_shared_and_overlapping_slots_are_written_once(self):
        per_position, tail = probe.request_slots([5], 64, page=64, ratio=2, circular=False)
        plan = probe.plan_writes([{'layer_regex': '.', 'positions': [10, 14]},
                                  {'layer_regex': '.', 'positions': [12, 14]}], 'l', per_position, tail)
        self.assertEqual([slot for slot, _ in plan], [5 * 64 + 5, 5 * 64 + 6])
        self.assertEqual(plan[0][1], [('positions', 10), ('positions', 11)])
        self.assertEqual(plan[1][1], [('positions', 12), ('positions', 13)])
        slots = [slot for slot, _ in plan]
        self.assertEqual(len(slots), len(set(slots)))

    def test_layout_key_and_source_rows(self):
        per_position, tail = probe.request_slots([0, 3], 200, page=128, ratio=1, circular=False)
        layer = {'spec': 'SlidingWindowMLASpec', 'page': 128, 'ratio': 1, 'circular': False,
                 'record_bytes': 528, 'slots': per_position, 'tail_slots': tail}
        self.assertEqual(probe.source_row(layer, 'positions', 128), ('values', 0))
        self.assertEqual(probe.source_row(layer, 'positions', 199), ('values', 71))
        self.assertEqual(probe.source_row(layer, 'tail', 0), ('tail', 0))
        with self.assertRaises(IndexError):
            probe.source_row(layer, 'positions', 5)
        with self.assertRaises(IndexError):
            probe.source_row(layer, 'tail', len(tail))
        other = dict(layer, slots=probe.request_slots([9, 3], 200, page=128, ratio=1, circular=False)[0])
        self.assertNotEqual(probe.layout_key(layer), probe.layout_key(other))       # backed pattern differs
        self.assertEqual(probe.layout_key(layer),
                         probe.layout_key(dict(layer, slots=probe.request_slots([0, 7], 200, page=128, ratio=1,
                                                                                circular=False)[0])))
        self.assertNotEqual(probe.layout_key(layer), probe.layout_key(dict(layer, record_bytes=288)))


class Selection(unittest.TestCase):
    def test_ranges(self):
        self.assertEqual(probe.ranges([0, 1, 2, 5, 6, 9]), [[0, 3], [5, 7], [9, 10]])
        self.assertEqual(probe.ranges([]), [])

    def test_bisect_both_domains(self):
        halves = probe.bisect_selects([{'layer_regex': 'a', 'positions': [0, 257]},
                                       {'layer_regex': 'b', 'tail': [0, 16]},
                                       {'layer_regex': 'c', 'positions': [7, 8]}])
        self.assertEqual(halves[0], [{'layer_regex': 'a', 'positions': [0, 128]},
                                     {'layer_regex': 'b', 'tail': [0, 8]},
                                     {'layer_regex': 'c', 'positions': [7, 8]}])
        self.assertEqual(halves[1], [{'layer_regex': 'a', 'positions': [128, 257]},
                                     {'layer_regex': 'b', 'tail': [8, 16]}])


@unittest.skipUnless((VLLM / '.git').exists(), 'no ~/git/vllm checkout')
class PinnedSource(unittest.TestCase):
    def test_slot_mapping_replicates_tokens_kernel(self):
        src = pinned('vllm/models/deepseek_v4_1/sparse_mla.py')
        for fragment in ('logical = pos // RATIO',
                         'page_col = tl.full((B,), 0, tl.int64) if CIRCULAR else logical // PAGE',
                         'valid = valid & (page_col < table_width)',
                         'valid = valid & (block > 0)',
                         'slot = block * PAGE + logical % PAGE',
                         'self.page = int(kv_cache_spec.num_states)',
                         'self.ratio = int(kv_cache_spec.tokens_per_state)',
                         'return (num_blocks, block_size, head_size)'):
            self.assertIn(fragment, src)

    def test_hook_points(self):
        state = pinned('vllm/models/deepseek_v4_1/nvidia/model_state.py')
        self.assertIn('    def prepare_attn(\n        self,\n        input_batch: InputBatch,\n        cudagraph_mode: CUDAGraphMode,\n'
                      '        block_tables: tuple[torch.Tensor, ...],\n        slot_mappings: torch.Tensor,\n'
                      '        attn_groups: list[list[AttentionGroup]],\n        kv_cache_config: KVCacheConfig,\n'
                      '        for_capture: bool = False,\n    ) -> dict[str, Any]:', state)
        self.assertIn('    def prepare_inputs(\n        self, input_batch: InputBatch, req_states: RequestState\n'
                      '    ) -> dict[str, torch.Tensor | None]:', state)
        runner = pinned('vllm/v1/worker/gpu/model_runner.py')
        attn, inputs = runner.index('attn_metadata = self.model_state.prepare_attn('), runner.index(
            '**self.model_state.prepare_inputs(input_batch, self.req_states),')
        self.assertLess(attn, inputs)            # runtime order: metadata, then inputs, then forward
        states = pinned('vllm/v1/worker/gpu/states.py')
        self.assertIn('self.prompt_len = UvaBackedTensor(self.max_num_reqs, dtype=torch.int32)', states)
        self.assertIn('self.all_token_ids = StagedWriteTensor(', states)

    def test_patch_applies_to_pinned_blob(self):
        patch = (HERE / 'claude-stale-state-model_state.patch').read_text()
        self.assertIn('+        from .claude_stale_state_probe import install as _stale_probe_install  # DIAGNOSTIC ONLY', patch)
        self.assertEqual(sum(1 for line in patch.splitlines() if line.startswith('+') and not line.startswith('+++')), 2)
        self.assertEqual(sum(1 for line in patch.splitlines() if line.startswith('-') and not line.startswith('---')), 0)
        state = pinned('vllm/models/deepseek_v4_1/nvidia/model_state.py')
        self.assertEqual(state.count('            if isinstance(module, DeepseekV4Model) and module.disk_engram\n        )\n'), 1)

    def test_default_coverage_names_match_pinned_prefixes(self):
        attention = pinned('vllm/models/deepseek_v4_1/attention.py')
        compressor = pinned('vllm/models/deepseek_v4_1/compressor.py')
        model = pinned('vllm/models/deepseek_v4_1/nvidia/model.py')
        dspark = pinned('vllm/models/deepseek_v4_1/nvidia/dspark.py')
        self.assertIn('f"{prefix}.swa_cache"', attention)
        self.assertIn('f"{prefix}.indexer.k_cache"', attention)
        self.assertIn('prefix=f"{prefix}.compressor"', attention)
        self.assertIn('CompressorStateCache(vllm_config, f"{prefix}.state_cache")', compressor)
        self.assertIn('prefix=f"{prefix}.attn"', model)
        self.assertIn('prefix=maybe_prefix(prefix, f"layers.{self.num_hidden_layers + i}")', dspark)
        self.assertIn('        if not self.is_kv_source:\n            return None\n        return MLAAttentionSpec(', attention)
        self.assertEqual(probe.DEFAULT_COVERAGE['swa']['suffix'], '.swa_cache')
        self.assertEqual(probe.DEFAULT_COVERAGE['ring']['suffix'], '.compressor.state_cache')

    def test_record_sizes_and_padding(self):
        attention = pinned('vllm/models/deepseek_v4_1/attention.py')
        self.assertIn('state_content_bytes=528 if self.kind == "swa" else 68,', attention)
        self.assertIn('state_content_bytes=288,', attention)
        self.assertIn('# Keep allocator page stride: padding belongs to the allocator, not the ABI.', attention)
        spec = pinned('vllm/v1/kv_cache_interface.py')
        self.assertIn('return self.num_heads * self.num_states * self.state_content_size_bytes', spec)

    def test_lifo_reuse_of_unhashed_blocks(self):
        pool = pinned('vllm/v1/core/block_pool.py')
        self.assertIn('# LIFO reuse of non-cached blocks for better GPU locality.', pool)
        self.assertIn('self.free_block_queue.prepend_n(blocks_to_evict_first)', pool)


if __name__ == '__main__':
    unittest.main()
