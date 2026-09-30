"""CPU-torch tests for the tensor paths of claude_stale_state_probe.py.

Intended to run inside the pinned image (its /opt/venv has torch), for example:
  podman run --rm --network=none -v $PWD:/diag:ro --entrypoint /opt/venv/bin/python <image> \
      -m unittest discover -s /diag -p 'claude_test_stale_state_probe_torch.py'
Skipped when torch is absent. Not run on the workstation (no borrowed venv).
`vllm.v1.kv_cache_interface` is replaced by a stub only when vllm is not importable.
"""
import importlib.util
import sys
import types
import unittest
import tempfile
from unittest import mock
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - workstation
    torch = None

HERE = Path(__file__).resolve().parent


def load_probe():
    spec = importlib.util.spec_from_file_location('claude_stale_state_probe', HERE / 'claude_stale_state_probe.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_spec_module():
    try:
        from vllm.v1.kv_cache_interface import CircularBufferSpec  # noqa: F401
        return None
    except Exception:
        stub = types.ModuleType('vllm.v1.kv_cache_interface')

        class CircularBufferSpec:  # noqa: D401 - stub
            pass

        stub.CircularBufferSpec = CircularBufferSpec
        class UniformTypeKVCacheSpecs:
            def __init__(self, block_size, kv_cache_specs):
                self.block_size = block_size
                self.kv_cache_specs = kv_cache_specs
        stub.UniformTypeKVCacheSpecs = UniformTypeKVCacheSpecs
        for name in ('vllm', 'vllm.v1'):
            sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules['vllm.v1.kv_cache_interface'] = stub
        return stub


def padded_cache(blocks, page, record, pad, dtype=None):
    """A (blocks, -1) cache with `pad` allocator bytes after each page payload."""
    raw = torch.arange(blocks * (page * record + pad), dtype=torch.int64).remainder(251).to(torch.uint8)
    cache = raw.view(blocks, page * record + pad)
    return cache if dtype is None else cache.view(dtype)


@unittest.skipIf(torch is None, 'torch not available; run inside the pinned image')
class TensorPaths(unittest.TestCase):
    def setUp(self):
        self.probe = load_probe()

    def layer(self, cache, row, page=4, record=8, ratio=1, circular=False, spec='MLAAttentionSpec'):
        return {'group': 0, 'spec': spec, 'page': page, 'ratio': ratio, 'circular': circular,
                'record_bytes': record, 'cache': cache, 'block_row': row}

    def test_payload_view_aliases_and_skips_padding(self):
        cache = padded_cache(5, 4, 8, pad=6)
        view = self.probe.payload_view(cache, 4, 8)
        self.assertEqual(tuple(view.shape), (5, 4, 8))
        self.assertEqual(view.data_ptr(), cache.data_ptr())
        view[2, 1] = 7
        self.assertTrue(bool((cache[2, 8:16] == 7).all()))
        self.assertFalse(bool((cache[2, 32:38] == 7).any()))           # padding untouched
        with self.assertRaises(ValueError):
            self.probe.payload_view(cache, 4, 16)

    def test_payload_view_of_float_ring(self):
        ring = torch.zeros(3, 16 * 1024, dtype=torch.float32)
        view = self.probe.payload_view(ring, 16, 4096)
        self.assertEqual(tuple(view.shape), (3, 16, 4096))
        view[1, 2] = 255
        self.assertTrue(torch.isnan(ring[1, 2 * 1024:3 * 1024]).all())

    def test_gather_records_by_position(self):
        cache = padded_cache(6, 4, 8, pad=0)
        view = self.probe.payload_view(cache, 4, 8)
        per_position, tail = self.probe.request_slots([3, 5], 6, page=4, ratio=1, circular=False)
        values = self.probe.gather_records(view, per_position, 4)
        self.assertTrue(torch.equal(values[0], view[3, 0]))
        self.assertTrue(torch.equal(values[5], view[5, 1]))
        tail_values = self.probe.gather_records(view, tail, 4)
        self.assertTrue(torch.equal(tail_values, view[5, 2:4]))
        self.assertIsNone(self.probe.gather_records(view, [None, None], 4))
        for slots in ([0], [-1], [24]):
            with self.assertRaisesRegex(ValueError, 'slot outside cache'):
                self.probe.gather_records(view, slots, 4)

    def test_poison_positions_and_tail_exactly(self):
        cache = padded_cache(6, 4, 8, pad=3)
        before = cache.clone()
        layer = self.layer(cache, [3, 5])
        action = {'run': 'p', 'mode': 'poison', 'byte': 255,
                  'select': [{'layer_regex': '.', 'positions': [4, 5]}, {'layer_regex': '.', 'tail': [1, 2]}]}
        self.assertEqual(self.probe.apply_write(action, 'l', layer, 6, {}), 2)
        view = self.probe.payload_view(cache, 4, 8)
        self.assertTrue(bool((view[5, 0] == 255).all()) and bool((view[5, 3] == 255).all()))
        changed = (cache != before).nonzero()[:, 0].unique().tolist()
        self.assertEqual(changed, [5])
        self.assertEqual(int((cache != before).sum()), 16)

    def test_transplant_positions_and_tail_from_snapshot(self):
        source_cache = padded_cache(6, 4, 8, pad=2)
        target_cache = torch.zeros_like(source_cache)
        row = [3, 5]
        per_position, tail = self.probe.request_slots(row, 6, page=4, ratio=1, circular=False)
        source_view = self.probe.payload_view(source_cache, 4, 8)
        snapshot = dict(self.layer(None, row), slots=per_position, tail_slots=tail,
                        values=self.probe.gather_records(source_view, per_position, 4),
                        tail=self.probe.gather_records(source_view, tail, 4))
        action = {'run': 't', 'mode': 'transplant', 'source': 's',
                  'select': [{'layer_regex': '.', 'positions': [0, 6]}, {'layer_regex': '.', 'tail': [0, 2]}]}
        written = self.probe.apply_write(action, 'l', self.layer(target_cache, row), 6, {'l': snapshot})
        self.assertEqual(written, 8)
        target_view = self.probe.payload_view(target_cache, 4, 8)
        self.assertTrue(torch.equal(target_view[3], source_view[3]))
        self.assertTrue(torch.equal(target_view[5], source_view[5]))
        self.assertTrue(bool((target_view[[0, 1, 2, 4]] == 0).all()))
        self.assertTrue(bool((target_cache[:, 32:] == 0).all()))        # padding untouched

    def test_transplant_rejects_layout_mismatch_and_inconsistent_shared_slot(self):
        cache = padded_cache(6, 4, 8, pad=0)
        view = self.probe.payload_view(cache, 4, 8)
        per_position, tail = self.probe.request_slots([3, 5], 6, page=4, ratio=1, circular=False)
        snapshot = dict(self.layer(None, [3, 5]), slots=per_position, tail_slots=tail,
                        values=self.probe.gather_records(view, per_position, 4),
                        tail=self.probe.gather_records(view, tail, 4))
        action = {'run': 't', 'mode': 'transplant', 'source': 's',
                  'select': [{'layer_regex': '.', 'positions': [0, 2]}]}
        with self.assertRaisesRegex(RuntimeError, 'layout differs'):
            self.probe.apply_write(action, 'l', self.layer(torch.zeros_like(cache), [3, 5], record=8,
                                                           spec='SlidingWindowMLASpec'), 6, {'l': snapshot})
        with self.assertRaisesRegex(RuntimeError, 'lacks layer'):
            self.probe.apply_write(action, 'l', self.layer(torch.zeros_like(cache), [3, 5]), 6, {})
        # ratio 2: positions 0 and 1 share a slot; different captured bytes must be refused
        per2, tail2 = self.probe.request_slots([3], 4, page=4, ratio=2, circular=False)
        values = torch.stack([torch.full((8,), v, dtype=torch.uint8) for v in (1, 2, 3, 3)])
        snap2 = dict(self.layer(None, [3], ratio=2), slots=per2, tail_slots=tail2, values=values, tail=None)
        with self.assertRaisesRegex(RuntimeError, 'different source bytes'):
            self.probe.apply_write(action, 'l', self.layer(torch.zeros_like(cache), [3], ratio=2), 4, {'l': snap2})
        ok = {'run': 't', 'mode': 'transplant', 'source': 's', 'select': [{'layer_regex': '.', 'positions': [2, 4]}]}
        self.assertEqual(self.probe.apply_write(ok, 'l', self.layer(torch.zeros_like(cache), [3], ratio=2), 4,
                                                {'l': snap2}), 1)

    def test_zero_write_rejected_by_probe(self):
        stub = ensure_spec_module()
        probe = self.probe
        instance = probe._Probe.__new__(probe._Probe)
        instance.node = 'cpu'
        cache = padded_cache(6, 4, 8, pad=0)
        layers = {'l': self.layer(cache, [0, 0])}                      # nothing backed
        action = {'run': 'p', 'mode': 'poison', 'byte': 1, 'select': [{'layer_regex': '.', 'positions': [0, 6]}]}
        with self.assertRaisesRegex(RuntimeError, 'selected no backed slot'):
            instance._write(action, layers, 6)
        del stub

    def test_later_invalid_layer_does_not_partially_write(self):
        instance = self.probe._Probe.__new__(self.probe._Probe)
        instance.node = 'cpu'
        first = torch.zeros(6, 32, dtype=torch.uint8)
        layers = {'first': self.layer(first, [3]),
                  'bad': self.layer(torch.zeros_like(first), [6])}
        action = {'run': 'p', 'mode': 'poison', 'byte': 99,
                  'select': [{'layer_regex': '.', 'positions': [0, 1]}]}
        with self.assertRaisesRegex(ValueError, 'slot outside cache'):
            instance._write(action, layers, 1)
        self.assertTrue(bool((first == 0).all()))

    def test_observe_layout_failure_retains_engine_and_failed_receipt(self):
        import json
        instance = self.probe._Probe.__new__(self.probe._Probe)
        instance.node, instance.next_action, instance.step = 'cpu', 0, ((), None, {})
        instance._load_control = lambda: None
        instance.control = {'scheduled_tokens': 8,
            'prompt_token_sha256': self.probe.token_sha256([1, 2]),
            'actions': [{'mode': 'observe', 'run': 'broken-layout'}]}
        instance.layers = mock.Mock(side_effect=RuntimeError('wrong layout'))
        batch = types.SimpleNamespace(num_reqs=1, num_scheduled_tokens=[8],
            idx_mapping_np=[0], seq_lens=[8])
        req = types.SimpleNamespace(prompt_len=types.SimpleNamespace(np=[2]),
            all_token_ids=types.SimpleNamespace(gpu=torch.tensor([[1, 2]])))
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(self.probe, 'OUTPUT', Path(directory)), \
             mock.patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False), \
             mock.patch.object(torch.cuda, 'current_stream', return_value=mock.Mock()):
            instance.after_prepare_inputs(batch, req)
            receipt = json.loads((Path(directory) / 'broken-layout-cpu.inventory.json').read_text())
            self.assertIn('wrong layout', receipt['error'])
            self.assertFalse(list(Path(directory).glob('*.pt')))
            self.assertEqual(instance.next_action, 1)

    def test_layers_classification_fail_closed(self):
        stub = ensure_spec_module()
        from vllm.v1.kv_cache_interface import CircularBufferSpec
        probe = self.probe

        class Spec:
            def __init__(self, page, ratio, record):
                self.num_states, self.tokens_per_state, self.page_size_bytes = page, ratio, page * record
                self.state_content_size_bytes, self.num_heads = record, 1

        class Ring(CircularBufferSpec):
            num_states, tokens_per_state, page_size_bytes = 4, 1, 4 * 8
            state_content_size_bytes, num_heads = 8, 1

        Spec.__name__ = 'MLAAttentionSpec'
        Ring.__name__ = 'CircularBufferSpec'

        def module(cache):
            return types.SimpleNamespace(kv_cache=cache)

        names = ['model.layers.2.attn.swa_cache', 'model.layers.2.attn.compressor.state_cache']
        context = {names[0]: module(padded_cache(4, 4, 8, pad=0)),
                   names[1]: module(padded_cache(4, 4, 8, pad=0, dtype=torch.float32))}
        groups = [types.SimpleNamespace(kv_cache_spec=Spec(4, 1, 8), layer_names=[names[0]]),
                  types.SimpleNamespace(kv_cache_spec=object.__new__(Ring), layer_names=[names[1]])]  # no dataclass init
        config = types.SimpleNamespace(kv_cache_groups=groups)
        tables = (torch.tensor([[1, 2]]), torch.tensor([[3]]))
        metadata = {name: types.SimpleNamespace(block_size=4, block_table=table)
                    for name, table in zip(names, tables)}
        instance = probe._Probe.__new__(probe._Probe)
        instance.context = context
        coverage = {'swa': {'suffix': '.swa_cache', 'layers': [[2, 3]]},
                    'ring': {'suffix': '.compressor.state_cache', 'layers': [[2, 3]]}}
        instance.control = {'allow_omitted': '', 'coverage': coverage}
        layers, omitted, counts = instance.layers(config, tables, metadata)
        self.assertEqual(sorted(layers), sorted(names))
        self.assertTrue(layers[names[1]]['circular'])
        self.assertEqual((omitted, counts), ([], {'swa': 1, 'ring': 1}))
        with self.assertRaisesRegex(RuntimeError, 'groups but 1 block tables'):
            instance.layers(config, tables[:1], metadata)
        context[names[1]] = module(torch.tensor([]))
        with self.assertRaisesRegex(RuntimeError, 'empty_cache'):
            instance.layers(config, tables, metadata)
        instance.control = {'allow_omitted': r'state_cache$', 'coverage': coverage}
        with self.assertRaisesRegex(RuntimeError, 'required coverage missing'):
            instance.layers(config, tables, metadata)
        instance.control = {'allow_omitted': r'state_cache$', 'coverage': {'swa': coverage['swa']}}
        layers, omitted, _ = instance.layers(config, tables, metadata)
        self.assertEqual(omitted, [(1, names[1], 'empty_cache')])
        del stub

    def test_real_unified_padded_and_split_geometry(self):
        from vllm.v1.core.kv_cache_utils import unify_kv_cache_spec_page_size
        from vllm.v1.kv_cache_interface import (
            SlidingWindowMLASpec, KVCacheTensor, create_kv_cache_views)
        from vllm.v1.kv_cache_layout import KVCacheLayout
        names = ['model.layers.2.attn.swa_cache', 'model.layers.3.attn.swa_cache',
                 'model.layers.4.attn.swa_cache']
        specs = unify_kv_cache_spec_page_size({
            name: SlidingWindowMLASpec(block_size=4, num_kv_heads=1,
                head_size=record, dtype=torch.uint8, sliding_window=64)
            for name, record in zip(names, (12, 8, 4))})
        self.assertEqual(specs[names[1]].page_size_padded, 48)
        self.assertEqual(specs[names[2]].block_size, 12)
        instance = self.probe._Probe.__new__(self.probe._Probe)
        instance.context = {}
        instance.control = {'allow_omitted': '', 'coverage': {
            'swa': {'suffix': '.swa_cache', 'layers': [[3, 5]]}}}
        groups, tables, metadata = [], [], {}
        for name in names[1:]:
            spec = specs[name]
            split = spec.block_size != 4
            layers = [name] if split else [name, 'unused-layer']
            block_stride = 48 * len(layers)
            raw = torch.full((4 * block_stride,), 233, dtype=torch.uint8)
            placement = KVCacheTensor(size=raw.numel(), layers=layers,
                                     layer_stride=48, block_stride=block_stride)
            view = create_kv_cache_views(raw, spec, 4, KVCacheLayout.BLHNC,
                                        placement, kernel_block_size=4)[0]
            for block in range(view.shape[0]):
                for row in range(4):
                    view[block, 0, row, :] = block * 8 + row
            cache = view.view(view.shape[0], -1)
            instance.context[name] = types.SimpleNamespace(kv_cache=cache)
            groups.append(types.SimpleNamespace(kv_cache_spec=spec, layer_names=[name]))
            tables.append(torch.tensor([[1, 2]]))
            metadata[name] = types.SimpleNamespace(block_size=4, block_table=torch.tensor([[2, 3]]))
        config = types.SimpleNamespace(kv_cache_groups=groups)
        found, omitted, _ = instance.layers(config, tables, metadata)
        self.assertEqual(omitted, [])
        for name, record in zip(names[1:], (8, 4)):
            layer = found[name]
            self.assertEqual((layer['page'], layer['record_bytes']), (4, record))
            view = self.probe.payload_view(layer['cache'], 4, record)
            positions, _ = self.probe.request_slots(layer['block_row'], 6, page=4, ratio=1, circular=False)
            values = self.probe.gather_records(view, positions, 4)
            self.assertEqual(values[:, 0].tolist(), [16, 17, 18, 19, 24, 25])
            before = layer['cache'].clone()
            action = {'mode': 'poison', 'byte': 99,
                      'select': [{'layer_regex': '.', 'positions': [4, 5]}]}
            self.assertEqual(self.probe.apply_write(action, name, layer, 6, {}), 1)
            self.assertEqual(int((layer['cache'] != before).sum()), record)
            self.assertTrue(bool((view[3, 0] == 99).all()))
        del metadata[names[1]]
        with self.assertRaisesRegex(RuntimeError, 'no_layer_metadata'):
            instance.layers(config, tables, metadata)
        from vllm.v1.kv_cache_interface import CircularBufferSpec, MLAAttentionSpec
        for spec in (
            CircularBufferSpec(block_size=16, num_kv_heads=1, head_size=1024,
                               head_size_v=0, dtype=torch.float32),
            MLAAttentionSpec(block_size=256, storage_block_size=64, num_kv_heads=1,
                             head_size=8, dtype=torch.uint8),
        ):
            name = names[1]
            kernel_size = getattr(spec, 'storage_block_size', None) or spec.block_size
            page = spec.get_num_kernel_states(kernel_size)
            record = spec.state_content_size_bytes
            raw = torch.zeros(4 * spec.page_size_bytes, dtype=torch.uint8)
            placement = KVCacheTensor(size=raw.numel(), layers=[name],
                layer_stride=spec.page_size_bytes, block_stride=spec.page_size_bytes)
            view = create_kv_cache_views(raw, spec, 4, KVCacheLayout.BLHNC,
                                        placement, kernel_block_size=kernel_size)[0]
            instance.context = {name: types.SimpleNamespace(kv_cache=view.view(view.shape[0], -1))}
            instance.control['coverage']['swa']['layers'] = [[3, 4]]
            config.kv_cache_groups = [types.SimpleNamespace(kv_cache_spec=spec, layer_names=[name])]
            metadata = {name: types.SimpleNamespace(block_size=page, block_table=torch.tensor([[2]]))}
            found, _, _ = instance.layers(config, [torch.tensor([[1]])], metadata)
            layer = found[name]
            self.assertEqual((layer['page'], layer['record_bytes']), (page, record))
            byte_view = self.probe.payload_view(layer['cache'], page, record)
            byte_view[2, 0] = 87
            self.assertTrue(bool((self.probe.gather_records(byte_view, [2 * page], page) == 87).all()))

    def test_grouped_specs_resolve_individual_layer_geometry(self):
        ensure_spec_module()
        from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
        probe = self.probe
        class Spec:
            def __init__(self, record):
                self.num_states, self.tokens_per_state, self.page_size_bytes = 4, 1, 4 * record
                self.state_content_size_bytes, self.num_heads = record, 1
        Spec.__name__ = 'MLAAttentionSpec'
        names = ['model.layers.2.attn', 'model.layers.8.attn']
        specs = {names[0]: Spec(8), names[1]: Spec(12)}
        grouped = UniformTypeKVCacheSpecs(block_size=4, kv_cache_specs=specs)
        instance = probe._Probe.__new__(probe._Probe)
        instance.context = {name: types.SimpleNamespace(kv_cache=padded_cache(4, 4, record, pad=0))
                            for name, record in zip(names, (8, 12))}
        instance.control = {'allow_omitted': '', 'coverage': {
            'main': {'suffix': '.attn', 'layers': [[2, 3], [8, 9]]}}}
        groups = [types.SimpleNamespace(kv_cache_spec=grouped, layer_names=names)]
        config = types.SimpleNamespace(kv_cache_groups=groups)
        metadata = {name: types.SimpleNamespace(block_size=4, block_table=torch.tensor([[2, 3]]))
                    for name in names}
        layers, omitted, counts = instance.layers(config, (torch.tensor([[1, 2]]),), metadata)
        self.assertEqual(layers[names[0]]['block_row'], [2, 3])
        self.assertEqual(layers[names[0]]['group_block_row'], [1, 2])
        self.assertEqual([layers[name]['record_bytes'] for name in names], [8, 12])
        self.assertEqual(omitted, [])
        self.assertEqual(counts, {'main': 2})
        del specs[names[1]]
        with self.assertRaisesRegex(RuntimeError, 'unsupported spec NoneType'):
            instance.layers(config, (torch.tensor([[1, 2]]),), metadata)


if __name__ == '__main__':
    unittest.main()
