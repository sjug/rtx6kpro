"""CPU tests for the window-capture kit (helper, attention patch, lock, installer, comparator, transport).
No GPU, nodes or builds:
  .venv-snapshot-cpu/bin/python claude-window-tests.py
"""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prep = load('claude_window_prep', HERE / 'claude-window-prepare.py')
compare = load('claude_window_compare', HERE / 'claude-window-compare.py')
run = load('claude_window_run', HERE / 'claude-window-run.py')
decision_prep = prep.decision_kit()
REF = compare.load_reference()
WINDOW, RECORDS, HIDDEN, HEADS, PAGE, BLOCKS = 128, 256, 64, 16, 128, 8
LAST = 524287


def fresh_helper():
    return load(f'claude_window_{time.monotonic_ns()}', HERE / 'claude-window-capture.py')


class FakeEvent:
    def record(self):
        pass

    def synchronize(self):
        pass


def slot_of(position):
    """Test slot mapping: position -> slot inside an 8-block cache (positions 524033..524287 -> 513..1023)."""
    return position % (BLOCKS * PAGE)


class Fixture:
    """Two fake full-row SWA layers with a packed SWA cache shaped like the DS4.1 layer 0/1 caches."""

    def __init__(self, seed=7):
        self.g = torch.Generator().manual_seed(seed)
        self.layers, self.kv_by_position = {}, {}
        for lid in (0, 1):
            cache = torch.zeros((BLOCKS, PAGE * 528), dtype=torch.uint8)
            positions = list(range(LAST - 254, LAST + 1))
            kv = torch.randn(len(positions), 512, generator=self.g).bfloat16()
            packed = REF.pack_deepseek_v41_cache_reference(kv.float(), page_size=1, cache_kind='swa')
            packed = packed.view(torch.uint8).reshape(-1, 528)
            for p, record in zip(positions, packed):
                slot = slot_of(p)
                cache[slot // PAGE, (slot % PAGE) * 528:(slot % PAGE + 1) * 528] = record
            self.kv_by_position[lid] = dict(zip(positions, kv))
            layer = types.SimpleNamespace(layer_id=lid, compress_ratio=0, is_ced_decoder=False, is_draft=False,
                                          hidden_size=HIDDEN, n_local_heads=HEADS, swa_width=WINDOW,
                                          attn_sink=torch.randn(HEADS, generator=self.g), forward_mqa=None,
                                          swa_cache_layer=types.SimpleNamespace(prefix=f'layers.{lid}.swa',
                                                                                block_size=PAGE, kv_cache=cache))
            self.layers[lid] = layer
        self.other = types.SimpleNamespace(layer_id=2, compress_ratio=2, is_ced_decoder=False, is_draft=False,
                                           swa_cache_layer=types.SimpleNamespace(prefix='layers.2.swa'))
        self.draft = types.SimpleNamespace(layer_id=0, is_draft=True,
                                           swa_cache_layer=types.SimpleNamespace(prefix='layers.0.swa'))

    def metadata(self, *, decode=False, reqs=1, tokens=8192, seq=524288):
        m = types.SimpleNamespace(is_decode=decode, num_reqs=reqs, num_actual_tokens=tokens, max_seq_len=seq)
        return {f'layers.{lid}.swa': m for lid in (0, 1, 2)}

    def tensors(self, lid, rows):
        """Forward tensors with distinct final rows; the last 128 kv_rot rows are the cache's own records."""
        g = self.g
        positions = torch.arange(524288 - rows, 524288, dtype=torch.int64)
        hidden = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16)
        hidden[-WINDOW:] = torch.randn(WINDOW, HIDDEN, generator=g).bfloat16()
        kv = torch.zeros(rows, 512, dtype=torch.bfloat16)
        kv[-WINDOW:] = torch.randn(WINDOW, 512, generator=g).bfloat16()
        rotated = torch.zeros(rows, 512, dtype=torch.bfloat16)
        rotated[-WINDOW:] = torch.stack([self.kv_by_position[lid][int(p)] for p in positions[-WINDOW:]])
        slots = torch.tensor([slot_of(int(p)) for p in positions], dtype=torch.int64)
        q = torch.zeros(rows, HEADS, 512, dtype=torch.bfloat16)
        q[-WINDOW:] = torch.randn(WINDOW, HEADS, 512, generator=g).bfloat16()
        indices = torch.full((rows, WINDOW), -1, dtype=torch.int32)
        lengths = torch.zeros(rows, dtype=torch.int32)
        for r in range(rows - WINDOW, rows):
            p = int(positions[r])
            indices[r] = torch.tensor([slot_of(x) for x in range(p - WINDOW + 1, p + 1)], dtype=torch.int32)
            lengths[r] = WINDOW
        out = torch.zeros(rows, HEADS, 512, dtype=torch.bfloat16)
        cache = self.layers[lid].swa_cache_layer.kv_cache
        records = torch.stack([cache[s // PAGE, (s % PAGE) * 528:(s % PAGE + 1) * 528]
                               for s in range(513, 1024)])
        for r in range(rows - WINDOW, rows):
            idx = (indices[r] - 513).reshape(1, -1)
            out[r] = REF.compressed_sparse_mla_reference(
                q[r].float().reshape(1, HEADS, 512), records, idx, torch.tensor([WINDOW], dtype=torch.int32),
                sm_scale=512 ** -0.5, attn_sink=self.layers[lid].attn_sink, swa_page_size=1,
                cache_format='deepseek_v41').reshape(HEADS, 512).bfloat16()
        wo = torch.zeros(rows, HIDDEN, 1, dtype=torch.bfloat16)
        wo[-WINDOW:] = torch.randn(WINDOW, HIDDEN, 1, generator=g).bfloat16()
        return {'positions': positions, 'hidden': hidden, 'kv': kv, 'rotated': rotated, 'slots': slots, 'q': q,
                'out': out, 'indices': indices, 'lengths': lengths, 'wo': wo, 'wo_reduced': wo * 2}

    def run_forward(self, helper, metadata, skip=(), order=None, metadata_capacity=None):
        """Call the hooks exactly where the patched source calls them, layer 0 then layer 1."""
        rows = metadata['layers.0.swa'].num_actual_tokens
        self.forward = {}
        for lid, layer in self.layers.items():
            t = self.tensors(lid, rows)
            self.forward[lid] = t
            entry = helper.begin(layer, metadata)
            if entry is None:
                continue
            steps = order or ('inputs', 'rotated', 'attention', 'wo_partial', 'wo_reduced')
            for step in steps:
                if step in skip:
                    continue
                if step == 'inputs':
                    entry.inputs(layer, hidden_states=t['hidden'], kv=t['kv'], positions=t['positions'])
                elif step == 'rotated':
                    helper.rotated(layer, rotated=t['rotated'], slot_mapping=t['slots'], positions=t['positions'])
                elif step == 'attention':
                    # Execute the generated call site, not a hand-written copy.
                    # The production buffers have prepared capacity, while q/output
                    # have only live rows. Poison unused metadata to catch tail slicing.
                    capacity = rows if metadata_capacity is None else metadata_capacity
                    indices = torch.full((capacity, WINDOW), -987654, dtype=t['indices'].dtype)
                    lengths = torch.full((capacity,), -987654, dtype=t['lengths'].dtype)
                    indices[:rows], lengths[:rows] = t['indices'], t['lengths']
                    tree = ast.parse((HERE / 'claude-window-attention.py').read_text())
                    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                             and isinstance(node.func, ast.Attribute) and node.func.attr == 'attention'
                             and isinstance(node.func.value, ast.Name) and node.func.value.id == '_claude_window']
                    if len(calls) != 1:
                        raise RuntimeError('Expected one generated window attention hook')
                    code = compile(ast.fix_missing_locations(ast.Expression(calls[0])), '<generated-window-hook>', 'eval')
                    eval(code, {'_claude_window': helper}, {'self': layer, 'rows': rows,
                         'q': t['q'], 'output': t['out'], 'swa_indices': indices, 'swa_lengths': lengths})
                elif step == 'wo_partial':
                    helper.wo_partial(layer, t['wo'])
                elif step == 'wo_reduced':
                    helper.wo_reduced(layer, t['wo_reduced'])

    def decision_capture(self, lid, rows):
        """A schema-v3 entry of the same forward, as the unchanged decision-row helper would stage it."""
        t = self.forward[lid]
        cache = self.layers[lid].swa_cache_layer.kv_cache
        slots = t['indices'][rows - 1].to(torch.int64)
        records = torch.stack([cache[s // PAGE, (s % PAGE) * 528:(s % PAGE + 1) * 528] for s in slots.tolist()])
        return {'q': t['q'][rows - 1], 'out': t['out'][rows - 1], 'attn_sink': self.layers[lid].attn_sink,
                'swa_len': WINDOW, 'swa_slots': slots, 'swa_records': records, 'position': LAST}


class HelperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.helper = fresh_helper()
        self.fixture = Fixture()
        h = self.helper
        self.capturing = False
        self.patches = [mock.patch.object(h, 'TRIGGER', str(self.dir / 'trigger.json')),
                        mock.patch.object(h, 'OUT_DIR', str(self.dir / 'out')),
                        mock.patch.object(h, 'POLL_SECONDS', 0.0),
                        mock.patch.object(h, '_rank', lambda: 0),
                        mock.patch.object(h, '_world_size', lambda: 4),
                        mock.patch.object(h, '_pinned', lambda shape, dtype: torch.empty(shape, dtype=dtype)),
                        mock.patch.object(h, '_target_layers', lambda: dict(self.fixture.layers)),
                        mock.patch.object(h, '_capturing', lambda: self.capturing),
                        mock.patch.object(h, '_source_trees', lambda: {'b12x': 't', 'vllm': 'v'}),
                        mock.patch.object(h.torch.cuda, 'Event', FakeEvent),
                        mock.patch.dict(os.environ, {'DS41_NODE': 'dusty', 'DS41_KIT_SHA256': 'kit'})]
        for p in self.patches:
            p.start()
        h._IMPORTED_AT = time.time() - 60

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def arm(self, token='window-0001', **extra):
        (self.dir / 'trigger.json').write_text(json.dumps({'token': token, 'prompt_tokens': 524288, **extra}))

    def saved(self):
        for thread in list(threading.enumerate()):
            if thread.name == 'claude-window-waiter':
                thread.join(timeout=30)
        return sorted((self.dir / 'out').glob('*.pt')) if (self.dir / 'out').exists() else []

    def test_unarmed_disabled_and_invalid_triggers_do_nothing(self):
        layer, metadata = self.fixture.layers[0], self.fixture.metadata()
        self.assertIsNone(self.helper.begin(layer, metadata))
        for spec in ({'token': 'BAD token', 'prompt_tokens': 524288}, {'token': 'window-0001', 'prompt_tokens': 1},
                     {'token': 'window-0001', 'prompt_tokens': 524288, 'window': False},
                     {'token': 'window-0001', 'prompt_tokens': 524288, 'window': {'rows': 64, 'layers': [0, 1]}},
                     {'token': 'window-0001', 'prompt_tokens': 524288, 'window': {'rows': 128, 'layers': [0]}},
                     {'token': 'window-0001', 'prompt_tokens': 524288, 'chunk_rows': 7936}):
            (self.dir / 'trigger.json').write_text(json.dumps(spec))
            self.assertIsNone(self.helper.begin(layer, metadata))
            self.assertIsNone(self.helper._state['armed'])
        self.arm()
        os.utime(self.dir / 'trigger.json', (0, 0))
        self.assertIsNone(self.helper.begin(layer, metadata))
        # module-level hooks are inert when nothing is armed
        self.helper.rotated(layer, rotated=None, slot_mapping=None, positions=None)
        self.helper.wo_reduced(layer, None)

    def test_arms_from_the_decision_row_payload_and_explicit_window_field(self):
        for extra in ({}, {'window': True}, {'window': {'rows': 128, 'layers': [0, 1]}}):
            helper = fresh_helper()
            with mock.patch.object(helper, 'TRIGGER', str(self.dir / 'trigger.json')), \
                    mock.patch.object(helper, 'OUT_DIR', str(self.dir / 'out')):
                helper._IMPORTED_AT = 0
                self.arm(**extra)
                self.assertIsNotNone(helper._read_trigger(0), extra)

    def test_only_layers_0_and_1_on_the_decision_chunk(self):
        self.arm()
        f = self.fixture
        for metadata in (f.metadata(decode=True), f.metadata(reqs=2), f.metadata(seq=524288 - 8192)):
            self.assertIsNone(self.helper.begin(f.layers[0], metadata))
        self.assertIsNone(self.helper.begin(f.draft, f.metadata()))
        self.assertIsNone(self.helper.begin(f.other, f.metadata()))
        self.assertIsNotNone(self.helper.begin(f.layers[0], f.metadata()))

    def test_final_chunk_size_mismatch_aborts_with_receipt(self):
        self.arm(chunk_rows=4096)
        self.assertIsNone(self.helper.begin(self.fixture.layers[0], self.fixture.metadata()))
        self.assertIsNone(self.helper._state['armed'])
        self.assertEqual(self.saved(), [])
        receipt = json.loads((self.dir / 'out' / 'window-0001-rank0.window-aborted').read_text())
        self.assertIn('final chunk rows 8192 differ from armed 4096', receipt['error'])

    def check_capture(self, capture, rows):
        f = self.fixture
        self.assertEqual(capture['schema'], 'claude-window-v1')
        meta = capture['meta']
        self.assertEqual((meta['chunk_rows'], meta['problems'], meta['tp_world_size'], meta['row_position']),
                         (rows, [], 4, LAST))
        self.assertEqual(meta['decision_row_file'], 'dusty-rank0-window-0001.pt')
        self.assertEqual(compare.validate(capture), meta)
        for lid in (0, 1):
            t, e = f.forward[lid], capture['layers'][lid]
            self.assertEqual((e['chunk_rows'], e['chunk_row_start'], e['hidden'], e['heads']), (rows, rows - WINDOW, HIDDEN, HEADS))
            for name, src in (('hidden_in', t['hidden']), ('kv_norm', t['kv']), ('kv_rot', t['rotated']),
                              ('write_slots', t['slots']), ('q', t['q']), ('out', t['out']),
                              ('swa_lengths', t['lengths']), ('positions', t['positions'])):
                self.assertTrue(torch.equal(e[name], src[-WINDOW:]), name)
            self.assertTrue(torch.equal(e['swa_indices'], t['indices'][-WINDOW:].to(torch.int64)))
            self.assertTrue(torch.equal(e['wo_partial'], t['wo'][-WINDOW:].reshape(WINDOW, HIDDEN)))
            self.assertTrue(torch.equal(e['wo_reduced'], t['wo_reduced'][-WINDOW:].reshape(WINDOW, HIDDEN)))
            self.assertTrue(torch.equal(e['record_slots'], torch.cat((t['indices'][rows - WINDOW], t['indices'][rows - 1])).long()))
            cache = f.layers[lid].swa_cache_layer.kv_cache
            for i, s in enumerate(e['record_slots'].tolist()):
                self.assertTrue(torch.equal(e['records'][i], cache[s // PAGE, (s % PAGE) * 528:(s % PAGE + 1) * 528]))
            self.assertEqual(compare.decision_consistency(capture, {
                'schema': 'claude-decision-row-v3', 'meta': {'rank': 0, 'chunk_rows': rows},
                'layers': {l: f.decision_capture(l, rows) for l in (0, 1)}})[lid]['all_equal'], True)

    def test_full_forward_saves_schema_and_is_one_shot(self):
        self.arm()
        self.fixture.run_forward(self.helper, self.fixture.metadata())
        for layer in self.fixture.layers.values():          # block reuse after the request cannot reach the capture
            layer.swa_cache_layer.kv_cache.fill_(0xA5)
        (path,) = self.saved()
        self.assertEqual(path.name, 'dusty-rank0-window-0001-window.pt')
        capture = torch.load(path, map_location='cpu', weights_only=True)
        for layer in self.fixture.layers.values():
            layer.swa_cache_layer.kv_cache.copy_(self.fixture_cache_backup(layer))
        receipt = json.loads((self.dir / 'out' / 'window-0001-rank0.window-consumed').read_text())
        self.assertEqual(receipt['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertIsNone(self.helper._state['armed'])
        again = fresh_helper()
        with mock.patch.object(again, 'OUT_DIR', str(self.dir / 'out')), \
                mock.patch.object(again, 'TRIGGER', str(self.dir / 'trigger.json')):
            again._IMPORTED_AT = 0
            self.assertIsNone(again._read_trigger(0))
        self.assertEqual(run.parse_window_consumed(json.dumps(receipt), 'window-0001', 0, lambda p: p)['sha256'],
                         receipt['sha256'])

    def fixture_cache_backup(self, layer):
        if not hasattr(self, '_backup'):
            self._backup = {}
        return self._backup.setdefault(layer.layer_id, self.rebuild_cache(layer.layer_id))

    def rebuild_cache(self, lid):
        cache = torch.zeros((BLOCKS, PAGE * 528), dtype=torch.uint8)
        kv = torch.stack([self.fixture.kv_by_position[lid][p] for p in range(LAST - 254, LAST + 1)])
        packed = REF.pack_deepseek_v41_cache_reference(kv.float(), page_size=1, cache_kind='swa').view(torch.uint8).reshape(-1, 528)
        for p, record in zip(range(LAST - 254, LAST + 1), packed):
            s = slot_of(p)
            cache[s // PAGE, (s % PAGE) * 528:(s % PAGE + 1) * 528] = record
        return cache

    def test_capture_contents_on_both_grids(self):
        for rows in (8192, 4096):
            with self.subTest(rows=rows):
                helper, fixture = fresh_helper(), Fixture(seed=rows)
                self.fixture = fixture
                with mock.patch.object(helper, 'TRIGGER', str(self.dir / 'trigger.json')), \
                        mock.patch.object(helper, 'OUT_DIR', str(self.dir / f'out{rows}')), \
                        mock.patch.object(helper, 'POLL_SECONDS', 0.0), mock.patch.object(helper, '_rank', lambda: 0), \
                        mock.patch.object(helper, '_world_size', lambda: 4), \
                        mock.patch.object(helper, '_pinned', lambda shape, dtype: torch.empty(shape, dtype=dtype)), \
                        mock.patch.object(helper, '_target_layers', lambda: dict(fixture.layers)), \
                        mock.patch.object(helper, '_capturing', lambda: False), \
                        mock.patch.object(helper, '_source_trees', lambda: {'b12x': 't', 'vllm': 'v'}), \
                        mock.patch.object(helper.torch.cuda, 'Event', FakeEvent):
                    helper._IMPORTED_AT = 0
                    self.arm(chunk_rows=rows)
                    fixture.run_forward(helper, fixture.metadata(tokens=rows))
                    for thread in list(threading.enumerate()):
                        if thread.name == 'claude-window-waiter':
                            thread.join(timeout=30)
                    (path,) = sorted((self.dir / f'out{rows}').glob('*.pt'))
                    self.check_capture(torch.load(path, map_location='cpu', weights_only=True), rows)

    def test_oversized_prepared_metadata_uses_only_live_rows(self):
        self.arm(chunk_rows=4096)
        self.fixture.run_forward(self.helper, self.fixture.metadata(tokens=4096), metadata_capacity=8192)
        files = self.saved()
        self.assertEqual(len(files), 1, 'Prepared metadata capacity must not abort the live-row capture')
        self.check_capture(torch.load(files[0], map_location='cpu', weights_only=True), 4096)
        self.assertFalse(list((self.dir / 'out').glob('*.window-aborted')))

    def test_boundary_order_is_enforced_and_errors_are_contained(self):
        self.arm()
        self.fixture.run_forward(self.helper, self.fixture.metadata(), order=('inputs', 'attention'))
        self.assertIsNone(self.helper._state['armed'])
        self.assertEqual(self.saved(), [])
        receipt = json.loads((self.dir / 'out' / 'window-0001-rank0.window-aborted').read_text())
        self.assertIn('attention before the cache write', receipt['error'])
        helper = fresh_helper()
        self.arm(token='window-0002')
        with mock.patch.object(helper, 'TRIGGER', str(self.dir / 'trigger.json')), \
                mock.patch.object(helper, 'OUT_DIR', str(self.dir / 'out')), mock.patch.object(helper, 'POLL_SECONDS', 0.0), \
                mock.patch.object(helper, '_rank', lambda: 0), mock.patch.object(helper, '_world_size', lambda: 4), \
                mock.patch.object(helper, '_pinned', lambda shape, dtype: torch.empty(shape, dtype=dtype)), \
                mock.patch.object(helper, '_target_layers', lambda: dict(self.fixture.layers)), \
                mock.patch.object(helper, '_capturing', lambda: False), \
                mock.patch.object(helper._LayerCapture, '_tail', side_effect=RuntimeError('boom')):
            helper._IMPORTED_AT = 0
            entry = helper.begin(self.fixture.layers[0], self.fixture.metadata())
            self.assertIsNone(entry.inputs(self.fixture.layers[0], hidden_states=None, kv=None, positions=None))
            self.assertIsNone(helper._state['armed'])
        self.assertIn('boom', (self.dir / 'out' / 'window-0002-rank0.window-aborted').read_text())

    def test_incomplete_forward_never_saves_and_disarms(self):
        self.arm()
        f = self.fixture
        entry = self.helper.begin(f.layers[0], f.metadata())
        session = self.helper._state['armed']
        self.assertIsNotNone(entry)
        session.finish()
        self.assertIsNone(self.helper._state['armed'])
        self.assertEqual(self.saved(), [])

    def test_no_arming_under_graph_capture(self):
        self.arm()
        self.capturing = True
        with mock.patch.object(self.helper, '_Session', side_effect=AssertionError('allocated under capture')):
            self.assertIsNone(self.helper.begin(self.fixture.layers[0], self.fixture.metadata()))
        self.assertIsNone(self.helper._state['armed'])


class ForwardDiscipline(unittest.TestCase):
    FORBIDDEN = {'item', 'cpu', 'synchronize', 'tolist', 'numpy', 'wait_event', 'wait_stream', 'Stream'}

    def test_forward_path_has_no_host_synchronization(self):
        tree = ast.parse((HERE / 'claude-window-capture.py').read_text())
        forward = {'begin', '_begin', '_active', 'rotated', 'attention', 'wo_partial', 'wo_reduced', 'inputs', '_d2h',
                   '_tail', '_gather', 'open', 'finish', '_abort', 'wrapper'}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in forward:
                calls = {c.func.attr for c in ast.walk(node) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)}
                self.assertFalse(calls & self.FORBIDDEN, (node.name, calls & self.FORBIDDEN))

    def test_d2h_is_non_blocking_and_waiter_touches_no_device_cache(self):
        text = (HERE / 'claude-window-capture.py').read_text()
        self.assertIn('staged.copy_(value, non_blocking=True)', text)
        tree = ast.parse(text)
        host = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'host')
        names = {a.attr for a in ast.walk(host) if isinstance(a, ast.Attribute)}
        self.assertFalse(names & {'kv_cache', 'index_select'}, names)


class Patch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = decision_prep.upstream_attention().decode()
        cls.decision = decision_prep.compose(cls.old)
        cls.new = prep.compose(cls.old, decision_prep)

    def test_only_guarded_additions_over_the_decision_row_source(self):
        import difflib
        diff = [l for l in difflib.ndiff(self.decision.splitlines(), self.new.splitlines()) if l.startswith(('- ', '+ '))]
        self.assertFalse([l for l in diff if l.startswith('- ')])
        added = [l[2:].strip() for l in diff]
        self.assertIn('import vllm.models.deepseek_v4_1.claude_window as _claude_window', added)
        self.assertEqual(sum('_claude_window.begin(self, metadata)' in l for l in added), 1)
        self.assertEqual([l for l in added if l.startswith('_claude_window.') and not l.startswith('_claude_window.begin')],
                         ['_claude_window.rotated(self, rotated=rotated, slot_mapping=slot_mapping, positions=positions)',
                          '_claude_window.attention(', '_claude_window.wo_partial(self, local)',
                          '_claude_window.wo_reduced(self, local)'])
        self.assertEqual(self.decision, (HERE / 'claude-decision-row-attention.py').read_text())

    def test_hooks_follow_their_producers_at_eager_boundaries(self):
        t = self.new
        self.assertLess(t.index('swa = self._query_metadata(original_swa)'), t.index('_window = _claude_window.begin('))
        self.assertLess(t.index('_window.inputs('), t.index('self._cache_context_kv(kv, positions)'))
        self.assertLess(t.index('plan=self._helper_plan("kv"),'), t.index('_claude_window.rotated('))
        self.assertLess(t.index('_claude_window.rotated('), t.index('mla.write_cache(\n            rotated,'))
        self.assertLess(t.index('_claude.attention('), t.index('_claude_window.attention('))
        self.assertLess(t.index('must return BF16'), t.index('_claude_window.wo_partial(self, local)'))
        self.assertLess(t.index('_claude_window.wo_partial(self, local)'), t.index('l2_prefetch.issue(self._l2pf_ffn, rows)'))
        self.assertLess(t.index('get_tp_group().all_reduce(local)'), t.index('_claude_window.wo_reduced(self, local)'))
        body = t[t.index('def _o_proj('):]
        self.assertLess(body.index('_claude_window.wo_reduced(self, local)'), body.index('return local'))
        for name in ('def _forward(', 'def insert_context_kv(', 'def forward_mqa(', 'def _o_proj('):
            self.assertIn(name, t)

    def test_lock_dockerfile_and_ignore_reproduce(self):
        lock = json.loads((HERE / 'claude-window.lock.json').read_text())
        for name, digest in lock['inputs'].items():
            self.assertEqual(hashlib.sha256((HERE / name).read_bytes()).hexdigest(), digest, name)
        self.assertEqual(lock['targets'][prep.ATTENTION]['output_sha256'], hashlib.sha256(self.new.encode()).hexdigest())
        self.assertEqual((HERE / prep.OUTPUT).read_text(), self.new)
        self.assertEqual((HERE / 'Dockerfile.claude-window').read_text(), prep.render_dockerfile(lock))
        names = sorted(lock['inputs']) + ['claude-window.lock.json']
        self.assertEqual((HERE / 'claude-window.ignore').read_text(), '**\n' + ''.join('!' + n + '\n' for n in names))
        decision = json.loads((HERE / 'claude-decision-row.lock.json').read_text())
        self.assertEqual(lock['base_image_id'], 'e06df11a8ca18fa514d9f28f67cc691aef296da2eeb22b113a734519853bccd7')
        self.assertEqual(lock['base_kind'], 'router-stage-release-candidate')
        self.assertEqual(lock['targets'][prep.DECISION_HELPER]['output_sha256'],
                         decision['targets'][prep.DECISION_HELPER]['output_sha256'])
        self.assertEqual(lock['decision_row_lock_sha256'],
                         hashlib.sha256((HERE / 'claude-decision-row.lock.json').read_bytes()).hexdigest())
        self.assertNotEqual(lock['trees']['vllm'], decision['trees']['vllm'])
        self.assertEqual(lock['trees']['b12x'], decision['trees']['b12x'])
        self.assertEqual((lock['install_dir'], lock['install_pass_marker'], lock['image_kind_label'], lock['build_args']),
                         ('/opt/ds41-window', 'WINDOW-INSTALL-PASS', 'window-capture', ['DECISION_LOCK', 'DECISION_CACHE']))
        self.assertIn('not numerics', lock['scope'])
        text = (HERE / 'Dockerfile.claude-window').read_text()
        for required in ('ARG DECISION_LOCK', 'ARG DECISION_CACHE', '/opt/ds41-window/claude-window-install.py',
                         'diagnostic.kind="window-capture"', 'decision-row-lock.sha256="' + lock['decision_row_lock_sha256']):
            self.assertIn(required, text)

    def test_dockerfile_continuations(self):
        lines = (HERE / 'Dockerfile.claude-window').read_text().splitlines()
        for line in [l for l in lines if l.endswith('\\')]:
            self.assertTrue(line.endswith(' \\') and not line.endswith('\\\\'), line)
        env = lines.index(next(l for l in lines if l.startswith('ENV ')))
        self.assertTrue(lines[env].endswith('\\') and not lines[env + 1].endswith('\\'))
        label = lines.index(next(l for l in lines if l.startswith('LABEL ')))
        end = next(i for i in range(label, len(lines)) if not lines[i].endswith('\\'))
        self.assertTrue(lines[end].startswith('    org.opencontainers.image.title='))


class Installer(unittest.TestCase):
    def simulate(self, tamper=None):
        lock = json.loads((HERE / 'claude-window.lock.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            kit, opt, router, stub = Path(tmp) / 'kit', Path(tmp) / 'opt', Path(tmp) / 'router', Path(tmp) / 'stub'
            kit.mkdir()
            router.mkdir()
            for name in [*lock['inputs'], 'claude-window.lock.json']:
                shutil.copy(HERE / name, kit / name)
            attention = opt / prep.ATTENTION
            attention.parent.mkdir(parents=True)
            attention.write_bytes(decision_prep.upstream_attention())
            fenced = opt / 'b12x' / lock['base_router_target']['path']
            fenced.parent.mkdir(parents=True)
            shutil.copy(HERE / 'router-prefill-release-before.py', fenced)
            (opt / 'b12x/b12x/native.so').write_bytes(b'x')
            shutil.copy(HERE / 'router-release.lock.json', router / 'router-release.lock.json')
            digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            preserved = {str(p): digest(p) for c in ('vllm', 'b12x') for p in (opt / c).rglob('*') if p.is_file()}
            (router / 'preserved-files.json').write_text(json.dumps(preserved, sort_keys=True) + '\n')
            if tamper:
                tamper(opt=opt, router=router, fenced=fenced, stub=stub)
            source = (HERE / 'claude-window-install.py').read_text()
            script = Path(tmp) / 'install.py'
            script.write_text(source.replace("Path('/opt/ds41-window')", f"Path({str(kit)!r})")
                              .replace("Path('/opt/jovian-judgement')", f"Path({str(opt)!r})")
                              .replace("Path('/opt/ds41-router-release')", f"Path({str(router)!r})")
                              .replace("Path('/opt/ds41-decision-row/claude-decision-row.lock.json')",
                                       f"Path({str(stub / 'claude-decision-row.lock.json')!r})"))
            result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
            files = {p.relative_to(opt).as_posix() for p in opt.rglob('*') if p.is_file()}
            stub_text = (stub / 'claude-decision-row.lock.json').read_text() if (stub / 'claude-decision-row.lock.json').exists() else None
            return result, attention.read_text(), files, stub_text

    def test_install_is_confined_and_writes_the_provenance_stub(self):
        result, attention, files, stub = self.simulate()
        self.assertIn('WINDOW-INSTALL-PASS', result.stdout, result.stderr)
        self.assertEqual(attention, (HERE / prep.OUTPUT).read_text())
        self.assertEqual(files, {'vllm/vllm/models/deepseek_v4_1/attention.py', 'vllm/vllm/models/deepseek_v4_1/claude_decision_row.py',
                                 'vllm/vllm/models/deepseek_v4_1/claude_window.py', 'b12x/b12x/gemm/bf16_gemv/_prefill.py',
                                 'b12x/b12x/native.so'})
        lock = json.loads((HERE / 'claude-window.lock.json').read_text())
        stub = json.loads(stub)
        self.assertEqual(stub['trees'], lock['trees'])
        self.assertEqual(stub['window_lock_sha256'], hashlib.sha256((HERE / 'claude-window.lock.json').read_bytes()).hexdigest())

    def test_other_bases_are_refused(self):
        for message, tamper in {
                'router source differs': lambda opt, router, fenced, stub: shutil.copy(HERE / 'claude-pinned-bf16_gemv-_prefill.py', fenced),
                'inventory differs': lambda opt, router, fenced, stub: (opt / 'vllm/extra.py').write_text('x'),
                'already carries a decision-row installation': lambda opt, router, fenced, stub: (stub.mkdir(), (stub / 'claude-decision-row.lock.json').write_text('{}')),
                'preexisting diagnostic file': lambda opt, router, fenced, stub: (opt / 'vllm/vllm/models/deepseek_v4_1/claude_window.py').write_text('x')}.items():
            result, *_ = self.simulate(tamper)
            self.assertNotEqual(result.returncode, 0, message)
            self.assertIn(message, result.stderr)


def synthetic(chunk_rows, seed):
    """A schema-v1 capture built from the fixture forward, without the helper."""
    fixture = Fixture(seed=seed)
    layers = {}
    for lid in (0, 1):
        t = fixture.tensors(lid, chunk_rows)
        cache = fixture.layers[lid].swa_cache_layer.kv_cache
        slots = torch.cat((t['indices'][chunk_rows - WINDOW], t['indices'][chunk_rows - 1])).long()
        records = torch.stack([cache[s // PAGE, (s % PAGE) * 528:(s % PAGE + 1) * 528] for s in slots.tolist()])
        layers[lid] = {'positions': t['positions'][-WINDOW:], 'hidden_in': t['hidden'][-WINDOW:], 'kv_norm': t['kv'][-WINDOW:],
                       'kv_rot': t['rotated'][-WINDOW:], 'write_slots': t['slots'][-WINDOW:], 'q': t['q'][-WINDOW:],
                       'out': t['out'][-WINDOW:], 'attn_sink': fixture.layers[lid].attn_sink,
                       'swa_indices': t['indices'][-WINDOW:].long(), 'swa_lengths': t['lengths'][-WINDOW:],
                       'record_slots': slots, 'records': records, 'wo_partial': t['wo'][-WINDOW:].reshape(WINDOW, HIDDEN),
                       'wo_reduced': t['wo_reduced'][-WINDOW:].reshape(WINDOW, HIDDEN), 'layer_id': lid,
                       'chunk_rows': chunk_rows, 'chunk_row_start': chunk_rows - WINDOW, 'hidden': HIDDEN, 'heads': HEADS,
                       'swa_width': WINDOW, 'page_size': PAGE}
    meta = {'rank': 0, 'node': 'dusty', 'generation': 'tok', 'kit_sha256': 'kit', 'source_trees': {'vllm': 'v', 'b12x': 'b'},
            'prompt_tokens': 524288, 'chunk_rows': chunk_rows, 'window_rows': WINDOW, 'record_rows': RECORDS, 'layers': [0, 1],
            'tp_world_size': 4, 'batch_requests': 1, 'row_position': LAST, 'problems': []}
    return {'schema': 'claude-window-v1', 'meta': meta, 'layers': layers}


class Comparator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.left, cls.right = synthetic(8192, 3), synthetic(4096, 3)

    def test_same_forward_on_two_grids_reports_no_difference_and_within_checks_hold(self):
        report = compare.compare_rank(self.left, self.right, REF)
        self.assertIsNone(report['first_differing_boundary'])
        self.assertEqual(report['scope'], 'descriptive-only; experiment comparability not established')
        for side in ('within_left', 'within_right'):
            for lid in (0, 1):
                w = report[side][lid]
                self.assertLess(w['reference']['rel_l2_max'], 1e-2)              # bf16 rounding of the fixture output only
                self.assertEqual(w['packed']['rows_bit_equal'], WINDOW)
                self.assertEqual(w['packed']['rows_not_in_gathered_records'], [])
                self.assertTrue(w['slots']['write_slots_equal_last_row_window'])
                self.assertTrue(w['slots']['write_offsets_equal_position_offsets'])
                self.assertTrue(w['slots']['write_slots_ascending'])

    def test_first_observed_boundary_and_row_profile(self):
        import copy
        right = copy.deepcopy(self.right)
        right['layers'][1]['kv_rot'][16:112] += 1
        right['layers'][1]['wo_reduced'][5] += 1
        report = compare.compare_rank(self.left, right, REF)
        self.assertEqual(report['first_differing_boundary'], {'layer': 1, 'boundary': 'kv_rot'})
        b = report['layers'][1]['boundaries']
        self.assertEqual((b['kv_rot']['first_changed_row'], b['kv_rot']['last_changed_row'], b['kv_rot']['changed_row_count']), (16, 111, 96))
        self.assertTrue(b['hidden_in']['equal_values'] and b['kv_norm']['equal_values'])
        self.assertEqual(b['wo_reduced']['changed_rows'], [5])
        self.assertIsNone(report['layers'][0]['first_differing_boundary'])
        # the packed check now shows the rotated rows no longer match the records read back
        self.assertLess(report['within_right'][1]['packed']['rows_bit_equal'], WINDOW)

    def test_refuses_mismatched_identity_or_geometry(self):
        import copy
        for mutate, message in ((lambda c: c['meta'].update(rank=1), 'identity differs'),
                                (lambda c: c['meta'].update(problems=['x']), 'reported problems'),
                                (lambda c: c['layers'][0].update(chunk_row_start=0), 'window geometry'),
                                (lambda c: c['layers'].pop(1), 'layers 0 and 1')):
            right = copy.deepcopy(self.right)
            mutate(right)
            with self.assertRaisesRegex(ValueError, message):
                compare.compare_rank(self.left, right, REF)

    def test_cross_rank_hashes(self):
        import copy
        ranks = []
        for r in range(4):
            c = copy.deepcopy(self.left)
            c['meta']['rank'] = r
            c['layers'][0]['q'] += r                      # per-rank heads differ; replicated tensors stay equal
            ranks.append(c)
        report = compare.cross_rank(ranks)
        self.assertTrue(report['replicated_all_equal'])
        self.assertFalse(report['layers'][0]['q']['equal_across_ranks'])
        self.assertTrue(report['layers'][1]['q']['equal_across_ranks'])
        ranks[2]['layers'][1]['hidden_in'][0, 0] += 1
        self.assertFalse(compare.cross_rank(ranks)['replicated_all_equal'])


class Transport(unittest.TestCase):
    def test_payload_extends_the_decision_row_trigger(self):
        payload = run.trigger_payload('window-0001', 4096)
        self.assertEqual(payload, {'token': 'window-0001', 'prompt_tokens': 524288, 'chunk_rows': 4096,
                                   'window': {'rows': 128, 'layers': [0, 1]}})
        self.assertEqual(run.trigger_payload('window-0001')['window'], {'rows': 128, 'layers': [0, 1]})
        self.assertIs(run.trigger_payload('window-0001', window=False)['window'], False)
        helper = fresh_helper()
        self.assertIsNotNone(helper.window_spec(payload))
        self.assertIsNone(helper.window_spec(run.trigger_payload('window-0001', window=False)))
        self.assertIsNotNone(helper.window_spec(run.driver().trigger_payload('window-0001', 8192)))

    def test_receipt_parsing_and_gather_script_reuse(self):
        text = json.dumps({'file': '/cache/claude-decision-row/toby-rank1-window-0001-window.pt', 'sha256': 'a' * 64, 'problems': []})
        record = run.parse_window_consumed(text, 'window-0001', 1)
        self.assertEqual(record['host_file'], '/home/jugs/.cache/vllm-jj-ds41-tp4/claude-decision-row/toby-rank1-window-0001-window.pt')
        with self.assertRaises(ValueError):
            run.parse_window_consumed(text, 'window-0001', 2)
        with self.assertRaises(ValueError):
            run.parse_window_consumed(text.replace('-window.pt', '.pt'), 'window-0001', 1)
        script = run.driver().gather_script('/remote/receipt', {'dusty': dict(record, host_file='/x/dusty-rank0-window-0001-window.pt'),
                                                                'toby': record})
        self.assertIn('rsync --whole-file', script)
        self.assertIn('sha256sum -c -', script)
        self.assertIn('toby-rank1-window-0001-window.pt', script)
        self.assertNotIn('rm ', script)


if __name__ == '__main__':
    unittest.main()
