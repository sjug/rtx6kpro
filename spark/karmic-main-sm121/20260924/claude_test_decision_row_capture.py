"""CPU tests for the decision-row capture kit (helper, attention patch, lock, installer). No GPU, nodes or builds.
  .venv-snapshot-cpu/bin/python -m unittest claude_test_decision_row_capture
"""
import ast
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

import torch

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prep = load('claude_decision_prep', HERE / 'claude_prepare_decision_row.py')
audit = load('claude_decision_audit', HERE / 'claude_decision_row_audit.py')
TABLE = audit.layer_table()


def fresh_helper():
    return load(f'claude_decision_row_{time.monotonic_ns()}', HERE / 'claude-decision-row-capture.py')


@dataclasses.dataclass
class Config:
    max_chunks_per_row: int = 1
    single_pass: bool = True
    split_chunk_size: int = 1
    v41_compute_mode: str = 'bf16'
    v41_heads_per_block: int = 16


class FakeEvent:
    def record(self):
        pass

    def synchronize(self):
        pass


def expanded(t, rows=8192):
    return t.unsqueeze(0).expand(rows, *t.shape)


def query_rows(lid, chunk_rows=8192):
    """Production row geometry of the decision chunk: CED decoder layers (>= 20) run the last 128 rows only."""
    return 128 if TABLE[lid]['ced_decoder'] else chunk_rows


class Fixture:
    """Fake target layers, caches and forward tensors shaped like the DS4.1 extend decision chunk."""

    COUNT = {2: 640, 8: 640, 14: 640, 20: 700}

    def __init__(self):
        g = torch.Generator().manual_seed(7)
        self.layers = {}
        k_caches = {}
        for lid, role in TABLE.items():
            layer = types.SimpleNamespace(layer_id=lid, compress_ratio=role['ratio'], _main_page=role['page'],
                                          _index_page=role['page'], _main_width=16, _index_width=16,
                                          kv_source_layer_id=role['kv_owner'],
                                          swa_width=128, is_draft=False, candidate_source_layer=20,
                                          is_ced_decoder=role['ced_decoder'],
                                          attn_sink=torch.randn(16, generator=g), forward_mqa=None)
            layer._query_metadata = (lambda m, layer=layer: m.decoder
                                     if layer.is_ced_decoder and m.decoder is not None else m)
            layer.swa_cache_layer = types.SimpleNamespace(
                prefix=f'layers.{lid}.swa', block_size=128,
                kv_cache=torch.randint(0, 255, (4, 128 * 528), dtype=torch.uint8, generator=g))
            layer.kv_cache = torch.randint(0, 255, (8, role['page'] * 288), dtype=torch.uint8, generator=g)
            layer.topk_indices_buffer = expanded(torch.arange(512, dtype=torch.int32))
            layer.indexer = None
            if role['index_source']:
                owner = role['kv_owner']
                if owner not in k_caches:
                    pages = -(-self.COUNT[owner] // role['page']) + 1
                    k_caches[owner] = torch.randint(0, 255, (pages, role['page'] * 68), dtype=torch.uint8, generator=g)
                layer.indexer = types.SimpleNamespace(k_cache=types.SimpleNamespace(kv_cache=k_caches[owner]),
                                                      owns_k=owner == lid)
                if lid == 20:
                    layer._candidates = expanded(torch.arange(16384, dtype=torch.int32))
                    layer._candidate_lens = expanded(torch.tensor(600, dtype=torch.int32))
            self.layers[lid] = layer
        for lid, layer in self.layers.items():
            role = TABLE[lid]
            layer._owner = (lambda o=self.layers.get(role['kv_owner'], layer): o)
        self.draft = types.SimpleNamespace(layer_id=40, is_draft=True)

    def metadata(self, *, decode=False, reqs=1, tokens=8192, seq=524288, decoder_rows=128, decoder_shift=0):
        """Original metadata plus the CED decoder view (ced.py decoder_metadata: the last 128 rows of the request)."""
        positions = torch.arange(seq - tokens, seq, dtype=torch.int64)
        decoder = None
        if not decode and tokens > decoder_rows:
            kept = positions[tokens - decoder_rows - decoder_shift:tokens - decoder_shift]
            decoder = types.SimpleNamespace(is_decode=False, num_reqs=reqs, num_actual_tokens=decoder_rows,
                                            max_seq_len=seq, positions=kept, decoder=None)
        m = types.SimpleNamespace(is_decode=decode, num_reqs=reqs, num_actual_tokens=tokens, max_seq_len=seq,
                                  positions=positions, decoder=decoder)
        return {layer.swa_cache_layer.prefix: m for layer in self.layers.values()}

    def run_forward(self, helper, metadata, page_override=None, rows_override=None, distinct=False):
        """Call the hooks exactly where the patched forward_mqa calls them, with production row counts per layer."""
        g = torch.Generator().manual_seed(11)
        self.forward_rows = {}
        for lid, layer in self.layers.items():
            capture = helper.begin(layer, metadata)
            if capture is None:
                continue
            role = TABLE[lid]
            n = (rows_override or {}).get(lid, query_rows(lid, metadata[layer.swa_cache_layer.prefix].num_actual_tokens))
            rows = lambda t: (t.unsqueeze(0).expand(n, *t.shape))
            if role['index_source']:
                count = self.COUNT[role['kv_owner']]
                used = -(-count // role['page'])
                pages = torch.full((min(n, 256), 16), -1, dtype=torch.int32)
                pages[:, :used] = torch.arange(1, used + 1, dtype=torch.int32)
                if page_override and lid in page_override:
                    pages[:, :used] = page_override[lid](used)
                capture.indexer(layer, row_in_chunk=min(n, 256) - 1,
                                iq_data=rows(torch.randint(0, 255, (32, 64), dtype=torch.uint8, generator=g)),
                                iq_scale=rows(torch.full((32, 4), 127, dtype=torch.uint8)),
                                iw=rows(torch.rand(32, generator=g).bfloat16()),
                                cache_lengths=rows(torch.tensor(count, dtype=torch.int32)), index_pages=pages)
            main = role['ratio'] != 0
            q = torch.randn(n, 16, 512, generator=g).bfloat16() if distinct else rows(torch.randn(16, 512, generator=g).bfloat16())
            self.forward_rows[lid] = q
            capture.attention(
                layer, state=types.SimpleNamespace(query=types.SimpleNamespace(mode='extend'), config=Config()),
                q=q, output=rows(torch.randn(16, 512, generator=g).bfloat16()),
                swa_indices=rows(torch.arange(128, dtype=torch.int32) + 128),
                swa_lengths=rows(torch.tensor(128, dtype=torch.int32)),
                top_lengths=rows(torch.tensor(512, dtype=torch.int32)),
                page_table=rows(torch.arange(16, dtype=torch.int32) % 7 + 1) if main else None,
                binding=types.SimpleNamespace(scratch=types.SimpleNamespace(
                    mapped_indices=rows(torch.arange(512, dtype=torch.int64) % (8 * role['page'])))),
                owner=layer)


class HelperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.helper = fresh_helper()
        self.fixture = Fixture()
        h = self.helper
        self.patches = [mock.patch.object(h, 'TRIGGER', str(self.dir / 'trigger.json')),
                        mock.patch.object(h, 'OUT_DIR', str(self.dir / 'out')),
                        mock.patch.object(h, 'POLL_SECONDS', 0.0),
                        mock.patch.object(h, '_rank', lambda: 0),
                        mock.patch.object(h, '_pinned', lambda shape, dtype: torch.empty(shape, dtype=dtype)),
                        mock.patch.object(h, '_target_layers', lambda: dict(self.fixture.layers)),
                        mock.patch.object(h, '_device_empty', lambda shape, dtype, device: torch.empty(shape, dtype=dtype)),
                        mock.patch.object(h, '_capturing', lambda: self.capturing),
                        mock.patch.object(h, '_source_trees', lambda: {'b12x': 't', 'vllm': 'v'}),
                        mock.patch.object(h.torch.cuda, 'Event', FakeEvent),
                        mock.patch.dict(os.environ, {'DS41_NODE': 'dusty', 'DS41_KIT_SHA256': 'kit'})]
        for p in self.patches:
            p.start()
        self.capturing = False
        h._IMPORTED_AT = time.time() - 60

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def arm(self, token='decision-0001', **extra):
        (self.dir / 'trigger.json').write_text(json.dumps({'token': token, 'prompt_tokens': 524288, **extra}))

    def saved(self):
        for thread in list(__import__('threading').enumerate()):
            if thread.name == 'claude-decision-row-waiter':
                thread.join(timeout=30)
        return sorted((self.dir / 'out').glob('*.pt')) if (self.dir / 'out').exists() else []

    def test_unarmed_and_invalid_triggers_do_nothing(self):
        layer, metadata = self.fixture.layers[0], self.fixture.metadata()
        self.assertIsNone(self.helper.begin(layer, metadata))
        for spec in ({'token': 'BAD token', 'prompt_tokens': 524288}, {'token': 'decision-0001', 'prompt_tokens': 1}):
            (self.dir / 'trigger.json').write_text(json.dumps(spec))
            self.assertIsNone(self.helper.begin(layer, metadata))
        self.arm()
        os.utime(self.dir / 'trigger.json', (0, 0))                       # older than import: stale
        self.assertIsNone(self.helper.begin(layer, metadata))

    def test_only_the_decision_chunk_of_target_layers_is_captured(self):
        self.arm()
        layer = self.fixture.layers[3]
        for metadata in (self.fixture.metadata(decode=True), self.fixture.metadata(reqs=2),
                         self.fixture.metadata(tokens=4096, seq=524288 - 8192), self.fixture.metadata(seq=524288 - 8192)):
            self.assertIsNone(self.helper.begin(layer, metadata))
        self.assertIsNone(self.helper.begin(self.fixture.draft, self.fixture.metadata()))
        self.assertIsNotNone(self.helper.begin(layer, self.fixture.metadata()))

    def snapshot_caches(self):
        layers = self.fixture.layers
        return {lid: {'swa': l.swa_cache_layer.kv_cache.clone(), 'kv': l.kv_cache.clone(),
                      'k': l.indexer.k_cache.kv_cache.clone() if l.indexer else None} for lid, l in layers.items()}

    def test_explicit_4096_capture_keeps_ced_geometry(self):
        self.arm(chunk_rows=4096)
        self.assertIsNone(self.helper.begin(self.fixture.layers[0], self.fixture.metadata(seq=524288 - 8192)))
        self.fixture.run_forward(self.helper, self.fixture.metadata(tokens=4096))
        (path,) = self.saved()
        capture = torch.load(path, map_location='cpu', weights_only=True)
        self.assertEqual(capture['meta']['chunk_rows'], 4096)
        self.assertEqual(capture['meta']['row_index'], 4095)
        self.assertEqual(capture['meta']['problems'], [])
        self.assertEqual(audit.validate(capture, TABLE, 0,
                         {'kit_sha256': 'kit', 'source_trees': {'b12x': 't', 'vllm': 'v'}}), [])
        for lid, entry in capture['layers'].items():
            rows = 128 if lid >= 20 else 4096
            self.assertEqual((entry['query_rows'], entry['row'], entry['position']),
                             (rows, rows - 1, 524287))

    def test_invalid_chunk_rows_do_not_arm(self):
        for value in (0, 128, 7936, '4096', 4096.0, True, None):
            self.arm(chunk_rows=value)
            with mock.patch.object(self.helper, '_Session', side_effect=AssertionError('invalid trigger allocated')):
                self.assertIsNone(self.helper._read_trigger(0))
                self.assertIsNone(self.helper.begin(self.fixture.layers[0], self.fixture.metadata()))
            self.assertIsNone(self.helper._state['armed'])

    def test_final_chunk_size_mismatch_aborts_with_receipt(self):
        self.arm(chunk_rows=4096)
        self.assertIsNone(self.helper.begin(self.fixture.layers[0], self.fixture.metadata()))
        self.assertIsNone(self.helper._state['armed'])
        self.assertEqual(self.saved(), [])
        receipt = json.loads((self.dir / 'out' / 'decision-0001-rank0.aborted').read_text())
        self.assertIn('final chunk rows 8192 differ from armed 4096', receipt['error'])

    def reuse_all_blocks(self):
        """The request ends and its blocks are reused: every cache byte changes."""
        for layer in self.fixture.layers.values():
            layer.swa_cache_layer.kv_cache.fill_(0xA5)
            layer.kv_cache.fill_(0x5A)
            if layer.indexer:
                layer.indexer.k_cache.kv_cache.fill_(0x3C)

    def test_completes_without_later_forward_and_survives_block_reuse(self):
        self.arm()
        before = self.snapshot_caches()
        self.fixture.run_forward(self.helper, self.fixture.metadata())
        self.reuse_all_blocks()                                   # zero begin() calls after the decision forward
        (path,) = self.saved()
        capture = torch.load(path, map_location='cpu', weights_only=True)
        observed = {'kit_sha256': 'kit', 'source_trees': {'b12x': 't', 'vllm': 'v'}}
        self.assertEqual(audit.validate(capture, TABLE, 0, observed), [])
        self.assertEqual((capture['meta']['row_position'], capture['meta']['problems']), (524287, []))
        receipt = json.loads((self.dir / 'out' / 'decision-0001-rank0.consumed').read_text())
        self.assertEqual(receipt['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
        for lid in (2, 20):                                       # staged bytes are the pre-reuse bytes
            ix = capture['layers'][lid]['indexer']
            used = -(-ix['cache_length'] // ix['page_size'])
            self.assertTrue(torch.equal(ix['key_pages'], before[lid]['k'][1:used + 1]))
        swa = capture['layers'][5]['swa_records']
        self.assertTrue(torch.equal(swa[0], before[5]['swa'][1, :528]))  # slot 128 = block 1, offset 0
        ix20, ix24 = capture['layers'][20]['indexer'], capture['layers'][24]['indexer']
        self.assertEqual(ix20['key_pages'].data_ptr(), ix24['key_pages'].data_ptr())   # shared owner saved once
        self.assertEqual(capture['layers'][9]['plan']['config']['v41_compute_mode'], 'bf16')
        self.assertIsNone(self.helper._state['armed'])                # disarmed after completion
        again = fresh_helper()                                    # one-shot across processes
        with mock.patch.object(again, 'OUT_DIR', str(self.dir / 'out')), \
                mock.patch.object(again, 'TRIGGER', str(self.dir / 'trigger.json')):
            again._IMPORTED_AT = 0
            self.assertIsNone(again._read_trigger(0))

    def test_ced_decoder_layers_capture_their_last_row(self):
        """Regression: layers >= 20 compute only the gathered last 128 rows; row 8191 does not exist there."""
        self.arm()
        self.fixture.run_forward(self.helper, self.fixture.metadata(), distinct=True)
        (path,) = self.saved()
        capture = torch.load(path, map_location='cpu', weights_only=True)
        self.assertEqual(capture['meta']['problems'], [])
        self.assertEqual(capture['schema'], 'claude-decision-row-v3')
        for lid in (0, 5, 19):
            entry = capture['layers'][lid]
            self.assertEqual((entry['ced_decoder'], entry['query_rows'], entry['row'], entry['position']),
                             (False, 8192, 8191, 524287))
            self.assertTrue(torch.equal(entry['q'], self.fixture.forward_rows[lid][8191]))
        for lid in (20, 25, 39):
            entry = capture['layers'][lid]
            self.assertEqual((entry['ced_decoder'], entry['query_rows'], entry['row'], entry['position']),
                             (True, 128, 127, 524287))
            self.assertTrue(torch.equal(entry['q'], self.fixture.forward_rows[lid][127]))
        self.assertEqual(audit.validate(capture, TABLE, 0, {'kit_sha256': 'kit', 'source_trees': {'b12x': 't', 'vllm': 'v'}}), [])

    def test_unexpected_row_geometry_aborts_without_saving(self):
        """A decoder layer handed the wrong number of rows (any absolute-index assumption) never captures."""
        self.arm()
        with mock.patch('builtins.print'):
            self.fixture.run_forward(self.helper, self.fixture.metadata(decoder_rows=64))   # query metadata says 64
        self.assertEqual(self.saved(), [])
        self.assertIsNone(self.helper._state['armed'])
        receipt = (self.dir / 'out' / 'decision-0001-rank0.aborted').read_text().splitlines()
        self.assertTrue(receipt and 'query rows 64' in json.loads(receipt[0])['error'], receipt)
        self.helper._state['done'].clear()
        self.arm('decision-0002')
        with mock.patch('builtins.print'):
            self.fixture.run_forward(self.helper, self.fixture.metadata(), rows_override={25: 8192})   # tensors disagree
        self.assertEqual(self.saved(), [])
        self.assertIsNone(self.helper._state['armed'])

    def test_decoder_rows_not_ending_at_the_decision_position_are_reported(self):
        self.arm()
        self.fixture.run_forward(self.helper, self.fixture.metadata(decoder_shift=1))
        (path,) = self.saved()
        capture = torch.load(path, map_location='cpu', weights_only=True)
        self.assertEqual(capture['layers'][30]['position'], 524286)
        self.assertEqual(len(capture['meta']['problems']), 20)
        self.assertIn('layer 20: captured row position 524286 is not the decision row', capture['meta']['problems'])
        problems = audit.validate(capture, TABLE, 0, {'kit_sha256': 'kit', 'source_trees': {'b12x': 't', 'vllm': 'v'}})
        self.assertTrue(any('captured position 524286' in p for p in problems))

    def test_consumer_with_a_different_page_table_is_not_deduplicated(self):
        self.arm()
        self.fixture.run_forward(self.helper, self.fixture.metadata(),
                                 page_override={28: lambda used: torch.arange(used, 0, -1, dtype=torch.int32)})
        (path,) = self.saved()
        capture = torch.load(path, map_location='cpu', weights_only=True)
        self.assertEqual(capture['meta']['problems'], ['layer 28: page table differs from key owner 20'])
        self.assertNotIn('key_pages', capture['layers'][28]['indexer'])
        problems = audit.validate(capture, TABLE, 0, {'kit_sha256': 'kit', 'source_trees': {'b12x': 't', 'vllm': 'v'}})
        self.assertTrue(any('capture reported problems' in p for p in problems))

    def test_no_arming_or_allocation_under_graph_capture(self):
        self.arm()
        self.capturing = True
        with mock.patch.object(self.helper, '_Session', side_effect=AssertionError('allocated while capturing')):
            self.assertIsNone(self.helper.begin(self.fixture.layers[0], self.fixture.metadata()))
        self.assertIsNone(self.helper._state['armed'])
        self.capturing = False
        self.assertIsNotNone(self.helper.begin(self.fixture.layers[0], self.fixture.metadata()))

    def test_incomplete_forward_never_saves_and_disarms(self):
        self.arm()
        layers = self.fixture.layers
        self.fixture.layers = {k: v for k, v in layers.items() if k != 17}
        self.fixture.run_forward(self.helper, self.fixture.metadata())   # layer 39 runs; 17 never did
        self.fixture.layers = layers
        self.assertEqual(self.saved(), [])
        self.assertIsNone(self.helper._state['armed'])

    def test_hook_errors_never_reach_the_forward_and_never_save(self):
        self.arm()
        broken = self.fixture.layers[11]
        original = broken.swa_cache_layer
        broken.swa_cache_layer = types.SimpleNamespace(prefix=original.prefix, block_size=128)   # no kv_cache
        with mock.patch('builtins.print'):
            self.fixture.run_forward(self.helper, self.fixture.metadata())     # must not raise
        broken.swa_cache_layer = original
        self.assertEqual(self.saved(), [])
        self.assertIsNone(self.helper._state['armed'])
        self.assertIn('decision-0001', self.helper._state['done'])

    def test_arming_errors_are_contained(self):
        self.arm()
        with mock.patch.object(self.helper, '_target_layers', side_effect=RuntimeError('layer registry')), \
                mock.patch('builtins.print'):
            self.assertIsNone(self.helper.begin(self.fixture.layers[0], self.fixture.metadata()))
        self.assertIsNone(self.helper._state['armed'])


class ForwardDiscipline(unittest.TestCase):
    FORBIDDEN = {'item', 'cpu', 'synchronize', 'tolist', 'numpy', 'wait_event', 'wait_stream', 'Stream'}

    def test_forward_path_has_no_host_synchronization(self):
        tree = ast.parse((HERE / 'claude-decision-row-capture.py').read_text())
        forward = {'begin', '_begin', 'expect', '_row', 'indexer', 'attention', '_d2h', '_gather', 'layer', 'stage_keys',
                   'finish', '_abort', 'wrapper'}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in forward:
                calls = {c.func.attr for c in ast.walk(node)
                         if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)}
                self.assertFalse(calls & self.FORBIDDEN, (node.name, calls & self.FORBIDDEN))

    def test_d2h_is_non_blocking_and_waiter_touches_no_device_cache(self):
        text = (HERE / 'claude-decision-row-capture.py').read_text()
        self.assertIn("self.staged[name].copy_(value, non_blocking=True)", text)
        self.assertIn("host.copy_(view, non_blocking=True)", text)
        tree = ast.parse(text)
        host = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'host')
        names = {a.attr for a in ast.walk(host) if isinstance(a, ast.Attribute)}
        self.assertFalse(names & {'kv_cache', 'k_cache', 'index_select'}, names)


class Dockerfiles(unittest.TestCase):
    def test_single_backslash_continuations(self):
        for name in ('Dockerfile.claude-decision-row', 'Dockerfile.claude-engram-fault-inject'):
            lines = (HERE / name).read_text().splitlines()
            continued = [l for l in lines if l.endswith('\\')]
            self.assertTrue(continued, name)
            for line in continued:
                self.assertTrue(line.endswith(' \\') and not line.endswith('\\\\'), (name, line))
            env = lines.index(next(l for l in lines if l.startswith('ENV ')))
            label = lines.index(next(l for l in lines if l.startswith('LABEL ')))
            self.assertEqual(lines[env].endswith('\\'), True)
            self.assertFalse(lines[env + 1].endswith('\\'), name)   # ENV has exactly two lines
            end = next(i for i in range(label, len(lines)) if not lines[i].endswith('\\'))
            self.assertTrue(lines[end].startswith('    org.opencontainers.image.title='), name)
            self.assertTrue(all(lines[i].startswith('    ') for i in range(label + 1, end + 1)), name)


class Patch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = prep.upstream_attention().decode()
        cls.new = prep.compose(cls.old)

    def test_only_guarded_additions(self):
        old, new = self.old.splitlines(), self.new.splitlines()
        self.assertEqual([l for l in new if l not in old or new.count(l) != old.count(l)][:0], [])
        diff = [l for l in __import__('difflib').ndiff(old, new) if l.startswith(('- ', '+ '))]
        self.assertFalse([l for l in diff if l.startswith('- ')])               # nothing removed or changed
        added = [l[2:].strip() for l in diff]
        self.assertEqual(added[:2], ['# DIAGNOSTIC ONLY: decision-row capture (claude-decision-row-audit-DESIGN.md).',
                                     'import vllm.models.deepseek_v4_1.claude_decision_row as _claude_row'])
        self.assertEqual(sum('_claude_row.begin(self, metadata)' in l for l in added), 1)
        self.assertEqual([l for l in added if l.startswith('if _claude')],
                         ['if _claude is not None and end == rows:', 'if _claude is not None:'])

    def test_begin_follows_state_and_hooks_follow_their_producers(self):
        text = self.new
        self.assertLess(text.index('state = require_prepared(plan, "attention.compressed_sparse_mla")'),
                        text.index('_claude = _claude_row.begin(self, metadata)'))
        self.assertLess(text.index('dsa_indexer.select(binding)'), text.index('_claude.indexer('))
        self.assertLess(text.index('cache_format="deepseek_v41",\n        )\n        if _claude'),
                        text.index('_claude.attention('))

    def test_lock_and_dockerfile_reproduce(self):
        lock = json.loads((HERE / 'claude-decision-row.lock.json').read_text())
        for name, digest in lock['inputs'].items():
            self.assertEqual(hashlib.sha256((HERE / name).read_bytes()).hexdigest(), digest, name)
        self.assertEqual(lock['targets'][prep.ATTENTION]['output_sha256'],
                         hashlib.sha256(self.new.encode()).hexdigest())
        self.assertEqual((HERE / prep.OUTPUT).read_text(), self.new)
        self.assertEqual((HERE / 'Dockerfile.claude-decision-row').read_text(), prep.render_dockerfile(lock))
        self.assertEqual(lock['base_image_id'], 'e06df11a8ca18fa514d9f28f67cc691aef296da2eeb22b113a734519853bccd7')
        router = json.loads((HERE / 'router-release.lock.json').read_text())
        self.assertNotEqual(lock['trees']['vllm'], router['trees']['vllm'])
        self.assertEqual(lock['trees']['b12x'], router['trees']['b12x'])
        self.assertIn('not numerics', lock['scope'])


class Installer(unittest.TestCase):
    def simulate(self, tamper=None):
        lock = json.loads((HERE / 'claude-decision-row.lock.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            kit, opt, router = Path(tmp) / 'kit', Path(tmp) / 'opt', Path(tmp) / 'router'
            kit.mkdir()
            router.mkdir()
            for name in [*lock['inputs'], 'claude-decision-row.lock.json']:
                shutil.copy(HERE / name, kit / name)
            attention = opt / prep.ATTENTION
            attention.parent.mkdir(parents=True)
            attention.write_bytes(prep.upstream_attention())
            fenced = opt / 'b12x' / lock['base_router_target']['path']
            fenced.parent.mkdir(parents=True)
            shutil.copy(HERE / 'router-prefill-release-before.py', fenced)
            (opt / 'b12x/b12x/native.so').write_bytes(b'x')
            shutil.copy(HERE / 'router-release.lock.json', router / 'router-release.lock.json')
            digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            preserved = {str(p): digest(p) for c in ('vllm', 'b12x') for p in (opt / c).rglob('*') if p.is_file()}
            (router / 'preserved-files.json').write_text(json.dumps(preserved, sort_keys=True) + '\n')
            if tamper:
                tamper(opt=opt, router=router, fenced=fenced)
            source = (HERE / 'claude_install_decision_row.py').read_text()
            script = Path(tmp) / 'install.py'
            script.write_text(source.replace("Path('/opt/ds41-decision-row')", f"Path({str(kit)!r})")
                              .replace("Path('/opt/jovian-judgement')", f"Path({str(opt)!r})")
                              .replace("Path('/opt/ds41-router-release')", f"Path({str(router)!r})"))
            result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
            return result, attention.read_text()

    def test_install_is_confined(self):
        result, attention = self.simulate()
        self.assertIn('DECISION-ROW-INSTALL-PASS', result.stdout, result.stderr)
        self.assertEqual(attention, (HERE / prep.OUTPUT).read_text())

    def test_non_router_bases_are_refused(self):
        for message, tamper in {
                'router source differs': lambda opt, router, fenced: shutil.copy(
                    HERE / 'claude-pinned-bf16_gemv-_prefill.py', fenced),
                'inventory differs': lambda opt, router, fenced: (opt / 'vllm/extra.py').write_text('x')}.items():
            result, _ = self.simulate(tamper)
            self.assertNotEqual(result.returncode, 0, message)
            self.assertIn(message, result.stderr)


if __name__ == '__main__':
    unittest.main()
