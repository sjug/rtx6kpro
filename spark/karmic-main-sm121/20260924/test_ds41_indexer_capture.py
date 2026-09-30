"""CPU tests for the layer-2 indexer capture kit (helper, attention seam, lock, installer). No GPU, nodes or builds:
  .venv-snapshot-cpu/bin/python -m unittest test_ds41_indexer_capture
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
import threading
import time
import types
import unittest
from unittest import mock

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ds41_indexer_prepare as prep  # noqa: E402

WINDOW, HIDDEN, HEADS, INDEX_HEADS, LAST = 128, 64, 16, 32, 524287


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fresh_helper():
    return load(f'ds41_indexer_capture_{time.monotonic_ns()}', HERE / 'ds41_indexer_capture.py')


class FakeEvent:
    def record(self):
        pass

    def synchronize(self):
        pass


@dataclasses.dataclass(frozen=True)
class GemvQuery:
    source_dtype: str = 'bfloat16'
    weight_dtype: str = 'bfloat16'
    output_dtype: str = 'bfloat16'
    max_rows: int = 8192
    in_features: int = HIDDEN
    out_features: int = INDEX_HEADS


@dataclasses.dataclass(frozen=True)
class GemvConfig:
    backend: str = 'torch'
    rows_per_tile: int = 8


def plan(max_rows, handle, backend='torch'):
    return types.SimpleNamespace(handle=handle, query=GemvQuery(max_rows=max_rows),
                                 selection=types.SimpleNamespace(config=GemvConfig(backend=backend), source='default'))


class Fixture:
    def __init__(self, seed=3, extra_plans=()):
        self.g = torch.Generator().manual_seed(seed)
        plans = {(torch.bfloat16, 8192): plan(8192, 11), (torch.float32, 8192): plan(8192, 12),
                 (torch.bfloat16, 32): plan(32, 13, 'simt')}
        for rows in extra_plans:
            plans[(torch.bfloat16, rows)] = plan(rows, 20 + rows)
        self.weight = torch.randn(INDEX_HEADS, HIDDEN, generator=self.g).bfloat16()
        proj = types.SimpleNamespace(weight=self.weight, b12x_linear_plans=plans, b12x_linear_capacity=8192)
        self.layer = types.SimpleNamespace(layer_id=2, is_draft=False, hidden_size=HIDDEN, n_local_heads=HEADS,
                                           swa_cache_layer=types.SimpleNamespace(prefix='layers.2.swa'),
                                           indexer=types.SimpleNamespace(heads=INDEX_HEADS, weights_proj=proj))
        self.other = types.SimpleNamespace(**{**vars(self.layer), 'layer_id': 3})
        self.draft = types.SimpleNamespace(**{**vars(self.layer), 'is_draft': True})

    def metadata(self, *, decode=False, reqs=1, tokens=8192, seq=524288):
        m = types.SimpleNamespace(is_decode=decode, num_reqs=reqs, num_actual_tokens=tokens, max_seq_len=seq)
        return {'layers.2.swa': m}

    def tensors(self, rows):
        g = self.g
        t = {'positions': torch.arange(524288 - rows, 524288, dtype=torch.int64)}
        for name, shape in (('hidden', (HIDDEN,)), ('kv', (512,)), ('q', (HEADS, 512)), ('iq', (INDEX_HEADS, 128)),
                            ('weights', (INDEX_HEADS,)), ('iw', (INDEX_HEADS,))):
            full = torch.zeros(rows, *shape, dtype=torch.bfloat16)
            full[-WINDOW:] = torch.randn(WINDOW, *shape, generator=g).bfloat16()
            t[name] = full
        return t

    def call(self, helper, metadata, rows=None, override=None):
        """Execute the generated call site from the composed attention source, not a hand-written copy."""
        rows = rows or metadata['layers.2.swa'].num_actual_tokens
        t = self.tensors(rows)
        t.update(override or {})
        self.forward = t
        tree = ast.parse((HERE / prep.OUTPUT).read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == 'capture' and isinstance(n.func.value, ast.Name) and n.func.value.id == '_indexer_capture']
        if len(calls) != 1:
            raise RuntimeError('Expected one generated indexer capture call')
        code = compile(ast.fix_missing_locations(ast.Expression(calls[0])), '<generated-indexer-hook>', 'eval')
        eval(code, {'_indexer_capture': helper}, {'self': self.layer, 'metadata': metadata, 'positions': t['positions'],
             'hidden_states': t['hidden'], 'kv': t['kv'], 'q': t['q'], 'iq': t['iq'], 'weights': t['weights'],
             'iw': t['iw']})


class HelperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.fixture = Fixture()
        self.helper = self.patch_helper(fresh_helper(), self.fixture)
        self.capturing = False

    def patch_helper(self, h, fixture):
        self.capturing = False
        patches = [mock.patch.object(h, 'TRIGGER', str(self.dir / 'trigger.json')),
                   mock.patch.object(h, 'OUT_DIR', str(self.dir / 'out')),
                   mock.patch.object(h, 'POLL_SECONDS', 0.0),
                   mock.patch.object(h, '_rank', lambda: 0),
                   mock.patch.object(h, '_world_size', lambda: 4),
                   mock.patch.object(h, '_pinned', lambda shape, dtype: torch.empty(shape, dtype=dtype)),
                   mock.patch.object(h, '_capturing', lambda: self.capturing),
                   mock.patch.object(h, '_source_trees', lambda: {'b12x': 't', 'vllm': 'v'}),
                   mock.patch.object(h.torch.cuda, 'Event', FakeEvent),
                   mock.patch.dict(os.environ, {'DS41_NODE': 'dusty', 'DS41_KIT_SHA256': 'kit'})]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        h._IMPORTED_AT = time.time() - 60
        return h

    def tearDown(self):
        self.tmp.cleanup()

    def arm(self, token='indexer-0001', **extra):
        (self.dir / 'trigger.json').write_text(json.dumps({'token': token, 'prompt_tokens': 524288, **extra}))

    def saved(self):
        for thread in list(threading.enumerate()):
            if thread.name == 'ds41-indexer-waiter':
                thread.join(timeout=30)
        return sorted((self.dir / 'out').glob('*.pt')) if (self.dir / 'out').exists() else []

    def test_unarmed_disabled_and_invalid_triggers_do_nothing(self):
        f, h = self.fixture, self.helper
        f.call(h, f.metadata())
        for spec in ({'token': 'BAD token', 'prompt_tokens': 524288}, {'token': 'indexer-0001', 'prompt_tokens': 1},
                     {'token': 'indexer-0001', 'prompt_tokens': 524288, 'indexer': False},
                     {'token': 'indexer-0001', 'prompt_tokens': 524288, 'chunk_rows': 7936}):
            (self.dir / 'trigger.json').write_text(json.dumps(spec))
            f.call(h, f.metadata())
            self.assertIsNone(h._state['armed'])
        self.arm()
        os.utime(self.dir / 'trigger.json', (0, 0))
        f.call(h, f.metadata())
        self.assertIsNone(h._state['armed'])
        self.assertEqual(self.saved(), [])
        self.assertIsNotNone(h.trigger_spec({'token': 'indexer-0001', 'prompt_tokens': 524288, 'window': True}))

    def test_only_layer_2_on_the_decision_chunk(self):
        f, h = self.fixture, self.helper
        self.arm()
        f.layer = f.other
        f.call(h, f.metadata())
        self.assertIsNone(h._state['armed'])
        f.layer = f.draft
        f.call(h, f.metadata())
        self.assertIsNone(h._state['armed'])
        f.layer = Fixture().layer
        for metadata in (f.metadata(decode=True), f.metadata(reqs=2), f.metadata(seq=524288 - 8192)):
            f.call(h, metadata)
            self.assertIsNotNone(h._state['armed'])          # armed on an earlier eligible forward, not captured
            self.assertFalse(h._state['armed'].fired)
        self.assertEqual(self.saved(), [])

    def test_final_chunk_size_mismatch_aborts_with_receipt(self):
        f, h = self.fixture, self.helper
        self.arm(chunk_rows=4096)
        f.call(h, f.metadata())
        self.assertIsNone(h._state['armed'])
        self.assertEqual(self.saved(), [])
        receipt = json.loads((self.dir / 'out' / 'indexer-0001-rank0.indexer-aborted').read_text())
        self.assertIn('final chunk rows 8192 differ from armed 4096', receipt['error'])

    def check_capture(self, capture, rows, lookup):
        f = self.fixture
        h = self.helper
        self.assertEqual(capture['schema'], 'ds41-indexer-capture-v1')
        meta = h.validate(capture)
        self.assertEqual((meta['chunk_rows'], meta['layer'], meta['problems'], meta['tp_world_size'], meta['row_position']),
                         (rows, 2, [], 4, LAST))
        self.assertEqual((meta['hidden'], meta['heads'], meta['index_heads']), (HIDDEN, HEADS, INDEX_HEADS))
        self.assertEqual(meta['window_file'], 'dusty-rank0-indexer-0001-window.pt')
        e, t = capture['layer'], f.forward
        for name, src in (('positions', 'positions'), ('hidden_input', 'hidden'), ('kv_norm', 'kv'), ('q_rotated', 'q'),
                          ('index_query_rotated', 'iq'), ('raw_weights', 'weights'), ('scaled_weights', 'iw')):
            self.assertTrue(torch.equal(e[name], t[src][-WINDOW:]), name)
        self.assertTrue(torch.equal(e['projection_weight'], f.weight))
        digest = hashlib.sha256(bytes(f.weight.contiguous().view(torch.uint8).flatten().tolist())).hexdigest()
        self.assertEqual(e['projection_weight_sha256'], digest)
        self.assertEqual((e['chunk_rows'], e['chunk_row_start']), (rows, rows - WINDOW))
        p = e['plan']
        self.assertEqual((p['lookup'], p['rows'], p['dtype'], p['capacity']), (lookup, rows, 'bfloat16', 8192))
        self.assertEqual(p['query']['max_rows'], rows if lookup == 'rows' else 8192)
        self.assertEqual(p['selection'], {'config': {'backend': 'torch', 'rows_per_tile': 8}, 'source': 'default'})
        self.assertIn(['bfloat16', 8192], p['available_keys'])

    def test_full_capture_records_tensors_plan_and_is_one_shot(self):
        f, h = self.fixture, self.helper
        self.arm()
        f.call(h, f.metadata())
        (path,) = self.saved()
        self.assertEqual(path.name, 'dusty-rank0-indexer-0001-indexer.pt')
        capture = torch.load(path, map_location='cpu', weights_only=True)
        self.check_capture(capture, 8192, 'rows')                # the capacity plan IS the exact-rows plan at 8192
        self.assertEqual(capture['layer']['plan']['plan_handle'], 11)
        receipt = json.loads((self.dir / 'out' / 'indexer-0001-rank0.indexer-consumed').read_text())
        self.assertEqual(receipt['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertIsNone(h._state['armed'])
        f.call(h, f.metadata())                                   # a second decision forward does not re-arm the token
        self.assertIsNone(h._state['armed'])
        again = fresh_helper()
        with mock.patch.object(again, 'OUT_DIR', str(self.dir / 'out')), \
                mock.patch.object(again, 'TRIGGER', str(self.dir / 'trigger.json')):
            again._IMPORTED_AT = 0
            self.assertIsNone(again._read_trigger(0))

    def test_4096_uses_the_exact_rows_plan_when_present_else_capacity(self):
        for extra, lookup, handle in ((), 'capacity', 11), ((4096,), 'rows', 20 + 4096):
            with self.subTest(lookup=lookup):
                fixture = Fixture(seed=5, extra_plans=extra)
                helper = self.patch_helper(fresh_helper(), fixture)
                with mock.patch.object(helper, 'OUT_DIR', str(self.dir / f'out-{lookup}')):
                    self.arm(chunk_rows=4096)
                    fixture.call(helper, fixture.metadata(tokens=4096))
                    for thread in list(threading.enumerate()):
                        if thread.name == 'ds41-indexer-waiter':
                            thread.join(timeout=30)
                    (path,) = sorted((self.dir / f'out-{lookup}').glob('*.pt'))
                capture = torch.load(path, map_location='cpu', weights_only=True)
                self.fixture, self.helper = fixture, helper
                self.check_capture(capture, 4096, lookup)
                self.assertEqual(capture['layer']['plan']['plan_handle'], handle)

    def test_dtype_or_shape_surprises_abort_and_never_raise(self):
        f, h = self.fixture, self.helper
        self.arm()
        bad = f.tensors(8192)
        bad['weights'] = bad['weights'].float()
        f.call(h, f.metadata(), override={'weights': bad['weights']})
        self.assertIsNone(h._state['armed'])
        self.assertEqual(self.saved(), [])
        self.assertIn('raw_weights has shape', (self.dir / 'out' / 'indexer-0001-rank0.indexer-aborted').read_text())

    def test_incomplete_plan_record_is_a_problem_not_evidence(self):
        f, h = self.fixture, self.helper
        proj = f.layer.indexer.weights_proj
        proj.b12x_linear_plans[(torch.bfloat16, 8192)] = types.SimpleNamespace(handle=11, query=None, selection=None)
        self.arm()
        f.call(h, f.metadata())
        (path,) = self.saved()
        capture = torch.load(path, map_location='cpu', weights_only=True)
        self.assertTrue(any('plan query' in p for p in capture['meta']['problems']))
        self.assertTrue(any('selection/backend' in p for p in capture['meta']['problems']))
        receipt = json.loads((self.dir / 'out' / 'indexer-0001-rank0.indexer-consumed').read_text())
        self.assertEqual(receipt['problems'], capture['meta']['problems'])
        with self.assertRaisesRegex(ValueError, 'geometry or reported problems'):
            h.validate(capture)
        capture['meta']['problems'] = []
        with self.assertRaisesRegex(ValueError, 'plan record is not evidence'):
            h.validate(capture)
        good = {'lookup': 'capacity', 'rows': 4096, 'dtype': 'bfloat16', 'capacity': 8192, 'plan_handle': 3,
                'query': {'max_rows': 8192, 'in_features': HIDDEN, 'out_features': INDEX_HEADS},
                'selection': {'config': {'backend': 'torch', 'rows_per_tile': 8}, 'source': 'default'}}
        self.assertEqual(h.plan_problems(good, rows=4096, hidden=HIDDEN, index_heads=INDEX_HEADS), [])
        self.assertTrue(h.plan_problems(dict(good, query=dict(good['query'], max_rows=4096)), rows=4096, hidden=HIDDEN, index_heads=INDEX_HEADS))
        self.assertTrue(h.plan_problems(dict(good, selection={'config': {'backend': ''}, 'source': 'default'}), rows=4096, hidden=HIDDEN, index_heads=INDEX_HEADS))

    def test_no_arming_under_graph_capture(self):
        f, h = self.fixture, self.helper
        self.arm()
        self.capturing = True
        with mock.patch.object(h, '_Session', side_effect=AssertionError('allocated under capture')):
            f.call(h, f.metadata())
        self.assertIsNone(h._state['armed'])

    def test_forward_path_has_no_host_synchronization(self):
        tree = ast.parse((HERE / 'ds41_indexer_capture.py').read_text())
        forbidden = {'item', 'cpu', 'synchronize', 'tolist', 'numpy', 'wait_event', 'wait_stream', 'Stream'}
        forward = {'capture', '_capture', '_d2h', '_tail', '_abort', 'selected_plan', '_asdict'}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in forward:
                calls = {c.func.attr for c in ast.walk(node) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)}
                self.assertFalse(calls & forbidden, (node.name, calls & forbidden))
        text = (HERE / 'ds41_indexer_capture.py').read_text()
        self.assertIn('staged.copy_(value, non_blocking=True)', text)


class Seam(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.window = (HERE / prep.WINDOW_ATTENTION).read_text()
        cls.new = prep.compose(cls.window)

    def test_only_the_guarded_observation_is_added(self):
        import difflib
        diff = [l for l in difflib.ndiff(self.window.splitlines(), self.new.splitlines()) if l.startswith(('- ', '+ '))]
        self.assertFalse([l for l in diff if l.startswith('- ')])
        added = [l[2:].strip() for l in diff]
        self.assertIn('import vllm.models.deepseek_v4_1.ds41_indexer_capture as _indexer_capture', added)
        self.assertEqual(sum('_indexer_capture.capture(' in l for l in added), 1)
        self.assertIn('if self.layer_id == 2:', added)
        t = self.new
        self.assertLess(t.index('plan=self._helper_plan("index_weights"),'), t.index('_indexer_capture.capture('))
        self.assertLess(t.index('_indexer_capture.capture('), t.index('index_query = (iq_data, iq_scale, iw)'))
        prep.verify_only_hook_changed(self.window, self.new)

    def test_numerical_or_extra_changes_are_rejected(self):
        for mutate in (lambda s: s.replace('weights = self.indexer.weights_proj(hidden_states)',
                                           'weights = self.indexer.weights_proj(hidden_states.float())'),
                       lambda s: s.replace('if self.layer_id == 2:', 'if self.layer_id in (2, 8):'),
                       lambda s: s + '\nX = 1\n'):
            with self.assertRaises(ValueError):
                prep.verify_only_hook_changed(self.window, mutate(self.new))
        with self.assertRaises(ValueError):
            prep.compose(self.window + '\n', prep.base_identity()['attention_sha256'])

    def test_lock_dockerfile_and_ignore_reproduce(self):
        lock = json.loads((HERE / 'ds41-indexer.lock.json').read_text())
        for name, digest in lock['inputs'].items():
            self.assertEqual(hashlib.sha256((HERE / name).read_bytes()).hexdigest(), digest, name)
        self.assertEqual(lock['targets'][prep.ATTENTION]['output_sha256'], hashlib.sha256(self.new.encode()).hexdigest())
        self.assertEqual((HERE / prep.OUTPUT).read_text(), self.new)
        self.assertEqual((HERE / 'Dockerfile.ds41-indexer').read_text(), prep.render_dockerfile(lock))
        names = sorted(lock['inputs']) + ['ds41-indexer.lock.json']
        self.assertEqual((HERE / 'ds41-indexer.ignore').read_text(), '**\n' + ''.join('!' + n + '\n' for n in names))
        window = json.loads((HERE / 'claude-window.lock.json').read_text())
        receipt = json.loads((HERE / 'receipts/window-build-receipt.json').read_text())
        self.assertEqual(lock['base_image_id'], receipt['image_id'])
        self.assertEqual(lock['base_image_id'][:8], 'c4a51be4')
        self.assertEqual((lock['base_kind'], lock['base_lock_sha256']), ('window-capture', receipt['lock_sha256']))
        self.assertEqual(lock['targets'][prep.ATTENTION]['input_sha256'], window['targets'][prep.ATTENTION]['output_sha256'])
        self.assertEqual(lock['preserved'], {prep.DECISION_HELPER: window['targets'][prep.DECISION_HELPER]['output_sha256'],
                                             prep.WINDOW_HELPER: window['targets'][prep.WINDOW_HELPER]['output_sha256']})
        self.assertNotEqual(lock['trees']['vllm'], window['trees']['vllm'])
        self.assertEqual(lock['trees']['b12x'], window['trees']['b12x'])
        self.assertEqual((lock['install_dir'], lock['install_pass_marker'], lock['image_kind_label'], lock['build_args']),
                         ('/opt/ds41-indexer', 'INDEXER-INSTALL-PASS', 'indexer-capture', ['DECISION_LOCK', 'DECISION_CACHE']))
        provenance = lock['inherited_helper_provenance']
        self.assertTrue(provenance['disposition'].startswith('metadata-only'))
        self.assertEqual((provenance['original_window_lock'], provenance['original_decision_stub']),
                         ('claude-window.lock.original.json', 'claude-decision-row.lock.original.json'))
        text = (HERE / 'Dockerfile.ds41-indexer').read_text()
        for required in ('FROM ' + receipt['image_id'], 'ARG DECISION_LOCK', 'ARG DECISION_CACHE',
                         '/opt/ds41-indexer/ds41_indexer_install.py', 'diagnostic.kind="indexer-capture"',
                         'window-lock.sha256="' + receipt['lock_sha256']):
            self.assertIn(required, text)
        for line in [l for l in text.splitlines() if l.endswith('\\')]:
            self.assertTrue(line.endswith(' \\') and not line.endswith('\\\\'), line)


class Installer(unittest.TestCase):
    def simulate(self, tamper=None):
        lock = json.loads((HERE / 'ds41-indexer.lock.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            kit, opt, window, decision = Path(tmp) / 'kit', Path(tmp) / 'opt', Path(tmp) / 'window', Path(tmp) / 'decision'
            kit.mkdir()
            window.mkdir()
            decision.mkdir()
            window_lock = json.loads((HERE / 'claude-window.lock.json').read_text())
            (decision / 'claude-decision-row.lock.json').write_text(json.dumps(
                {'kind': 'window-capture-provenance-stub', 'trees': window_lock['trees']}))
            for name in [*lock['inputs'], 'ds41-indexer.lock.json']:
                shutil.copy(HERE / name, kit / name)
            model = opt / 'vllm/vllm/models/deepseek_v4_1'
            model.mkdir(parents=True)
            (model / 'attention.py').write_text((HERE / prep.WINDOW_ATTENTION).read_text())
            shutil.copy(HERE / 'claude-decision-row-capture.py', model / 'claude_decision_row.py')
            shutil.copy(HERE / 'claude-window-capture.py', model / 'claude_window.py')
            fenced = opt / 'b12x' / lock['base_router_target']['path']
            fenced.parent.mkdir(parents=True)
            shutil.copy(HERE / 'router-prefill-release-before.py', fenced)
            (opt / 'b12x/b12x').mkdir(parents=True, exist_ok=True)
            (opt / 'b12x/b12x/native.so').write_bytes(b'x')
            shutil.copy(HERE / 'claude-window.lock.json', window / 'claude-window.lock.json')
            digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            inventory = {str(p): digest(p) for c in ('vllm', 'b12x') for p in (opt / c).rglob('*') if p.is_file()}
            (window / 'after-files.json').write_text(json.dumps(inventory, sort_keys=True) + '\n')
            if tamper:
                tamper(opt=opt, window=window, model=model, decision=decision)
            source = (HERE / 'ds41_indexer_install.py').read_text()
            script = Path(tmp) / 'install.py'
            script.write_text(source.replace("Path('/opt/ds41-indexer')", f"Path({str(kit)!r})")
                              .replace("Path('/opt/jovian-judgement')", f"Path({str(opt)!r})")
                              .replace("Path('/opt/ds41-window')", f"Path({str(window)!r})")
                              .replace("Path('/opt/ds41-decision-row')", f"Path({str(decision)!r})"))
            result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
            files = sorted(p.relative_to(opt).as_posix() for p in opt.rglob('*') if p.is_file())
            provenance = {p.name: p.read_bytes() for d in (window, decision) for p in d.iterdir() if p.is_file()}
            return result, (model / 'attention.py').read_text(), files, provenance

    def test_install_is_confined_to_attention_and_the_new_helper(self):
        result, attention, files, provenance = self.simulate()
        self.assertIn('INDEXER-INSTALL-PASS', result.stdout, result.stderr)
        lock_raw = (HERE / 'ds41-indexer.lock.json').read_bytes()
        lock = json.loads(lock_raw)
        # the original window lock is preserved byte for byte, both read paths now name this image's trees
        self.assertEqual(hashlib.sha256(provenance['claude-window.lock.original.json']).hexdigest(), lock['base_lock_sha256'])
        self.assertEqual(json.loads(provenance['claude-decision-row.lock.original.json'])['kind'], 'window-capture-provenance-stub')
        for name in ('claude-window.lock.json', 'claude-decision-row.lock.json'):
            stub = json.loads(provenance[name])
            self.assertEqual((stub['kind'], stub['trees'], stub['indexer_lock_sha256'], stub['disposition']),
                             ('indexer-capture-provenance-stub', lock['trees'], hashlib.sha256(lock_raw).hexdigest(),
                              lock['inherited_helper_provenance']['disposition']), name)
        self.assertEqual(attention, (HERE / prep.OUTPUT).read_text())
        self.assertEqual(files, ['b12x/b12x/gemm/bf16_gemv/_prefill.py', 'b12x/b12x/native.so',
                                 'vllm/vllm/models/deepseek_v4_1/attention.py',
                                 'vllm/vllm/models/deepseek_v4_1/claude_decision_row.py',
                                 'vllm/vllm/models/deepseek_v4_1/claude_window.py',
                                 'vllm/vllm/models/deepseek_v4_1/ds41_indexer_capture.py'])

    def test_other_bases_are_refused(self):
        cases = {
            'window lock differs': lambda opt, window, model, decision: (window / 'claude-window.lock.json').write_text('{}'),
            'Base helper differs': lambda opt, window, model, decision: (model / 'claude_window.py').write_text('x'),
            'Base attention differs': lambda opt, window, model, decision: (model / 'attention.py').write_text('x'),
            'inventory differs': lambda opt, window, model, decision: (opt / 'vllm/extra.py').write_text('x'),
            'preexisting diagnostic file': lambda opt, window, model, decision: (model / 'ds41_indexer_capture.py').write_text('x'),
            'already exist': lambda opt, window, model, decision: (window / 'claude-window.lock.original.json').write_text('x'),
            'window-capture provenance stub': lambda opt, window, model, decision: (decision / 'claude-decision-row.lock.json').write_text('{"kind": "other"}'),
        }
        for message, tamper in cases.items():
            result, *_ = self.simulate(tamper)
            self.assertNotEqual(result.returncode, 0, message)
            self.assertIn(message, result.stderr, message)


if __name__ == '__main__':
    unittest.main()
