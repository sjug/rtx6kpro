"""Local tests for the expanded-mHC real-input capture kit (no GPU, no nodes, no image build).

Covers: the capture helper end to end on CPU fakes (trigger, decision filter, wrapped pre passthrough,
staging, per-rank records and digests, fail-open); image preparation and the installer on a temporary
tree; the comparator's gates and classification; the recorder's pure steps and a faked record; and
the unapplied integration patch (ds41-mhc-expanded-capture.patch) on temporary copies, which must fail
loudly unless the live files are the reviewed base or the exact patched tree.

    .venv-snapshot-cpu/bin/python -m unittest test_mhc_expanded_capture
"""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import torch
import torch.nn as nn

KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT))
# Kit modules first: the comparator and replay strip their own directory from sys.path when loaded.
import claude_compare_selections  # noqa: E402,F401
import run_selection_replay  # noqa: E402,F401
import run_selection_transplant  # noqa: E402,F401
import seed_selection_replay  # noqa: E402,F401
import seed_selection_transplant  # noqa: E402,F401

PATCH = KIT / 'ds41-mhc-expanded-capture.patch'
BASES = {'launch_contract.py': '9c1d8615e8e07f851e3e8735878bb319189810fcee36e19b8d73a483238f62f2',
         'run_node.py': '42a6005fe16f87434705d578550c303c25d036016ca36f17f2832b2ffbb58682',
         'start_moe_repaired.py': '6c700bdaac753d5d295d12a999a8f56da364e539cf05c8ff9548ecc2989e02da',
         # In-container KV-block guard (the user's addition, carried in the patch so base + patch is the tree).
         'runtime.py': 'a826e97e5c52f6a9a823a42f4b68659f5ff9fa9ff42c35861badf62a889ad270'}
SUITES = ('test_launch_contract', 'test_precision_orchestration', 'test_start_diagnostic_profile',
          'test_chunking_contract', 'test_router_fixed_control', 'test_window_orchestration',
          'test_indexer_orchestration', 'test_distribute_diagnostic')
RELEASE = '1989e16daf38d2966b03a2e2abcb878263fb7d17183c8d6c5b084a51fbed7e8f'
H, ROWS = 8, 8192


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helper = load(KIT / 'claude-mhc-expanded-capture.py', 'claude_mhc_expanded_under_test')
compare = load(KIT / 'ds41_mhc_expanded_compare.py', 'ds41_mhc_expanded_compare_under_test')
prepare = load(KIT / 'claude-mhc-expanded-prepare.py', 'claude_mhc_expanded_prepare_under_test')
installer = load(KIT / 'claude-mhc-expanded-install.py', 'claude_mhc_expanded_install_under_test')
recorder = load(KIT / 'run_mhc_expanded_capture.py', 'run_mhc_expanded_capture_under_test')
replay = compare.load_replay()
sys.path.insert(0, str(KIT))


# ---------------------------------------------------------------- capture helper on CPU fakes

class FakeMHC(nn.Module):
    """The pinned B12xMHC call shape (vLLM 1794dcf1 b12x_layers.py): post_pre calls self.pre with
    previous_output=x, so an instance wrapper on pre sees every fused call too."""
    hidden_size = H

    def __init__(self):
        super().__init__()
        self.calls = []
        self._plans = {}

    def pre(self, residual, fn, scale, base, norm, pre, *, previous_output=None, previous_post=None,
            previous_comb=None):
        self.calls.append('fused' if previous_output is not None else 'unfused')
        rows = residual.shape[0]
        streams = residual if residual.dim() == 3 else residual[:, None, :].expand(-1, 4, -1)
        y = (pre.unsqueeze(-1) * streams.float()).sum(1).to(torch.bfloat16)
        return (streams.clone(), torch.full((rows, 4), 0.5), torch.full((rows, 4, 4), 0.25), y, pre + 1)

    def post_pre(self, x, residual, post, comb, fn, scale, base, norm, pre):
        return self.pre(residual, fn, scale, base, norm, pre, previous_output=x, previous_post=post,
                        previous_comb=comb)

    def _plan_for(self, operation, tokens):
        selection = SimpleNamespace(component_id='norm.mhc', source='cached', query={'operation': operation,
                                    'max_tokens': 8192}, config={'backend': 'tf32_tma'})
        return SimpleNamespace(selection=selection)


class FakeAttn(nn.Module):
    def __init__(self, layer_id, draft=False):
        super().__init__()
        self.layer_id, self.is_draft = layer_id, draft
        self.swa_cache_layer = SimpleNamespace(prefix=f'layers.{layer_id}.attn')

    def forward(self, positions, x, _unused, global_kv_ready=None):
        return x * 2


class FakeLayer(nn.Module):
    """The engram decoder-layer order at 1794dcf1: unfused pre, attention, fused post_pre."""

    def __init__(self, layer_id, engram=True, draft=False, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(100 + layer_id + seed)
        self.engram = nn.Identity() if engram else None
        self.attn, self._b12x_mhc = FakeAttn(layer_id, draft), FakeMHC()
        self.fn, self.scale, self.base = torch.randn(24, 4 * H, generator=g), torch.randn(3, generator=g), torch.randn(24, generator=g)
        self.norm = torch.ones(H, dtype=torch.bfloat16)
        self.ffn_norm = SimpleNamespace(weight=torch.full((H,), 2.0, dtype=torch.bfloat16))

    def forward(self, hidden, positions, input_ids, pre_mix, residual):
        residual_out, post, comb, y, attn_pre = self._b12x_mhc.pre(
            residual, self.fn, self.scale, self.base, self.norm, pre_mix,
            previous_output=None, previous_post=None, previous_comb=None)
        x = self.attn(positions, y, None, global_kv_ready=None)
        fused = self._b12x_mhc.post_pre(x, residual_out, post, comb, self.fn, self.scale, self.base,
                                        self.ffn_norm.weight, attn_pre)
        return x, (residual_out, post, comb, y, attn_pre), fused


def metadata(rows=ROWS, seq=524288, reqs=1, decode=False):
    return SimpleNamespace(is_decode=decode, num_reqs=reqs, max_seq_len=seq, num_actual_tokens=rows)


class Helper(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.trigger, self.out = root / 'trigger.json', root / 'out'
        self.rank, self.available = 0, 64 << 30
        self.forward_context = SimpleNamespace(attn_metadata={})
        vllm = ModuleType('vllm')
        context = ModuleType('vllm.forward_context')
        context.get_forward_context = lambda: self.forward_context
        self.patches = [mock.patch.dict(sys.modules, {'vllm': vllm, 'vllm.forward_context': context}),
                        mock.patch.object(helper, 'TRIGGER', str(self.trigger)),
                        mock.patch.object(helper, 'OUT_DIR', str(self.out)),
                        mock.patch.object(helper, '_IMPORTED_AT', time.time() - 5),
                        mock.patch.object(helper, 'POLL_SECONDS', 0.0),
                        mock.patch.object(helper, '_rank', lambda: self.rank),
                        mock.patch.object(helper, '_world_size', lambda: 4),
                        mock.patch.object(helper, 'mem_available', lambda path=None: self.available),
                        mock.patch.object(helper, '_event', lambda: SimpleNamespace(synchronize=lambda: None)),
                        mock.patch.dict(os.environ, {'DS41_KIT_SHA256': 'k' * 64})]
        for patch in self.patches:
            patch.start()
        self.reset()

    def reset(self):
        helper._state.update(armed=None, checked=0.0, done=set(), layers={})
        self.layers = [FakeLayer(0, engram=False), FakeLayer(1), FakeLayer(2, engram=False), FakeLayer(14),
                       FakeLayer(3, draft=True)]
        self.model = SimpleNamespace(layers=self.layers)

    def tearDown(self):
        for patch in reversed(self.patches):
            patch.stop()
        helper._state.update(armed=None, checked=0.0, done=set(), layers={})
        self.tmp.cleanup()

    def arm(self, token='mhc-expanded-test-0001', layer=1, reserve=1024):
        self.trigger.write_text(json.dumps({'token': token, 'prompt_tokens': 524288, 'chunk_rows': 8192,
                                            'layer': layer, 'reserve_mib': reserve}))
        return token

    def forward(self, meta=None, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.forward_context.attn_metadata = {f'layers.{l}.attn': meta or metadata() for l in (0, 1, 2, 3, 14)}
        positions = torch.arange(524288 - ROWS, 524288)
        results = {}
        for layer in (self.layers[1], self.layers[3]):
            residual = torch.randn((ROWS, 4, H), generator=g).to(torch.bfloat16)
            pre_mix = torch.softmax(torch.randn((ROWS, 4), generator=g), -1)
            with mock.patch.dict(os.environ, {'DS41_NODE': compare.NODES[self.rank]}):
                results[layer.attn.layer_id] = (residual, pre_mix, layer(None, positions, None, pre_mix, residual))
        return results

    def receipt(self, token, rank):
        consumed = self.out / f'{token}-rank{rank}.mhc-consumed'
        aborted = self.out / f'{token}-rank{rank}.mhc-aborted'
        if aborted.exists():
            self.fail(aborted.read_text())
        self.assertTrue(consumed.exists(), 'capture did not complete')
        return json.loads(consumed.read_text())

    def wait_threads(self):
        for thread in threading.enumerate():
            if thread.name == 'claude-mhc-expanded-waiter':
                thread.join(20)

    def test_trigger_and_decision_rules(self):
        good = {'token': 'mhc-expanded-x1', 'prompt_tokens': 524288, 'chunk_rows': 8192, 'layer': 1, 'reserve_mib': 1024}
        self.assertEqual(helper.trigger_spec(good, {1, 14}), good)
        for bad in (dict(good, token='UPPER-case'), dict(good, chunk_rows=4096), dict(good, prompt_tokens=524287),
                    dict(good, extra=1), dict(good, chunk_rows=8192.0), dict(good, layer=2), dict(good, reserve_mib=100),
                    dict(good, reserve_mib=20000), {k: v for k, v in good.items() if k != 'reserve_mib'}, [good]):
            self.assertIsNone(helper.trigger_spec(bad, {1, 14}), bad)
        self.assertTrue(helper.is_decision(metadata(), ROWS, good))
        for meta, rows in ((metadata(decode=True), ROWS), (metadata(reqs=2), ROWS), (metadata(seq=524287), ROWS),
                           (metadata(rows=4096), ROWS), (metadata(), 4096)):
            self.assertFalse(helper.is_decision(meta, rows, good))

    def test_memory_plan_quantified(self):
        plan = helper.memory_plan(5120, 8192, 4)
        residual_shard = 2048 * 4 * 5120 * 2
        self.assertEqual([r['host_registered_bytes'] for r in plan['ranks']][1], 84131840)
        self.assertGreater(plan['ranks'][1]['host_registered_bytes'], residual_shard)
        self.assertLess(plan['ranks'][1]['host_registered_bytes'], residual_shard + (1 << 20))
        self.assertEqual(plan['device_digest_transient_bound_bytes'], 15 << 20)
        self.assertLess(plan['max_peak_bytes'], 104 << 20)       # about 103 MiB on the last rank, every rank < 110 MB
        lock = json.loads((KIT / 'claude-mhc-expanded.lock.json').read_text())
        self.assertEqual(lock['memory_plan'], {'hidden': 5120, 'rows': 8192, **plan})
        offsets, total = helper.layout_offsets(helper.staging_layout(5120, 8192, 3, 4))
        self.assertEqual(set(offsets), set(helper.ROW_SHARDED) | set(helper.TAILS))
        self.assertEqual(set(helper.layout_offsets(helper.staging_layout(5120, 8192, 0, 4))[0]),
                         set(helper.ROW_SHARDED) | set(helper.WEIGHTS))
        spans = sorted((o, o + b) for o, b, _, _ in offsets.values())
        self.assertTrue(all(a[1] <= b[0] for a, b in zip(spans, spans[1:])))
        self.assertLessEqual(spans[-1][1], total)

    def test_digest_bounded_and_order_free(self):
        tensor = torch.randn(3000, 700).to(torch.bfloat16)
        with mock.patch.object(helper, 'DIGEST_BLOCK_BYTES', 4096):
            small = helper.digest(tensor)
        whole = helper.digest(tensor)
        self.assertTrue(torch.equal(small, whole))
        self.assertTrue(torch.equal(whole, compare.digest(tensor)))
        strided = torch.randn(64, 40)[:, ::2]
        self.assertTrue(torch.equal(helper.digest(strided), helper.digest(strided.contiguous())))
        moved = tensor.clone()
        moved[5, 7], moved[5, 8] = tensor[5, 8], tensor[5, 7]
        if not torch.equal(moved, tensor):
            self.assertNotEqual(helper.digest(moved)[1].item(), whole[1].item())
        words = tensor.reshape(-1).view(torch.int32).to(torch.int64)
        idx = torch.arange(words.numel()) % 65521 + 1
        self.assertEqual(whole.tolist(), [int(words.sum()), int((words * idx).sum())])

    def test_write_region_streams(self):
        region = torch.arange(3 * (1 << 20) + 17, dtype=torch.int64).view(torch.uint8)[:5 * (1 << 20) + 3]
        with mock.patch.object(helper, 'WRITE_SLICE', 1 << 20):
            got = helper.write_region(str(self.out.parent / 'r.bin'), region)
        data = (self.out.parent / 'r.bin').read_bytes()
        self.assertEqual((got, len(data)), (sha(data), region.numel()))
        self.assertEqual(data, bytes(region.tolist()))

    def test_install_and_fused_passthrough(self):
        helper.install(self.model)
        helper.install(self.model)  # idempotent
        self.assertEqual(sorted(helper._state['layers']), [1, 14])
        for layer in self.layers:
            self.assertEqual('pre' in vars(layer._b12x_mhc), layer.attn.layer_id in (1, 14), layer.attn.layer_id)
        for layer_id, (residual, pre_mix, (attn, outputs, fused)) in self.forward().items():
            reference = FakeMHC()
            expected = reference.pre(residual, None, None, None, None, pre_mix)
            for got, want in zip(outputs, expected):
                self.assertTrue(torch.equal(got, want))
        self.assertEqual(self.layers[1]._b12x_mhc.calls, ['unfused', 'fused'])
        self.assertIsNone(helper._state['armed'])
        self.assertFalse(self.out.exists())

    def test_fused_post_pre_ignored_before_and_during_decision(self):
        helper.install(self.model)
        token = self.arm('mhc-expanded-test-0007')
        reached = []
        original = helper._after_pre
        with mock.patch.object(helper, '_after_pre', lambda *a: (reached.append(a[0]), original(*a))):
            self.forward(meta=metadata(seq=100000), seed=1)    # arming chunk: unfused and fused calls, not decision
            self.assertIsNotNone(helper._state['armed'])
            self.assertFalse(helper._state['armed'].captured)
            self.assertFalse((self.out / f'{token}-rank0.mhc-aborted').exists())
            self.forward(seed=2)                                # decision chunk, fused post_pre inside it as well
            self.wait_threads()
            # A fused post_pre on the captured layer before its own attention (not the served order, but the
            # wrapper must not depend on order): still never reaches the capture path.
            layer = self.layers[1]
            layer._b12x_mhc.post_pre(torch.zeros(ROWS, H, dtype=torch.bfloat16), torch.zeros((ROWS, 4, H), dtype=torch.bfloat16),
                                     torch.zeros(ROWS, 4), torch.zeros(ROWS, 4, 4), layer.fn, layer.scale, layer.base,
                                     layer.norm, torch.zeros(ROWS, 4))
        self.receipt(token, 0)
        self.assertEqual(self.layers[1]._b12x_mhc.calls, ['unfused', 'fused'] * 2 + ['fused'])
        self.assertEqual(self.layers[3]._b12x_mhc.calls, ['unfused', 'fused'] * 2)
        # Only unfused calls on engram layers while armed ever reach the capture path.
        self.assertEqual(reached, [1, 14, 1])
        self.assertFalse(any(p.name.endswith('.mhc-aborted') for p in self.out.iterdir()))

    def test_four_ranks_shard_and_reassemble(self):
        token = 'mhc-expanded-test-0002'
        captured = None
        for rank in range(4):
            self.rank = rank
            self.reset()
            helper.install(self.model)
            self.arm(token, layer=14)
            self.forward(meta=metadata(seq=100000), seed=1)
            captured = self.forward(seed=2)                 # identical inputs on every rank: replicated state
            self.wait_threads()
            self.receipt(token, rank)
        records, shards, shas = {}, {}, {}
        for rank, node in enumerate(compare.NODES):
            stem = self.out / f'{node}-rank{rank}-{token}'
            records[node] = json.loads(stem.with_suffix('.json').read_text())
            receipt = json.loads((self.out / f'{token}-rank{rank}.mhc-consumed').read_text())
            shards[node], streamed = compare.load_staging(stem.with_suffix('.bin'), records[node]['staging'])
            shas[node] = (receipt['bin_sha256'], streamed)
            self.assertEqual(records[node]['shard'], [rank * 2048, (rank + 1) * 2048])
            self.assertLess(records[node]['staging']['total'], 2048 * 4 * H * 2 + (1 << 20))
        data = compare.assemble(shards)
        recomputed = {n: [int(v) for v in compare.digest(data[n])] for n in compare.STAGED_FULL}
        self.assertEqual(compare.capture_problems(records, recomputed, shas), [])
        residual, pre_mix, (attn, outputs, _) = captured[14]
        layer = self.layers[3]
        _, post, comb, y, pre_out = outputs
        for name, want in (('residual', residual), ('pre_mix', pre_mix), ('post', post), ('comb', comb),
                           ('pre_out', pre_out), ('fn', layer.fn), ('norm', layer.norm),
                           ('ffn_norm', layer.ffn_norm.weight), ('positions', torch.arange(524288 - ROWS, 524288)),
                           ('y_tail', y[-128:]), ('attn_out_tail', attn[-128:]), ('residual_out_tail', outputs[0][-128:])):
            self.assertTrue(torch.equal(data[name], want), name)
        head = records['dusty']
        self.assertEqual(head['digests']['y'], [int(v) for v in compare.digest(y)])
        self.assertEqual(head['digests']['attn_out'], [int(v) for v in compare.digest(attn)])
        self.assertEqual(head['plan'], {'component_id': 'norm.mhc', 'source': 'cached',
                                        'query': {'operation': 'pre', 'max_tokens': 8192}, 'config': {'backend': 'tf32_tma'}})
        self.assertEqual(set(head['memory']), {'plan', 'available_at_arm', 'available_after_allocation',
                                               'available_at_save', 'available_after_release'})
        tampered = json.loads(json.dumps(records))
        tampered['kirby']['digests']['residual'][0] += 1
        self.assertIn('kirby differs from rank 0', ' '.join(compare.capture_problems(tampered, recomputed, shas)))
        shifted = json.loads(json.dumps(records))
        shifted['toby']['shard'] = [0, 2048]
        self.assertTrue(compare.capture_problems(shifted, recomputed, shas))
        bad = dict(recomputed, residual=[0, 0])
        self.assertIn('reassembled residual', ' '.join(compare.capture_problems(records, bad, shas)))
        self.assertTrue(compare.capture_problems(records, recomputed, dict(shas, rusty=('a', 'b'))))
        self.assertTrue(compare.capture_problems({k: v for k, v in records.items() if k != 'kirby'}, recomputed, shas))

    def test_memory_refusal_allocates_nothing(self):
        self.available = 200 << 20
        helper.install(self.model)
        token = self.arm('mhc-expanded-test-0003', reserve=256)
        with mock.patch.object(helper, '_Session', side_effect=AssertionError('must not allocate')):
            self.forward(seed=1)
        self.assertIsNone(helper._state['armed'])
        refusal = json.loads((self.out / f'{token}-rank0.mhc-aborted').read_text())
        self.assertIn('reserve', refusal['error'])
        self.forward(seed=2)                                 # the token is spent; no second attempt
        self.assertEqual([p.name for p in self.out.iterdir()], [f'{token}-rank0.mhc-aborted'])

    def test_fail_open_on_shape_mismatch_releases_after_stream(self):
        helper.install(self.model)
        self.layers[3].fn = torch.randn(24, 4 * H + 1)
        token = self.arm('mhc-expanded-test-0004', layer=14)
        released = []
        original = helper._Session._release
        with mock.patch.object(helper._Session, '_release', lambda s: (released.append(1), original(s))):
            captured = self.forward(seed=5)
            self.wait_threads()
        self.assertEqual(set(captured), {1, 14})          # serving outputs still returned
        self.assertTrue((self.out / f'{token}-rank0.mhc-aborted').exists())
        self.assertFalse((self.out / f'{token}-rank0.mhc-consumed').exists())
        self.assertIsNone(helper._state['armed'])
        self.assertTrue(released)

    def test_decision_without_positions_aborts(self):
        helper.install(self.model)
        token = self.arm('mhc-expanded-test-0005')
        helper._positions(1, (None, torch.arange(10)))      # arms with a wrong-length positions tensor
        self.forward_context.attn_metadata = {'layers.1.attn': metadata()}
        layer = self.layers[1]
        residual = torch.randn((ROWS, 4, H)).to(torch.bfloat16)
        layer._b12x_mhc.pre(residual, layer.fn, layer.scale, layer.base, layer.norm, torch.rand(ROWS, 4),
                            previous_output=None)
        self.assertTrue((self.out / f'{token}-rank0.mhc-aborted').exists())


# ---------------------------------------------------------------- preparation and installer

class Image(unittest.TestCase):
    def test_prepared_outputs_are_fresh_and_minimal(self):
        fresh = prepare.prepare()
        for name, data in fresh.items():
            self.assertEqual((KIT / name).read_bytes(), data, name)
        base = (KIT / prepare.BASE_MODEL).read_text().splitlines(keepends=True)
        model = fresh['claude-mhc-expanded-model.py'].decode().splitlines(keepends=True)
        added = [l for l in model if l not in base]
        self.assertEqual(len(model) - len(base), 5)
        self.assertEqual([l.strip() for l in added if not l.strip().startswith('#')],
                         ['import vllm.models.deepseek_v4_1.claude_mhc_expanded as _mhc_expanded',
                          '_mhc_expanded.install(self)'])
        lock = json.loads(fresh['claude-mhc-expanded.lock.json'])
        release = json.loads((KIT / 'ds41-precision-release.lock.json').read_text())
        self.assertEqual(lock['targets'][prepare.MODEL_TARGET]['input_sha256'],
                         release['after']['vllm']['vllm/models/deepseek_v4_1/nvidia/model.py']['sha256'])
        self.assertEqual((lock['base_image_id'], lock['base_lock_sha256']), (RELEASE, release_sha()))
        docker = fresh['Dockerfile.claude-mhc-expanded'].decode()
        copied = docker.split('COPY ', 1)[1].split('\n', 1)[0].split()[:-1]
        self.assertEqual(sorted(copied), sorted(list(lock['inputs']) + ['claude-mhc-expanded.lock.json']))
        self.assertNotIn('ENV ', docker)
        self.assertIn(f'FROM {RELEASE}\n', docker)

    def tree(self, root):
        root = Path(root)
        here, opt, release = root / 'here', root / 'opt', root / 'release'
        for d in (here, opt / 'vllm/vllm/models/deepseek_v4_1/nvidia', opt / 'b12x/b12x', release):
            d.mkdir(parents=True)
        lock = json.loads((KIT / 'claude-mhc-expanded.lock.json').read_text())
        for name in lock['inputs']:
            shutil.copyfile(KIT / name, here / name)
        shutil.copyfile(KIT / 'claude-mhc-expanded.lock.json', here / 'claude-mhc-expanded.lock.json')
        shutil.copyfile(KIT / 'engram-progress-model.py', opt / prepare.MODEL_TARGET)
        (opt / 'b12x/b12x/__init__.py').write_text('# b12x\n')
        shutil.copyfile(KIT / 'ds41-precision-release.lock.json', release / 'ds41-precision-release.lock.json')
        (release / 'preserved-files.json').write_text(json.dumps(installer.inventory(opt)))
        return here, opt, release

    def test_install_on_temporary_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            here, opt, release = self.tree(tmp)
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                installer.install(here, opt, release, require_native=False)
            self.assertIn('MHC-EXPANDED-INSTALL-PASS', printed.getvalue())
            self.assertEqual((opt / prepare.MODEL_TARGET).read_bytes(), (KIT / 'claude-mhc-expanded-model.py').read_bytes())
            self.assertEqual((opt / prepare.HELPER_TARGET).read_bytes(), (KIT / 'claude-mhc-expanded-capture.py').read_bytes())
            before = json.loads((here / 'before-files.json').read_text())
            after = json.loads((here / 'after-files.json').read_text())
            self.assertEqual(set(after) - set(before), {str(opt / prepare.HELPER_TARGET)})
            self.assertEqual({k for k in before if before[k] != after[k]}, {str(opt / prepare.MODEL_TARGET)})

    def test_install_refusals(self):
        cases = {
            'input drift': lambda h, o, r: (h / 'claude-mhc-expanded-capture.py').write_text('x'),
            'release lock differs': lambda h, o, r: (r / 'ds41-precision-release.lock.json').write_text('{}'),
            'preexisting': lambda h, o, r: (o / prepare.HELPER_TARGET).write_text('x'),
            'preimage': lambda h, o, r: (o / prepare.MODEL_TARGET).write_text('x'),
            'inventory': lambda h, o, r: (o / 'b12x/b12x/extra.py').write_text('x'),
        }
        messages = {'input drift': 'input drift', 'release lock differs': 'release lock differs',
                    'preexisting': 'preexisting', 'preimage': 'preimage', 'inventory': 'inventory differs'}
        for name, mutate in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                here, opt, release = self.tree(tmp)
                mutate(here, opt, release)
                with self.assertRaisesRegex(RuntimeError, messages[name]):
                    installer.install(here, opt, release, require_native=False)


RELEASE_AMENDMENT = 'ratio1-restore-20260927T015500Z-amendment.json'


def release_pin():
    """The release pin as last restored (a fixed record), never the live candidate.json, which the
    operator swaps to the capture pin for the capture boot."""
    pin = json.loads((KIT / RELEASE_AMENDMENT).read_text())['candidate']
    if (pin['image_id'] != RELEASE or pin['diagnostic']['kind'] != 'precision-release-candidate'
            or pin['diagnostic']['lock_sha256'] != release_sha()):
        raise RuntimeError('Restore amendment does not hold the release pin')
    return pin


def release_sha():
    return sha((KIT / 'ds41-precision-release.lock.json').read_bytes())


# ---------------------------------------------------------------- comparator logic

def record(rank, digests, plans):
    return {'schema': 'claude-mhc-expanded-v1', 'token': 't', 'rank': rank, 'digests': digests, 'plans': plans}


class Comparator(unittest.TestCase):
    def test_y_contract_uses_pinned_rounding(self):
        residual = torch.tensor([[[1., 2., 3., 4.]] * 4], dtype=torch.bfloat16)
        incoming = torch.tensor([[.1, .2, .3, .4]])
        weight = torch.full((4,), 16., dtype=torch.bfloat16)
        namespace = {}
        exec(subprocess.check_output(['git', '-C', str(Path.home() / 'git/b12x'),
             'show', 'a7d7d29b:b12x/testing/mhc.py'], text=True), namespace)
        source = subprocess.check_output(['git', '-C', str(Path.home() / 'git/b12x'),
             'show', 'a7d7d29b:tests/norm/test_mhc_lagged.py'], text=True)
        begin = source.index('def _lagged_reference')
        scope = {'torch': torch, 'F': torch.nn.functional,
                 '_mhc_pre_reference': namespace['pre_reference']}
        exec(source[begin:source.index('\n@', begin)], scope)
        expected = scope['_lagged_reference'](residual, torch.zeros((24,16)),
                   torch.ones(3), torch.zeros(24), incoming, weight)[2]
        got = compare.upstream_y_reference(residual, incoming, weight, 1e-20)
        self.assertEqual(got.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(got, expected))
        exact = residual[:, 0].double()
        exact = exact * torch.rsqrt(exact.square().mean(-1, keepdim=True)) * weight.double()
        tol = {'atol': .008, 'rtol': 2e-5}
        self.assertGreater(replay.error_stats(got, exact, tol)['outside_tolerance'], 0)
        self.assertEqual(replay.error_stats(got, expected, tol)['outside_tolerance'], 0)
        evidence = compare.y_rounding_samples(got, exact, expected, tol, limit=1)
        self.assertGreater(evidence['unrounded_outliers'], 0)
        self.assertEqual(len(evidence['samples']), 1)
        self.assertEqual(evidence['samples'][0]['actual'], evidence['samples'][0]['upstream_bf16'])

    def test_contract_problems(self):
        selections = replay.helper()
        manifest = json.loads((KIT / 'ds41-mhc-layer0-inputs.json').read_text())
        want = dict(selections.MHC_QUERY, max_tokens=8192, **selections.mhc_invocation('pre', True))
        key = selections.mhc_key('pre', 8192, True)
        passing = manifest['configs']['passing']['pre_expanded']
        plan = {'component_id': 'norm.mhc', 'source': 'cached', 'query': want, 'config': passing}
        self.assertEqual(key, manifest['keys']['pre_expanded'])
        self.assertEqual(compare.contract_problems(plan, want, passing, key, key, replay.plain), [])
        release = manifest['configs']['release']['pre_expanded']
        self.assertEqual(len(compare.contract_problems(plan, want, release, key, key, replay.plain)), 1)
        for bad in (dict(plan, source='tuned'), dict(plan, source='default'), dict(plan, component_id='other')):
            self.assertIn('seeded cache', ' '.join(compare.contract_problems(bad, want, passing, key, key, replay.plain)))
        self.assertTrue(compare.contract_problems(plan, dict(want, max_tokens=4096), passing, key, key, replay.plain))
        self.assertTrue(compare.contract_problems(plan, want, passing, key, '0' * 64, replay.plain))

    def test_faithful(self):
        y = torch.randn(300, 8).to(torch.bfloat16)
        served = {'post': torch.zeros(3), 'comb': torch.zeros(3), 'pre_out': torch.zeros(3), 'y_tail': y[-128:]}
        replayed = {'post': torch.zeros(3), 'comb': torch.zeros(3), 'pre_out': torch.zeros(3), 'y': y.clone()}
        full = [int(v) for v in compare.digest(y)]
        self.assertTrue(compare.faithful(served, replayed, True, full)['passed'])
        self.assertFalse(compare.faithful(served, replayed, False, full)['passed'])
        early = y.clone()
        early[3, 1] = early[3, 1] + 1                          # outside the tail: only the full digest sees it
        result = compare.faithful(served, dict(replayed, y=early), True, full)
        self.assertEqual((result['passed'], result['y_full_digest_equal'], result['changed_elements']['y_tail']), (False, False, 0))
        moved = dict(replayed, comb=torch.tensor([0.0, 1e-9, 0.0]))
        self.assertEqual(compare.faithful(served, moved, True, full)['changed_elements']['comb'], 1)

    def test_representatives(self):
        base = {'backend': 'tf32_tma', 'projection_k_splits': 8, 'projection_tile_k': 128, 'lagged_prepare': False,
                'partials_per_cta': 4, 'projection_tile_m': 128}
        candidates = [dict(base, projection_tile_m=64), dict(base, projection_k_splits=2), dict(base, projection_k_splits=2,
                      projection_tile_m=32), dict(base, projection_tile_k=64), dict(base, backend='native')]
        chosen = compare.representatives(candidates, [base])
        self.assertEqual([compare.config_class(c) for c in chosen],
                         [('tf32_tma', 2, 128, False, 4), ('tf32_tma', 8, 64, False, 4), ('native', 8, 128, False, 4)])

    def result(self, errors, distance, gates=None):
        stats = lambda e: {n: {'max_abs': e, 'outside_tolerance': 0} for n in compare.OUTPUTS}
        out = {'gates': gates or {'faithful': True, 'all_repeat': True, 'population_complete': True},
               'errors': {k: stats(v) for k, v in errors.items()},
               'distance': {k: {n: v for n in compare.OUTPUTS} for k, v in distance.items()}}
        return out

    def test_classify(self):
        ok = self.result({'passing': 4e-7, 'release': 9e-7, 'candidate-0': 6e-7, 'candidate-1': 1e-6},
                         {'passing': 0.0, 'release': 1e-6, 'candidate-0': 8e-7, 'candidate-1': 1.2e-6})
        self.assertEqual(compare.classify(ok), 'within-valid-population')
        outlier = self.result({'passing': 4e-7, 'release': 9e-6, 'candidate-0': 6e-7},
                              {'passing': 0.0, 'release': 1e-6, 'candidate-0': 8e-7})
        self.assertEqual(compare.classify(outlier), 'release-outlier')
        spread = self.result({'passing': 4e-7, 'release': 5e-7, 'candidate-0': 6e-7},
                             {'passing': 0.0, 'release': 5e-6, 'candidate-0': 8e-7})
        self.assertEqual(compare.classify(spread), 'release-outlier')
        passing = self.result({'passing': 9e-6, 'release': 5e-7, 'candidate-0': 6e-7},
                              {'passing': 0.0, 'release': 5e-7, 'candidate-0': 8e-7})
        self.assertEqual(compare.classify(passing), 'passing-outlier')
        violation = self.result({'passing': 4e-7, 'release': 5e-7, 'candidate-0': 6e-7},
                                {'passing': 0.0, 'release': 5e-7, 'candidate-0': 8e-7})
        violation['errors']['release']['post']['outside_tolerance'] = 3
        self.assertEqual(compare.classify(violation), 'contract-violation')
        population = self.result({'passing': 4e-7, 'release': 5e-7, 'candidate-0': 6e-7},
                                 {'passing': 0.0, 'release': 5e-7, 'candidate-0': 8e-7})
        population['errors']['candidate-0']['comb']['outside_tolerance'] = 1
        self.assertEqual(compare.classify(population), 'population-violation')
        for gate in ('faithful', 'all_repeat', 'population_complete'):
            failed = self.result({'passing': 4e-7, 'release': 5e-7, 'candidate-0': 6e-7},
                                 {'passing': 0.0, 'release': 5e-7, 'candidate-0': 8e-7})
            failed['gates'][gate] = False
            self.assertEqual(compare.classify(failed), 'not-faithful', gate)
        self.assertEqual(compare.classify(self.result({'passing': 1e-7, 'release': 1e-7}, {'passing': 0, 'release': 0})),
                         'no-population')

    def test_consumer_matches_layer0_readout_and_pinned_post(self):
        g = torch.Generator().manual_seed(9)
        emb = torch.randn((6, 16), generator=g).to(torch.bfloat16)
        x = torch.randn((6, 16), generator=g).to(torch.bfloat16)
        post = torch.rand((6, 4), generator=g) * 2
        comb = torch.softmax(torch.randn((6, 4, 4), generator=g), -1)
        pre_out = torch.rand((6, 4), generator=g)
        norm = torch.linspace(0.5, 1.5, 16).to(torch.bfloat16)
        streams = emb[:, None, :].expand(-1, 4, -1).contiguous()
        got = compare.consumer(post, comb, pre_out, x, streams, norm, 1e-20)
        want = replay.propagate(post, comb, pre_out, x, emb, norm, 1e-20)
        for name in ('residual', 'ffn_in'):
            self.assertTrue(torch.equal(got[name], want[name]), name)
        distinct = torch.randn((6, 4, 16), generator=g).to(torch.bfloat16)
        pinned = {}
        exec(subprocess.run(['git', '-C', str(Path.home() / 'git/b12x'), 'show', 'a7d7d29b:b12x/testing/mhc.py'],
                            capture_output=True, text=True, check=True).stdout, pinned)
        self.assertTrue(torch.equal(compare.consumer(post, comb, pre_out, x, distinct, norm, 1e-20)['residual'],
                                    pinned['post_reference'](x, distinct, post, comb)))
        diff = compare.consumer_diff(got, dict(got, ffn_in=got['ffn_in'].clone().index_fill_(0, torch.tensor([2]), 0)))
        self.assertEqual((diff['residual']['changed_elements'], diff['ffn_in']['rows_changed']), (0, 1))

    def test_own_entry_removed(self):
        own = str(KIT / 'ds41_mhc_expanded_compare.py')
        files = {'__main__': own, 'torch': '/x/torch/__init__.py'}
        self.assertEqual(compare.own_entry_removed(files, own), {'torch': '/x/torch/__init__.py'})
        main = object()
        alias_files = dict(files, __mp_main__=own)
        self.assertEqual(compare.own_entry_removed(alias_files, own,
                         {'__main__': main, '__mp_main__': main}), {'torch': '/x/torch/__init__.py'})
        with self.assertRaisesRegex(RuntimeError, 'not an identity alias'):
            compare.own_entry_removed(alias_files, own,
                                      {'__main__': main, '__mp_main__': object()})
        with self.assertRaisesRegex(RuntimeError, 'not an identity alias'):
            compare.own_entry_removed(alias_files, own, {'__mp_main__': main})
        with self.assertRaisesRegex(RuntimeError, 'also loaded'):
            compare.own_entry_removed(dict(files, alias=own), own)
        problems, _ = replay.loaded_problems({'root': '/opt/jovian-judgement'}, {'__main__': own}, {}, lambda p: None)
        self.assertTrue(problems)  # the shared check alone would refuse this harness; hence the exemption

    def test_uses_replay_by_path_and_reports_hash(self):
        text = (KIT / 'ds41_mhc_expanded_compare.py').read_text()
        self.assertIn("'replay_sha256'", text)
        self.assertNotIn('import ds41_mhc_layer0_replay', text)


# ---------------------------------------------------------------- recorder

class Recorder(unittest.TestCase):
    def release_pin(self):
        return release_pin()

    def test_capture_pin(self):
        pin = recorder.capture_pin(self.release_pin(), 'c' * 64)
        lock = sha((KIT / 'claude-mhc-expanded.lock.json').read_bytes())
        self.assertEqual(pin['diagnostic'], {'kind': 'mhc-expanded-capture', 'lock_sha256': lock,
                                             'base_image_id': RELEASE, 'b12x_tree': self.release_pin()['diagnostic']['b12x_tree']})
        self.assertEqual({k: v for k, v in pin.items() if k not in ('image_id', 'diagnostic')},
                         {k: v for k, v in self.release_pin().items() if k not in ('image_id', 'diagnostic')})
        for bad in ('c' * 63, 'C' * 64):
            with self.assertRaises(ValueError):
                recorder.capture_pin(self.release_pin(), bad)
        with self.assertRaises(RuntimeError):
            recorder.capture_pin(dict(self.release_pin(), image_id='d' * 64), 'c' * 64)

    def test_live_pin_is_release_or_its_exact_capture_derivation(self):
        live = json.loads((KIT / 'candidate.json').read_text())
        if live == release_pin():
            return
        self.assertEqual(live, recorder.capture_pin(release_pin(), live['image_id']))
        self.assertEqual(live['diagnostic']['lock_sha256'], sha((KIT / 'claude-mhc-expanded.lock.json').read_bytes()))
        self.assertEqual(recorder.pin_problems(live), [])

    def test_trigger_arm_and_receipts(self):
        payload = recorder.trigger('mhc-expanded-passing-20260927t000000z', 1, 1024)
        self.assertEqual(helper.trigger_spec(payload, {1, 14}), payload)
        for layer, reserve in ((2, 1024), (1, 100), (14, 20000)):
            with self.assertRaises(ValueError):
                recorder.trigger('mhc-expanded-x', layer, reserve)
        command = recorder.arm_command('/home/jugs/git/ds41-r38/x root', payload)
        self.assertIn("'/home/jugs/git/ds41-r38/x root/claude-mhc-expanded.json.tmp'", command)
        self.assertTrue(command.endswith('echo ARMED'))
        self.assertEqual(recorder.receipt_state('a\nt-rank1.mhc-consumed\n', 't', 1), 'consumed')
        self.assertEqual(recorder.receipt_state('t-rank1.mhc-consumed t-rank1.mhc-aborted', 't', 1), 'aborted')
        self.assertEqual(recorder.receipt_state('t-rank2.mhc-consumed', 't', 1), 'pending')

    def test_token0_and_compare_command(self):
        import seed_selection_transplant
        refs = seed_selection_transplant.references(KIT)
        trial = {'response': {'choices': [{'logprobs': {'content': [refs['bf16-ratio1-variant']]}}]}}
        self.assertEqual(recorder.token0_verdict(trial, refs)['equals'], 'bf16-ratio1-variant')
        command = recorder.compare_command('/r/receipts/x/captures', Path('/r/receipts/x'), 'tok', 'release')
        argv = command['argv']
        self.assertEqual(argv[argv.index('--manifest-sha256') + 1], sha((KIT / 'ds41-mhc-layer0-inputs.json').read_bytes()))
        self.assertEqual(argv[argv.index('--served') + 1], 'release')
        self.assertEqual(argv[:3], ['python3', '-P', '/gate/ds41_mhc_expanded_compare.py'])
        self.assertEqual(command['env']['PYTHONPATH'], '/opt/jovian-judgement/vllm:/opt/jovian-judgement/b12x')
        self.assertEqual(argv[argv.index('--captures-dir') + 1], '/captures')

    def test_memory_preflight_and_gather(self):
        plan = json.loads((KIT / 'claude-mhc-expanded.lock.json').read_text())['memory_plan']
        avail = {'dusty': 2 << 30, 'toby': 900 << 20, 'rusty': 2 << 30, 'kirby': 2 << 30}
        rows = recorder.memory_preflight(lambda node, cmd: f'MemAvailable: {avail[node] >> 10} kB\n', 1024)
        self.assertEqual({n: r['ok'] for n, r in rows.items()}, {'dusty': True, 'toby': False, 'rusty': True, 'kirby': True})
        self.assertEqual([rows[n]['need'] for n in recorder.NODES], [r['peak_bytes'] for r in plan['ranks']])
        with self.assertRaises(RuntimeError):
            recorder.memory_preflight(lambda node, cmd: 'garbage', 1024)
        script = recorder.gather_script('/remote/receipts/x', {n: [(f'/h/{n}.bin', 'a' * 64)] for n in recorder.NODES})
        self.assertIn('cp --reflink=auto /h/dusty.bin /remote/receipts/x/captures/dusty.bin', script)
        for node, address in (('toby', '10.11.11.6'), ('rusty', '10.11.11.5'), ('kirby', '10.11.11.8')):
            self.assertIn(f"ip -j route get {address} |", script)
            self.assertIn(f"{address}:/h/{node}.bin /remote/receipts/x/captures/{node}.bin", script)
        self.assertEqual(script.count('sha256sum -c -'), 4)
        self.assertEqual([l for l in script.splitlines() if l.startswith('echo GATHERED')],
                         [f'echo GATHERED {n}' for n in recorder.NODES])
        self.assertIn("r.get('prefsrc')=='10.11.11.7'", script)

    def test_record_flow_with_fakes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'rec'
            host = '/home/jugs/git/ds41-r38/karmic-main-20260924/selection-replay-cache-standard8192-233036Z'
            import seed_selection_transplant
            refs = seed_selection_transplant.references(KIT)
            state = {'armed': set(), 'gather': None}

            def ssh(node, command):
                if command.startswith('grep -m1 ^MemAvailable'):
                    return 'MemAvailable: 4194304 kB\n'
                if command.startswith('set -eu'):
                    state['armed'].add(node)
                    return 'ARMED\n'
                if command.startswith('bash -c'):
                    state['gather'] = (node, command)
                    return ''.join(f'GATHERED {n}\n' for n in recorder.NODES)
                token = state['token']
                rank = recorder.NODES.index(node)
                if command.startswith('ls -1'):
                    return f'{token}-rank{rank}.mhc-consumed\n'
                if command.endswith('.mhc-consumed'):
                    return json.dumps({'json': f'/cache/claude-mhc-expanded/{node}-rank{rank}-{token}.json',
                                       'bin': f'/cache/claude-mhc-expanded/{node}-rank{rank}-{token}.bin',
                                       'bin_sha256': str(rank) * 64})
                if command.startswith('cat ' + host):
                    return json.dumps({'rank': rank})
                raise AssertionError(command)

            class Process:
                def __init__(self, command, **kwargs):
                    if command[command.index('--repeats') + 1] != '3':
                        raise AssertionError('The unchanged qualification driver requires three trials')
                    target = Path(command[command.index('--out') + 1])
                    target.mkdir(parents=True)
                    (target / 'report.json').write_text(json.dumps([{'repeat': i, 'cached_tokens': 0, 'correct': True,
                                                                     'first_token': {'margin_nats': 0.125}}
                                                                    for i in range(3)]))
                    choice = {'logprobs': {'content': [refs['passing-anchor']]}}
                    (target / '0.json').write_text(json.dumps({'response': {'choices': [choice]}}))
                    self.stdout = io.StringIO('trial\n')

                def wait(self):
                    return 0

            original_trigger = recorder.trigger

            def trigger(token, layer, reserve):
                state['token'] = token
                return original_trigger(token, layer, reserve)
            with mock.patch.object(recorder, 'pin_problems', lambda pin, root: []), \
                    mock.patch.dict(sys.modules, {'runtime': SimpleNamespace(kit_digest=lambda: 'k' * 64)}), \
                    mock.patch.object(recorder, 'seeded_bytes', lambda sel, root: (host, {})), \
                    mock.patch.object(recorder, 'identity', lambda *a: {n: {'Id': n} for n in recorder.NODES}), \
                    mock.patch.object(recorder, 'trigger', trigger), \
                    contextlib.redirect_stdout(io.StringIO()):
                summary = recorder.record('passing', out, ssh, 1, 1024, root=KIT, run=Process, sleep=lambda s: None)
            self.assertEqual(state['armed'], set(recorder.NODES))
            self.assertEqual(summary['token0']['equals'], 'passing-anchor')
            self.assertTrue(summary['token0_as_expected'])
            self.assertEqual((summary['served'], summary['layer']), ('passing', 1))
            self.assertEqual(state['gather'][0], 'dusty')
            self.assertEqual(summary['gathered_dir'], recorder.REMOTE + '/receipts/rec/captures')
            self.assertTrue(all(r['ok'] for r in summary['memory_preflight'].values()))
            script = (out / 'gather.sh').read_text()
            for rank, node in enumerate(recorder.NODES):
                self.assertIn(f'{host}/claude-mhc-expanded/{node}-rank{rank}-{state["token"]}.bin', script)

    def test_record_refuses_before_arming_when_memory_short(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []

            def ssh(node, command):
                calls.append(command)
                if command.startswith('grep -m1 ^MemAvailable'):
                    return 'MemAvailable: 524288 kB\n'
                raise AssertionError(command)
            with mock.patch.object(recorder, 'pin_problems', lambda pin, root: []), \
                    mock.patch.dict(sys.modules, {'runtime': SimpleNamespace(kit_digest=lambda: 'k' * 64)}), \
                    mock.patch.object(recorder, 'seeded_bytes', lambda sel, root: ('/h', {})), \
                    mock.patch.object(recorder, 'identity', lambda *a: {}):
                with self.assertRaisesRegex(RuntimeError, 'Not armed'):
                    recorder.record('passing', Path(tmp) / 'r', ssh, 1, 1024, root=KIT, run=None)
            self.assertFalse(any(c.startswith('set -eu') for c in calls))


# ---------------------------------------------------------------- unapplied integration patch

_tmp = BASE = PATCHED = None


def setUpModule():
    global _tmp, BASE, PATCHED
    live = {n: sha((KIT / n).read_bytes()) for n in BASES}
    if live == BASES:
        applied = False
    else:
        _tmp_probe = tempfile.TemporaryDirectory()
        probe = Path(_tmp_probe.name)
        for n in BASES:
            shutil.copy2(KIT / n, probe / n)
        reverse = subprocess.run(['patch', '-p1', '--batch', '--reverse', '--dry-run', '-s', '-d', str(probe), '-i', str(PATCH)],
                                 capture_output=True)
        _tmp_probe.cleanup()
        if reverse.returncode:
            raise RuntimeError(f'Live files are neither the reviewed one-key base nor this patch applied: {live}')
        applied = True
    _tmp = tempfile.TemporaryDirectory(prefix='mhc-expanded-capture-')
    BASE, PATCHED = Path(_tmp.name) / 'base', Path(_tmp.name) / 'patched'
    for target in (BASE, PATCHED):
        target.mkdir()
        for path in KIT.iterdir():
            if path.is_file() and path.name != PATCH.name:
                shutil.copy2(path, target / path.name)
        (target / 'receipts').mkdir()
        for path in (KIT / 'receipts').iterdir():
            if path.is_file() and path.suffix == '.json':
                shutil.copy2(path, target / 'receipts' / path.name)
        for extra in ('router-fence-20260926T045016Z', 'router-release-build-20260925T220934Z',
                      'decision-row-matched8192-nccl-standard-upstream-capture-20260926T233036Z',
                      'ratio1-namespace-snapshot-20260927T010531Z', 'ratio1-control-20260927T0122Z',
                      'ratio1-variant-20260927T0141Z', 'precision-release-full-qualification-20260927-r2/initial-selections',
                      Path(json.loads((KIT / 'receipts/precision-release-build-receipt.json').read_text())['directory']).name):
            if (KIT / 'receipts' / extra).is_dir():
                shutil.copytree(KIT / 'receipts' / extra, target / 'receipts' / extra)
    if applied:
        subprocess.run(['patch', '-p1', '--batch', '--reverse', '-s', '-d', str(BASE), '-i', str(PATCH)], check=True)
    else:
        subprocess.run(['patch', '-p1', '--batch', '--forward', '-s', '-d', str(PATCHED), '-i', str(PATCH)], check=True)
    for name, digest in BASES.items():
        if sha((BASE / name).read_bytes()) != digest:
            raise RuntimeError(f'base/{name} does not match the reviewed hash')


def tearDownModule():
    _tmp.cleanup()


TREE_MODULES = ('launch_contract', 'runtime', 'diagnostic_overlay', 'run_node', 'seed_selection_replay',
                'seed_selection_transplant', 'run_selection_transplant', 'run_selection_replay')


def load_tree(tree, module):
    """One module from a temporary tree; the tree's modules never stay registered for later tests."""
    saved = {name: sys.modules.pop(name) for name in TREE_MODULES if name in sys.modules}
    sys.path.insert(0, str(tree))
    try:
        return load(tree / f'{module}.py', module)
    finally:
        sys.path.remove(str(tree))
        for name in TREE_MODULES:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


def capture_pin(tree):
    release = release_pin()
    return recorder.capture_pin(release, 'c' * 64, tree)


class Integration(unittest.TestCase):
    def test_contract(self):
        c, base = load_tree(PATCHED, 'launch_contract'), load_tree(BASE, 'launch_contract')
        self.assertEqual(c.MHC_EXPANDED_CAPTURE_LOCK, sha((KIT / 'claude-mhc-expanded.lock.json').read_bytes()))
        self.assertEqual(c.SELECTION_ROOTS, base.SELECTION_ROOTS)
        geometry = {'DS41_DECISION_ROW_BLOCKS': '81389', 'DS41_PREFILL_THRESHOLD': '8192'}
        for selector in base.SELECTION_ROOTS:
            with mock.patch.dict(os.environ, dict(geometry, DS41_SELECTION_REPLAY=selector), clear=True):
                self.assertEqual(c.render('toby'), base.render('toby'))
        release = release_pin()
        pin = capture_pin(PATCHED)
        for selector in base.SELECTION_ROOTS:
            env = dict(geometry, DS41_SELECTION_REPLAY=selector)
            base.validate_chunking_image(release, env)
            c.validate_chunking_image(release, env)
            if selector in (c.REPLAY_SELECTOR, c.MHC_EXPANDED8192_SELECTOR):
                c.validate_chunking_image(pin, env)
            else:
                with self.assertRaises(ValueError):
                    c.validate_chunking_image(pin, env)
            with self.assertRaises(ValueError):
                base.validate_chunking_image(pin, env)
        env = dict(geometry, DS41_SELECTION_REPLAY=c.REPLAY_SELECTOR)
        for mutate in (lambda d: d.update(lock_sha256='0' * 64), lambda d: d.update(base_image_id='0' * 64),
                       lambda d: d.pop('base_image_id')):
            bad = json.loads(json.dumps(pin))
            mutate(bad['diagnostic'])
            with self.assertRaises(ValueError):
                c.validate_chunking_image(bad, env)
        for bad_env in ({}, geometry, dict(geometry, DS41_SELECTION_REPLAY=c.REPLAY_SELECTOR, DS41_PREFILL_THRESHOLD='4096'),
                        dict(env, DS41_NCCL_ARM='standard-upstream')):
            with self.assertRaises(ValueError, msg=bad_env):
                c.validate_chunking_image(pin, bad_env)
        with self.assertRaises(ValueError):
            c.validate_chunking_image(dict(pin, diagnostic_overlay={}), env)

    def test_run_node(self):
        c = load_tree(PATCHED, 'launch_contract')
        run_node, base_node = load_tree(PATCHED, 'run_node'), load_tree(BASE, 'run_node')
        pin = capture_pin(PATCHED)
        release = release_pin()
        geometry = {'DS41_DECISION_ROW_BLOCKS': '81389', 'DS41_PREFILL_THRESHOLD': '8192'}
        for selector in (c.REPLAY_SELECTOR, c.MHC_EXPANDED8192_SELECTOR):
            with mock.patch.dict(os.environ, dict(geometry, DS41_SELECTION_REPLAY=selector), clear=True):
                argv, _, cache = run_node.command('dusty', pin, 'b' * 64)
                release_argv, _, release_cache = base_node.command('dusty', release, 'b' * 64)
            self.assertEqual(cache, release_cache)
            norm = lambda args, tree, image: [a.replace(str(tree), 'TREE') for a in args if a != image]
            self.assertEqual(norm(argv, PATCHED, pin['image_id']), norm(release_argv, BASE, release['image_id']))
            self.assertEqual(argv.count(pin['image_id']), 1)

    def test_start(self):
        def parse(tree, argv):
            nodes = ast.parse((tree / 'start_moe_repaired.py').read_text()).body
            cut = next(i for i, node in enumerate(nodes) if isinstance(node, ast.Assign)
                       and getattr(node.targets[0], 'id', None) == 'stamp')
            body = [n for n in nodes[:cut] if not (isinstance(n, ast.ImportFrom) and n.module == 'runtime')]
            with mock.patch.object(sys, 'argv', ['s', *argv]), contextlib.redirect_stderr(io.StringIO()):
                ns = {'__file__': str(tree / 'start_moe_repaired.py')}
                exec(compile(ast.Module(body=body, type_ignores=[]), 'start', 'exec'), ns)
            return ns['args']

        def line(tree, args):
            fn = next(n for n in ast.parse((tree / 'start_moe_repaired.py').read_text()).body
                      if isinstance(n, ast.FunctionDef) and n.name == 'launch_command')
            ns = {'REMOTE': '/r', 'args': args}
            exec(compile(ast.Module(body=[fn], type_ignores=[]), 'start', 'exec'), ns)
            return ns['launch_command']('dusty')
        geometry = ['--decision-row-blocks', '81389', '--prefill-threshold', '8192']
        existing = [['--arm', 'selection-replay', *geometry], ['--arm', 'selection-transplant', *geometry],
                    ['--arm', 'selection-transplant', *geometry, '--transplant-set', 'mhc-expanded8192'],
                    ['--arm', 'precision-release'], ['--arm', 'moe-control-repaired']]
        for argv in existing:
            self.assertEqual(line(PATCHED, parse(PATCHED, argv)), line(BASE, parse(BASE, argv)), argv)
        capture = ['--arm', 'mhc-expanded-capture', *geometry, '--capture-selections']
        self.assertEqual(line(PATCHED, parse(PATCHED, capture + ['passing'])),
                         line(BASE, parse(BASE, ['--arm', 'selection-replay', *geometry])))
        self.assertEqual(line(PATCHED, parse(PATCHED, capture + ['mhc-expanded8192'])),
                         line(BASE, parse(BASE, ['--arm', 'selection-transplant', *geometry, '--transplant-set', 'mhc-expanded8192'])))
        for argv in (['--arm', 'mhc-expanded-capture', *geometry], capture + ['other'],
                     ['--arm', 'selection-replay', *geometry, '--capture-selections', 'passing'],
                     ['--arm', 'mhc-expanded-capture', '--decision-row-blocks', '81389', '--prefill-threshold', '4096',
                      '--capture-selections', 'passing'],
                     capture + ['passing', '--transplant-set', 'mhc8192']):
            with self.assertRaises(SystemExit, msg=argv):
                parse(PATCHED, argv)
        source = (PATCHED / 'start_moe_repaired.py').read_text()
        self.assertIn("'mhc-expanded-capture': 'mhc-expanded-capture',", source)
        self.assertIn("if args.arm in ('precision-release', 'ratio1-diag', 'selection-replay', 'selection-transplant', "
                      "'mhc-expanded-capture'):", source)

    def test_scope_and_regression(self):
        changed = {p.name for p in PATCHED.iterdir() if p.is_file() and sha(p.read_bytes()) != sha((BASE / p.name).read_bytes())}
        self.assertEqual(changed, set(BASES))
        runtime = json.loads((KIT / 'runtime-files.json').read_text())
        self.assertEqual(changed & set(runtime), {'launch_contract.py', 'run_node.py', 'runtime.py'})
        # The recorded runtime manifest names either the base or the patched bytes of every changed runtime
        # file, never anything else; the patch leaves the manifest refresh to the operator.
        for name in changed & set(runtime):
            self.assertIn(runtime[name], (sha((BASE / name).read_bytes()), sha((PATCHED / name).read_bytes())), name)

        def outcome(root, suite):
            result = subprocess.run([sys.executable, '-m', 'unittest', suite], cwd=root, capture_output=True, text=True,
                                    timeout=900, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
            return result.returncode, [l.split(' in ')[0] if l.startswith('Ran ') else l for l in result.stderr.splitlines()
                                       if l.startswith(('Ran ', 'OK', 'FAILED', 'ERROR:', 'FAIL:'))]
        for suite in SUITES:
            with self.subTest(suite=suite):
                base = outcome(BASE, suite)
                self.assertEqual(base[0], 0, base)
                self.assertEqual(base, outcome(PATCHED, suite))


class Guards(unittest.TestCase):
    """The KV-block kind guard exists twice: run_node.command on the host and runtime.main in the container.
    Each is exercised alone (the contract check is neutralized for the host guard; the container run
    stops at the first step after the guard)."""
    geometry = {'DS41_DECISION_ROW_BLOCKS': '81389', 'DS41_PREFILL_THRESHOLD': '8192'}

    def pins(self, tree):
        return capture_pin(tree), release_pin()

    def test_host_guard(self):
        for tree, admits in ((PATCHED, True), (BASE, False)):
            run_node = load_tree(tree, 'run_node')
            capture, release = self.pins(tree)
            env = dict(self.geometry, DS41_SELECTION_REPLAY='standard8192-233036Z')
            with mock.patch.dict(os.environ, env, clear=True), \
                    mock.patch.object(run_node, 'validate_chunking_image', lambda pin, env: None):
                run_node.command('dusty', release, 'b' * 64)
                if admits:
                    run_node.command('dusty', capture, 'b' * 64)
                else:
                    with self.assertRaisesRegex(RuntimeError, 'KV block pin is restricted'):
                        run_node.command('dusty', capture, 'b' * 64)
                with self.assertRaisesRegex(RuntimeError, 'KV block pin is restricted'):
                    run_node.command('dusty', dict(capture, diagnostic=dict(capture['diagnostic'], kind='mhc-expanded')), 'b' * 64)

    def run_container(self, tree, pin, blocks=True):
        class PastGuard(Exception):
            pass
        runtime = load_tree(tree, 'runtime')
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / 'candidate.json').write_text(json.dumps(pin))
            env = {'DS41_KIT_SHA256': 'k' * 64, 'DS41_NODE': 'dusty'}
            if blocks:
                env['DS41_DECISION_ROW_BLOCKS'] = '81389'
            with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(runtime, 'ROOT', Path(root)), \
                    mock.patch.object(runtime, 'kit_digest', lambda: 'k' * 64), \
                    mock.patch.object(runtime, 'render', mock.Mock(side_effect=PastGuard)):
                try:
                    runtime.main()
                except PastGuard:
                    return 'past-guard'
                except RuntimeError as error:
                    return str(error)
        return 'returned'

    def test_container_guard(self):
        for tree, admits in ((PATCHED, True), (BASE, False)):
            capture, release = self.pins(tree)
            self.assertEqual(self.run_container(tree, release), 'past-guard', tree.name)
            result = self.run_container(tree, capture)
            if admits:
                self.assertEqual(result, 'past-guard')
            else:
                self.assertIn('KV block pin is restricted', result)
            self.assertEqual(self.run_container(tree, capture, blocks=False), 'past-guard')  # guard only with a block pin
            wrong = dict(capture, diagnostic=dict(capture['diagnostic'], kind='mhc-expanded'))
            self.assertIn('KV block pin is restricted', self.run_container(tree, wrong))

    def test_kind_lists_agree(self):
        def kinds(path, function):
            tree = ast.parse(path.read_text())
            fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function)
            found = [n for n in ast.walk(fn) if isinstance(n, ast.Tuple)
                     and any(isinstance(e, ast.Constant) and e.value == 'decision-row-capture' for e in n.elts)]
            self.assertEqual(len(found), 1, (path.name, function))
            return {e.value for e in found[0].elts}
        for tree in (BASE, PATCHED):
            host, container = kinds(tree / 'run_node.py', 'command'), kinds(tree / 'runtime.py', 'main')
            self.assertEqual(host, container, tree.name)
            self.assertEqual('mhc-expanded-capture' in host, tree is PATCHED)
        self.assertEqual(kinds(PATCHED / 'runtime.py', 'main') - kinds(BASE / 'runtime.py', 'main'), {'mhc-expanded-capture'})


if __name__ == '__main__':
    unittest.main()
