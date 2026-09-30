"""Tests for the MoE-capture diagnostic layer (v3). Run with the task-local CPU venv:
  .venv-snapshot-cpu/bin/python -m unittest claude_test_moe_capture
Torch CPU only (SyncTransport); CUDA streams, events and pinned copies are not
exercised here. Pinned-source and packaging checks read the existing
~/git/vllm checkout (read-only) and skip, saying so, without it.
"""
import ast
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trace = load('claude_moe_capture_trace_under_test', HERE / 'claude-moe-capture-act_trace.py')
analyze = load('claude_moe_capture_analyze_under_test', HERE / 'claude-moe-capture-act_analyze.py')
prep = load('claude_prepare_moe_capture_under_test', HERE / 'claude_prepare_moe_capture.py')
HAVE_VLLM = (prep.VLLM_CHECKOUT / '.git').exists()
LAYERS, WIDTH, EXPERTS, TOPK = 40, 8, 6, 2


class Gate(nn.Module):
    """ReplicatedLinear-like: returns (logits, bias)."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(WIDTH, EXPERTS, bias=False)

    def forward(self, x):
        return self.lin(x).float(), None


class Router:
    """Plain object with select_experts, like FusedMoERouter."""

    def select_experts(self, hidden_states, router_logits, topk_indices_dtype=None, *, input_ids=None):
        weights, ids = torch.topk(torch.softmax(router_logits, -1), TOPK, dim=-1)
        return weights, ids.to(torch.int32)


class FakeRunner(nn.Module):
    """Mirrors MoERunner: gate module and router called inside the op, and the
    two forward callsites; `fault` perturbs one routed call."""

    def __init__(self):
        super().__init__()
        self.gate = Gate()
        self.router = Router()
        self.shared_lin = nn.Linear(WIDTH, WIDTH)
        self.experts = nn.Parameter(torch.randn(EXPERTS, WIDTH, WIDTH) * 0.1)
        self.fault = None                        # callable(routed) -> None, applied once
        self.seen = None
        self._forward_entry = self._entry

    def _entry(self, hidden_states, router_logits, shared_experts_input, input_ids, layer_name, dim, dtype):
        logits, _ = self.gate(hidden_states)
        weights, ids = self.router.select_experts(hidden_states=hidden_states, router_logits=logits)
        routed = torch.einsum('tk,tkij,tj->ti', weights, self.experts[ids.long()], hidden_states)
        if self.fault is not None:
            self.fault(routed)
            self.fault = None
        self.seen = {'input': hidden_states, 'logits': logits, 'topk_weights': weights,
                     'topk_ids': ids, 'routed': routed.clone()}
        return self.shared_lin(shared_experts_input), routed

    def _maybe_reduce_final_output(self, states, trunc_size, output_is_reduced=None):
        return states * 4

    def forward(self, hidden_states, router_logits):
        shared, fused = self._forward_entry(hidden_states, router_logits, hidden_states, None, 'x', 0, None)
        return self._maybe_reduce_final_output(shared + fused, None, False)


class Block(nn.Module):
    def __init__(self, engram=False):
        super().__init__()
        self.attn = nn.Linear(WIDTH, WIDTH)
        self.ffn = nn.Module()
        self.ffn.experts = FakeRunner()
        self.engram = nn.Linear(WIDTH, WIDTH) if engram else None

    def forward(self, h):
        if self.engram is not None:
            h = self.engram(h)
        a = self.attn(torch.tanh(h))
        return self.ffn.experts(hidden_states=a, router_logits=a) * 1e-2 + h, a


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(100, WIDTH)
        self.layers = nn.ModuleList(Block(engram=i in (1, 14)) for i in range(LAYERS))

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        h = self.embed(input_ids)
        for layer in self.layers:
            h, _ = layer(h)
        return h


def runner(model, i):
    return model.layers[i].ffn.experts


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.trigger = Path(self.tmp.name) / 'trigger.json'
        self.records = []
        self.model = Toy()
        self.transport = trace.SyncTransport(None, slots=2, capacity=1024)
        trace._INSTALLED.clear()
        self.tracer = trace.install(self.model, transport=self.transport, trigger=str(self.trigger),
                                    out_dir=self.tmp.name, rank=1, node='toby')

        def sink(meta):                 # production wires the transport to Tracer._sink
            self.records.append(dict(meta))
            self.tracer._sink(meta)
        self.transport.sink = sink

    def tearDown(self):
        self.tmp.cleanup()

    def arm(self, **spec):
        spec = {'arm': 'capture-arm-0001', 'min_tokens': 4, 'max_tokens': 64, 'post': ['layers.39'],
                'moe': ['layers.*.ffn.experts'], **spec}
        self.trigger.write_text(json.dumps(spec))
        os.utime(self.trigger, ns=(time.time_ns(), time.time_ns()))

    def step(self, rows=8):
        self.model._engram_epoch = getattr(self.model, '_engram_epoch', 0) + 1
        with torch.inference_mode():
            return self.model(torch.arange(rows) % 100, torch.arange(rows), None)

    def steps(self):
        return [r for r in self.records if r.get('type') == 'step']

    def receipts(self):
        out = []
        for file in Path(self.tmp.name).glob('*.jsonl'):
            out += [json.loads(l) for l in file.read_text().splitlines() if l.strip()]
        return out

    def statuses(self, generation=None):
        return [r['state'] for r in self.receipts() if r.get('type') == 'capture'
                and (generation is None or r.get('generation') == generation)]

    def pristine(self):
        for i in range(LAYERS):
            r = runner(self.model, i)
            self.assertEqual(vars(r)['_forward_entry'], r._entry)
            self.assertNotIn('_maybe_reduce_final_output', vars(r))
            self.assertNotIn('select_experts', vars(r.router))
            self.assertEqual(len(r.gate._forward_hooks), 0)


class Seams(Base):
    def test_unarmed_is_pristine(self):
        self.step()
        self.pristine()
        self.assertEqual(self.records, [])

    def test_route_inputs_recorded_in_order_with_exact_digests(self):
        baseline = self.step()
        self.arm()
        out = self.step()
        self.assertTrue(torch.equal(out, baseline))
        (record,) = self.steps()
        names = [n for n in record['names'] if n.startswith('layers.3.ffn')]
        self.assertEqual(names, [f'layers.3.ffn.experts{s}' for s in trace.SEAMS])
        index = {n: i for i, n in enumerate(record['names'])}
        seen = runner(self.model, 3).seen
        for key, name in (('input', '#input'), ('logits', '#logits'), ('topk_weights', '#topk_weights'),
                          ('topk_ids', '#topk_ids'), ('input', '#input_after'), ('routed', '#routed')):
            self.assertEqual(record['hashes'][index['layers.3.ffn.experts' + name]],
                             trace.digest(seen[key]).tolist(), name)
        self.assertNotIn('<capture>', record['names'])      # no capture configured

    def test_disarm_restores_gate_router_and_callsites(self):
        self.arm()
        self.step()
        self.trigger.unlink()
        self.step()
        self.pristine()

    def test_already_wrapped_router_is_refused_and_rolled_back(self):
        foreign = runner(self.model, 5).router
        foreign.select_experts = foreign.select_experts        # someone else's instance wrapper
        self.arm()
        with mock.patch('builtins.print') as printed:
            self.step()
        self.assertIn('trigger refused', printed.call_args_list[-1].args[0])
        self.assertEqual((self.tracer.handles, self.tracer.capture), ([], None))
        del foreign.select_experts
        self.pristine()

    def test_inside_window_hooks_launch_nothing(self):
        tree = ast.parse((HERE / 'claude-moe-capture-act_trace.py').read_text())
        seams = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MoeSeams')
        for name in ('gate_hook', 'select_experts'):
            fn = next(n for n in ast.walk(seams) if isinstance(n, ast.FunctionDef) and n.name == name)
            calls = {ast.unparse(c.func) for c in ast.walk(fn) if isinstance(c, ast.Call)}
            self.assertLessEqual(calls, {'select', 'isinstance'}, name)


class Capturing(Base):
    def spec(self, **extra):
        return {'capture': {'rows': [8], 'moe': ['layers.1[0-3].ffn.experts']}, **extra}

    def test_latches_first_disagreeing_call_and_saves_it(self):
        self.arm(**self.spec())
        for _ in range(3):
            self.step()
        target = runner(self.model, 12)

        def fault(routed):
            routed[2, 3] += 0.5
        target.fault = fault
        self.step()                                     # faulty call at epoch 4
        faulty = target.seen
        self.assertTrue(self.tracer.capture.ready)
        self.step()                                     # next step start freezes and saves
        path = Path(self.tmp.name) / f'toby-{os.getpid()}-g1-capture.pt'
        saved = torch.load(path)
        self.assertEqual((saved['runner'], saved['meta']['epoch'], saved['meta']['rows'], saved['rank']),
                         ('layers.12.ffn.experts', 4, 8, 1))
        for key in trace.CAPTURE_FIELDS:
            self.assertTrue(torch.equal(saved['tensors'][key], faulty[key].detach()), key)
        self.assertEqual([saved['meta']['digest0'], saved['meta']['digest1']],
                         trace.digest(faulty['routed']).tolist())
        self.assertNotEqual([saved['meta']['ref0'], saved['meta']['ref1']],
                            [saved['meta']['digest0'], saved['meta']['digest1']])
        latches = [r['hashes'][r['names'].index('<capture>')] for r in self.steps() if '<capture>' in r['names']]
        self.assertEqual([l[0] for l in latches], [0, 0, 0, 1])
        self.assertTrue(self.tracer.capture.frozen)

    def test_only_first_fault_is_kept(self):
        self.arm(**self.spec())
        self.step()
        runner(self.model, 11).fault = lambda routed: routed.mul_(1.5)
        runner(self.model, 13).fault = lambda routed: routed.add_(1.0)
        self.step()
        first = runner(self.model, 11).seen['routed']
        self.step()
        saved = torch.load(Path(self.tmp.name) / f'toby-{os.getpid()}-g1-capture.pt')
        self.assertEqual(saved['runner'], 'layers.11.ffn.experts')
        self.assertTrue(torch.equal(saved['tensors']['routed'], first))

    def test_status_lines_and_step_status_are_written(self):
        self.arm(**self.spec())
        self.step()
        runner(self.model, 12).fault = lambda routed: routed.add_(1.0)
        self.step()
        self.step()
        self.assertEqual(self.statuses(1), ['fetching', 'saved'])
        states = [r['capture'] for r in self.steps()]
        self.assertEqual([c['generation'] for c in states], [1, 1, 1])
        self.assertEqual([c['state'] for c in states], ['allocated', 'allocated', 'saved'])
        saved = [r for r in self.receipts() if r.get('state') == 'saved'][0]
        self.assertEqual((saved['epoch'], saved['runner'], saved['rank']), (2, 'layers.12.ffn.experts', 1))

    def test_stale_generation_record_never_readies_the_new_capture(self):
        held = []
        self.transport.sink = held.append                   # the writer is behind
        self.arm(**self.spec())
        self.step()
        runner(self.model, 12).fault = lambda routed: routed.add_(1.0)
        self.step()                                         # generation 1 latched on the device
        self.arm(arm='capture-arm-0002', **self.spec())
        self.step()                                         # generation 2 armed and allocated
        fresh = self.tracer.capture
        self.assertEqual(fresh.generation, 2)
        for meta in held:                                   # late delivery of generation-1 records
            self.tracer._sink(meta)
        self.assertFalse(fresh.ready)
        self.assertIn(fresh.state, ('armed', 'allocated'))
        self.assertEqual(self.statuses(1), ['discarded', 'latched-discarded'])
        self.assertFalse(any(Path(self.tmp.name).glob('*-g1-capture.pt')))

    def test_new_capture_is_not_readied_before_allocation(self):
        self.arm(**self.spec())
        self.step(rows=9)                                   # capture armed but never allocated
        capture = self.tracer.capture
        forged = {'type': 'step', 'arm': capture.arm, 'session': capture.session,
                  'capture': {'generation': capture.generation, 'state': 'armed'},
                  'names': ['<capture>'], 'hashes': [[1, 7]], 'rank': 1, 'node': 'toby'}
        self.tracer._sink(forged)
        self.assertFalse(capture.ready)

    def test_save_failure_is_an_explicit_error_and_analysis_flags_it(self):
        self.arm(**self.spec())
        self.step()
        runner(self.model, 12).fault = lambda routed: routed.add_(1.0)
        self.step()
        with mock.patch.object(trace.torch, 'save', side_effect=OSError('disk full')), \
                mock.patch('builtins.print'):
            self.step()
        self.assertEqual(self.statuses(1), ['fetching', 'error'])
        self.assertEqual(self.tracer.capture.state, 'error')
        records = [dict(r, rank=1) for r in self.receipts() if r.get('type') == 'step']
        statuses = [r for r in self.receipts() if r.get('type') == 'capture']
        summary, incomplete = analyze.capture_summary([analyze.strip_capture(dict(r)) for r in records],
                                                      statuses, 'capture-arm-0001')
        self.assertEqual(len(incomplete), 1)
        self.assertEqual(incomplete[0]['latched_epochs'], [2])

    def test_observe_failure_never_raises_into_the_forward(self):
        self.arm(**self.spec())
        with mock.patch.object(trace.Capture, '_observe', side_effect=RuntimeError('boom')), \
                mock.patch('builtins.print'):
            out = self.step()
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(self.tracer.capture.state, 'error')
        self.assertIn('error', self.statuses(1))

    def test_different_prompt_with_same_rows_never_latches(self):
        self.arm(**self.spec())
        self.step()
        self.model._engram_epoch += 1
        with torch.inference_mode():
            self.model((torch.arange(8) + 7) % 100, torch.arange(8), None)   # other tokens, 8 rows
        self.step()
        self.assertFalse(self.tracer.capture.ready)
        self.assertTrue(all(r['hashes'][r['names'].index('<capture>')][0] == 0 for r in self.steps()))

    def test_other_rows_and_unselected_runners_are_ignored(self):
        self.arm(**self.spec())
        self.step(rows=9)
        runner(self.model, 20).fault = lambda routed: routed.add_(1.0)
        self.step()
        self.step()
        self.assertFalse(self.tracer.capture.ready)
        self.assertFalse(any(Path(self.tmp.name).glob('*capture.pt')))

    def test_clean_run_never_latches_and_analysis_ignores_latch_words(self):
        self.arm(**self.spec())
        for _ in range(4):
            self.step()
        steps = self.steps()
        self.assertTrue(all(r['hashes'][r['names'].index('<capture>')][0] == 0 for r in steps))
        records = [dict(r, rank=k, session=f's{k}', _source='x', names=list(r['names']),
                        hashes=[list(h) for h in r['hashes']]) for k in range(4) for r in steps]
        report = analyze.analyze(records, min_group=3)
        self.assertEqual(report['verdict'], 'no-divergence-observed')

    def test_capture_pattern_must_select_armed_seams(self):
        for capture in ({'rows': [8], 'moe': ['layers.99.ffn.experts']},
                        {'rows': [8, 8], 'moe': ['layers.1.ffn.experts']},
                        {'rows': [100], 'moe': ['layers.1.ffn.experts']},
                        {'rows': [8]}):
            self.arm(capture=capture)
            with mock.patch('builtins.print') as printed:
                self.step()
            self.assertTrue(any('trigger' in c.args[0] for c in printed.call_args_list), capture)
            self.assertIsNone(self.tracer.capture)
            self.pristine()

    def test_forward_path_has_no_host_sync(self):
        tree = ast.parse((HERE / 'claude-moe-capture-act_trace.py').read_text())
        forbidden = {'item', 'cpu', 'tolist', 'numpy', 'synchronize', 'elapsed_time', 'query'}
        capture = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Capture')
        for fn in (n for n in capture.body if isinstance(n, ast.FunctionDef) and n.name in ('observe', '_allocate')):
            self.assertFalse({a.attr for a in ast.walk(fn) if isinstance(a, ast.Attribute)} & forbidden, fn.name)
        seams = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MoeSeams')
        entry = next(n for n in ast.walk(seams) if isinstance(n, ast.FunctionDef) and n.name == 'forward_entry')
        self.assertFalse({a.attr for a in ast.walk(entry) if isinstance(a, ast.Attribute)} & forbidden)


@unittest.skipUnless(HAVE_VLLM, 'no ~/git/vllm checkout: pinned-source checks not run')
class PinnedContract(unittest.TestCase):
    def test_callsites_and_router_conform(self):
        runner_text, layer_text, model_text, router_text = (
            prep.pinned_source(p).decode() for p in (prep.RUNNER, prep.PINNED[2], prep.PINNED[3], prep.PINNED[4]))
        prep.check_callsites(runner_text, layer_text, model_text)
        prep.check_router(router_text)
        with self.assertRaises(RuntimeError):
            prep.check_callsites(runner_text.replace('router_logits, _ = self.gate(hidden_states)',
                                                     'router_logits = self.gate(hidden_states)[0]', 1),
                                 layer_text, model_text)
        with self.assertRaises(RuntimeError):
            prep.check_router(router_text.replace('    def select_experts(', '    @abstractmethod\n    def select_experts(', 1))


class Packaging(unittest.TestCase):
    def test_v2_kit_is_untouched(self):
        amendment = json.loads((HERE / prep.BASE_AMENDMENT).read_text())['candidate']
        v2 = (HERE / prep.V1_LOCK).read_bytes()
        self.assertEqual(hashlib.sha256(v2).hexdigest(), amendment['diagnostic']['lock_sha256'])
        for name, value in json.loads(v2)['inputs'].items():
            self.assertEqual(hashlib.sha256((HERE / name).read_bytes()).hexdigest(), value, name)

    def test_lock_patch_and_dockerfile(self):
        lock = json.loads((HERE / prep.LOCK).read_text())
        for name, value in lock['inputs'].items():
            self.assertEqual(hashlib.sha256((HERE / name).read_bytes()).hexdigest(), value, name)
        docker = (HERE / 'Dockerfile.claude-moe-capture').read_text()
        copy = [l for l in docker.splitlines() if l.startswith('COPY ')][0].split()[1:-1]
        self.assertEqual(sorted(copy), sorted([*lock['inputs'], prep.LOCK]))
        self.assertIn('FROM ' + lock['base_image_id'], docker)
        files = [l for l in (HERE / prep.PATCH).read_text().splitlines() if l.startswith(('--- ', '+++ '))]
        self.assertEqual(files, ['--- a/' + prep.HELPER_TARGET, '+++ b/' + prep.HELPER_TARGET])
        self.assertEqual(len(lock['verify_unchanged']), 5)

    @unittest.skipUnless(HAVE_VLLM, 'no ~/git/vllm checkout: simulated install not run')
    def test_simulated_install_is_confined(self):
        lock = json.loads((HERE / prep.LOCK).read_text())
        with tempfile.TemporaryDirectory() as tmp:
            kit, opt = Path(tmp) / 'kit', Path(tmp) / 'opt'
            kit.mkdir()
            for name in [*lock['inputs'], prep.LOCK]:
                shutil.copy(HERE / name, kit / name)
            helper = opt / prep.HELPER_TARGET
            helper.parent.mkdir(parents=True)
            shutil.copy(HERE / 'claude-moe-seams-act_trace.py', helper)
            for path in prep.PINNED:
                target = opt / 'vllm' / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(prep.pinned_source(path))
            (opt / 'b12x/b12x').mkdir(parents=True)
            (opt / 'b12x/b12x/native.so').write_bytes(b'x')
            source = (HERE / 'claude_install_moe_capture.py').read_text()
            script = Path(tmp) / 'install.py'
            script.write_text(source.replace("Path('/opt/ds41-moe-capture')", f"Path({str(kit)!r})")
                              .replace("Path('/opt/jovian-judgement')", f"Path({str(opt)!r})"))
            run = lambda: subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
            first = run()
            self.assertIn('MOE-CAPTURE-INSTALL-PASS', first.stdout, first.stderr)
            self.assertEqual(helper.read_bytes(), (HERE / prep.HELPER_SOURCE).read_bytes())
            self.assertNotEqual(run().returncode, 0)


if __name__ == '__main__':
    unittest.main()
