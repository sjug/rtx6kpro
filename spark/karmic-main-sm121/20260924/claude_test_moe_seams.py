"""Tests for the MoE-seam diagnostic layer. Run with the task-local CPU venv:
  .venv-snapshot-cpu/bin/python -m unittest claude_test_moe_seams
Torch CPU only; CUDA transport, streams and events are not exercised here.
The callsite-conformance and packaging tests read the pinned vLLM blobs from
the existing ~/git/vllm checkout (read-only) and skip, saying so, without it.
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


trace = load('claude_moe_seams_trace_under_test', HERE / 'claude-moe-seams-act_trace.py')
analyze = load('claude_moe_seams_analyze_under_test', HERE / 'claude-moe-seams-act_analyze.py')
prep = load('claude_prepare_moe_seams_under_test', HERE / 'claude_prepare_moe_seams.py')
HAVE_VLLM = (prep.VLLM_CHECKOUT / '.git').exists()
LAYERS = 40


class FakeRunner(nn.Module):
    """The two seam callsites exactly as MoERunner.forward uses them at 1794dcf1:
    `_forward_entry` is an instance attribute set in __init__ and returns
    (shared, fused); `_maybe_reduce_final_output(states, trunc, reduced)` is a
    class method whose result is the layer output."""

    def __init__(self, width):
        super().__init__()
        self.shared_lin = nn.Linear(width, width)
        self.routed_lin = nn.Linear(width, width)
        self.collective_fault = 0.0
        self.seen = None
        self._forward_entry = self._entry

    def _entry(self, hidden_states, router_logits, shared_experts_input, input_ids, layer_name, dim, dtype):
        return self.shared_lin(shared_experts_input), self.routed_lin(hidden_states)

    def _maybe_reduce_final_output(self, states, trunc_size, output_is_reduced=None):
        out = states * 4 + self.collective_fault            # stands in for the TP4 sum
        return out[..., :trunc_size] if trunc_size is not None else out

    def forward(self, hidden_states, router_logits):
        result = self._forward_entry(hidden_states, router_logits, hidden_states, None, 'x', 0, torch.float32)
        shared_output, fused_output = result
        combined = shared_output + fused_output
        final = self._maybe_reduce_final_output(combined, None, False)
        self.seen = (shared_output, fused_output, combined, final)
        return final


class Ffn(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.experts = FakeRunner(width)

    def forward(self, h):
        return self.experts(hidden_states=h, router_logits=h)


class Block(nn.Module):
    def __init__(self, width, engram=False):
        super().__init__()
        self.attn = nn.Linear(width, width)
        self.ffn = Ffn(width)
        self.engram = nn.Linear(width, width) if engram else None

    def forward(self, h):
        if self.engram is not None:
            h = self.engram(h)
        a = self.attn(torch.tanh(h))
        return self.ffn(a) * 1e-2 + h, a


class Toy(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(100, width)
        self.layers = nn.ModuleList(Block(width, engram=i in (1, 14)) for i in range(LAYERS))

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        h = self.embed(input_ids)
        for layer in self.layers:
            h, _ = layer(h)
        return h


def runners(model):
    return [layer.ffn.experts for layer in model.layers]


class Seams(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.trigger = Path(self.tmp.name) / 'trigger.json'
        self.records = []
        self.model = Toy()
        self.originals = [vars(r)['_forward_entry'] for r in runners(self.model)]
        self.transport = trace.SyncTransport(self.records.append, slots=2, capacity=512)
        trace._INSTALLED.clear()
        self.tracer = trace.install(self.model, transport=self.transport, trigger=str(self.trigger),
                                    out_dir=self.tmp.name, rank=1, node='toby')

    def tearDown(self):
        self.tmp.cleanup()

    def arm(self, **spec):
        spec = {'arm': 'seam-arm-0001', 'min_tokens': 4, 'max_tokens': 64, **spec}
        self.trigger.write_text(json.dumps(spec))
        os.utime(self.trigger, ns=(time.time_ns(), time.time_ns()))

    def step(self, rows=8):
        self.model._engram_epoch = getattr(self.model, '_engram_epoch', 0) + 1
        with torch.inference_mode():
            return self.model(torch.arange(rows) % 100, torch.arange(rows), None)

    def assert_pristine(self):
        for runner, original in zip(runners(self.model), self.originals):
            self.assertIs(vars(runner)['_forward_entry'], original)
            self.assertNotIn('_maybe_reduce_final_output', vars(runner))

    def all_seams(self):
        return [f'layers.{i}.ffn.experts{s}' for i in range(LAYERS) for s in trace.SEAMS]

    def test_unarmed_runner_is_the_pinned_path(self):
        out = self.step()
        self.assert_pristine()
        self.assertEqual(self.records, [])
        self.assertEqual(self.tracer.seams, 0)
        self.assertTrue(torch.isfinite(out).all())

    def test_armed_seams_cover_every_layer_in_call_order_with_exact_digests(self):
        baseline = self.step()
        self.arm(post=['layers.39'], moe=['layers.*.ffn.experts'])
        out = self.step()
        self.assertTrue(torch.equal(out, baseline))                # no numerical change
        (record,) = self.records
        self.assertEqual(record['names'], ['<input>', '<positions>', *self.all_seams(),
                                           'layers.39:0', 'layers.39:1', '<output>'])
        self.assertEqual(record['seam_runners'], LAYERS)
        index = {n: i for i, n in enumerate(record['names'])}
        for i, runner in enumerate(runners(self.model)):
            for seam, tensor in zip(trace.SEAMS, runner.seen):
                self.assertEqual(record['hashes'][index[f'layers.{i}.ffn.experts{seam}']],
                                 trace.digest(tensor).tolist())

    def test_wrappers_pass_arguments_and_return_the_original_objects(self):
        runner = runners(self.model)[3]
        pair, final, calls = (torch.ones(2), torch.zeros(2)), torch.full((2,), 7.0), []
        runner._forward_entry = lambda *a, **k: calls.append(('entry', a, k)) or pair
        self.originals[3] = vars(runner)['_forward_entry']
        with mock.patch.object(FakeRunner, '_maybe_reduce_final_output',
                               lambda s, *a, **k: calls.append(('reduce', a, k)) or final):
            self.arm(moe=['layers.3.ffn.experts'])
            self.tracer._refresh()
            self.assertIs(runner._forward_entry(1, 2, x=3), pair)
            self.assertIs(runner._maybe_reduce_final_output(final, None, output_is_reduced=False), final)
        self.assertEqual(calls, [('entry', (1, 2), {'x': 3}),
                                 ('reduce', (final, None), {'output_is_reduced': False})])

    def test_disarm_and_rearm_restore_identity_without_double_wrapping(self):
        self.arm(moe=['layers.*.ffn.experts'])
        self.step()
        wrapped = vars(runners(self.model)[0])['_forward_entry']
        self.assertTrue(getattr(wrapped, '_claude_seam', False))
        self.arm(moe=['layers.*.ffn.experts'])                     # new mtime: rearm
        self.step()
        self.assertEqual(self.records[-1]['names'].count('layers.0.ffn.experts#routed'), 1)
        self.trigger.unlink()
        self.step()
        self.assert_pristine()
        self.assertEqual(self.tracer.handles, [])

    def test_inactive_steps_pass_through(self):
        self.arm(min_tokens=6, max_tokens=10, moe=['layers.*.ffn.experts'])
        baseline = self.step(rows=5)
        self.assertEqual(self.records, [])
        self.assertEqual(self.tracer.seams, LAYERS)
        with mock.patch.object(torch.compiler, 'is_compiling', return_value=True):
            self.step(rows=8)
        self.assertEqual(self.records, [])
        self.trigger.unlink()
        self.assertTrue(torch.equal(self.step(rows=5), baseline))

    def test_capture_and_busy_slots_do_not_record_or_block(self):
        self.arm(moe=['layers.*.ffn.experts'])
        with mock.patch.object(torch.cuda, 'is_available', return_value=True), \
                mock.patch.object(torch.cuda, 'is_current_stream_capturing', return_value=True):
            self.step()
        self.transport.free[:] = [False, False]
        self.step()
        self.assertEqual((self.records, self.tracer.dropped), ([], 1))

    def test_patterns_matching_nothing_refuse_the_trigger(self):
        for spec in ({'moe': ['layers.*.ffn.experts.routed_experts']},   # the v1 silent miss
                     {'post': ['layers.99']},
                     {'pre': ['layers.*.nothing']},
                     {'moe': ['layers.*.attn']},                          # module without callsites
                     {'moe': ['layers.*.ffn.experts', 'layers.*.attn']}):  # partial arming rolled back
            self.arm(**spec)
            with mock.patch('builtins.print') as printed:
                self.step()
            self.assertIn('trigger refused', printed.call_args_list[-1].args[0])
            self.assertEqual((self.records, self.tracer.handles, self.tracer.seams), ([], [], 0))
            self.assertIsNone(self.tracer.config)
            self.assert_pristine()

    def test_refused_then_valid_trigger_arms_cleanly(self):
        self.arm(moe=['layers.*.nope'])
        with mock.patch('builtins.print'):
            self.step()
        self.arm(moe=['layers.*.ffn.experts'])
        self.step()
        self.assertEqual(self.records[-1]['seam_runners'], LAYERS)


class Attribution(unittest.TestCase):
    """End to end: tracer records -> four simulated ranks -> analyzer."""

    def run_ranks(self, fault):
        model = Toy()
        records = []
        transport = trace.SyncTransport(records.append, slots=2, capacity=512)
        trace._INSTALLED.clear()
        with tempfile.TemporaryDirectory() as tmp:
            trigger = Path(tmp) / 't.json'
            trigger.write_text(json.dumps({'arm': 'seam-arm-0002', 'min_tokens': 4, 'max_tokens': 64,
                                           'post': ['layers.39'], 'moe': ['layers.*.ffn.experts']}))
            trace.install(model, transport=transport, trigger=str(trigger), out_dir=tmp, rank=0, node='n')
            for epoch in range(1, 6):
                model._engram_epoch = epoch
                if epoch == 5:
                    fault(model)
                with torch.inference_mode():
                    model(torch.arange(8), torch.arange(8), None)
        out = []
        for rank in range(4):
            for record in records:
                clone = dict(record, rank=rank, session=f's{rank}', _source='x')
                if rank != 2 and record['epoch'] == 5:
                    clone['hashes'] = records[0]['hashes']
                out.append(clone)
        return analyze.analyze(out, require=['layers.*.ffn.experts#post_reduce=40'])

    def test_rank_local_routed_partial(self):
        def fault(model):
            with torch.no_grad():
                model.layers[13].ffn.experts.routed_lin.bias[0] += 1e-3
        (step,) = self.run_ranks(fault)['divergent_steps']
        finding = step['first_by_rank']['2']
        self.assertEqual(finding['name'], 'layers.13.ffn.experts#routed')
        self.assertEqual(finding['differing_names'][:3], ['layers.13.ffn.experts#routed',
                                                          'layers.13.ffn.experts#pre_reduce',
                                                          'layers.13.ffn.experts#post_reduce'])

    def test_collective_only(self):
        def fault(model):
            model.layers[13].ffn.experts.collective_fault = 1e-3
        (step,) = self.run_ranks(fault)['divergent_steps']
        self.assertEqual(step['first_by_rank']['2']['name'], 'layers.13.ffn.experts#post_reduce')

    def test_coverage_gate_excludes_short_records(self):
        base = {'type': 'step', 'arm': 'a' * 8, 'rows': 8, 'wall': 0.0, 'gpu_ms': 1.0, '_source': 's'}
        names = ['<input>', '<positions>', 'layers.0.ffn.experts#post_reduce', 'layers.0:0']
        records = [dict(base, rank=r, session=f's{r}', epoch=e, seq=e, names=names,
                        hashes=[[1, 2], [3, 4], [5, 6], [7, 8]]) for r in range(4) for e in range(1, 5)]
        ok = analyze.analyze(records, require=['layers.*.ffn.experts#post_reduce'])
        self.assertEqual((ok['verdict'], ok['counts']['covered']), ('no-divergence-observed', 4))
        short = analyze.analyze(records, require=['layers.*.ffn.experts#post_reduce=40'])
        self.assertEqual((short['verdict'], short['counts']['missing_required']), ('no-coverage', 4))


class NoHostSync(unittest.TestCase):
    FORBIDDEN = {'item', 'cpu', 'tolist', 'numpy', 'synchronize', 'elapsed_time', 'query',
                 'wait_stream', 'wait_event', 'record_stream', 'clone', 'copy_'}

    def test_seam_wrappers_only_call_originals_and_record(self):
        tree = ast.parse((HERE / 'claude-moe-seams-act_trace.py').read_text())
        seams = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MoeSeams')
        inner = [n for n in ast.walk(seams) if isinstance(n, ast.FunctionDef)
                 and n.name in ('forward_entry', 'reduce_final_output')]
        self.assertEqual(len(inner), 2)
        for function in inner:
            attrs = {n.attr for n in ast.walk(function) if isinstance(n, ast.Attribute)}
            self.assertFalse(attrs & self.FORBIDDEN, function.name)
            self.assertEqual(attrs - {'active', '_record', 'Tensor'}, set(), function.name)


@unittest.skipUnless(HAVE_VLLM, 'no ~/git/vllm checkout: pinned-source checks not run')
class PinnedContract(unittest.TestCase):
    def texts(self):
        return [prep.pinned_source(p).decode() for p in (prep.RUNNER, prep.PINNED[2], prep.PINNED[3])]

    def test_pinned_callsites_conform(self):
        self.assertEqual(prep.check_callsites(*self.texts())[-1], '_maybe_reduce_final_output')

    def test_drift_is_rejected(self):
        runner, layer, model = self.texts()
        mutants = [
            (runner.replace('result = self._maybe_reduce_final_output(',
                            'result = self._maybe_reduce_final_output(result, None)\n'
                            '        result = self._maybe_reduce_final_output(', 1), layer, model),
            (runner.replace('self._forward_entry = self._select_forward()',
                            'self._entry_fn = self._select_forward()', 1), layer, model),
            (runner, layer.replace('if apply_routed_scale_to_output\n        else 1.0',
                                   'if True\n        else 1.0', 1), model),
            (runner, layer, model.replace('            is_sequence_parallel=self.use_sequence_parallel,\n        )',
                                          '            is_sequence_parallel=self.use_sequence_parallel,\n'
                                          '            apply_routed_scale_to_output=True,\n        )', 1)),
            (runner, layer, model.replace('            is_sequence_parallel=self.use_sequence_parallel,\n        )',
                                          '            is_sequence_parallel=self.use_sequence_parallel,\n'
                                          '            reduce_results=False,\n        )', 1)),
        ]
        for index, texts in enumerate(mutants):
            self.assertNotEqual(texts, (runner, layer, model), index)
            with self.assertRaises(RuntimeError, msg=str(index)):
                prep.check_callsites(*texts)

    def test_pins_match_the_image_manifest(self):
        manifest = json.loads((HERE / 'runtime.lock.json').read_text())['sources']['vllm']['files']
        lock = json.loads((HERE / prep.LOCK).read_text())
        for path, digest in lock['verify_unchanged'].items():
            self.assertEqual(manifest[path.removeprefix('vllm/')]['sha256'], digest)


class Packaging(unittest.TestCase):
    def test_v1_kit_is_untouched(self):
        amendment = json.loads((HERE / prep.BASE_AMENDMENT).read_text())['candidate']
        v1 = (HERE / prep.V1_LOCK).read_bytes()
        self.assertEqual(hashlib.sha256(v1).hexdigest(), amendment['diagnostic']['lock_sha256'])
        for name, digest in json.loads(v1)['inputs'].items():
            self.assertEqual(hashlib.sha256((HERE / name).read_bytes()).hexdigest(), digest, name)

    def test_patch_is_the_helper_only(self):
        patch = (HERE / prep.PATCH).read_text()
        files = [l for l in patch.splitlines() if l.startswith(('--- ', '+++ '))]
        self.assertEqual(files, ['--- a/' + prep.HELPER_TARGET, '+++ b/' + prep.HELPER_TARGET])

    def test_lock_and_dockerfile(self):
        lock = json.loads((HERE / prep.LOCK).read_text())
        for name, digest in lock['inputs'].items():
            self.assertEqual(hashlib.sha256((HERE / name).read_bytes()).hexdigest(), digest, name)
        docker = (HERE / 'Dockerfile.claude-moe-seams').read_text()
        copy = [l for l in docker.splitlines() if l.startswith('COPY ')][0].split()[1:-1]
        self.assertEqual(sorted(copy), sorted([*lock['inputs'], prep.LOCK]))
        self.assertIn('FROM ' + lock['base_image_id'], docker)
        v1 = json.loads((HERE / prep.V1_LOCK).read_text())
        self.assertEqual(lock['targets'][prep.HELPER_TARGET]['input_sha256'],
                         v1['targets'][prep.HELPER_TARGET]['output_sha256'])

    @unittest.skipUnless(HAVE_VLLM, 'no ~/git/vllm checkout: simulated install not run')
    def test_simulated_install_is_confined_and_pins_the_runner(self):
        lock = json.loads((HERE / prep.LOCK).read_text())
        with tempfile.TemporaryDirectory() as tmp:
            kit, opt = Path(tmp) / 'kit', Path(tmp) / 'opt'
            kit.mkdir()
            for name in [*lock['inputs'], prep.LOCK]:
                shutil.copy(HERE / name, kit / name)
            helper = opt / prep.HELPER_TARGET
            helper.parent.mkdir(parents=True)
            shutil.copy(HERE / 'claude_act_trace.py', helper)
            for path in prep.PINNED:
                target = opt / 'vllm' / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(prep.pinned_source(path))
            (opt / 'b12x/b12x').mkdir(parents=True)
            (opt / 'b12x/b12x/native.so').write_bytes(b'x')
            source = (HERE / 'claude_install_moe_seams.py').read_text()
            script = Path(tmp) / 'install.py'
            script.write_text(source.replace("Path('/opt/ds41-moe-seams')", f"Path({str(kit)!r})")
                              .replace("Path('/opt/jovian-judgement')", f"Path({str(opt)!r})"))
            run = lambda: subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
            first = run()
            self.assertIn('MOE-SEAMS-INSTALL-PASS', first.stdout, first.stderr)
            self.assertEqual(helper.read_bytes(), (HERE / prep.HELPER_SOURCE).read_bytes())
            self.assertNotEqual(run().returncode, 0)                 # base is no longer v1
            shutil.copy(HERE / 'claude_act_trace.py', helper)
            runner = opt / 'vllm' / prep.RUNNER
            runner.write_bytes(runner.read_bytes() + b'\n')
            drift = run()
            self.assertNotEqual(drift.returncode, 0)
            self.assertIn('pinned seam contract', drift.stderr)


if __name__ == '__main__':
    unittest.main()
