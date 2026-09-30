"""Tests for the activation-digest diagnostic. Run with the task-local CPU venv:
  .venv-snapshot-cpu/bin/python -m unittest claude_test_act_trace
Torch CPU only; CUDA transport, streams and events are not exercised here.
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


trace = load('claude_act_trace_under_test', HERE / 'claude_act_trace.py')
analyze = load('claude_act_analyze_under_test', HERE / 'claude_act_analyze.py')
prep = load('claude_prepare_act_trace_under_test', HERE / 'claude_prepare_act_trace.py')


def reference(tensor, period, chunk):
    """Pure-Python digest: int64 two's-complement sums of raw words."""
    raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8).tolist()
    data = bytes(raw)
    width = 4 if len(data) % 4 == 0 else 2 if len(data) % 2 == 0 else 1
    words = [int.from_bytes(data[i:i + width], 'little', signed=width > 1) for i in range(0, len(data), width)]
    wrap = lambda v: (v + 2**63) % 2**64 - 2**63
    lane0 = wrap(sum(words))
    lane1 = wrap(sum(w * ((i % chunk) % period + 1) for i, w in enumerate(words)))
    return [lane0, lane1]


class Digest(unittest.TestCase):
    def setUp(self):
        self.saved = (trace.PERIOD, trace.CHUNK)
        trace.PERIOD, trace.CHUNK = 7, 21                    # small: exercise many chunks
        trace._WEIGHTS.clear()

    def tearDown(self):
        trace.PERIOD, trace.CHUNK = self.saved
        trace._WEIGHTS.clear()

    def test_matches_reference_across_chunks_and_dtypes(self):
        g = torch.Generator().manual_seed(1)
        for tensor in (torch.randn(50, 3, generator=g).to(torch.bfloat16), torch.randn(97, generator=g),
                       torch.randint(-2**31, 2**31 - 1, (43,), dtype=torch.int32, generator=g),
                       torch.tensor([True, False, True]), torch.tensor([1, 2, 3], dtype=torch.uint8),
                       torch.zeros(0)):
            self.assertEqual(trace.digest(tensor).tolist(), reference(tensor, 7, 21), tensor.dtype)

    def test_sensitivity_and_layout_independence(self):
        x = torch.randn(64, 8).to(torch.bfloat16)
        base = trace.digest(x).tolist()
        flipped = x.clone().view(torch.int16)
        flipped[37, 5] ^= 1
        self.assertNotEqual(trace.digest(flipped.view(torch.bfloat16)).tolist()[0], base[0])
        swapped = x.clone()
        swapped[[3, 40]] = swapped[[40, 3]]
        other = trace.digest(swapped).tolist()
        if not torch.equal(x[3], x[40]):
            self.assertEqual(other[0], base[0])               # same multiset of words
            self.assertNotEqual(other[1], base[1])            # position-sensitive lane
        strided = torch.randn(8, 64).to(torch.bfloat16).t()   # non-contiguous
        self.assertEqual(trace.digest(strided).tolist(), trace.digest(strided.contiguous()).tolist())

    def test_weight_buffer_is_bounded(self):
        for n in (5, 50, 500):
            trace.digest(torch.randn(n))
        self.assertEqual([w.numel() for w in trace._WEIGHTS.values()], [21])


class Block(nn.Module):
    def __init__(self, width, engram=False):
        super().__init__()
        self.attn = nn.Module()
        self.attn.wo_b = nn.Linear(width, width)
        self.ffn = nn.Linear(width, width)
        self.engram = nn.Linear(width, width) if engram else None

    def forward(self, h):
        if self.engram is not None:
            h = self.engram(h)
        a = self.attn.wo_b(torch.tanh(h))
        return self.ffn(a) + h, a


class Toy(nn.Module):
    def __init__(self, layers=4, width=16):
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(100, width)
        self.layers = nn.ModuleList(Block(width, engram=i == 1) for i in range(layers))

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        h = self.embed(input_ids)
        for layer in self.layers:
            h, _ = layer(h)
        return h


class TracerCases(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.trigger = Path(self.tmp.name) / 'trigger.json'
        self.records = []
        self.model = Toy()
        self.transport = trace.SyncTransport(self.records.append, slots=2, capacity=64)
        trace._INSTALLED.clear()
        self.tracer = trace.install(self.model, transport=self.transport, trigger=str(self.trigger),
                                    out_dir=self.tmp.name, rank=2, node='rusty')

    def tearDown(self):
        self.tmp.cleanup()

    def arm(self, **spec):
        spec = {'arm': 'test-arm-0001', 'min_tokens': 4, 'max_tokens': 64, **spec}
        self.trigger.write_text(json.dumps(spec))
        os.utime(self.trigger, ns=(time.time_ns(), time.time_ns()))

    def step(self, rows=8, ids=None):
        self.model._engram_epoch = getattr(self.model, '_engram_epoch', 0) + 1
        ids = torch.arange(rows) % 100 if ids is None else ids
        with torch.inference_mode():
            self.model(ids, torch.arange(rows), None)

    def module_hooks(self):
        return sum(len(m._forward_hooks) + len(m._forward_pre_hooks)
                   for name, m in self.model.named_modules() if name)

    def test_second_model_in_process_is_not_traced(self):
        other = Toy()
        with mock.patch('builtins.print'):
            self.assertIsNone(trace.install(other, transport=self.transport, trigger=str(self.trigger)))
        self.assertEqual(len(other._forward_pre_hooks), 0)

    def test_disarmed_registers_nothing_and_records_nothing(self):
        self.step()
        self.assertEqual(self.module_hooks(), 0)
        self.assertEqual(self.records, [])

    def test_armed_records_call_order_and_disarms(self):
        self.arm()
        self.step()
        self.assertGreater(self.module_hooks(), 0)
        names = self.records[0]['names']
        self.assertEqual(names, ['<input>', '<positions>', 'layers.0:0', 'layers.0:1', 'layers.1.engram',
                                 'layers.1:0', 'layers.1:1', 'layers.2:0', 'layers.2:1', '<output>'])
        self.assertEqual(len(self.records[0]['hashes']), len(names))
        record = self.records[0]
        self.assertEqual((record['rank'], record['node'], record['seq'], record['epoch'], record['arm']),
                         (2, 'rusty', 1, 1, 'test-arm-0001'))
        self.assertEqual(record['session'], self.tracer.session)
        self.trigger.unlink()
        self.step()
        self.assertEqual(self.module_hooks(), 0)
        self.assertEqual(len(self.records), 1)

    def test_row_window_and_pattern_selection(self):
        self.arm(min_tokens=6, max_tokens=10, post=['layers.*:1', 'layers.*.ffn'], pre=['layers.*.attn.wo_b'])
        self.step(rows=5)
        self.step(rows=11)
        self.assertEqual(self.records, [])
        self.step(rows=6)
        expected = ['<input>', '<positions>']
        for i in range(4):
            expected += [f'>layers.{i}.attn.wo_b:0', f'layers.{i}.ffn', f'layers.{i}:1']
        self.assertEqual(self.records[0]['names'], expected + ['<output>'])

    def test_new_arm_token_restarts_seq(self):
        self.arm()
        self.step()
        self.step()
        self.arm(arm='test-arm-0002')
        self.step()
        self.assertEqual([(r['arm'], r['seq']) for r in self.records],
                         [('test-arm-0001', 1), ('test-arm-0001', 2), ('test-arm-0002', 1)])

    def test_identical_steps_identical_records(self):
        self.arm()
        self.step()
        self.step()
        self.assertEqual(self.records[0]['hashes'], self.records[1]['hashes'])
        self.assertEqual(self.records[1]['seq'], 2)

    def test_capture_and_compile_skip_everything(self):
        self.arm()
        with mock.patch.object(torch.cuda, 'is_available', return_value=True), \
                mock.patch.object(torch.cuda, 'is_current_stream_capturing', return_value=True):
            self.step()
        with mock.patch.object(torch.compiler, 'is_compiling', return_value=True):
            self.step()
        self.assertEqual((self.records, self.tracer.seq), ([], 0))

    def test_busy_slots_drop_without_blocking(self):
        self.arm()
        self.transport.free[:] = [False, False]
        self.step()
        self.assertEqual((self.records, self.tracer.seq, self.tracer.dropped), ([], 1, 1))
        self.transport.free[:] = [True, True]
        self.step()
        self.assertEqual(self.records[0]['dropped_before'], 1)
        self.assertEqual(self.records[0]['seq'], 2)

    def test_overflow_is_flagged(self):
        small = trace.SyncTransport(self.records.append, slots=1, capacity=5)
        self.tracer.transport = small
        self.arm()
        self.step()
        self.assertTrue(self.records[0]['overflow'])
        self.assertEqual(len(self.records[0]['hashes']), 5)

    def test_invalid_trigger_is_ignored(self):
        for text in ('{bad', json.dumps({'arm': 'test-arm-0001', 'min_tokens': 0, 'max_tokens': 5}),
                     json.dumps({'arm': 'test-arm-0001', 'min_tokens': 4, 'max_tokens': 4096}),
                     json.dumps({'arm': 'test-arm-0001', 'min_tokens': 4, 'max_tokens': 8, 'extra': 1}),
                     json.dumps({'min_tokens': 4, 'max_tokens': 8}),
                     json.dumps({'arm': 'SHORT', 'min_tokens': 4, 'max_tokens': 8})):
            self.trigger.write_text(text)
            os.utime(self.trigger, ns=(time.time_ns(), time.time_ns()))
            with mock.patch('builtins.print'):
                self.step()
            self.assertEqual((self.records, self.module_hooks()), ([], 0))

    def test_localizes_an_injected_single_layer_divergence(self):
        self.arm()
        for _ in range(4):
            self.step()
        with torch.no_grad():
            self.model.layers[2].ffn.bias[0] += 1e-3           # the "race" on one rank, one layer
        self.step()
        records = []
        for rank in range(4):
            for record in self.records:
                clone = dict(record, rank=rank, session=f's{rank}', _source='x')
                if rank != 2 and record['epoch'] == 5:
                    clone['hashes'] = self.records[0]['hashes']
                records.append(clone)
        report = analyze.analyze(records)
        (step,) = report['divergent_steps']
        self.assertEqual((step['epoch'], step['divergent_ranks'], step['clean_ranks']), (5, [2], [0, 1, 3]))
        self.assertEqual(step['earliest_name'], 'layers.2:0')
        self.assertEqual(report['verdict'], 'divergence-localized')


class Patterns(unittest.TestCase):
    def test_member_suffix(self):
        self.assertEqual(trace.split_member('layers.14:1'), ('layers.14', 1))
        self.assertEqual(trace.split_member('layers.14'), ('layers.14', None))

    def test_sparse_defaults_have_no_bypassed_modules(self):
        spec = trace.parse_config(json.dumps({'arm': 'a-12345678', 'min_tokens': 1, 'max_tokens': 8}))
        self.assertEqual(spec['pre'], [])
        self.assertFalse(any('wo_' in p or p.endswith('.attn') or p.endswith('.ffn') for p in spec['post']))
        self.assertEqual(len(spec['post']), 10)

    def test_star_is_one_segment(self):
        self.assertTrue(trace.matches('layers.12', 'layers.*'))
        self.assertFalse(trace.matches('layers.12.attn.wo_b', 'layers.*'))
        self.assertTrue(trace.matches('layers.12.attn.wo_b', 'layers.*.attn.wo_b'))
        self.assertTrue(trace.matches('layers.3.engram', 'layers.[0-9].engram'))
        self.assertFalse(trace.matches('layers.14.engram', 'layers.[0-9].engram'))


class NoHostSync(unittest.TestCase):
    FORWARD = {'digest', '_begin', '_record', '_post', '_pre', '_end'}
    FORBIDDEN = {'item', 'cpu', 'tolist', 'numpy', 'synchronize', 'elapsed_time', 'query'}

    def test_forward_path_has_no_host_synchronizing_call(self):
        tree = ast.parse((HERE / 'claude_act_trace.py').read_text())
        found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in self.FORWARD:
                for call in ast.walk(node):
                    if isinstance(call, ast.Attribute) and call.attr in self.FORBIDDEN:
                        found.append((node.name, call.attr))
        self.assertEqual(found, [])

    def test_cuda_submit_waits_only_on_side_stream(self):
        tree = ast.parse((HERE / 'claude_act_trace.py').read_text())
        cuda = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CudaTransport')
        submit = next(n for n in cuda.body if isinstance(n, ast.FunctionDef) and n.name == 'submit')
        attrs = {n.attr for n in ast.walk(submit) if isinstance(n, ast.Attribute)}
        self.assertTrue({'wait_event', 'record', 'put'} <= attrs)
        self.assertFalse(attrs & self.FORBIDDEN)


class Analyzer(unittest.TestCase):
    GOOD = [[1, 1], [2, 2], [3, 3]]

    def rec(self, rank, epoch, hashes=None, names=None, rows=8, key=(1, 2), **extra):
        names = names or ['<input>', '<positions>', 'a', 'b', 'c']
        hashes = self.GOOD if hashes is None else hashes
        record = {'type': 'step', 'arm': 'arm-00000001', 'session': f's{rank}', 'epoch': epoch,
                  'rank': rank, 'seq': epoch, 'rows': rows, 'names': names, 'wall': float(epoch or 0),
                  'hashes': [list(key), [3, 4]] + hashes, 'gpu_ms': 10.0 + (epoch or 0), '_source': 's'}
        record.update(extra)
        return record

    def base(self, epochs=range(1, 6), ranks=range(4)):
        return [self.rec(r, e) for r in ranks for e in epochs]

    def test_localizes_and_reports_clean_ranks(self):
        records = self.base()
        records[3 * 5 + 2] = self.rec(3, 3, [[1, 1], [9, 9], [8, 8]])
        report = analyze.analyze(records)
        (step,) = report['divergent_steps']
        self.assertEqual((step['epoch'], step['divergent_ranks'], step['earliest_name']), (3, [3], 'b'))
        self.assertEqual(step['first_by_rank']['3']['differing'], 2)
        self.assertEqual(report['counts']['clean'], 4)

    def test_mixed_arms_and_sessions_are_refused(self):
        records = self.base() + [self.rec(0, 9, arm='arm-00000002')]
        with self.assertRaisesRegex(analyze.AnalysisRefused, 'span arms'):
            analyze.analyze(records)
        self.assertEqual(analyze.analyze(records, arm='arm-00000001')['verdict'], 'no-divergence-observed')
        restarted = self.base() + [self.rec(1, 7, session='s1-restarted')]
        with self.assertRaisesRegex(analyze.AnalysisRefused, 'several sessions'):
            analyze.analyze(restarted)

    def test_seq_is_not_the_join_key(self):
        # Rank 1 armed one step late: its seq is shifted but epochs still align.
        records = [self.rec(r, e, seq=e - (r == 1)) for r in range(4) for e in range(1, 6)]
        self.assertEqual(analyze.analyze(records)['counts']['aligned'], 5)

    def test_incomplete_misaligned_duplicate_and_no_epoch_are_not_evidence(self):
        records = self.base()
        records.append(self.rec(0, 6))                                    # other ranks missing
        records += [self.rec(r, 7, key=(1, 2) if r else (5, 5)) for r in range(4)]   # keys disagree
        records += [self.rec(r, 8) for r in range(4)] + [self.rec(2, 8)]  # duplicate on rank 2
        records += [self.rec(r, None) for r in range(4)]
        counts = analyze.analyze(records)['counts']
        self.assertEqual((counts['incomplete'], counts['misaligned'], counts['duplicate'], counts['no_epoch']),
                         (1, 1, 1, 4))
        self.assertEqual(counts['aligned'], 5)
        self.assertEqual(analyze.analyze(records, ranks=[0, 1, 2, 3, 4])['verdict'], 'no-coverage')

    def test_uncovered_and_overflow_never_imply_coverage(self):
        only_keys = [self.rec(r, e, hashes=[[7, 7]], names=['<input>', '<positions>', '<output>'])
                     for r in range(4) for e in range(1, 6)]
        report = analyze.analyze(only_keys)
        self.assertEqual((report['verdict'], report['counts']['uncovered']), ('no-coverage', 5))
        overflowed = [self.rec(r, e, overflow=True) for r in range(4) for e in range(1, 6)]
        report = analyze.analyze(overflowed)
        self.assertEqual((report['verdict'], report['counts']['overflow']), ('no-coverage', 5))

    def test_small_groups_are_undecided(self):
        report = analyze.analyze(self.base(epochs=range(1, 3)))
        self.assertEqual(report['verdict'], 'no-coverage')
        self.assertEqual(report['counts']['undecided_steps'], 2)

    def test_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            lines = self.base()
            lines[4] = self.rec(0, 5, [[1, 1], [0, 0], [3, 3]])
            (Path(tmp) / 'x.jsonl').write_text(''.join(json.dumps(line) + '\n' for line in lines))
            result = subprocess.run([sys.executable, str(HERE / 'claude_act_analyze.py'), tmp, '--ranks', '0,1,2,3'],
                                    capture_output=True, text=True)
            self.assertIn('epoch 5 rows 8: divergent ranks [0] clean [1, 2, 3] earliest b', result.stdout)
            (Path(tmp) / 'y.jsonl').write_text(json.dumps(self.rec(0, 1, arm='arm-00000009')) + '\n')
            refused = subprocess.run([sys.executable, str(HERE / 'claude_act_analyze.py'), tmp],
                                     capture_output=True, text=True)
            self.assertEqual(refused.returncode, 2)
            self.assertIn('refused', refused.stdout)


class Packaging(unittest.TestCase):
    def test_model_patch_scope(self):
        image, _, old, new = prep.compose()
        repair = json.loads((HERE / 'engram-repair.lock.json').read_text())
        self.assertEqual(hashlib.sha256(old.encode()).hexdigest(),
                         repair['targets'][prep.MODEL_TARGET]['output_sha256'])
        added = [l.strip() for l in new.splitlines() if l not in old.splitlines()]
        self.assertEqual(added, ['# DIAGNOSTIC ONLY: activation digests of eager prefill steps.',
                                 'import vllm.models.deepseek_v4_1.claude_act_trace as _act_trace',
                                 '# DIAGNOSTIC ONLY: two step hooks; module hooks exist only while the',
                                 '# /cache/ds41-act-trace.json trigger is present.',
                                 '_act_trace.install(self)'])
        init = next(n for n in ast.walk(ast.parse(new)) if isinstance(n, ast.ClassDef)
                    and n.name == 'DeepseekV4Model')
        init = next(n for n in init.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        self.assertEqual(ast.unparse(init.body[-1]), '_act_trace.install(self)')

    def test_lock_and_dockerfile(self):
        lock = json.loads((HERE / 'claude-act-trace.lock.json').read_text())
        for name, digest in lock['inputs'].items():
            self.assertEqual(hashlib.sha256((HERE / name).read_bytes()).hexdigest(), digest, name)
        docker = (HERE / 'Dockerfile.claude-act-trace').read_text()
        copy = [l for l in docker.splitlines() if l.startswith('COPY ')][0].split()[1:-1]
        self.assertEqual(sorted(copy), sorted([*lock['inputs'], 'claude-act-trace.lock.json']))
        self.assertIn('FROM ' + lock['base_image_id'], docker)

    def test_simulated_install_is_confined(self):
        lock = json.loads((HERE / 'claude-act-trace.lock.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            kit, opt = Path(tmp) / 'kit', Path(tmp) / 'opt'
            kit.mkdir()
            for name in [*lock['inputs'], 'claude-act-trace.lock.json']:
                shutil.copy(HERE / name, kit / name)
            model = opt / prep.MODEL_TARGET
            model.parent.mkdir(parents=True)
            shutil.copy(HERE / prep.MODEL_SOURCE, model)
            (opt / 'b12x/b12x').mkdir(parents=True)
            (opt / 'b12x/b12x/native.so').write_bytes(b'x')
            source = (HERE / 'claude_install_act_trace.py').read_text()
            script = Path(tmp) / 'install.py'
            script.write_text(source.replace("Path('/opt/ds41-act-trace')", f"Path({str(kit)!r})")
                              .replace("Path('/opt/jovian-judgement')", f"Path({str(opt)!r})"))
            first = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
            self.assertIn('ACT-TRACE-INSTALL-PASS', first.stdout, first.stderr)
            again = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
            self.assertNotEqual(again.returncode, 0)


if __name__ == '__main__':
    unittest.main()
