"""CPU tests for claude_verify_moe_capture.py, on receipts produced by the real v3 helper.
  .venv-snapshot-cpu/bin/python -m unittest claude_test_verify_moe_capture
"""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
import unittest

import torch

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fixtures = load('claude_verify_fixtures', HERE / 'claude_test_moe_capture.py')
trace = fixtures.trace
verifier = load('claude_verify_under_test', HERE / 'claude_verify_moe_capture.py')
ARM = 'capture-arm-0001'


class Receipts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.model = fixtures.Toy()
        self.transport = trace.SyncTransport(None, slots=2, capacity=1024)
        trace._INSTALLED.clear()
        trigger = self.dir / 'trigger.json'
        self.tracer = trace.install(self.model, transport=self.transport, trigger=str(trigger),
                                    out_dir=str(self.dir), rank=1, node='toby')
        self.transport.sink = self.tracer._sink
        trigger.write_text(json.dumps({'arm': ARM, 'min_tokens': 4, 'max_tokens': 64, 'post': ['layers.39'],
                                       'moe': ['layers.*.ffn.experts'],
                                       'capture': {'rows': [8], 'moe': ['layers.1[0-3].ffn.experts']}}))
        os.utime(trigger, ns=(time.time_ns(), time.time_ns()))

    def tearDown(self):
        self.tmp.cleanup()

    def step(self, fault=False):
        if fault:
            fixtures.runner(self.model, 12).fault = lambda routed: routed.add_(0.25)
        self.model._engram_epoch = getattr(self.model, '_engram_epoch', 0) + 1
        with torch.inference_mode(), unittest.mock.patch('builtins.print'):
            self.model(torch.arange(8) % 100, torch.arange(8), None)

    def run_steps(self, faults, total):
        for index in range(1, total + 1):
            self.step(fault=index in faults)

    def capture_file(self):
        (path,) = self.dir.glob('*-capture.pt')
        return path

    def verify(self, **kwargs):
        return verifier.verify(verifier.expand([self.dir]), kwargs.pop('arm', ARM), **kwargs)


import unittest.mock  # noqa: E402


class Verify(Receipts):
    def test_captured_fault_is_nonmodal_and_verified(self):
        self.run_steps({5}, 7)
        report = self.verify()
        self.assertEqual((report['verdict'], report['failures']), ('pass', []))
        (row,) = report['verified']
        self.assertEqual((row['classification'], row['epoch'], row['rank'], row['runner'], row['group_size']),
                         ('nonmodal', 5, 1, 'layers.12.ffn.experts', 7))
        self.assertTrue(row['reference_is_modal'])
        self.assertTrue(all(row['route_inputs_modal'].values()))
        self.assertFalse(row['input_changed_by_op'])

    def test_faulty_reference_captures_a_modal_call(self):
        self.run_steps({1}, 6)
        (row,) = self.verify()['verified']
        self.assertEqual((row['classification'], row['epoch']), ('modal', 2))
        self.assertFalse(row['reference_is_modal'])

    def test_small_group_is_undecided(self):
        self.run_steps({2}, 4)
        (row,) = self.verify(min_group=10)['verified']
        self.assertEqual(row['classification'], 'undecided')

    def test_downloads_manifest_is_checked(self):
        self.run_steps({3}, 5)
        path = self.capture_file()
        good = [{'node': 'toby', 'file': str(path), 'bytes': path.stat().st_size,
                 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}]
        manifest = self.dir / 'downloads.json'
        manifest.write_text(json.dumps(good))
        self.assertEqual(self.verify(downloads=manifest)['verdict'], 'pass')
        manifest.write_text(json.dumps([dict(good[0], sha256='0' * 64)]))
        self.assertEqual(self.verify(downloads=manifest)['verdict'], 'fail')
        manifest.write_text(json.dumps(good + [dict(good[0], file='toby-1-g9-capture.pt')]))
        self.assertIn('not found', self.verify(downloads=manifest)['failures'][0]['error'])

    def test_no_capture_is_reported_as_no_captures(self):
        self.run_steps(set(), 4)
        self.assertEqual(self.verify()['verdict'], 'no-captures')


class Mismatches(Receipts):
    def tamper(self, change):
        path = self.capture_file()
        saved = torch.load(path, weights_only=True)
        change(saved)
        torch.save(saved, path)

    def failing(self, **kwargs):
        report = self.verify(**kwargs)
        self.assertEqual(report['verdict'], 'fail')
        return report['failures'][0]['error']

    def setUp(self):
        super().setUp()
        self.run_steps({4}, 6)

    def test_tensor_bytes(self):
        self.tamper(lambda s: s['tensors']['routed'].view(-1)[0].add_(1))
        self.assertIn('routed digest differs', self.failing())

    def test_route_input_bytes(self):
        self.tamper(lambda s: s['tensors']['topk_ids'].view(-1)[0].copy_((s['tensors']['topk_ids'].view(-1)[0] + 1) % 6))
        self.assertIn('topk_ids digest differs', self.failing())

    def test_identity(self):
        for change, text in ((lambda s: s.update(rank=2), 'expected one saved status line'),
                             (lambda s: s.update(node='rusty'), 'node differs'),
                             (lambda s: s['meta'].update(latched=0), 'latch word is clear'),
                             (lambda s: s['meta'].update(epoch=3), 'status epoch')):
            original = torch.load(self.capture_file(), weights_only=True)
            self.tamper(change)
            self.assertIn(text, self.failing())
            torch.save(original, self.capture_file())

    def test_layout(self):
        self.tamper(lambda s: s['tensors'].update(logits=s['tensors']['logits'].double()))
        self.assertIn('logits layout', self.failing())

    def test_other_arm(self):
        self.assertIn('arm', self.failing(arm='another-arm-0002'))

    def test_renamed_file_and_missing_file(self):
        path = self.capture_file()
        path.rename(path.with_name('toby-1-g1-capture.pt'))
        errors = [f['error'] for f in self.verify()['failures']]
        self.assertTrue(any('session pid' in e for e in errors))
        self.assertTrue(any('without a capture file' in e for e in errors))

    def test_loader_is_weights_only(self):
        tree = ast.parse((HERE / 'claude_verify_moe_capture.py').read_text())
        loads = [c for c in ast.walk(tree) if isinstance(c, ast.Call) and ast.unparse(c.func) == 'torch.load']
        self.assertTrue(loads)
        for call in loads:
            self.assertEqual({k.arg: ast.unparse(k.value) for k in call.keywords},
                             {'map_location': "'cpu'", 'weights_only': 'True'})


if __name__ == '__main__':
    unittest.main()
