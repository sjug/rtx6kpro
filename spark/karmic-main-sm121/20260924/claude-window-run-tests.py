"""Focused tests for the collector's separate indexer kind (claude-window-run.py). Not part of any image lock.
  .venv-snapshot-cpu/bin/python claude-window-run-tests.py
"""
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run = load('claude_window_run', HERE / 'claude-window-run.py')


class IndexerCollect(unittest.TestCase):
    def test_indexer_receipts_are_a_separate_collect_kind(self):
        text = json.dumps({'file': '/cache/claude-decision-row/toby-rank1-window-0001-indexer.pt', 'sha256': 'b' * 64, 'problems': []})
        record = run.parse_consumed(text, 'window-0001', 1, 'indexer')
        self.assertTrue(record['host_file'].endswith('/toby-rank1-window-0001-indexer.pt'))
        for kind in ('window',):
            with self.assertRaises(ValueError):
                run.parse_consumed(text, 'window-0001', 1, kind)
        with self.assertRaises(ValueError):
            run.parse_consumed(text, 'window-0001', 2, 'indexer')
        self.assertEqual(run.capture_name('indexer', 'dusty', 0, 'tok-00000001'), 'dusty-rank0-tok-00000001-indexer.pt')
        self.assertEqual(run.capture_name('window', 'dusty', 0, 'tok-00000001'), run.window_name('dusty', 0, 'tok-00000001'))

    def fake_ssh(self, node, command):
        if 'podman logs' in command:
            prefix = 'DS41-INDEXER' if 'DS41-INDEXER' in command else 'CLAUDE-WINDOW'
            self.log_queries.append(prefix)
            return self.logs.get((node, prefix), '')
        if command.startswith('cat '):
            path = command.split()[1]
            return self.receipts.get((node, path.rsplit('/', 1)[1]), '')
        raise AssertionError('unexpected ssh: ' + command)

    def test_wait_and_collect_indexer_without_touching_window_outputs(self):
        rank_of = {n: i for i, n in enumerate(run.NODES)}
        self.logs, self.log_queries = {}, []
        self.receipts = {(n, f'tok-00000001-rank{rank_of[n]}.indexer-consumed'): json.dumps({
            'file': f'/cache/claude-decision-row/{n}-rank{rank_of[n]}-tok-00000001-indexer.pt', 'sha256': 'c' * 64,
            'problems': []}) for n in run.NODES}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / 'summary.json').write_text(json.dumps({'token': 'tok-00000001', 'remote_dir': '/remote/receipt'}))
            fake_driver = types.SimpleNamespace(ssh=self.fake_ssh, OUT_DIR='/cache/claude-decision-row', NAME='c',
                                                host_path=run.driver().host_path, gather_script=run.driver().gather_script)
            with mock.patch.object(run, 'driver', lambda: fake_driver):
                captures = run.collect(out, 5, 'indexer', run=lambda argv, input, text: 'GATHERED x\n' * 4)
            self.assertEqual(set(captures), set(run.NODES))
            self.assertEqual(set(self.log_queries), {'DS41-INDEXER'})
            self.assertTrue((out / 'indexer-captures.json').is_file())
            self.assertTrue((out / 'indexer-gather.sh').is_file() and (out / 'indexer-gather.log').is_file())
            self.assertIn('kirby-rank3-tok-00000001-indexer.pt', (out / 'indexer-gather.sh').read_text())
            self.assertFalse(any(p.name.startswith('window-') for p in out.iterdir()))
            # an indexer abort line stops the wait; window helper lines are never consulted for the indexer kind
            self.logs[('rusty', 'DS41-INDEXER')] = 'DS41-INDEXER aborted token=tok-00000002 RuntimeError: x\n'
            (out / 'summary.json').write_text(json.dumps({'token': 'tok-00000002', 'remote_dir': '/remote/receipt'}))
            with mock.patch.object(run, 'driver', lambda: fake_driver), self.assertRaisesRegex(RuntimeError, 'indexer helper aborted'):
                run.collect(out, 5, 'indexer', run=lambda argv, input, text: '')

    def test_window_collect_path_is_unchanged(self):
        rank_of = {n: i for i, n in enumerate(run.NODES)}
        self.logs, self.log_queries = {}, []
        self.receipts = {(n, f'tok-00000003-rank{rank_of[n]}.window-consumed'): json.dumps({
            'file': f'/cache/claude-decision-row/{n}-rank{rank_of[n]}-tok-00000003-window.pt', 'sha256': 'd' * 64,
            'problems': []}) for n in run.NODES}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / 'summary.json').write_text(json.dumps({'token': 'tok-00000003', 'remote_dir': '/remote/receipt'}))
            fake_driver = types.SimpleNamespace(ssh=self.fake_ssh, OUT_DIR='/cache/claude-decision-row', NAME='c',
                                                host_path=run.driver().host_path, gather_script=run.driver().gather_script)
            with mock.patch.object(run, 'driver', lambda: fake_driver):
                run.collect(out, 5, run=lambda argv, input, text: 'GATHERED x\n' * 4)
            self.assertEqual(set(self.log_queries), {'CLAUDE-WINDOW'})
            self.assertEqual(sorted(p.name for p in out.iterdir()),
                             ['dusty-window-helper.log', 'kirby-window-helper.log', 'rusty-window-helper.log',
                              'summary.json', 'toby-window-helper.log', 'window-captures.json', 'window-gather.log',
                              'window-gather.sh', 'window-partial-captures.json'])



if __name__ == '__main__':
    unittest.main()
