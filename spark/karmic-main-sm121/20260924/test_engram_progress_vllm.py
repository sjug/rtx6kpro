"""CPU integration-order tests, not evidence of GPU progress or determinism."""
import ast
import contextlib
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import prepare_engram_progress_vllm as prepare


class ProgressIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.log = []
        log = self.log

        class Tensor:
            def __getitem__(self, index):
                return self

            def fill_(self, value):
                log.append(('expected', value))

        class Event:
            def record(self):
                log.append('record-ready')

        class Stream:
            def wait_event(self, event):
                log.append('wait-ready')

        @contextlib.contextmanager
        def stream_context(stream):
            log.append('enter-side')
            yield
            log.append('exit-side')

        class Publish:
            def __getitem__(self, grid):
                def run(status, ready, epoch):
                    log.append(('publish', status, epoch))
                return run

        self.job = object()

        def enqueue(bindings, counts, *, status_host, clear_tail):
            log.append(('enqueue', bindings, counts, status_host, clear_tail))
            return self.job

        self.native = SimpleNamespace(enqueue_lookups=enqueue)
        namespace = {
            'torch': SimpleNamespace(cuda=SimpleNamespace(Event=Event, stream=stream_context)),
            'engram_native': self.native, '_publish_engram_epoch': Publish(),
        }
        # Exercise the actual generated method, not a second implementation.
        code = 'class Model:\n' + prepare.START
        exec(compile(code, '<generated-start-engram>', 'exec'), namespace)
        self.model = namespace['Model']()
        self.model._engram_epoch = 7
        self.model._engram_epochs = Tensor()
        self.model._engram_stream = lambda: Stream()
        self.model._engram_io_status = SimpleNamespace(host_view='host', device_view='device')

    def test_producer_and_publication_submitted_before_return(self):
        self.model._start_engram_job(['a', 'b'], [385, 385])
        self.assertEqual(self.log, [
            ('expected', 8), 'record-ready', 'enter-side', 'wait-ready',
            ('enqueue', ['a', 'b'], [385, 385], 'host', False),
            ('publish', 'device', 8), 'exit-side',
        ])
        self.assertIs(self.model._engram_job, self.job)

    def test_each_publication_captures_its_own_epoch(self):
        self.model._start_engram_job([], [])
        self.model._start_engram_job([], [])
        self.assertEqual([entry for entry in self.log if isinstance(entry, tuple)
                          and entry[0] == 'publish'],
                         [('publish', 'device', 8), ('publish', 'device', 9)])

    def test_enqueue_error_does_not_publish_success(self):
        def fail(*args, **kwargs):
            raise RuntimeError('read submission failed')
        self.native.enqueue_lookups = fail
        with self.assertRaisesRegex(RuntimeError, 'read submission failed'):
            self.model._start_engram_job([], [])
        self.assertFalse(any(isinstance(entry, tuple) and entry[0] == 'publish'
                             for entry in self.log))

    def test_composition_matches_durable_outputs(self):
        parent, outputs = prepare.compose()
        record = json.loads((prepare.ROOT / 'engram-progress-vllm.lock.json').read_text())
        self.assertEqual(record['failclosed_lock_sha256'], parent)
        for path, before, after in outputs:
            ast.parse(after)
            target = record['targets'][path]
            self.assertEqual(hashlib.sha256(after).hexdigest(), target['output_sha256'])
            self.assertEqual((prepare.ROOT / target['source']).read_bytes(), after)
            if path.endswith('/nvidia/model.py'):
                tree = ast.parse(after)
                method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                              and n.name == 'prepare_disk_engram')
                calls = [n for n in ast.walk(method) if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Attribute)]
                finish = next(n for n in calls if n.func.attr == '_finish_engram_job')
                stage = next(n for n in calls if n.func.attr == 'stage_disk')
                self.assertLess(finish.lineno, stage.lineno)
                self.assertNotIn(b'ThreadPoolExecutor', after)
                self.assertEqual(after.count(b'_engram_io_status.host_view.zero_()'), 1)
                self.assertIn(b'job.result(timeout=60.0)', after)
                self.assertNotIn(b'job.result()', after)


if __name__ == '__main__':
    unittest.main()
