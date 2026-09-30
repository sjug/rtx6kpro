"""Stdlib tests for the Python side of the proposed PLE batch API (no torch, CUDA or GPU).

Runs the REAL patched launch_batch_read, DiskBatchJob and enqueue_lookups
(extracted from the patched B12X files) against fakes, and checks the exact
order of transaction entry, ID staging, the single native launch, lookups
and transaction exit, plus every refusal path. Also checks reproducibility
and the export surface. CUDA stream semantics are not exercised.
"""
import array
import ast
import contextlib
import hashlib
import importlib.util
import json
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('prep', HERE / 'claude_prepare_ple_batch.py')
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)
COMPOSED = {path: (old, new) for path, old, new in prep.compose()}
LOCK = json.loads((HERE / 'claude-ple-batch.lock.json').read_text())


def node(path, name):
    return next(n for n in ast.parse(COMPOSED[path][1]).body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)


def extract(path, names, namespace):
    module = ast.Module(body=[node(path, n) for n in names], type_ignores=[])
    exec(compile(module, path, 'exec'), namespace)
    return namespace


class FakeTensor:
    def __init__(self, *, device='cpu', dtype='int64', numel=1, contiguous=True):
        self.device = types.SimpleNamespace(type=device)
        self.dtype, self._numel, self._contiguous = dtype, numel, contiguous
        self.storage = array.array('q', [0] * numel)

    def numel(self):
        return self._numel

    def is_contiguous(self):
        return self._contiguous

    def numpy(self):
        return self.storage


TORCH = types.SimpleNamespace(Tensor=FakeTensor, int64='int64')


class FakeNative:
    def __init__(self, log, fail_launch=False, results=()):
        self.log, self.fail_launch, self.results = log, fail_launch, list(results)

    def ple_batch(self, readers, ids, weights, scales, counts, status):
        self.log.append(('ple_batch', readers, ids, weights, scales, counts, status.nbytes, status.format))
        return 'handle'

    def ple_batch_launch(self, handle, stream):
        self.log.append(('launch', handle, stream))
        if self.fail_launch:
            raise RuntimeError('PLE batch launch failed: invalid resource handle')

    def ple_batch_result(self, handle, timeout):
        self.log.append(('result', handle, timeout))
        return self.results.pop(0)


class FakeCache:
    def __init__(self, name, log, native, stream=7, gds=False):
        self.name, self.log, self._native = name, log, native
        self._gds = object() if gds else None
        self._reader, self._ids_buffer = f'reader-{name}', f'ids-{name}'
        self._weight_buffer, self._scale_buffer = f'w-{name}', None
        self._transaction_thread = self._transaction_stream = None
        self.stream, self.max_lookups = stream, 1000

    def _require_open(self):
        self.log.append(('open', self.name))

    @contextlib.contextmanager
    def transaction(self):
        self.log.append(('enter', self.name))
        self._transaction_thread = threading.get_ident()
        self._transaction_stream = types.SimpleNamespace(cuda_stream=self.stream)
        try:
            yield self
        finally:
            self._transaction_thread = self._transaction_stream = None
            self.log.append(('exit', self.name))

    def _stage_ids(self, ids, count):
        self.log.append(('stage', self.name, count))
        self._staged_count = count


DISK_NS = extract(prep.DISK, ['DiskBatchJob', 'launch_batch_read'],
                  {'torch': TORCH, 'operator': __import__('operator'), 'threading': threading,
                   'math': __import__('math')})
launch_batch_read, DiskBatchJob = DISK_NS['launch_batch_read'], DISK_NS['DiskBatchJob']


class LaunchBatchRead(unittest.TestCase):
    def setUp(self):
        self.log = []
        self.native = FakeNative(self.log)
        self.caches = [FakeCache('a', self.log, self.native), FakeCache('b', self.log, self.native)]

    def staged(self, counts=(48, 24)):
        stack = contextlib.ExitStack()
        for cache, count in zip(self.caches, counts):
            stack.enter_context(cache.transaction())
            cache._stage_ids(None, count)
        return stack

    def test_one_native_batch_on_the_transaction_stream(self):
        status = FakeTensor()
        with self.staged():
            job = launch_batch_read(self.caches, [48, 24], status)
        batch = next(e for e in self.log if e[0] == 'ple_batch')
        self.assertEqual(batch[1:6], (('reader-a', 'reader-b'), ('ids-a', 'ids-b'),
                                      ('w-a', 'w-b'), (None, None), (48, 24)))
        self.assertEqual(batch[6:], (8, 'B'))
        self.assertIn(('launch', 'handle', 7), self.log)
        self.assertIsInstance(job, DiskBatchJob)

    def test_refusals(self):
        status = FakeTensor()
        with self.assertRaisesRegex(RuntimeError, 'active disk row transaction'):
            launch_batch_read(self.caches, [48, 24], status)            # no transaction
        with self.staged((48, 25)):
            with self.assertRaisesRegex(RuntimeError, 'staged ID count'):
                launch_batch_read(self.caches, [48, 24], status)
        self.caches[1].stream = 8
        with self.staged():
            with self.assertRaisesRegex(RuntimeError, 'one stream'):
                launch_batch_read(self.caches, [48, 24], status)
        self.caches[1].stream = 7
        self.caches[1]._gds = object()
        with self.staged():
            with self.assertRaisesRegex(NotImplementedError, 'GDS is refused'):
                launch_batch_read(self.caches, [48, 24], status)
        self.caches[1]._gds = None
        with self.staged():
            with self.assertRaisesRegex(ValueError, 'distinct'):
                launch_batch_read([self.caches[0], self.caches[0]], [48, 48], status)
            for bad in (FakeTensor(device='cuda'), FakeTensor(dtype='int32'), FakeTensor(numel=2),
                        FakeTensor(contiguous=False), array.array('q', [0])):
                with self.assertRaisesRegex(ValueError, 'status_host'):
                    launch_batch_read(self.caches, [48, 24], bad)
            with self.assertRaisesRegex(ValueError, 'one staged count'):
                launch_batch_read(self.caches, [48], status)
        with self.staged():
            result = [None]

            def other_thread():
                try:
                    launch_batch_read(self.caches, [48, 24], status)
                except RuntimeError as error:
                    result[0] = error
            worker = threading.Thread(target=other_thread)
            worker.start()
            worker.join()
            self.assertIn('one stream', str(result[0]))              # wrong thread
        self.assertNotIn('launch', [e[0] for e in self.log])

    def test_job_result_mapping(self):
        native = FakeNative([], results=[('pending', None), ('done', None), ('done', 'short PLE read'),
                                         ('pending', None), ('launch_failed', 'cudaLaunchHostFunc: x'),
                                         ('done', None), ('pending', None)])
        job = DiskBatchJob(native, 'h')
        self.assertFalse(job.done())
        job.result()
        with self.assertRaisesRegex(RuntimeError, 'short PLE read'):
            job.result(timeout=1.5)
        with self.assertRaises(TimeoutError):
            job.result(timeout=0)
        with self.assertRaisesRegex(RuntimeError, 'cudaLaunchHostFunc'):
            job.result()
        job.result(timeout=float('inf'))
        with self.assertRaises(TimeoutError):
            job.result(timeout=-3)                          # negative polls
        with self.assertRaisesRegex(ValueError, 'NaN'):
            job.result(timeout=float('nan'))
        self.assertEqual([c[2] for c in native.log], [0.0, -1.0, 1.5, 0.0, -1.0, float('inf'), 0.0])

    def test_ownership_contract_is_documented(self):
        source = COMPOSED[prep.DISK][1] + COMPOSED[prep.IMPL][1] + COMPOSED[prep.READER][1]
        for phrase in ('does not own anything the GPU uses later', 'reports the native read only',
                       'is NOT covered', 'Reports the host function only'):
            self.assertIn(phrase, source)


class LookupBinding:
    def __init__(self, name, table, rows=100):
        self.plan = types.SimpleNamespace(handle=f'plan-{name}')
        self.weight = self.scale_bytes = self.num_tokens = None
        self.out = f'out-{name}'
        self.hash_ids = types.SimpleNamespace(shape=(rows, 24))
        self.disk_table = table


class EnqueueLookups(unittest.TestCase):
    def setUp(self):
        self.log = []
        self.launch_error = self.lookup_error = None

        def fake_launch(caches, counts, status):
            self.log.append(('launch', tuple(c.name for c in caches), tuple(counts), status))
            if self.launch_error:
                raise self.launch_error
            return 'job'

        def fake_lookup(handle, weight, scales, hashes, num_tokens, out, n, clear_tail):
            self.log.append(('lookup', handle, n, clear_tail))
            if self.lookup_error:
                raise self.lookup_error
        modules = {
            'b12x': types.ModuleType('b12x'), 'b12x.sequence': types.ModuleType('b12x.sequence'),
            'b12x.sequence._shared': types.ModuleType('b12x.sequence._shared'),
            'b12x.sequence._shared.disk_table': types.SimpleNamespace(launch_batch_read=fake_launch),
            'b12x.sequence.engram': types.ModuleType('b12x.sequence.engram'),
            'b12x.sequence.engram._kernels': types.SimpleNamespace(lookup_op=fake_lookup),
        }
        for name in ('b12x', 'b12x.sequence', 'b12x.sequence._shared', 'b12x.sequence.engram'):
            modules[name].__path__ = []
        self.patch = mock.patch.dict(sys.modules, modules)
        self.patch.start()
        ns = extract(prep.IMPL, ['enqueue_lookups'],
                     {'operator': __import__('operator'), 'LookupBinding': LookupBinding,
                      '__package__': 'b12x.sequence.engram', '__name__': 'b12x.sequence.engram._impl'})
        self.enqueue = ns['enqueue_lookups']
        tables = [types.SimpleNamespace(_require_open=lambda n=n: self.log.append(('open', n)),
                                        _cache=FakeCache(n, self.log, None)) for n in 'ab']
        self.bindings = [LookupBinding(n, t) for n, t in zip('ab', tables)]

    def tearDown(self):
        self.patch.stop()

    def test_everything_is_queued_in_stream_order(self):
        self.assertEqual(self.enqueue(self.bindings, [2, 1], status_host='status'), 'job')
        self.assertEqual(self.log, [
            ('open', 'a'), ('enter', 'a'), ('stage', 'a', 48),
            ('open', 'b'), ('enter', 'b'), ('stage', 'b', 24),
            ('launch', ('a', 'b'), (48, 24), 'status'),
            ('lookup', 'plan-a', 2, False), ('lookup', 'plan-b', 1, False),
            ('exit', 'b'), ('exit', 'a')])

    def test_launch_failure_queues_no_lookup_and_closes_transactions(self):
        self.launch_error = RuntimeError('PLE batch launch failed')
        with self.assertRaisesRegex(RuntimeError, 'launch failed'):
            self.enqueue(self.bindings, [2, 1], status_host='status')
        self.assertNotIn('lookup', [e[0] for e in self.log])
        self.assertEqual(self.log[-2:], [('exit', 'b'), ('exit', 'a')])

    def test_lookup_failure_after_launch_still_closes_transactions(self):
        self.lookup_error = RuntimeError('lookup launch failed')
        with self.assertRaisesRegex(RuntimeError, 'lookup launch failed'):
            self.enqueue(self.bindings, [2, 1], status_host='status')
        self.assertEqual(self.log[-2:], [('exit', 'b'), ('exit', 'a')])

    def test_validation_before_any_transaction(self):
        self.bindings[1].disk_table = None
        with self.assertRaisesRegex(ValueError, 'disk-backed'):
            self.enqueue(self.bindings, [2, 1], status_host='s')
        with self.assertRaisesRegex(ValueError, 'one token count'):
            self.enqueue(self.bindings, [2], status_host='s')
        with self.assertRaisesRegex(ValueError, 'one token count'):
            self.enqueue([], [], status_host='s')
        with self.assertRaisesRegex(TypeError, 'Engram lookup binding'):
            self.enqueue([object()], [1], status_host='s')
        with self.assertRaisesRegex(ValueError, 'capacity'):
            self.enqueue(self.bindings[:1], [101], status_host='s')
        self.assertEqual(self.log, [])


class Surface(unittest.TestCase):
    def test_reproducible_and_locked(self):
        for path, (old, new) in COMPOSED.items():
            entry = LOCK['targets'][path]
            self.assertEqual(entry['input_sha256'], hashlib.sha256(old.encode()).hexdigest())
            self.assertEqual(entry['output_sha256'], hashlib.sha256(new.encode()).hexdigest())
            self.assertEqual((HERE / entry['source']).read_text(), new)
        for name, digest in LOCK['inputs'].items():
            self.assertEqual(hashlib.sha256((HERE / name).read_bytes()).hexdigest(), digest, name)

    def test_exports(self):
        api = COMPOSED[prep.API][1]
        self.assertIn('run_lookups, enqueue_lookups\n', api)
        self.assertIn('"enqueue_lookups"', api.split('__all__')[1])
        self.assertIn('"enqueue_lookups"', COMPOSED[prep.INIT][1].split('entry_points=')[1].split(')')[0])

    def test_existing_python_unchanged_except_additions(self):
        import difflib
        for path in (prep.DISK, prep.IMPL, prep.API, prep.INIT):
            old, new = COMPOSED[path]
            removed = [l for l in difflib.ndiff(old.splitlines(), new.splitlines()) if l.startswith('- ')]
            allowed = {prep.API: 2, prep.INIT: 1}.get(path, 0)
            self.assertEqual(len(removed), allowed, (path, removed))
        stage = ast.unparse(next(n for n in ast.walk(ast.parse(COMPOSED[prep.DISK][1]))
                                 if isinstance(n, ast.FunctionDef) and n.name == '_stage_ids'))
        self.assertTrue(stage.rstrip().endswith('self._staged_count = count'))

    def test_native_run_only_changes_its_critical_section(self):
        import difflib
        old, new = COMPOSED[prep.READER]
        removed = [l[2:] for l in difflib.ndiff(old.splitlines(), new.splitlines()) if l.startswith('- ')]
        self.assertEqual(removed, [l for l in prep.RUN_BLOCK.splitlines()
                                   if l not in prep.RUN_BLOCK_NEW.splitlines()])


if __name__ == '__main__':
    unittest.main()
