"""Native tests for the proposed PLE batch host-function API (no CUDA calls, no GPU).

Builds the pinned and patched _b12x_loader_storage extensions locally with the
same flags as b12x.loader._native (plus liburing) into TMPDIR, then:
  * checks the refactored ple_reader_run is byte-identical to the pinned one;
  * runs the REAL ple_batch_callback on foreign pthreads via a test-only
    launcher compiled in with -DB12X_PLE_BATCH_TEST_LAUNCH (stream value
    selects: 1 run on a thread, 2 fail the launch, 3 defer, 4 never run);
  * checks lifetime: launch failure, handle dropped while pending (orphan),
    skipped callback, buffer export retention, and GIL independence.
The production build (real cudaLaunchHostFunc) is compiled with -Werror and
its symbol checked, but never called: CUDA behaviour is not exercised here.
"""
import array
import faulthandler
import importlib.util
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import threading
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('prep', HERE / 'claude_prepare_ple_batch.py')
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)

CUDA = Path(os.environ.get('CUDA_HOME', '/opt/cuda'))
FLAGS = ['-O2', '-std=c99', '-shared', '-fPIC', '-pthread', '-D_FILE_OFFSET_BITS=64',
         '-Wall', '-Wextra', '-Werror', '-DB12X_HAVE_LIBURING=1',
         f'-I{sysconfig.get_path("include")}', f'-I{CUDA / "include"}']
LINK = [f'-L{CUDA / "lib64"}', f'-Wl,-rpath,{CUDA / "lib64"}', '-lcudart', '-lcuda', '-luring']
if os.environ.get('PLE_BATCH_SANITIZE'):
    # Run as: PLE_BATCH_SANITIZE=1 ASAN_OPTIONS=detect_leaks=0 \
    #   LD_PRELOAD=$(cc -print-file-name=libasan.so) python3 -m unittest claude_test_ple_batch_native
    FLAGS = [f for f in FLAGS if f != '-O2'] + ['-O1', '-g', '-fno-omit-frame-pointer',
                                                '-fsanitize=address,undefined', '-fno-sanitize-recover=all']
BUILD = Path(tempfile.mkdtemp(prefix='ple-batch-'))
SOURCES = {path: new for path, _old, new in prep.compose()}


def build(name, patched, extra=()):
    folder = BUILD / name
    folder.mkdir()
    listing = subprocess.check_output(['git', '-C', str(prep.B12X), 'ls-tree', '--name-only',
                                       prep.COMMIT, 'b12x/loader/'], text=True).split()
    for path in listing:
        if path.endswith(('.c', '.h')):
            text = SOURCES[path] if patched and path in SOURCES else prep.pinned(path)
            (folder / Path(path).name).write_text(text)
    target = folder / ('_b12x_loader_storage' + sysconfig.get_config_var('EXT_SUFFIX'))
    subprocess.run(['cc', *FLAGS, *extra, str(folder / '_storage.c'), *LINK, '-o', str(target)],
                   check=True, capture_output=True, text=True)
    return target


def load(target):
    module_spec = importlib.util.spec_from_file_location('_b12x_loader_storage', target)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


PINNED = load(build('pinned', False))
PRODUCTION = build('production', True)
TEST = load(build('test', True, ['-DB12X_PLE_BATCH_TEST_LAUNCH=1']))
ROW, SCALE, ROWS = 5003, 513, 10


def setUpModule():
    faulthandler.dump_traceback_later(120, exit=True)   # a GIL deadlock fails instead of hanging


def tearDownModule():
    faulthandler.cancel_dump_traceback_later()
    shutil.rmtree(BUILD, ignore_errors=True)


class Table:
    def __init__(self, module, folder, seed, *, scales=True):
        self.weights = bytes((i * (17 + seed) + seed) % 256 for i in range(ROWS * ROW))
        self.scales = bytes((i * (31 + seed) + 5) % 256 for i in range(ROWS * SCALE)) if scales else b''
        self.path = Path(folder) / f'table{seed}'
        self.path.write_bytes(self.weights + self.scales)
        self.reader = module.ple_reader(ROWS, ROWS, 0, ROWS, ROW, SCALE if scales else 0, 16, 4)
        module.ple_reader_add(self.reader, 0, str(self.path), 0, False)
        if scales:
            module.ple_reader_add(self.reader, 0, str(self.path), len(self.weights), True)

    def buffers(self, ids):
        ids = array.array('q', ids)
        return ids, bytearray(len(ids) * ROW), bytearray(len(ids) * SCALE) if self.scales else None

    def expected(self, ids):
        rows = b''.join(self.weights[i * ROW:(i + 1) * ROW] if 0 <= i < ROWS else bytes(ROW) for i in ids)
        scales = b''.join(self.scales[i * SCALE:(i + 1) * SCALE] if 0 <= i < ROWS else bytes(SCALE)
                          for i in ids) if self.scales else None
        return rows, scales


def exported(buffer):
    """True while some native owner still holds a buffer export."""
    try:
        buffer.append(0)
    except BufferError:
        return True
    buffer.pop()
    return False


class Build(unittest.TestCase):
    def test_production_build_uses_real_launch(self):
        symbols = subprocess.check_output(['nm', '-D', str(PRODUCTION)], text=True)
        self.assertIn('cudaLaunchHostFunc', symbols)
        self.assertNotIn('ple_batch_test', subprocess.check_output(['nm', str(PRODUCTION)], text=True))
        self.assertFalse(hasattr(load(PRODUCTION), 'ple_batch_test_run_deferred'))

    def test_read_core_is_the_original_block(self):
        core = prep.core_function()
        body = prep.RUN_BLOCK.split('    Py_BEGIN_ALLOW_THREADS\n', 1)[1].rsplit('    Py_END_ALLOW_THREADS\n', 1)[0]
        for old, new, _ in prep.CORE_SUBSTITUTIONS:
            body = body.replace(old, new)
        self.assertIn(body, core)
        reader = SOURCES['b12x/loader/_ple_reader.c']
        self.assertEqual(reader.count('ple_read_core('), 3)      # definition, run, batch thread


class ReadCoreEquivalence(unittest.TestCase):
    def test_refactored_run_matches_pinned(self):
        ids = [2, 2, 5, 9, 0, -1, 7, 100]
        with tempfile.TemporaryDirectory() as folder:
            results = []
            for module in (PINNED, TEST):
                table = Table(module, folder, 3)
                ids_buffer, rows, scales = table.buffers(ids)
                module.ple_reader_run(table.reader, ids_buffer, rows, scales, len(ids))
                stats = module.ple_reader_stats(table.reader)
                results.append((bytes(rows), bytes(scales),
                                {k: v for k, v in stats.items() if not k.endswith('seconds')}))
            self.assertEqual(results[0], results[1])
            self.assertEqual((results[0][0], results[0][1]), table.expected(ids))


class Batch(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.tables = [Table(TEST, self.folder.name, 1), Table(TEST, self.folder.name, 2, scales=False)]
        self.ids = [[1, 4, 4, 9, -1], [0, 8, 3]]
        self.io = [t.buffers(i) for t, i in zip(self.tables, self.ids)]
        self.status = array.array('q', [0])

    def tearDown(self):
        self.folder.cleanup()

    def batch(self, counts=None, status=None):
        return TEST.ple_batch(tuple(t.reader for t in self.tables), tuple(io[0] for io in self.io),
                              tuple(io[1] for io in self.io), tuple(io[2] for io in self.io),
                              tuple(counts or (len(i) for i in self.ids)), status or self.status)

    def assert_rows(self):
        for table, ids, io in zip(self.tables, self.ids, self.io):
            rows, scales = table.expected(ids)
            self.assertEqual(bytes(io[1]), rows)
            if scales is not None:
                self.assertEqual(bytes(io[2]), scales)

    def test_callback_reads_every_table_and_publishes_success(self):
        handle = self.batch()
        TEST.ple_batch_launch(handle, 1)
        self.assertEqual(TEST.ple_batch_result(handle, 30.0), ('done', None))
        self.assert_rows()
        self.assertEqual(self.status[0], 1)
        self.assertEqual(TEST.ple_reader_stats(self.tables[0].reader)['lookups'], 5)
        self.assertTrue(exported(self.io[0][1]))            # result() does not release
        del handle
        self.assertFalse(exported(self.io[0][1]))           # dropping the handle does

    def test_read_failure_publishes_minus_one_and_reports(self):
        self.tables[1].path.write_bytes(b'')                 # truncate after registration
        handle = self.batch()
        TEST.ple_batch_launch(handle, 1)
        state, message = TEST.ple_batch_result(handle, 30.0)
        self.assertEqual(state, 'done')
        self.assertIn('short PLE read', message)
        self.assertEqual(self.status[0], -1)
        rows, _ = self.tables[0].expected(self.ids[0])
        self.assertEqual(bytes(self.io[0][1]), rows)         # the healthy table still read

    def test_status_is_never_reset_between_jobs(self):
        self.status[0] = 7
        handle = self.batch()
        self.assertEqual(self.status[0], 7)                   # preparation does not touch it
        TEST.ple_batch_launch(handle, 3)
        self.assertEqual(self.status[0], 7)                   # nor does a pending launch
        TEST.ple_batch_test_run_deferred()
        self.assertEqual(self.status[0], 1)

    def test_launch_failure_releases_everything_at_once(self):
        handle = self.batch()
        with self.assertRaisesRegex(RuntimeError, 'launch failed'):
            TEST.ple_batch_launch(handle, 2)
        state, message = TEST.ple_batch_result(handle, 0.0)
        self.assertEqual(state, 'launch_failed')
        self.assertIn('cudaLaunchHostFunc', message)
        self.assertFalse(exported(self.io[0][1]))
        self.assertEqual(self.status[0], 0)
        with self.assertRaisesRegex(ValueError, 'already launched'):
            TEST.ple_batch_launch(handle, 1)

    def test_concurrent_waiter_wakes_when_launch_fails(self):
        handle = self.batch()
        outcome = []

        def waiter():
            import time
            time.sleep(0.05)                                 # launch is inside its 200 ms window
            outcome.append(TEST.ple_batch_result(handle, -1.0))
        thread = threading.Thread(target=waiter)
        thread.start()
        with self.assertRaisesRegex(RuntimeError, 'launch failed'):
            TEST.ple_batch_launch(handle, 5)
        thread.join(10)
        self.assertFalse(thread.is_alive(), 'waiter missed the launch-failure broadcast')
        self.assertEqual(outcome[0][0], 'launch_failed')

    def test_timeout_values(self):
        handle = self.batch()
        with self.assertRaisesRegex(ValueError, 'NaN'):
            TEST.ple_batch_result(handle, float('nan'))
        TEST.ple_batch_launch(handle, 3)
        self.assertEqual(TEST.ple_batch_result(handle, 0.0), ('pending', None))
        self.assertEqual(TEST.ple_batch_result(handle, 0.02), ('pending', None))
        with self.assertRaisesRegex(ValueError, 'NaN'):
            TEST.ple_batch_result(handle, float('nan'))
        TEST.ple_batch_test_run_deferred()
        for timeout in (float('inf'), 1e300, 1e9 + 1, -1.0, -float('inf'), 5.5):
            self.assertEqual(TEST.ple_batch_result(handle, timeout), ('done', None), timeout)

    def test_overlap_boundaries(self):
        # Rows occupy bytes [8, 8 + 5*ROW) = [8, 25023) of one 8-aligned block.
        r = tuple(t.reader for t in self.tables)
        ids, scales = tuple(io[0] for io in self.io), tuple(io[2] for io in self.io)
        raw = memoryview(array.array('q', [0]) * 3200).cast('B')
        rows = raw[8:8 + ROW * 5]

        def prepare(status):
            return TEST.ple_batch(r, ids, (rows, self.io[1][1]), scales, (5, 3), status)
        prepare(raw[0:8])                                      # ends exactly where rows begin
        prepare(raw[25024:25032])                              # first aligned word after rows
        for start in (8, 12504, 25016):                        # first, middle, last rows word
            with self.assertRaisesRegex(ValueError, 'must not overlap'):
                prepare(raw[start:start + 8])

    def test_dropped_handle_while_pending_keeps_buffers_until_callback(self):
        baseline = TEST.ple_batch_test_orphans()              # skipped-callback tests leave some
        handle = self.batch()
        TEST.ple_batch_launch(handle, 3)
        self.assertEqual(TEST.ple_batch_result(handle, 0.0), ('pending', None))
        del handle                                            # Python drops it before the callback
        self.assertEqual(TEST.ple_batch_test_orphans(), baseline + 1)
        self.assertTrue(exported(self.io[0][1]))              # buffers still owned by the job
        TEST.ple_batch_test_run_deferred()                    # callback runs later: no use after free
        self.assert_rows()
        self.batch(counts=(0, 0))                             # any later batch call sweeps
        self.assertEqual(TEST.ple_batch_test_orphans(), baseline)
        self.assertFalse(exported(self.io[0][1]))

    def test_skipped_callback_is_a_bounded_leak(self):
        handle = self.batch()
        TEST.ple_batch_launch(handle, 4)                      # CUDA never runs it (context error)
        self.assertEqual(TEST.ple_batch_result(handle, 0.05), ('pending', None))
        del handle
        self.batch(counts=(0, 0))
        self.assertGreaterEqual(TEST.ple_batch_test_orphans(), 1)
        self.assertTrue(exported(self.io[0][1]))              # retained forever, never freed early

    def test_callback_never_needs_the_gil(self):
        handle = self.batch()
        TEST.ple_batch_launch(handle, 3)
        TEST.ple_batch_test_run_deferred()                    # joins while HOLDING the GIL
        self.assertEqual(TEST.ple_batch_result(handle, 0.0), ('done', None))
        self.assert_rows()

    def test_prepared_but_unlaunched(self):
        handle = self.batch()
        with self.assertRaisesRegex(ValueError, 'not launched'):
            TEST.ple_batch_result(handle, 0.0)
        del handle
        self.assertFalse(exported(self.io[0][1]))

    def test_validation(self):
        r = tuple(t.reader for t in self.tables)
        ids = tuple(io[0] for io in self.io)
        rows = tuple(io[1] for io in self.io)
        scales = tuple(io[2] for io in self.io)
        counts = (5, 3)
        cases = [
            ((r[0], r[0]), ids, rows, scales, counts, self.status, 'distinct readers'),
            (r, ids, rows, scales, (5, 99), self.status, 'exceeds batch capacity'),
            (r, ids, rows, scales, (6, 3), self.status, 'cover count rows'),
            (r, ids, rows, (scales[0], bytearray(9999)), counts, self.status, 'no scale plane'),
            (r, ids, rows, scales, counts, array.array('q', [0, 0]), '8-byte aligned int64'),
            (r, ids, rows, scales, counts, bytes(8), None),       # read-only status buffer
            (r[:1], ids, rows, scales, counts, self.status, '1 to 8 tables'),
        ]
        for readers, i, w, s, c, status, message in cases:
            with self.assertRaises((ValueError, TypeError, BufferError)) as caught:
                TEST.ple_batch(readers, i, w, s, c, status)
            if message:
                self.assertIn(message, str(caught.exception))
        overlap = array.array('q', [0]) * (ROW * 5 // 8 + 2)   # 8-aligned, covers 5 rows
        status_view = memoryview(overlap).cast('B')[24992:25000]  # inside the row range
        with self.assertRaisesRegex(ValueError, 'must not overlap'):
            TEST.ple_batch(r, ids, (overlap, rows[1]), scales, counts, status_view)
        self.assertFalse(exported(self.io[0][1]))             # failed preparation released all

    def test_concurrent_python_reader_use_serializes_on_native_mutex(self):
        handle = self.batch()
        TEST.ple_batch_launch(handle, 3)
        ids, rows, scales = self.tables[0].buffers([2, 3])
        worker = threading.Thread(target=TEST.ple_batch_test_run_deferred)
        worker.start()
        TEST.ple_reader_run(self.tables[0].reader, ids, rows, scales, 2)   # same reader, other buffers
        worker.join()
        self.assertEqual(TEST.ple_batch_result(handle, 30.0), ('done', None))
        self.assertEqual(bytes(rows), self.tables[0].expected([2, 3])[0])
        self.assert_rows()


if __name__ == '__main__':
    unittest.main()
