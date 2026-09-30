"""Stdlib tests for the Engram host-timing layer; no torch, no GPU, no nodes.

The pinned B12X functions and their timed copies run side by side on the same
fake objects, and the tests require identical calls in identical order, with
timing both disabled and enabled.
"""
import ast
import contextlib
import difflib
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prep = load('claude_prepare_engram_timing', HERE / 'claude_prepare_engram_timing.py')
LOCK = json.loads((HERE / 'claude-engram-timing.lock.json').read_text())
MODEL = 'vllm/vllm/models/deepseek_v4_1/nvidia/model.py'
IMPL = 'b12x/b12x/sequence/engram/_impl.py'
DISK = 'b12x/b12x/sequence/_shared/disk_table.py'
ADDED_OK = ('_cet.event(', '_cet.job_begin(epoch)', '_cet.job_end()', 'try:', 'finally:',
            'import vllm.models.deepseek_v4_1.claude_engram_timing as _cet', '_cet.install_b12x()')


class Preparation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.composed = {image_path: (old, new, name) for image_path, old, new, name in prep.compose()}

    def test_outputs_and_lock_reproduce_from_pins(self):
        for image_path, (old, new, name) in self.composed.items():
            self.assertEqual((HERE / name).read_text(), new, name)
            entry = LOCK['targets'][image_path]
            self.assertEqual(entry['input_sha256'], hashlib.sha256(old.encode()).hexdigest())
            self.assertEqual(entry['output_sha256'], hashlib.sha256(new.encode()).hexdigest())
        for name, digest in LOCK['inputs'].items():
            self.assertEqual(hashlib.sha256((HERE / name).read_bytes()).hexdigest(), digest, name)
        self.assertEqual(LOCK['helper']['sha256'],
                         hashlib.sha256((HERE / 'claude_engram_timing.py').read_bytes()).hexdigest())
        self.assertEqual({p: e['delivery'] for p, e in LOCK['targets'].items()},
                         {MODEL: 'file', IMPL: 'runtime-bound', DISK: 'runtime-bound'})

    def test_edits_only_add_timing_lines(self):
        for image_path, (old, new, _) in self.composed.items():
            self.assertNotIn('_cet', old)
            added = [line[2:].strip() for line in difflib.ndiff(old.splitlines(), new.splitlines())
                     if line.startswith('+ ') and line[2:].strip()]
            for line in added:
                ok = line.startswith(ADDED_OK) or (image_path != MODEL and line in {
                    'if prepared:', 'finally:',
                    'self.programs[0][(prepared,24,1)](binding.weight,binding.scale_bytes,binding.hash_ids,binding.num_tokens,binding.out,prepared)',
                    'errors=[f.exception() for f in futures]'}) or (image_path == MODEL and not line.startswith('_cet') and line in old)
                self.assertTrue(ok, f'{image_path}: {line}')
            self.assertEqual(prep.original_code(new, {'try:', 'finally:'} if image_path == MODEL else set()),
                             prep.original_code(old, {'try:', 'finally:'} if image_path == MODEL else set()))

    def test_no_new_synchronisation_or_wait(self):
        risky = ('synchronize', 'wait', 'result(', '.item(', '.cpu(', '.tolist(', 'sleep', 'torch.')
        for image_path, (old, new, _) in self.composed.items():
            for line in new.splitlines():
                if '_cet.' in line and not line.strip().startswith('import'):
                    call = ast.parse(line.strip()).body[0].value
                    arguments = ' '.join(ast.unparse(a) for a in [*call.args[1:], *call.keywords])
                    self.assertFalse(any(word in arguments for word in risky), line)

    def test_lookup_job_end_is_guaranteed(self):
        _, new, _ = self.composed[MODEL]
        tree = ast.parse(new)
        (lookup,) = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'lookup']
        self.assertEqual(ast.unparse(lookup.body[0]), "_cet.event('lookup-begin')")
        self.assertIsInstance(lookup.body[1], ast.Try)
        self.assertEqual([ast.unparse(s) for s in lookup.body[1].finalbody], ['_cet.job_end()'])
        self.assertEqual(len(lookup.body), 2)
        (start,) = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_start_engram_job']
        body = [ast.unparse(s) for s in start.body]
        self.assertEqual(body[-2:], ['_cet.job_begin(epoch)', 'self._engram_job = pool.submit(lookup)'])

    def test_patch_names_only_intended_paths(self):
        patch = (HERE / 'claude-engram-timing.patch').read_text()
        targets = sorted({line[6:] for line in patch.splitlines() if line.startswith('+++ b/')})
        self.assertEqual(targets, sorted([MODEL, IMPL, DISK, LOCK['helper']['target']]))

    def test_dockerfile_copies_every_locked_input(self):
        docker = (HERE / 'Dockerfile.claude-engram-timing').read_text()
        copy = [l for l in docker.splitlines() if l.startswith('COPY ')][0].split()[1:-1]
        self.assertEqual(sorted(copy), sorted([*LOCK['inputs'], 'claude-engram-timing.lock.json']))
        self.assertIn('FROM ' + LOCK['base_image_id'], docker)


class Helper(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cet = load('cet_under_test', HERE / 'claude_engram_timing.py')
        self.cet.ENABLE = os.path.join(self.tmp.name, 'enable')
        self.cet.OUT_DIR = os.path.join(self.tmp.name, 'out')

    def tearDown(self):
        self.tmp.cleanup()

    def lines(self):
        files = list(Path(self.cet.OUT_DIR).glob('*.log'))
        return files[0].read_text().splitlines() if files else []

    def test_disabled_writes_nothing(self):
        self.cet.event('x', a=1)
        self.cet.job_begin(3)
        self.cet.job_end()
        self.assertFalse(os.path.exists(self.cet.OUT_DIR))

    def test_enabled_line_format_and_job_tag(self):
        Path(self.cet.ENABLE).write_text('')
        with mock.patch.dict(os.environ, {'DS41_NODE': 'dusty'}):
            self.cet.job_begin(7)
            self.cet.event('stage-copy-begin', cache='abc', count=48)
            self.cet.job_end()
            self.cet.event('after')
        rows = [line.split() for line in self.lines()]
        self.assertEqual([r[6] for r in rows], ['job-begin', 'stage-copy-begin', 'job-end', 'after'])
        self.assertEqual([r[5] for r in rows], ['job=7', 'job=7', 'job=7', 'job=-'])
        self.assertEqual(rows[1][1:5], ['node=dusty', f'pid={os.getpid()}',
                                        f'tid={threading.get_native_id()}', 'thread=MainThread'])
        self.assertEqual(rows[1][7:], ['cache=abc', 'count=48'])
        stamps = [int(r[0]) for r in rows]
        self.assertEqual(stamps, sorted(stamps))

    def test_enable_file_rechecked_at_most_once_per_second(self):
        now = [10_000_000_000]
        with mock.patch.object(self.cet.time, 'monotonic_ns', lambda: now[0]):
            self.cet.event('a')
            Path(self.cet.ENABLE).write_text('')
            now[0] += 500_000_000
            self.cet.event('b')                     # still cached as disabled
            now[0] += 600_000_000
            self.cet.event('c')
        self.assertEqual([l.split()[6] for l in self.lines()], ['c'])

    def test_never_raises(self):
        Path(self.cet.ENABLE).write_text('stacks')
        Path(self.cet.OUT_DIR).write_text('not a directory')
        self.cet.event('x')
        self.cet.job_begin(1)
        self.cet.job_end()

    def test_stack_dumps_armed_only_with_stacks_and_cancelled(self):
        for content, armed in (('', False), ('stacks', True)):
            Path(self.cet.ENABLE).write_text(content)
            self.cet._state['checked'] = -10**12
            with mock.patch.object(self.cet.faulthandler, 'dump_traceback_later') as arm, \
                    mock.patch.object(self.cet.faulthandler, 'cancel_dump_traceback_later') as cancel:
                self.cet.job_begin(5)
                self.cet.job_end()
            self.assertEqual(arm.called, armed)
            self.assertEqual(cancel.called, armed)
            if armed:
                self.assertEqual(arm.call_args.args, (self.cet.STACK_AFTER_S,))
                self.assertEqual(arm.call_args.kwargs['repeat'], True)
                self.assertEqual(arm.call_args.kwargs['exit'], False)
            self.assertFalse(self.cet._state['armed'])


# ---- behavioural equivalence of the runtime-bound B12X functions ----------

class Recorder:
    def __init__(self):
        self.calls, self.lock = [], threading.Lock()

    def __call__(self, *item):
        with self.lock:
            self.calls.append(item)


def fake_modules(root, rec):
    """Register fake b12x modules whose __file__ bytes are the pinned inputs."""
    pinned = {IMPL: HERE / 'claude-engram-timing-_impl.py', DISK: HERE / 'claude-engram-timing-disk_table.py'}
    files = {}
    for target in (IMPL, DISK):
        old = prep.subprocess.check_output(['git', '-C', str(prep.B12X), 'show',
                                            f'{prep.B12X_COMMIT}:{target[5:]}'])
        path = Path(root) / target.replace('/', '_')
        path.write_bytes(old)
        files[target] = path
    mods = {}
    for name in ('b12x', 'b12x.sequence', 'b12x.sequence.engram', 'b12x.sequence._shared'):
        mods[name] = types.ModuleType(name)
        mods[name].__path__ = []
    kernels = types.ModuleType('b12x.sequence.engram._kernels')
    kernels.lookup_op = lambda handle, *args: rec('lookup_op', handle, args[-2], args[-1])
    mods[kernels.__name__] = kernels

    class LookupBinding:
        def __init__(self, i, table):
            self.plan = types.SimpleNamespace(handle=100 + i)
            self.weight = self.scale_bytes = self.num_tokens = self.out = None
            self.hash_ids = types.SimpleNamespace(shape=(64, 24), i=i)
            self.disk_table = table

    class _State:
        def run_lookup(self):
            raise AssertionError('must be replaced')

    impl = types.ModuleType('b12x.sequence.engram._impl')
    impl.__file__, impl.__package__ = str(files[IMPL]), 'b12x.sequence.engram'
    impl.operator = __import__('operator')
    impl.LookupBinding, impl._State, impl._READ_POOL = LookupBinding, _State, None
    impl.run_lookup = lambda b, n, clear_tail: rec('serial', b.plan.handle, n)
    api = types.ModuleType('b12x.sequence.engram.api')
    api.run_lookups = 'original'

    class DiskRowCache:
        pass

    disk = types.ModuleType('b12x.sequence._shared.disk_table')
    disk.__file__, disk.__package__ = str(files[DISK]), 'b12x.sequence._shared'
    disk.DiskRowCache, disk.threading, disk.operator = DiskRowCache, threading, __import__('operator')
    disk.torch = types.SimpleNamespace(int64='int64',
                                       cuda=types.SimpleNamespace(device=lambda d: contextlib.nullcontext()))
    for module in (impl, api, disk):
        mods[module.__name__] = module
    mods['b12x.sequence.engram'].run_lookups = 'original'
    return mods, impl, disk


def pinned_functions(target, owner, names, module):
    old = prep.subprocess.check_output(['git', '-C', str(prep.B12X), 'show',
                                        f'{prep.B12X_COMMIT}:{target[5:]}']).decode()
    cet = sys.modules['claude_engram_timing_bound']
    namespace = dict(vars(module))
    exec(cet._functions(old, target, owner, names), namespace)
    out = {}
    for n in names:
        out[n] = types.FunctionType(namespace[n].__code__, vars(module), n, namespace[n].__defaults__)
        out[n].__kwdefaults__ = namespace[n].__kwdefaults__
    return out


class Bound(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        kit = Path(self.tmp.name) / 'kit'
        kit.mkdir()
        for name in ['claude-engram-timing.lock.json', *(e['source'] for e in LOCK['targets'].values())]:
            shutil.copy(HERE / name, kit / name)
        self.rec = Recorder()
        mods, self.impl, self.disk = fake_modules(self.tmp.name, self.rec)
        self.patch = mock.patch.dict(sys.modules, mods)
        self.patch.start()
        self.cet = load('claude_engram_timing_bound', HERE / 'claude_engram_timing.py')
        sys.modules['claude_engram_timing_bound'] = self.cet
        self.cet.ENABLE = str(Path(self.tmp.name) / 'enable')
        self.cet.OUT_DIR = str(Path(self.tmp.name) / 'out')
        self.cet.install_b12x(str(kit))

    def tearDown(self):
        pool = self.impl._READ_POOL
        if pool is not None:
            pool.shutdown()
        self.patch.stop()
        self.tmp.cleanup()

    def test_install_binds_everywhere_and_is_idempotent(self):
        engram = sys.modules['b12x.sequence.engram']
        self.assertIs(engram.run_lookups, self.impl.run_lookups)
        self.assertIs(sys.modules['b12x.sequence.engram.api'].run_lookups, self.impl.run_lookups)
        self.assertIs(self.impl.run_lookups.__globals__, vars(self.impl))
        self.assertTrue(self.impl.run_lookups.__code__.co_filename.endswith('claude-engram-timing-_impl.py'))
        self.assertEqual(self.impl.run_lookups.__kwdefaults__, {'clear_tail': True})
        first = self.impl.run_lookups
        self.cet.install_b12x('/nonexistent')
        self.assertIs(self.impl.run_lookups, first)

    def test_install_refuses_unpinned_live_module(self):
        Path(self.impl.__file__).write_bytes(b'# drift\n')
        self.cet._installed.clear()
        with self.assertRaises(RuntimeError):
            self.cet.install_b12x(str(Path(self.tmp.name) / 'kit'))

    def make_cache(self, i):
        rec = self.rec

        class Txn:
            def __enter__(s):
                rec('txn-enter', i)
                return cache

            def __exit__(s, *exc):
                rec('txn-exit', i)

        cache = types.SimpleNamespace(transaction=Txn,
                                      _stage_ids=lambda ids, n: rec('stage', ids.i, n),
                                      _read_staged=lambda n: rec('read', i, n))
        return types.SimpleNamespace(_cache=cache, _require_open=lambda: rec('open', i))

    def run_both(self, fn_new, fn_old, call):
        results = []
        for fn in (fn_old, fn_new):
            self.rec.calls.clear()
            call(fn)
            primary = [c for c in self.rec.calls if c[0] != 'read']
            reads = sorted(c for c in self.rec.calls if c[0] == 'read')
            results.append((primary, reads))
        return results

    def test_run_lookups_equivalent_disabled_and_enabled(self):
        old = pinned_functions(IMPL, None, ('run_lookups',), self.impl)['run_lookups']
        for enabled in (False, True):
            if enabled:
                Path(self.cet.ENABLE).write_text('')
            self.cet._state['checked'] = -10**12
            bindings = [self.impl.LookupBinding(i, self.make_cache(i)) for i in range(2)]
            orig, timed = self.run_both(self.impl.run_lookups, old,
                                        lambda fn: fn(bindings, [3, 5], clear_tail=False))
            self.assertEqual(orig, timed)
            self.assertEqual(orig[1], [('read', 0, 72), ('read', 1, 120)])
            self.assertEqual([c for c in orig[0] if c[0] == 'lookup_op'],
                             [('lookup_op', 100, 3, False), ('lookup_op', 101, 5, False)])
        names = [line.split()[6] for line in Path(self.cet.OUT_DIR).glob('*.log').__next__().read_text().splitlines()]
        self.assertEqual(names[:5], ['run-lookups-begin', 'txn-enter-begin', 'txn-entered',
                                     'txn-enter-begin', 'txn-entered'])
        self.assertEqual(names[-7:], ['reads-primary-returned', 'reads-joined', 'lookup-op-begin',
                                      'lookup-op-end', 'lookup-op-begin', 'lookup-op-end', 'run-lookups-end'])

    def test_run_lookups_serial_path_equivalent(self):
        old = pinned_functions(IMPL, None, ('run_lookups',), self.impl)['run_lookups']
        bindings = [self.impl.LookupBinding(i, None) for i in range(2)]
        orig, timed = self.run_both(self.impl.run_lookups, old, lambda fn: fn(bindings, [2, 4]))
        self.assertEqual(orig, timed)

    def test_run_lookups_error_propagation_equivalent(self):
        old = pinned_functions(IMPL, None, ('run_lookups',), self.impl)['run_lookups']
        outcomes = []
        for fn in (old, self.impl.run_lookups):
            bindings = [self.impl.LookupBinding(i, self.make_cache(i)) for i in range(2)]

            def boom(n):
                raise OSError('read failed')
            bindings[1].disk_table._cache._read_staged = boom
            self.rec.calls.clear()
            with self.assertRaises(OSError):
                fn(bindings, [1, 1])
            outcomes.append([c for c in self.rec.calls if c[0] != 'read'])
        self.assertEqual(outcomes[0], outcomes[1])

    def test_state_run_lookup_equivalent(self):
        old = pinned_functions(IMPL, '_State', ('run_lookup',), self.impl)['run_lookup']
        rec = self.rec

        class Out:
            def __getitem__(s, key):
                rec('slice', key.start)
                return types.SimpleNamespace(zero_=lambda: rec('zero'))
        for prepared in (0, 7, 64):
            state = types.SimpleNamespace(operation='lookup', caps=types.SimpleNamespace(max_tokens=64),
                                          programs=({(prepared, 24, 1): lambda *a: rec('launch', a[-1])},))
            binding = types.SimpleNamespace(weight=1, scale_bytes=2, hash_ids=3, num_tokens=4, out=Out())
            got = []
            for fn in (old, self.impl._State.run_lookup):
                rec.calls.clear()
                self.assertIs(fn(state, binding, prepared, clear_tail=True), binding.out)
                got.append(list(rec.calls))
            self.assertEqual(got[0], got[1])

    def disk_self(self, gds):
        rec = self.rec

        class Host:
            def __getitem__(s, key):
                return types.SimpleNamespace(copy_=lambda src, non_blocking: rec('copy', key.stop, non_blocking))
        ids = types.SimpleNamespace(device='cuda:0', dtype='int64', is_contiguous=lambda: True,
                                    numel=lambda: 480, view=lambda n: {slice(None, 48): 'ids48'} and _View())

        class _View:
            def __getitem__(s, key):
                return ('ids', key.stop)
        return types.SimpleNamespace(
            _transaction_thread=threading.get_ident(), max_lookups=480, device='cuda:0',
            ids_host=Host(), _ids_ready=types.SimpleNamespace(record=lambda s: rec('record', s),
                                                              synchronize=lambda: rec('sync')),
            _transaction_stream='stream', _ids_buffer='ids', _weight_buffer='w', _scale_buffer='s',
            _reader='reader', _gds=gds,
            _native=types.SimpleNamespace(ple_reader_run=lambda *a: rec('ple', *a))), ids

    def test_disk_stage_and_read_equivalent(self):
        old = pinned_functions(DISK, 'DiskRowCache', ('_stage_ids', '_read_staged'), self.disk)
        Path(self.cet.ENABLE).write_text('')
        self.cet._state['checked'] = -10**12
        for gds in (None, types.SimpleNamespace(read=lambda *a: self.rec('gds', *a))):
            got = []
            for stage, read in ((old['_stage_ids'], old['_read_staged']),
                                (self.disk.DiskRowCache._stage_ids, self.disk.DiskRowCache._read_staged)):
                self.rec.calls.clear()
                owner, ids = self.disk_self(gds)
                stage(owner, ids, 48)
                read(owner, 48)
                got.append(list(self.rec.calls))
            self.assertEqual(got[0], got[1])
            self.assertIn(('sync',), got[0])


if __name__ == '__main__':
    unittest.main()
