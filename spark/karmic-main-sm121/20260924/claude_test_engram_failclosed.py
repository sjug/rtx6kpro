"""Stdlib tests for the proposed Engram fail-closed patch; no torch, GPU or nodes.

Covers reproducibility from pinned blobs, the exact scope of the edits, the
ordering of the new calls in the patched source, and the real extracted
Python helpers (fault check, model-state snapshot, runner hook) on fakes.
Triton kernel semantics are only emulated in claude_test_engram_failclosed_torch.py.
"""
import ast
import contextlib
import difflib
import hashlib
import importlib.util
import json
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('prep', HERE / 'claude_prepare_engram_failclosed.py')
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)
LOCK = json.loads((HERE / 'claude-engram-failclosed.lock.json').read_text())
COMPOSED = {path: (old, new) for path, old, new in prep.compose()}


def patched(path):
    return COMPOSED[path][1]


def function(source, name, owner=None):
    tree = ast.parse(source)
    scope = tree.body
    if owner:
        scope = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner).body
    return next(n for n in scope if isinstance(n, ast.FunctionDef) and n.name == name)


def load_function(source, name, owner=None, namespace=None):
    node = function(source, name, owner)
    namespace = dict(namespace or {})
    exec(compile(ast.Module(body=[node], type_ignores=[]), name, 'exec'), namespace)
    return namespace[name]


class Preparation(unittest.TestCase):
    def test_outputs_and_lock_reproduce(self):
        for path, (old, new) in COMPOSED.items():
            entry = LOCK['targets'][path]
            self.assertEqual(entry['input_sha256'], hashlib.sha256(old.encode()).hexdigest())
            self.assertEqual(entry['output_sha256'], hashlib.sha256(new.encode()).hexdigest())
            self.assertEqual((HERE / entry['source']).read_text(), new)
        for name, digest in LOCK['inputs'].items():
            self.assertEqual(hashlib.sha256((HERE / name).read_bytes()).hexdigest(), digest, name)

    def test_only_intended_lines_removed(self):
        expected = {
            prep.ENGRAM: {
                'def _wait_engram_rows(ready, expected, failed, timeout_ms):',
                '"""Spin until the side-stream lookup publishes this step\'s epoch."""',
                '"staged_rows",', 'caps.max_tokens,',
                '_wait_engram_rows[(1,)](*self.overlap_epochs, 5000)',
                'rows = tensor_model_parallel_all_reduce(self.staged_rows[: hash_ids.shape[0]])'},
            prep.MODEL: set(),
            prep.STATE: set(), prep.ASYNC: set(), prep.RUNNER: set(),
        }
        for path, (old, new) in COMPOSED.items():
            removed = {l[2:].strip() for l in difflib.ndiff(old.splitlines(), new.splitlines())
                       if l.startswith('- ') and l[2:].strip()}
            self.assertEqual(removed, expected[path], path)

    def test_no_b12x_or_launch_contract_file_touched(self):
        self.assertEqual(set(COMPOSED), {prep.ENGRAM, prep.MODEL, prep.STATE, prep.ASYNC, prep.RUNNER})
        patch = (HERE / 'claude-engram-failclosed.patch').read_text()
        self.assertNotIn('b12x/', ''.join(l for l in patch.splitlines(True) if l.startswith(('---', '+++'))))


class SourceOrdering(unittest.TestCase):
    def test_existing_functions_keep_decorators_and_signatures(self):
        def table(source):
            out = {}

            def walk(nodes, prefix):
                for n in nodes:
                    if isinstance(n, ast.ClassDef):
                        walk(n.body, prefix + n.name + '.')
                    elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        out[prefix + n.name] = ([ast.unparse(d) for d in n.decorator_list],
                                                ast.unparse(n.args))
            walk(ast.parse(source).body, '')
            return out
        changed_signatures = {'_wait_engram_rows', 'AsyncOutput.__init__'}
        for path, (old, new) in COMPOSED.items():
            before, after = table(old), table(new)
            self.assertLessEqual(set(before), set(after), path)
            for name, (decorators, args) in before.items():
                self.assertEqual(after[name][0], decorators, f'{path}: {name}')
                if name not in changed_signatures:
                    self.assertEqual(after[name][1], args, f'{path}: {name}')
            for name in set(after) - set(before):
                self.assertEqual(after[name][0], [], f'{path}: {name}')
        added = set().union(*(set(table(new)) - set(table(old)) for old, new in COMPOSED.values()))
        self.assertEqual(added, {'Engram._check_fault_io_alias', 'DeepseekV4Model.snapshot_engram_fault',
                                 'DeepseekV41ModelState.step_fault_snapshot', '_check_model_fault',
                                 'GPUModelRunner._model_step_fault'})

    def test_wait_kernel_always_writes_flag_last(self):
        wait = function(patched(prep.ENGRAM), '_wait_engram_rows')
        self.assertEqual([a.arg for a in wait.args.args], ['ready', 'expected', 'failed', 'timeout_ms', 'flag'])
        self.assertEqual(ast.unparse(wait.body[-1]), 'tl.store(flag, timed_out.to(tl.bfloat16))')
        loop = next(n for n in wait.body if isinstance(n, ast.While))
        branch = next(n for n in loop.body if isinstance(n, ast.If))
        self.assertEqual([ast.unparse(s) for s in branch.body],
                         ['tl.store(failed, 1)', 'timed_out = 1', 'waiting = 0'])

    def test_forward_overlap_order(self):
        forward = function(patched(prep.ENGRAM), 'forward', 'Engram')
        branch = next(n for n in forward.body if isinstance(n, ast.If)
                      and ast.unparse(n.test) == 'self.overlap_epochs is not None')
        self.assertEqual([ast.unparse(s) for s in branch.body], [
            'io = self.engram_io_rows',
            'fault = self.overlap_fault_epoch',
            '_wait_engram_rows[1,](*self.overlap_epochs, 5000, io)',
            'reduced = tensor_model_parallel_all_reduce(io[:hash_ids.shape[0] + 1])',
            'raised = reduced[0, :1] != 0',
            'fault.copy_(torch.where(raised, torch.maximum(fault, self.overlap_epochs[1]), fault))',
            'rows = reduced[1:]'])
        self.assertNotIn('data_ptr', ast.unparse(forward))
        self.assertEqual([ast.unparse(s) for s in branch.orelse],
                         ['rows = tensor_model_parallel_all_reduce(self.staged_rows[:hash_ids.shape[0]])'])
        rest = [ast.unparse(s) for s in forward.body[forward.body.index(branch) + 1:]]
        self.assertEqual(rest[0], 'kv = self.wkv(rows)')

    def test_flag_row_is_outside_every_lookup_binding(self):
        init = function(patched(prep.ENGRAM), '__init__', 'Engram')
        text = [ast.unparse(s) for s in init.body]
        self.assertIn('self.engram_io_rows[0].zero_()', text)
        self.assertIn('self.staged_rows = self.engram_io_rows[1:]', text)
        # Every lookup and zeroing path binds self.staged_rows, never the io buffer.
        source = patched(prep.ENGRAM)
        uses = [l.strip() for l in source.splitlines() if 'engram_io_rows' in l]
        self.assertEqual(uses, ['"engram_io_rows",', 'self.engram_io_rows[0].zero_()',
                                'self.staged_rows = self.engram_io_rows[1:]',
                                '"""Host-only check that lookups still write inside engram_io_rows.',
                                'if self.staged_rows.data_ptr() != self.engram_io_rows[1:].data_ptr():',
                                'io = self.engram_io_rows'])

    def test_alias_check_runs_in_every_eager_staging_seam(self):
        source = patched(prep.ENGRAM)
        invalidate = function(source, 'invalidate_disk_output', 'Engram')
        self.assertEqual(ast.unparse(invalidate.body[0]), 'self._check_fault_io_alias()')
        for name in ('prepare_disk', 'stage_disk', 'prepare_dummy_output'):
            body = function(source, name, 'Engram').body
            first = [ast.unparse(s) for s in body if not isinstance(s, ast.Expr)
                     or not isinstance(s.value, ast.Constant)][0]
            self.assertTrue(first.startswith('self.invalidate_disk_output('), (name, first))
        model = patched(prep.MODEL)
        prepare = function(model, 'prepare_disk_engram', 'DeepseekV4Model')
        self.assertIn('engram.invalidate_disk_output()', ast.unparse(prepare))
        # Both model-level entry points refuse to run under compile or capture,
        # so the host-only alias check always runs eagerly.
        for name in ('prepare_disk_engram', 'prepare_dummy_engram'):
            guard = function(model, name, 'DeepseekV4Model').body[0]
            self.assertEqual(ast.unparse(guard.test),
                             'torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing()')

    def test_fault_epoch_set_with_overlap_only(self):
        source = patched(prep.MODEL)
        block = source[source.index('            if self._engram_overlap:'):source.index('        if get_pp_group().is_last_rank:')]
        self.assertIn('self._engram_fault_epoch = torch.zeros(', block)
        self.assertIn('layer.engram.overlap_fault_epoch = self._engram_fault_epoch', block)
        self.assertEqual(source.count('_engram_fault_epoch = torch.zeros('), 1)

    def test_runner_passes_snapshot_at_both_output_sites(self):
        runner = patched(prep.RUNNER)
        calls = [n for n in ast.walk(ast.parse(runner)) if isinstance(n, ast.Call)
                 and ast.unparse(n.func) == 'AsyncOutput']
        self.assertEqual(len(calls), 2)
        for call in calls:
            kw = {k.arg: ast.unparse(k.value) for k in call.keywords}
            self.assertEqual(kw['model_fault'], 'self._model_step_fault()')

    def test_get_output_checks_right_after_existing_sync(self):
        get = function(patched(prep.ASYNC), 'get_output', 'AsyncOutput')
        self.assertEqual([ast.unparse(s) for s in get.body[:2]], [
            'self.copy_event.synchronize()',
            'if self._model_fault is not None:\n    _check_model_fault(self._model_fault.tolist())'])
        init = function(patched(prep.ASYNC), '__init__', 'AsyncOutput')
        copy_block = next(n for n in ast.walk(init) if isinstance(n, ast.With))
        body = [ast.unparse(s) for s in copy_block.body]
        self.assertEqual(body[0], 'copy_stream.wait_stream(main_stream)')
        self.assertEqual(body[-1], 'self.copy_event.record(copy_stream)')
        self.assertIn("if model_fault is not None:\n    self._model_fault = model_fault.to('cpu', non_blocking=True)", body)


class ExtractedHelpers(unittest.TestCase):
    def test_check_model_fault(self):
        check = load_function(patched(prep.ASYNC), '_check_model_fault')
        check([0, 7])                                     # no fault since start
        check([0, 7, 0, 7])
        with self.assertRaisesRegex(RuntimeError, 'in this step .fault epoch 11, step epoch 11'):
            check([11, 11])
        with self.assertRaisesRegex(RuntimeError, 'in an earlier step .fault epoch 3, step epoch 9'):
            check([3, 9])                                 # warmup or unchecked step: still fatal
        with self.assertRaises(RuntimeError):
            check([0, 9, 4, 9])                           # second model faulted

    def test_step_fault_snapshot(self):
        cat = []
        torch = types.SimpleNamespace(cat=lambda xs: cat.append(xs) or ('cat', tuple(xs)))
        snap = load_function(patched(prep.STATE), 'step_fault_snapshot', 'DeepseekV41ModelState',
                             namespace={'torch': torch})
        model = lambda value: types.SimpleNamespace(snapshot_engram_fault=lambda: value)
        self.assertIsNone(snap(types.SimpleNamespace(disk_engram_models=())))
        self.assertIsNone(snap(types.SimpleNamespace(disk_engram_models=(model(None),))))
        self.assertEqual(snap(types.SimpleNamespace(disk_engram_models=(model('a'),))), 'a')
        self.assertEqual(snap(types.SimpleNamespace(disk_engram_models=(model('a'), model('b')))),
                         ('cat', ('a', 'b')))

    def test_runner_hook_uses_main_stream(self):
        entered = []

        @contextlib.contextmanager
        def stream(s):
            entered.append(('enter', s))
            yield
            entered.append(('exit', s))
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(stream=stream))
        hook = load_function(patched(prep.RUNNER), '_model_step_fault', 'GPUModelRunner', namespace={'torch': torch})
        state = types.SimpleNamespace(step_fault_snapshot=lambda: entered.append('snap') or 'tensor')
        runner = types.SimpleNamespace(model_state=state, main_stream='main')
        self.assertEqual(hook(runner), 'tensor')
        self.assertEqual(entered, [('enter', 'main'), 'snap', ('exit', 'main')])
        self.assertIsNone(hook(types.SimpleNamespace(model_state=object(), main_stream='main')))


if __name__ == '__main__':
    unittest.main()
