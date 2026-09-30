"""CPU tests for the worker-level precision kit (helper, worker seam, lock, installer). No GPU, nodes or builds:
  .venv-snapshot-cpu/bin/python -m unittest test_ds41_precision
"""
import ast
import difflib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ds41_precision_prepare as prep  # noqa: E402

FLAG = 'allow_bf16_reduced_precision_reduction'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fresh_helper():
    return load(f'ds41_precision_{time.monotonic_ns()}', HERE / 'ds41_precision.py')


def sha(data):
    return hashlib.sha256(data).hexdigest()


class Stop(Exception):
    pass


class FakeWorker:
    """Any device or config access after the precision call stops the extracted init_device."""

    def __init__(self, rank):
        self.rank = rank

    @property
    def device_config(self):
        raise Stop


def extracted_init_device(namespace):
    """The composed Worker.init_device, compiled alone from the generated source, not a hand-written copy."""
    tree = ast.parse((HERE / prep.OUTPUT).read_text())
    (worker,) = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Worker']
    (method,) = [n for n in worker.body if isinstance(n, ast.FunctionDef) and n.name == 'init_device']
    method.decorator_list = []          # the tracing decorator is not under test
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(module, '<generated-init_device>', 'exec'), namespace)
    return namespace['init_device']


class HelperTests(unittest.TestCase):
    def setUp(self):
        original = getattr(torch.backends.cuda.matmul, FLAG)
        self.addCleanup(setattr, torch.backends.cuda.matmul, FLAG, original)

    def test_apply_sets_verifies_and_prints_the_marker(self):
        h = fresh_helper()
        setattr(torch.backends.cuda.matmul, FLAG, True)
        out = io.StringIO()
        record = h.apply(rank=3, stream=out)
        self.assertIs(getattr(torch.backends.cuda.matmul, FLAG), False)
        self.assertEqual((record['rank'], record['before'], record['after'], record['count']), (3, True, False, 1))
        line = out.getvalue()
        self.assertTrue(line.startswith(f'DS41-PRECISION-APPLIED rank=3 {FLAG}=False before=True torch={torch.__version__} pid='), line)
        self.assertEqual(line.count('\n'), 1)
        status = h.status()
        self.assertEqual((status['schema'], status[FLAG], status['applications']), ('ds41-precision-v1', False, [record]))
        again = h.apply(rank=3, stream=io.StringIO())
        self.assertEqual((again['before'], again['count']), (False, 2))

    def test_apply_fails_closed_when_the_flag_does_not_take(self):
        h = fresh_helper()

        class Sticky:
            allow_bf16_reduced_precision_reduction = property(lambda self: True, lambda self, value: None)

        with mock.patch.object(h, 'backend', Sticky):
            with self.assertRaisesRegex(RuntimeError, 'did not read back False'):
                h.apply(rank=0, stream=io.StringIO())
        self.assertEqual(h.status()['applications'], [])

    def test_helper_has_no_hot_path_or_other_math_settings(self):
        text = (HERE / 'ds41_precision.py').read_text()
        tree = ast.parse(text)
        called = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        forbidden = {'mm', 'matmul', 'synchronize', 'set_float32_matmul_precision', 'use_deterministic_algorithms', 'compile'}
        self.assertFalse(called & forbidden, called & forbidden)
        assigned = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)}
        self.assertEqual(assigned, set())        # the flag is set through setattr on the backend only
        self.assertNotIn('allow_tf32', text)
        self.assertEqual(text.count(FLAG), 2)                     # the docstring and the FLAG constant only
        self.assertNotIn('allow_fp16', text)
        imports = {n.names[0].name for n in ast.walk(tree) if isinstance(n, ast.Import)}
        self.assertEqual(imports, {'os', 'sys', 'time', 'torch'})


class Seam(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = (HERE / prep.BASE_WORKER).read_text()
        cls.new = prep.compose(cls.base)

    def test_only_the_import_and_leading_call_are_added(self):
        diff = [l for l in difflib.ndiff(self.base.splitlines(), self.new.splitlines()) if l.startswith(('- ', '+ '))]
        self.assertFalse([l for l in diff if l.startswith('- ')])
        added = [l[2:] for l in diff]
        self.assertEqual(added, ['import vllm.v1.worker.ds41_precision as _ds41_precision',
                                 '        # DIAGNOSTIC ONLY: BF16 reduced-precision reduction off, once per worker process (ds41_precision.py).',
                                 '        _ds41_precision.apply(rank=self.rank)'])
        t = self.new
        self.assertLess(t.index('    def init_device(self):'), t.index('_ds41_precision.apply(rank=self.rank)'))
        self.assertLess(t.index('_ds41_precision.apply(rank=self.rank)'), t.index('if self.device_config.device_type == "cuda":'))
        self.assertEqual(t.count('_ds41_precision.'), 1)
        prep.verify_only_hook_changed(self.base, self.new)

    def test_other_changes_or_placements_are_rejected(self):
        call = '        _ds41_precision.apply(rank=self.rank)\n'
        for mutate in (lambda s: s.replace(call, call + '        torch.backends.cuda.matmul.allow_tf32 = False\n'),
                       lambda s: s.replace(call, '').replace('            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)\n',
                                                             '            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)\n    ' + call),
                       lambda s: s.replace(call, call + call),
                       lambda s: s.replace('apply(rank=self.rank)', 'apply(rank=self.local_rank)'),
                       lambda s: s.replace('torch.set_float32_matmul_precision(precision)', 'pass'),
                       lambda s: s + '\nX = 1\n'):
            with self.assertRaises(ValueError):
                prep.verify_only_hook_changed(self.base, mutate(self.new))
        with self.assertRaises(ValueError):
            prep.compose(self.base + '\n', sha(self.base.encode()))
        with self.assertRaises(ValueError):
            prep.compose(self.base.replace(prep.IMPORT_ANCHOR, prep.IMPORT_ANCHOR * 2))

    def test_extracted_init_device_applies_before_any_device_work(self):
        calls = []
        recorder = types.SimpleNamespace(apply=lambda **kw: calls.append(kw))
        init_device = extracted_init_device({'_ds41_precision': recorder, 'torch': torch})
        with self.assertRaises(Stop):
            init_device(FakeWorker(rank=2))
        self.assertEqual(calls, [{'rank': 2}])

    def test_extracted_init_device_makes_the_real_flag_effective(self):
        original = getattr(torch.backends.cuda.matmul, FLAG)
        self.addCleanup(setattr, torch.backends.cuda.matmul, FLAG, original)
        setattr(torch.backends.cuda.matmul, FLAG, True)
        h = fresh_helper()
        init_device = extracted_init_device({'_ds41_precision': h, 'torch': torch})
        out = io.StringIO()
        with mock.patch.object(h.sys, 'stdout', out):
            with self.assertRaises(Stop):
                init_device(FakeWorker(rank=1))
        self.assertIs(getattr(torch.backends.cuda.matmul, FLAG), False)
        self.assertTrue(out.getvalue().startswith(f'DS41-PRECISION-APPLIED rank=1 {FLAG}=False before=True'))
        self.assertEqual(h.status()['applications'][0]['rank'], 1)


class Lock(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lock = json.loads((HERE / 'ds41-precision.lock.json').read_text())
        cls.indexer = json.loads((HERE / 'ds41-indexer.lock.json').read_text())
        cls.receipt = json.loads((HERE / 'receipts/indexer-build-receipt.json').read_text())

    def test_inputs_targets_dockerfile_and_ignore_reproduce(self):
        lock = self.lock
        for name, digest in lock['inputs'].items():
            self.assertEqual(sha((HERE / name).read_bytes()), digest, name)
        base = (HERE / prep.BASE_WORKER).read_bytes()
        new = prep.compose(base.decode())
        self.assertEqual((HERE / prep.OUTPUT).read_text(), new)
        self.assertEqual(lock['targets'][prep.WORKER], {'input_sha256': sha(base), 'output_sha256': sha(new.encode()),
                                                        'source': prep.OUTPUT})
        self.assertEqual(lock['targets'][prep.PRECISION_HELPER],
                         {'input_sha256': None, 'output_sha256': sha((HERE / 'ds41_precision.py').read_bytes()),
                          'source': 'ds41_precision.py'})
        self.assertEqual((HERE / 'Dockerfile.ds41-precision').read_text(), prep.render_dockerfile(lock))
        names = sorted(lock['inputs']) + ['ds41-precision.lock.json']
        self.assertEqual((HERE / 'ds41-precision.ignore').read_text(), '**\n' + ''.join('!' + n + '\n' for n in names))
        self.assertEqual(lock['apply_marker'], fresh_helper().MARKER)
        self.assertEqual(lock['precision_schema'], fresh_helper().SCHEMA)
        self.assertEqual(set(lock['inputs']), set(prep.PACKAGED) | {prep.OUTPUT, 'ds41-precision.patch'})

    def test_base_is_the_exact_indexer_image_and_helpers_are_preserved(self):
        lock, indexer, receipt = self.lock, self.indexer, self.receipt
        self.assertEqual(lock['base_image_id'], receipt['image_id'])
        self.assertEqual(lock['base_image_id'][:8], '25c92dde')
        self.assertEqual((lock['base_kind'], lock['base_lock'], lock['base_lock_sha256']),
                         ('indexer-capture', 'ds41-indexer.lock.json', receipt['lock_sha256']))
        self.assertEqual(sha((HERE / 'ds41-indexer.lock.json').read_bytes()), lock['base_lock_sha256'])
        self.assertEqual(lock['base_trees'], indexer['trees'])
        self.assertEqual(lock['window_lock_sha256'], indexer['base_lock_sha256'])
        self.assertEqual(lock['decision_row_lock_sha256'], indexer['decision_row_lock_sha256'])
        expected = {**indexer['preserved'], **{t: e['output_sha256'] for t, e in indexer['targets'].items()}}
        self.assertEqual(lock['preserved'], expected)
        self.assertEqual(len(expected), 4)
        self.assertNotEqual(lock['trees']['vllm'], indexer['trees']['vllm'])
        self.assertEqual(lock['trees']['b12x'], indexer['trees']['b12x'])
        self.assertEqual((lock['install_dir'], lock['install_pass_marker'], lock['image_kind_label'], lock['build_args']),
                         ('/opt/ds41-precision', 'PRECISION-INSTALL-PASS', 'precision-capture', ['DECISION_LOCK', 'DECISION_CACHE']))
        provenance = lock['inherited_helper_provenance']
        self.assertTrue(provenance['disposition'].startswith('metadata-only'))
        self.assertEqual((provenance['window_read_path'], provenance['decision_read_path'], provenance['indexer_read_path']),
                         ('/opt/ds41-window/claude-window.lock.json', '/opt/ds41-decision-row/claude-decision-row.lock.json',
                          '/opt/ds41-indexer/ds41-indexer.lock.json'))
        text = (HERE / 'Dockerfile.ds41-precision').read_text()
        for required in ('FROM ' + receipt['image_id'], 'ARG DECISION_LOCK', 'ARG DECISION_CACHE',
                         '/opt/ds41-precision/ds41_precision_install.py', 'diagnostic.kind="precision-capture"',
                         'base-kind="indexer-capture"', 'indexer-lock.sha256="' + receipt['lock_sha256'],
                         'window-lock.sha256="' + indexer['base_lock_sha256'],
                         'vllm.source-tree="' + lock['trees']['vllm']):
            self.assertIn(required, text)
        for line in [l for l in text.splitlines() if l.endswith('\\')]:
            self.assertTrue(line.endswith(' \\') and not line.endswith('\\\\'), line)

    def test_full_tree_identities_reproduce_from_the_router_reconstruction(self):
        base = prep.base_identity()
        worker = prep.router_worker_source()
        self.assertEqual(worker, (HERE / prep.BASE_WORKER).read_bytes())
        self.assertEqual(sha(worker), self.lock['targets'][prep.WORKER]['input_sha256'])
        trees = prep.child_trees(base, (HERE / prep.OUTPUT).read_bytes(), (HERE / 'ds41_precision.py').read_bytes())
        self.assertEqual(trees, self.lock['trees'])


class Installer(unittest.TestCase):
    def simulate(self, tamper=None):
        lock = json.loads((HERE / 'ds41-precision.lock.json').read_text())
        indexer_lock_raw = (HERE / 'ds41-indexer.lock.json').read_bytes()
        indexer_lock = json.loads(indexer_lock_raw)
        with tempfile.TemporaryDirectory() as tmp:
            kit, opt = Path(tmp) / 'kit', Path(tmp) / 'opt'
            window, decision, indexer = Path(tmp) / 'window', Path(tmp) / 'decision', Path(tmp) / 'indexer'
            for d in (kit, window, decision, indexer):
                d.mkdir()
            for name in [*lock['inputs'], 'ds41-precision.lock.json']:
                shutil.copy(HERE / name, kit / name)
            model = opt / 'vllm/vllm/models/deepseek_v4_1'
            worker_dir = opt / 'vllm/vllm/v1/worker'
            model.mkdir(parents=True)
            worker_dir.mkdir(parents=True)
            shutil.copy(HERE / 'ds41-indexer-attention.py', model / 'attention.py')
            shutil.copy(HERE / 'claude-decision-row-capture.py', model / 'claude_decision_row.py')
            shutil.copy(HERE / 'claude-window-capture.py', model / 'claude_window.py')
            shutil.copy(HERE / 'ds41_indexer_capture.py', model / 'ds41_indexer_capture.py')
            shutil.copy(HERE / prep.BASE_WORKER, worker_dir / 'gpu_worker.py')
            fenced = opt / 'b12x' / lock['base_router_target']['path']
            fenced.parent.mkdir(parents=True)
            shutil.copy(HERE / 'router-prefill-release-before.py', fenced)
            (opt / 'b12x/b12x').mkdir(parents=True, exist_ok=True)
            (opt / 'b12x/b12x/native.so').write_bytes(b'x')
            stub = {'kind': 'indexer-capture-provenance-stub', 'trees': indexer_lock['trees'],
                    'indexer_lock_sha256': sha(indexer_lock_raw)}
            (window / 'claude-window.lock.json').write_text(json.dumps(stub))
            (decision / 'claude-decision-row.lock.json').write_text(json.dumps(stub))
            shutil.copy(HERE / 'claude-window.lock.json', window / 'claude-window.lock.original.json')
            (decision / 'claude-decision-row.lock.original.json').write_text(json.dumps(
                {'kind': 'window-capture-provenance-stub', 'trees': indexer_lock['base_trees']}))
            (indexer / 'ds41-indexer.lock.json').write_bytes(indexer_lock_raw)
            inventory = {str(p): sha(p.read_bytes()) for c in ('vllm', 'b12x') for p in (opt / c).rglob('*') if p.is_file()}
            (indexer / 'after-files.json').write_text(json.dumps(inventory, sort_keys=True) + '\n')
            if tamper:
                tamper(opt=opt, window=window, decision=decision, indexer=indexer, model=model, worker=worker_dir)
            source = (HERE / 'ds41_precision_install.py').read_text()
            script = Path(tmp) / 'install.py'
            script.write_text(source.replace("Path('/opt/ds41-precision')", f"Path({str(kit)!r})")
                              .replace("Path('/opt/jovian-judgement')", f"Path({str(opt)!r})")
                              .replace("Path('/opt/ds41-window')", f"Path({str(window)!r})")
                              .replace("Path('/opt/ds41-decision-row')", f"Path({str(decision)!r})")
                              .replace("Path('/opt/ds41-indexer')", f"Path({str(indexer)!r})"))
            result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
            files = sorted(p.relative_to(opt).as_posix() for p in opt.rglob('*') if p.is_file())
            provenance = {d.name + '/' + p.name: p.read_bytes() for d in (window, decision, indexer)
                          for p in d.iterdir() if p.is_file()}
            return result, (worker_dir / 'gpu_worker.py').read_text(), files, provenance

    def test_install_is_confined_to_the_worker_and_the_new_helper(self):
        result, worker, files, provenance = self.simulate()
        self.assertIn('PRECISION-INSTALL-PASS', result.stdout, result.stderr)
        lock_raw = (HERE / 'ds41-precision.lock.json').read_bytes()
        lock = json.loads(lock_raw)
        self.assertEqual(worker, (HERE / prep.OUTPUT).read_text())
        self.assertEqual(files, ['b12x/b12x/gemm/bf16_gemv/_prefill.py', 'b12x/b12x/native.so',
                                 'vllm/vllm/models/deepseek_v4_1/attention.py',
                                 'vllm/vllm/models/deepseek_v4_1/claude_decision_row.py',
                                 'vllm/vllm/models/deepseek_v4_1/claude_window.py',
                                 'vllm/vllm/models/deepseek_v4_1/ds41_indexer_capture.py',
                                 'vllm/vllm/v1/worker/ds41_precision.py', 'vllm/vllm/v1/worker/gpu_worker.py'])
        # all three read paths now name this image's trees and the precision lock
        for name in ('window/claude-window.lock.json', 'decision/claude-decision-row.lock.json', 'indexer/ds41-indexer.lock.json'):
            stub = json.loads(provenance[name])
            self.assertEqual((stub['kind'], stub['trees'], stub['precision_lock_sha256'], stub['indexer_lock_sha256']),
                             ('precision-provenance-stub', lock['trees'], sha(lock_raw), lock['base_lock_sha256']), name)
        # indexer-era files preserved byte for byte, window-era originals untouched
        self.assertEqual(sha(provenance['indexer/ds41-indexer.lock.original.json']), lock['base_lock_sha256'])
        for name in ('window/claude-window.lock.indexer-stub.json', 'decision/claude-decision-row.lock.indexer-stub.json'):
            self.assertEqual(json.loads(provenance[name])['kind'], 'indexer-capture-provenance-stub', name)
        self.assertEqual(sha(provenance['window/claude-window.lock.original.json']), lock['window_lock_sha256'])
        self.assertEqual(json.loads(provenance['decision/claude-decision-row.lock.original.json'])['kind'],
                         'window-capture-provenance-stub')
        self.assertEqual(len(provenance), 9)

    def test_other_bases_are_refused(self):
        cases = {
            'indexer lock differs': lambda **d: (d['indexer'] / 'ds41-indexer.lock.json').write_text('{}'),
            'Base helper differs': lambda **d: (d['model'] / 'attention.py').write_text('x'),
            'Base worker differs': lambda **d: (d['worker'] / 'gpu_worker.py').write_text('x'),
            'inventory differs': lambda **d: (d['opt'] / 'vllm/extra.py').write_text('x'),
            'preexisting diagnostic file': lambda **d: (d['worker'] / 'ds41_precision.py').write_text('x'),
            'already exist': lambda **d: (d['indexer'] / 'ds41-indexer.lock.original.json').write_text('x'),
            'expected indexer-capture-provenance-stub': lambda **d: (d['window'] / 'claude-window.lock.json').write_text('{"kind": "other"}'),
            'different indexer lock': lambda **d: (d['decision'] / 'claude-decision-row.lock.json').write_text(json.dumps(
                {**json.loads((d['decision'] / 'claude-decision-row.lock.json').read_text()), 'indexer_lock_sha256': 'x'})),
            'Window-era original lock': lambda **d: (d['window'] / 'claude-window.lock.original.json').unlink(),
            'Window-era original decision-row stub': lambda **d: (d['decision'] / 'claude-decision-row.lock.original.json').write_text('{}'),
        }
        for message, tamper in cases.items():
            result, *_ = self.simulate(tamper)
            self.assertNotEqual(result.returncode, 0, message)
            self.assertIn(message, result.stderr, message)


if __name__ == '__main__':
    unittest.main()
