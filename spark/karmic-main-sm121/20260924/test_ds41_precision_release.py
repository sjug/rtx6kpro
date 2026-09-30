"""Precision release derivative: exact source delta over the router parent, no capture helpers, fail-closed install.

    .venv-snapshot-cpu/bin/python -m unittest test_ds41_precision_release
"""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prep = load('ds41_precision_release_prepare', 'ds41_precision_release_prepare.py')
installer = load('ds41_precision_release_install', 'ds41_precision_release_install.py')
build = load('build_precision_release', 'build_precision_release.py')
LOCK_RAW = (ROOT / 'ds41-precision-release.lock.json').read_bytes()
LOCK = json.loads(LOCK_RAW)
ROUTER = json.loads((ROOT / 'router-release.lock.json').read_text())
PRECISION = json.loads((ROOT / 'ds41-precision.lock.json').read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


class SourceDelta(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from contracts import git_tree, manifest
        cls.git_tree, cls.manifest = staticmethod(git_tree), staticmethod(manifest)
        cls.before, cls.after = prep.release()

    def test_before_is_the_router_parent(self):
        self.assertEqual(self.manifest(self.before), ROUTER['after']['vllm'])
        self.assertEqual(self.git_tree(self.before), ROUTER['trees']['vllm'])
        self.assertEqual(LOCK['before'], ROUTER['after'])
        self.assertEqual(LOCK['base_trees'], ROUTER['trees'])

    def test_only_worker_and_helper_change(self):
        before, after = LOCK['before']['vllm'], LOCK['after']['vllm']
        self.assertEqual(LOCK['before']['b12x'], LOCK['after']['b12x'])
        self.assertEqual(LOCK['trees']['b12x'], ROUTER['trees']['b12x'])
        self.assertEqual(set(after) - set(before), {prep.HELPER})
        self.assertEqual(set(before) - set(after), set())
        self.assertEqual([p for p in before if before[p] != after[p]], [prep.WORKER])
        self.assertEqual(after[prep.HELPER], {'mode': '100644', 'sha256': LOCK['targets'][prep.HELPER]['output_sha256']})
        self.assertEqual(self.manifest(self.after), after)
        self.assertEqual(self.git_tree(self.after), LOCK['trees']['vllm'])

    def test_tree_matches_the_established_reconstruction(self):
        import claude_prepare_decision_row as decision
        changes = {path: data for path, (_, data) in self.after.items() if path in prep.SOURCES}
        self.assertEqual(decision.vllm_tree_after(changes), LOCK['trees']['vllm'])

    def test_targets_are_the_tested_precision_bytes(self):
        for path, entry in LOCK['targets'].items():
            tested = PRECISION['targets']['vllm/' + path]
            self.assertEqual(entry['output_sha256'], tested['output_sha256'])
            self.assertEqual(entry['input_sha256'], tested['input_sha256'])
            self.assertEqual(sha((ROOT / entry['source']).read_bytes()), entry['output_sha256'])
        self.assertEqual(LOCK['targets'][prep.WORKER]['input_sha256'], ROUTER['after']['vllm'][prep.WORKER]['sha256'])
        self.assertEqual(LOCK['precision_lock_sha256'], sha((ROOT / 'ds41-precision.lock.json').read_bytes()))

    def test_worker_delta_is_one_import_and_one_leading_call(self):
        dp = load('ds41_precision_prepare', 'ds41_precision_prepare.py')
        base = self.before[prep.WORKER][1].decode()
        self.assertEqual(dp.compose(base, LOCK['targets'][prep.WORKER]['input_sha256']).encode(), self.after[prep.WORKER][1])
        dp.verify_only_hook_changed(base, self.after[prep.WORKER][1].decode())

    def test_no_capture_instrumentation(self):
        after = LOCK['after']['vllm']
        for path in prep.CAPTURE_PATHS:
            self.assertNotIn(path, after)
        self.assertEqual(after['vllm/models/deepseek_v4_1/attention.py'],
                         ROUTER['after']['vllm']['vllm/models/deepseek_v4_1/attention.py'])
        named = lambda files: sorted(p for p in files if 'claude_' in Path(p).name or 'capture' in Path(p).name)
        self.assertEqual(named(after), named(ROUTER['after']['vllm']))
        helper = (ROOT / 'ds41_precision.py').read_text()
        imports = {n.names[0].name for n in ast.walk(ast.parse(helper)) if isinstance(n, ast.Import)}
        self.assertEqual(imports, {'os', 'sys', 'time', 'torch'})


class LockAndBuild(unittest.TestCase):
    def test_inputs_and_allowlist_reproduce(self):
        lock, digest = build.frozen()
        self.assertEqual(digest, sha(LOCK_RAW))
        self.assertEqual(set(lock['inputs']), set(prep.PACKAGED))

    def test_router_identity(self):
        router, router_sha, image = prep.router_identity()
        self.assertEqual(image, 'e06df11a8ca18fa514d9f28f67cc691aef296da2eeb22b113a734519853bccd7')
        self.assertEqual((LOCK['base_image_id'], LOCK['base_lock_sha256']), (image, router_sha))
        self.assertEqual(router_sha, '7122753a3206df8b1ff23b1fa1b0727daf4e1378f940a4ab7358dfeb76065a77')

    def test_gates_are_the_router_builds(self):
        for name in [g for g, _ in LOCK['gates']] + LOCK['gate_data']:
            self.assertEqual(LOCK['inputs'][name], ROUTER['inputs'][name], name)
        router_build = (ROOT / 'build_router_release.py').read_text()
        for name, marker in LOCK['gates']:
            self.assertIn(f"('{name}', '{marker}')", router_build)

    def test_dockerfile(self):
        text = (ROOT / 'Dockerfile.ds41-precision-release').read_text()
        self.assertEqual([l for l in text.splitlines() if l.startswith('FROM ')], ['FROM ' + LOCK['base_image_id']])
        self.assertEqual(text.count('\nRUN '), 1)
        self.assertIn('RUN /opt/venv/bin/python /opt/ds41-precision-release/ds41_precision_release_install.py', text)
        copy = next(l for l in text.splitlines() if l.startswith('COPY '))
        self.assertEqual(copy.split()[1:-1], sorted(prep.PACKAGED) + ['ds41-precision-release.lock.json'])
        for key, value in build.expected_labels(LOCK, '${PRECISION_LOCK}').items():
            value = '${PRECISION_CACHE}' if key == 'local-inference.cache.fingerprint' else value
            self.assertIn(f'{key}="{value}"', text)
        for word in ('DS41_DECISION_ROW_BLOCKS', 'NCCL_', 'claude_', 'capture'):
            self.assertNotIn(word, '\n'.join(l for l in text.split('LABEL')[0].splitlines() if not l.startswith('#')))
        self.assertTrue(LOCK['cache_fingerprint'].startswith('ds41-precision-release-'))
        self.assertNotEqual(LOCK['cache_fingerprint'], ROUTER['cache_fingerprint'])


class Installer(unittest.TestCase):
    """Runs the real installer against a small synthetic parent built from the lock's own shape."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.here, self.root, self.parent = base / 'here', base / 'jj', base / 'router'
        for d in (self.here, self.parent):
            d.mkdir()
        worker_base = (ROOT / 'ds41-precision-gpu_worker-base.py').read_bytes()
        self.files = {('vllm', prep.WORKER): worker_base, ('vllm', 'vllm/models/x.py'): b'x = 1\n',
                      ('b12x', 'b12x/y.py'): b'y = 2\n', ('b12x', 'b12x/native.so'): b'\x7fELF'}
        for (component, name), data in self.files.items():
            path = self.root / component / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        router_lock = b'{"router": true}\n'
        (self.parent / 'router-release.lock.json').write_bytes(router_lock)
        (self.parent / 'preserved-files.json').write_text(json.dumps(installer.inventory(self.root)))
        for name in ('ds41-precision-gpu_worker.py', 'ds41_precision.py', 'ds41_precision_release_install.py'):
            (self.here / name).write_bytes((ROOT / name).read_bytes())
        manifest = lambda comp: {n: {'mode': '100644', 'sha256': sha(d)} for (c, n), d in self.files.items()
                                 if c == comp and not n.endswith('.so')}
        after = manifest('vllm')
        for path, entry in LOCK['targets'].items():
            after[path] = {'mode': '100644', 'sha256': entry['output_sha256']}
        self.lock = dict(LOCK, base_lock_sha256=sha(router_lock),
                         before={'vllm': manifest('vllm'), 'b12x': manifest('b12x')},
                         after={'vllm': after, 'b12x': manifest('b12x')},
                         inputs={n: sha((self.here / n).read_bytes()) for n in
                                 ('ds41-precision-gpu_worker.py', 'ds41_precision.py', 'ds41_precision_release_install.py')})
        self.write_lock()

    def write_lock(self):
        (self.here / 'ds41-precision-release.lock.json').write_text(json.dumps(self.lock))

    def tearDown(self):
        self.tmp.cleanup()

    def run_install(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            installer.install(self.here, self.root, self.parent)
        return out.getvalue()

    def test_success_is_confined(self):
        before = installer.inventory(self.root)
        self.assertEqual(self.run_install().strip(), LOCK['install_pass'])
        after = installer.inventory(self.root)
        changed = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
        self.assertEqual(changed, {str(self.root / 'vllm' / p) for p in LOCK['targets']})
        self.assertEqual(json.loads((self.here / 'preserved-files.json').read_text()), after)
        self.assertEqual((self.parent / 'router-release.lock.json').read_bytes(), b'{"router": true}\n')

    def refuse(self, message, mutate):
        mutate()
        with self.assertRaisesRegex(RuntimeError, message):
            self.run_install()

    def test_refusals(self):
        cases = {
            'Recipe drift': lambda: (self.here / 'ds41_precision.py').write_text('# changed\n'),
            'Parent router lock differs': lambda: (self.parent / 'router-release.lock.json').write_text('{}'),
            'Base source drift': lambda: (self.root / 'b12x/b12x/y.py').write_text('y = 3\n'),
            'inventory differs': lambda: (self.root / 'b12x/b12x/extra.py').write_text(''),
            'Capture instrumentation': lambda: self.plant('vllm/vllm/models/deepseek_v4_1/claude_window.py'),
            'Unexpected preexisting file': lambda: self.plant('vllm/vllm/v1/worker/ds41_precision.py'),
        }
        for message, mutate in cases.items():
            with self.subTest(message=message):
                self.tearDown()
                self.setUp()
                self.refuse(message, mutate)

    def test_missing_native_and_wrong_preimage(self):
        (self.root / 'b12x/b12x/native.so').unlink()
        (self.parent / 'preserved-files.json').write_text(json.dumps(installer.inventory(self.root)))
        self.refuse('Missing inherited native', lambda: None)
        self.tearDown()
        self.setUp()
        self.lock['targets'] = {p: dict(e, input_sha256='0' * 64) if e['input_sha256'] else e
                                for p, e in self.lock['targets'].items()}
        self.lock['before']['vllm'][prep.WORKER]['sha256'] = sha(self.files[('vllm', prep.WORKER)])
        self.write_lock()
        self.refuse('Worker preimage differs', lambda: None)

    def plant(self, relative):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('')
        (self.parent / 'preserved-files.json').write_text(json.dumps(installer.inventory(self.root)))


class Smoke(unittest.TestCase):
    def test_static_contract(self):
        text = (ROOT / 'ds41_precision_release_smoke.py').read_text()
        for needle in ("lock['smoke_pass']", "lock['apply_marker']", "worker._ds41_precision is not helper",
                       "lock['capture_paths_absent']", "'/opt/ds41-precision'", "before is not True"):
            self.assertIn(needle, text)
        self.assertLess(text.index("print(line, flush=True)"), text.index('PRECISION-RELEASE-GEMM-OBSERVATION'))
        self.assertEqual(text.count("raise RuntimeError"), text[:text.index('generator =')].count("raise RuntimeError"))


if __name__ == '__main__':
    unittest.main()
