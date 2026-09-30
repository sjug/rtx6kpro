import os
import runpy
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path
from contracts import git_tree, load_lock, overlay, require, safe_path
from contracts import sha
from install import entry_sha

ROOT = Path(__file__).resolve().parent

class Contracts(unittest.TestCase):
    def test_empty_git_tree(self):
        self.assertEqual(git_tree({}), '4b825dc642cb6eb9a060e54bf8d69288fbee4904')

    def test_failure_survives_optimization(self):
        run = subprocess.run(['python', '-O', '-c', 'from contracts import require; require(False,"refused")'], cwd=ROOT, capture_output=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn(b'refused', run.stderr)

    def test_overlay_refuses_drift(self):
        with self.assertRaises((RuntimeError, KeyError)):
            overlay({'CMakeLists.txt': ('100644', b'drift')})

    def test_unsafe_path(self):
        for path in ('/tmp/wrong', '../wrong', 'a/../../wrong'):
            with self.subTest(path=path), self.assertRaises(RuntimeError):
                safe_path(ROOT, path)

    def test_parent_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'escape').symlink_to('/')
            with self.assertRaises(RuntimeError):
                safe_path(root, 'escape/etc/passwd')

    def test_tracked_dangling_symlink_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'tracked-link'
            path.symlink_to('../../.agents/skills/ci-fails-buildkite')
            self.assertFalse(path.exists())
            self.assertEqual(entry_sha(path, '120000'), sha(os.readlink(path).encode()))

    def test_preflight(self):
        subprocess.run(['python', 'preflight.py'], cwd=ROOT, check=True)

    def test_dry_run_has_no_host_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Check the dry run adds no receipt, lock or other task files.
            env = {**os.environ, 'DRY_RUN': '1'}
            before = {str(p) for p in ROOT.rglob('*') if '__pycache__' not in p.parts}
            run = subprocess.run(['bash', 'build.sh'], cwd=ROOT, env=env, capture_output=True)
            self.assertEqual(run.returncode, 0, run.stderr.decode())
            self.assertIn(b'DRY-RUN', run.stdout)
            after = {str(p) for p in ROOT.rglob('*') if '__pycache__' not in p.parts}
            self.assertEqual(before, after)

    def test_runner_render(self):
        for role in ('head', 'worker'):
            run = subprocess.run(['bash', 'run-qwen-tp2-node.sh'], cwd=ROOT,
                                 env={**os.environ, 'DRY_RUN': '1', 'ROLE': role}, capture_output=True)
            self.assertEqual(run.returncode, 0, run.stderr.decode())
            self.assertIn(b':karmic-main-spark-sm121', run.stdout)
            self.assertIn(b'NUM_SPECULATIVE_TOKENS=3', run.stdout)
            self.assertIn(b'NCCL_PROTO=LL', run.stdout)
            self.assertNotIn(b'NCCL_MIN_NCHANNELS', run.stdout)
            self.assertNotIn(b'NCCL_MAX_NCHANNELS', run.stdout)

    def test_launchers_preserve_settings(self):
        old = ROOT.parents[1] / 'karmic-beta-sm121/launchers'
        for path in (ROOT / 'inherited/launchers').glob('*.sh'):
            expected = (old / path.name).read_text()
            if 'glm53' in path.name:
                expected = expected.replace('roce_abi_version() == 3', 'roce_abi_version() == 4')
            self.assertEqual(path.read_text(), expected)

    def test_hc_diagnostic_passthrough(self):
        env = {k: v for k, v in os.environ.items() if k != 'VLLM_QWEN3_8_FLASH_NEXT_HC_TP'}
        for role in ('head', 'worker'):
            for value in (None, '0', '1', '', '2'):
                current = {**env, 'DRY_RUN': '1', 'ROLE': role}
                if value is not None:
                    current['VLLM_QWEN3_8_FLASH_NEXT_HC_TP'] = value
                run = subprocess.run(['bash', 'run-qwen-hc-diagnostic.sh'], cwd=ROOT,
                                     env=current, capture_output=True)
                with self.subTest(role=role, value=value):
                    self.assertEqual(run.returncode, 0 if value in (None, '0', '1') else 2)
                    if value in ('0', '1'):
                        self.assertIn(f'VLLM_QWEN3_8_FLASH_NEXT_HC_TP={value}'.encode(), run.stdout)
                    elif value is None:
                        self.assertNotIn(b'VLLM_QWEN3_8_FLASH_NEXT_HC_TP', run.stdout)

    def test_hc_diagnostic_render_has_only_one_delta(self):
        env = {k: v for k, v in os.environ.items() if k != 'VLLM_QWEN3_8_FLASH_NEXT_HC_TP'}
        for role in ('head', 'worker'):
            def render(script, extra):
                run = subprocess.run(['bash', script], cwd=ROOT,
                    env={**env, 'DRY_RUN': '1', 'ROLE': role, **extra},
                    check=True, capture_output=True, text=True)
                return shlex.split(run.stdout)
            baseline = render('run-qwen-tp2-node.sh', {})
            self.assertEqual(baseline, render('run-qwen-hc-diagnostic.sh', {}))
            for value in ('0', '1'):
                actual = render('run-qwen-hc-diagnostic.sh', {'VLLM_QWEN3_8_FLASH_NEXT_HC_TP': value})
                i = actual.index(f'VLLM_QWEN3_8_FLASH_NEXT_HC_TP={value}')
                self.assertEqual(actual[i - 1], '-e')
                del actual[i - 1:i + 1]
                self.assertEqual(baseline, actual)

    def test_hc_log_gate(self):
        module = runpy.run_path(str(ROOT / 'verify-hc-boot.py'))
        check = module['check_log']
        loaded = 'Model loading took 51 GiB'
        sharded = loaded + '\n' + module['MARKER']
        check(loaded, False)
        check(sharded, True)
        for log, enabled in [('', False), ('', True), (loaded, True), (sharded, False)]:
            with self.assertRaises(RuntimeError):
                check(log, enabled)

    def test_saved_qwen_profile(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ('VLLM_QWEN3_8_FLASH_NEXT_HC_TP', 'EXPECTED_IMAGE_ID')}
        for role in ('head', 'worker'):
            current = {**env, 'DRY_RUN': '1', 'ROLE': role}
            saved = subprocess.run(['bash', 'run-qwen-profile.sh'], cwd=ROOT,
                env=current, capture_output=True, text=True)
            self.assertEqual(saved.returncode, 0, saved.stderr)
            diagnostic = subprocess.run(['bash', 'run-qwen-hc-diagnostic.sh'], cwd=ROOT,
                env={**current, 'VLLM_QWEN3_8_FLASH_NEXT_HC_TP': '0'},
                check=True, capture_output=True, text=True)
            self.assertEqual(saved.stdout, diagnostic.stdout)
            for override in ({'VLLM_QWEN3_8_FLASH_NEXT_HC_TP': '1'},
                             {'EXPECTED_IMAGE_ID': '0' * 64}):
                refused = subprocess.run(['bash', 'run-qwen-profile.sh'], cwd=ROOT,
                    env={**current, **override}, capture_output=True)
                self.assertEqual(refused.returncode, 78)

    def test_retention_and_publication_order(self):
        script = (ROOT / 'build.sh').read_text()
        self.assertLess(script.index('--tag "$retention"'), script.index('bash gate.sh'))
        self.assertLess(script.index('image-inspect.json'), script.index('bash gate.sh'))
        self.assertLess(script.index('bash gate.sh'), script.index('podman tag'))
        self.assertIn('--network=none', script)
        self.assertIn('--pull=never', script)

    def test_proxy_and_source_pins(self):
        lock = load_lock(ROOT)
        self.assertEqual(lock['proxy_abi'], 4)
        self.assertEqual(lock['sources']['vllm']['package_tree'], '40f42170a07ed7533cd1af0ed6a01935c8c4e220')
        self.assertEqual(lock['sources']['b12x']['package_tree'], 'ce26f8523539be672eae5928eaa2b10f5c19c69c')

if __name__ == '__main__':
    unittest.main()
