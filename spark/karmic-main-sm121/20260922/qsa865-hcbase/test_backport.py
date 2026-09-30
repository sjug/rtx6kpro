"""CPU artifact checks, not claims of a GPU crash reproduction."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent


class BackportTests(unittest.TestCase):
    def test_changed_set_and_native_reuse(self):
        pin = json.loads((ROOT / 'backport.lock.json').read_text())
        old = json.loads((ROOT.parent / 'source.lock.json').read_text())
        new = json.loads((ROOT / 'runtime.lock.json').read_text())
        before = old['sources']['vllm']['files']
        after = new['sources']['vllm']['files']
        changed = {p for p in before.keys() | after.keys() if before.get(p) != after.get(p)}
        self.assertEqual(changed, set(new['sources']['vllm']['changed_paths']))
        self.assertEqual(set(pin['after']), changed)
        self.assertTrue(all(p.endswith(('.py', '.json', '.md')) for p in changed))
        for p in before:
            if p.startswith(('csrc/', 'cmake/', 'rust/', 'requirements/')):
                self.assertEqual(before[p], after[p])
        self.assertEqual(old['sources']['b12x'], new['sources']['b12x'])
        self.assertNotIn('lmcache_refresh', new)
        self.assertEqual(old['assets'], new['assets'])

    def test_replay_from_frozen_archive(self):
        pin = json.loads((ROOT / 'backport.lock.json').read_text())
        import hashlib
        with tempfile.TemporaryDirectory() as scratch:
            for path in pin['before']:
                target = Path(scratch) / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(subprocess.check_output(['git', '-C', '/home/jugs/git/vllm',
                                   'show', pin['applied_to_commit'] + ':' + path]))
            subprocess.run(['git', 'apply', '--check', str(ROOT / 'upstream.patch')], cwd=scratch, check=True)
            subprocess.run(['git', 'apply', str(ROOT / 'upstream.patch')], cwd=scratch, check=True)
            for path, entry in pin['after'].items():
                data = (Path(scratch) / path).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), entry['sha256'])
                ast.parse(data)

    def test_tracked_map_refresh_is_complete(self):
        # Execute the actual simple manifest-refresh statements on an isolated receipt.
        tree = ast.parse((ROOT / 'install.py').read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        statements = [n for n in main.body if isinstance(n, ast.Assign) and
                      any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and
                          t.value.id == 'original' for t in n.targets)]
        self.assertEqual(len(statements), 1)
        new = json.loads((ROOT / 'runtime.lock.json').read_text())
        original = {'tracked': {}, 'build_products': {'preserved': 'native'}}
        scope = {'original': original, 'new': new}
        exec(compile(ast.Module(body=statements, type_ignores=[]), '<tracked-refresh>', 'exec'), scope)
        self.assertEqual(original['tracked'], {p: e['sha256'] for p, e in new['sources']['vllm']['files'].items()})
        self.assertEqual(original['build_products'], {'preserved': 'native'})

    def test_runner_preserves_32_row_capture_both_roles(self):
        for role in ('head', 'worker'):
            env = {**os.environ, 'DRY_RUN': '1', 'ROLE': role,
                   'VLLM_QWEN3_8_FLASH_NEXT_HC_TP': '0'}
            result = subprocess.run(['bash', str(ROOT / 'run-qwen.sh')], env=env, capture_output=True, text=True, check=True)
            self.assertIn('VLLM_QWEN3_8_FLASH_NEXT_HC_TP=0', result.stdout)
            self.assertIn('--max-cudagraph-capture-size 32', result.stdout)
            self.assertNotIn('--max-cudagraph-capture-size 16', result.stdout)
            self.assertIn('karmic-main-hcbase-qsa865-spark-sm121', result.stdout)

    def test_invalid_controls_fail(self):
        for extra in ({'NUM_SPECULATIVE_TOKENS': '0'},
                      {'MAX_NUM_SEQS': '8'}, {'ROLE': 'stop'}):
            env = {**os.environ, 'DRY_RUN': '1', 'ROLE': 'head', **extra}
            result = subprocess.run(['bash', str(ROOT / 'run-qwen.sh')], env=env, capture_output=True)
            self.assertNotEqual(result.returncode, 0)

    def test_preflight_normal_and_optimized(self):
        for flags in ([], ['-O']):
            subprocess.run([sys.executable, *flags, str(ROOT / 'preflight.py')], check=True)

    def test_build_dry_run(self):
        result = subprocess.run(['bash', str(ROOT / 'build.sh')],
                                env={**os.environ, 'DRY_RUN': '1'}, capture_output=True, text=True, check=True)
        self.assertIn('DRY-RUN:', result.stdout)


if __name__ == '__main__':
    unittest.main()
