import ast
import hashlib
import json
from pathlib import Path
import subprocess
import unittest

from prepare_deterministic_diagnostic import COMMIT, REPO, ROOT, TARGET, object_id, tree_id


class PackagingTests(unittest.TestCase):
    def test_only_selector_body_changes(self):
        old = subprocess.check_output(['git', '-C', str(REPO), 'show', f'{COMMIT}:{TARGET}']).decode()
        new = (ROOT / 'diagnostic-mxfp4.py').read_text()
        def definitions(text):
            return {n.name: ast.dump(n) for n in ast.parse(text).body
                    if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))}
        before, after = definitions(old), definitions(new)
        self.assertEqual(before.keys(), after.keys())
        self.assertEqual([name for name in before if before[name] != after[name]], ['select_mxfp4'])

    def test_payload_hashes(self):
        lock = json.loads((ROOT / 'diagnostic-topk.lock.json').read_text())
        for filename, key in [('diagnostic-mxfp4.py', 'output_sha256'),
                              ('deterministic_topk.py', 'module_sha256'),
                              ('diagnostic-topk.patch', 'patch_sha256')]:
            self.assertEqual(hashlib.sha256((ROOT / filename).read_bytes()).hexdigest(), lock[key])

    def test_git_hashing(self):
        self.assertEqual(object_id('blob', b''), 'e69de29bb2d1d6434b8b29ae775ad8c2e48c5391')
        self.assertEqual(tree_id({}), '4b825dc642cb6eb9a060e54bf8d69288fbee4904')


if __name__ == '__main__':
    unittest.main()
