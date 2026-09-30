#!/usr/bin/env python3
"""Local, stdlib-only checks of the Karmic beta refresh kit (no GPU, no image)."""
import hashlib
import importlib.util
import json
import subprocess
import sys
import tarfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FOUNDATION = ROOT.parents[1] / 'karmic-main-sm121/20260922'
sys.path.insert(0, str(FOUNDATION))
from contracts import git_tree, manifest

spec = importlib.util.spec_from_file_location('beta_prepare', ROOT / 'prepare.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class KitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lock = json.loads((ROOT / 'runtime.lock.json').read_text())
        cls.pin = json.loads((ROOT / 'build.lock.json').read_text())
        with tarfile.open(ROOT / 'refresh.tar') as bundle:
            cls.members = {m.name: bundle.extractfile(m).read() for m in bundle.getmembers()}

    def test_preflight_passes(self):
        out = subprocess.run([sys.executable, str(ROOT / 'preflight.py')], check=True,
                             capture_output=True, text=True).stdout
        self.assertIn('KARMIC-BETA-INPUTS-PASS', out)

    def test_prepare_is_deterministic(self):
        before = {n: digest(ROOT / n) for n in ('runtime.lock.json', 'refresh.tar', 'build.lock.json')}
        subprocess.run([sys.executable, str(ROOT / 'prepare.py')], check=True, capture_output=True)
        after = {n: digest(ROOT / n) for n in before}
        self.assertEqual(before, after)

    def test_install_replay_reaches_locked_trees(self):
        # Pure replay of install.py semantics over the independently replayed base.
        bases = {'vllm': prepare.base_vllm(self.lock['sources']['vllm']['base_commit']),
                 'b12x': prepare.previous.source('b12x', self.lock['sources']['b12x']['base_commit'])}
        for component, files in bases.items():
            row = self.lock['sources'][component]
            self.assertEqual(manifest(files), row['base_files'])
            result = dict(files)
            for name in row['base_files'].keys() - row['files'].keys():
                del result[name]
            for name, entry in row['files'].items():
                if row['base_files'].get(name) != entry:
                    data = self.members[component + '/' + name]
                    self.assertEqual(hashlib.sha256(data).hexdigest(), entry['sha256'])
                    result[name] = (entry['mode'], data)
            self.assertEqual(manifest(result), row['files'])
            self.assertEqual(git_tree(result), row['refreshed_tree'])

    def test_inventory_matches_changes(self):
        expected = {c + '/' + p for c in ('vllm', 'b12x')
                    for p, e in self.lock['sources'][c]['files'].items()
                    if self.lock['sources'][c]['base_files'].get(p) != e}
        self.assertEqual(set(self.members), expected)
        for name in self.members:
            self.assertIn(Path(name).suffix, prepare.SHIPPED_SUFFIXES, name)
            self.assertFalse(name.startswith('vllm/rust/'), name)

    def test_rust_parser_stays_at_base(self):
        row = self.lock['sources']['vllm']
        for path in prepare.RUST_DEFERRED:
            self.assertEqual(row['files'].get(path), row['base_files'].get(path), path)
        self.assertEqual(sorted(row['deferred_native']['paths']), sorted(prepare.RUST_DEFERRED))

    def test_key_beta_fixes_are_shipped(self):
        # vllm #943: the ring spec no longer disables worker slot mapping.
        ring = self.members['vllm/vllm/v1/kv_cache_interface.py'].decode()
        block = ring[ring.index('class CircularBufferSpec'):]
        block = block[:block.index('\nclass ', 1)]
        if 'def uses_slot_mapping' in block:
            body = block.split('def uses_slot_mapping', 1)[1].split('\n    @', 1)[0]
            self.assertNotIn('return False', body)
        # 4211209df5: the late side-stream Engram overlap switch is gone.
        self.assertNotIn('VLLM_DS41_ENGRAM_OVERLAP', self.members['vllm/vllm/envs.py'].decode())

    def test_foundation_and_publication_are_separate(self):
        base = json.loads((FOUNDATION / 'qsa865/runtime.lock.json').read_text())
        for key in prepare.INHERITED:
            self.assertNotIn(key, self.lock, key)
            self.assertEqual(self.lock['foundation'][key], base[key], key)
        self.assertEqual(self.lock['foundation']['image_id'], prepare.BASE)
        self.assertEqual(self.lock['foundation']['source_lock_sha256'], self.pin['base_lock_sha256'])
        pub = self.lock['publication']
        self.assertEqual((pub['tag'], pub['image'], pub['recipe_commit']),
                         (prepare.PUBLICATION, prepare.PUBLISHED_IMAGE, prepare.RECIPE))
        self.assertEqual((pub['vllm_commit'], pub['b12x_commit']), (prepare.VLLM, prepare.B12X))
        self.assertEqual((self.pin['publication'], self.pin['recipe_commit']), (pub['tag'], pub['recipe_commit']))

    def test_overlay_and_pins(self):
        cmake = self.lock['sources']['vllm']['files']['CMakeLists.txt']
        self.assertEqual(cmake, self.lock['sources']['vllm']['base_files']['CMakeLists.txt'])
        self.assertEqual(self.pin['vllm_commit'], prepare.VLLM)
        self.assertEqual(self.pin['b12x_commit'], prepare.B12X)
        self.assertEqual(self.pin['base_image_id'], prepare.BASE)


if __name__ == '__main__':
    unittest.main(verbosity=2)
