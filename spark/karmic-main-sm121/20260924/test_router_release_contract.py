import hashlib
import json
from pathlib import Path
import unittest

from prepare_router_release_probe import transform

ROOT = Path(__file__).resolve().parent


class RouterReleaseContractTests(unittest.TestCase):
    def test_all_inputs_match_frozen_lock(self):
        lock = json.loads((ROOT / 'router-release.lock.json').read_text())
        for name, expected in lock['inputs'].items():
            self.assertEqual(Path(name).name, name)
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected, name)

    def test_only_router_source_changes(self):
        lock = json.loads((ROOT / 'router-release.lock.json').read_text())
        self.assertEqual(lock['before']['vllm'], lock['after']['vllm'])
        before, after = lock['before']['b12x'], lock['after']['b12x']
        self.assertEqual(set(before), set(after))
        self.assertEqual([p for p in before if before[p] != after[p]], [lock['target']])
        self.assertEqual(before[lock['target']]['sha256'], lock['input_sha256'])
        self.assertEqual(after[lock['target']]['sha256'], lock['output_sha256'])
        pinned = (ROOT / 'claude-pinned-bf16_gemv-_prefill.py').read_text()
        self.assertEqual(hashlib.sha256(pinned.encode()).hexdigest(), lock['input_sha256'])
        self.assertEqual(transform(pinned), (ROOT / 'router-prefill-release-before.py').read_text())

    def test_parent_manifest_and_image_are_anchored(self):
        lock = json.loads((ROOT / 'router-release.lock.json').read_text())
        parent = (ROOT / 'engram-repair.lock.json').read_bytes()
        self.assertEqual(hashlib.sha256(parent).hexdigest(), lock['base_lock_sha256'])
        self.assertEqual(json.loads(parent)['after'], lock['before'])
        self.assertEqual((ROOT / 'Dockerfile.router-release').read_text().splitlines()[0],
                         'FROM ' + lock['base_image_id'])


if __name__ == '__main__':
    unittest.main()
