import hashlib
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parent
LOCK = json.loads((ROOT / 'build.lock.json').read_text())


class BuildLockTests(unittest.TestCase):
    def test_component_and_runtime_pins(self):
        runtime = (ROOT / 'Dockerfile.runtime').read_text()
        for component, image in LOCK['components'].items():
            suffix = '' if component == 'runtime-deps' else f' AS {component}'
            self.assertIn(f'FROM {image}{suffix}\n', runtime)
        for filename in ('Dockerfile.serving', 'build-serving.sh'):
            self.assertIn(LOCK['runtime_image_id'], (ROOT / filename).read_text())

    def test_serving_id_consumers(self):
        for filename in ('restart-qwen.sh', 'distribute-clusters.sh', 'benchmark-qwen.sh',
                         'distribute-qwen.sh', 'ds4-vision/run-node.sh', 'prepare_runtime.py'):
            text = (ROOT / filename).read_text()
            self.assertIn(LOCK['serving_image_id'], text, filename)
            pattern = r"(?:image|expected|new_image)=['\"]?([a-f0-9]{64})"
            for image in re.findall(pattern, text):
                self.assertEqual(image, LOCK['serving_image_id'], filename)

    def test_overlay_identity(self):
        self.assertEqual(hashlib.sha256((ROOT / 'spark-overlay.patch').read_bytes()).hexdigest(),
                         LOCK['overlay_sha256'])
        self.assertIn(LOCK['vllm_overlay_tree'], (ROOT / 'Dockerfile.runtime').read_text())
        self.assertIn(LOCK['b12x_tree'], (ROOT / 'Dockerfile.runtime').read_text())

    def test_wheel_manifest_shape(self):
        self.assertEqual(len(LOCK['wheel_sha256']), 5)
        for filename, digest in LOCK['wheel_sha256'].items():
            self.assertTrue(filename.endswith('.whl'))
            self.assertRegex(digest, r'^[a-f0-9]{64}$')

    def test_pass_count_pattern_is_exact(self):
        for count in (1, 6, 7, 235):
            pattern = rf'^Karmic-REGRESSION-PASS .+: {count} passed$'
            self.assertRegex(f'Karmic-REGRESSION-PASS tests/test.py: {count} passed', pattern)
            self.assertNotRegex(f'Karmic-REGRESSION-PASS tests/test.py: 1{count} passed', pattern)


if __name__ == '__main__':
    unittest.main()
