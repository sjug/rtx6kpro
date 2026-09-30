import json
from pathlib import Path
import unittest

import precision_release as release


class EnvironmentTests(unittest.TestCase):
    def test_inherited_lazy_loading_is_normal_profile(self):
        env = {'B12X_DYNAMIC_DETERMINISTIC_OUTPUT': '1',
               'B12X_DENSE_SPLITK_TURBO': '0', 'CUDA_MODULE_LOADING': 'LAZY'}
        self.assertEqual(release.environment_problems(env), [])

    def test_changed_or_missing_loading_policy_rejected(self):
        for value in (None, 'EAGER'):
            env = {'B12X_DYNAMIC_DETERMINISTIC_OUTPUT': '1',
                   'B12X_DENSE_SPLITK_TURBO': '0'}
            if value is not None:
                env['CUDA_MODULE_LOADING'] = value
            self.assertTrue(release.environment_problems(env))

    def test_every_diagnostic_override_rejected(self):
        for key in release.DIAGNOSTIC_ENV:
            env = dict(release.REQUIRED_ENV)
            env[key] = '1'
            self.assertTrue(release.environment_problems(env), key)

    def test_parent_and_release_receipts_pin_lazy(self):
        root = Path(__file__).resolve().parent
        for kind in ('router-release', 'precision-release'):
            receipt = json.loads((root / f'receipts/{kind}-build-receipt.json').read_text())
            image = json.loads((Path(receipt['directory']) / 'image-inspect.json').read_text())[0]
            self.assertEqual(image['Id'].removeprefix('sha256:'), receipt['image_id'])
            env = dict(x.split('=', 1) for x in image['Config']['Env'])
            self.assertEqual(env['CUDA_MODULE_LOADING'], 'LAZY')


if __name__ == '__main__':
    unittest.main()
