import unittest
from probe_engram_fail_closed import validate


class LiveIdentity(unittest.TestCase):
    def info(self):
        return {'Image': 'sha256:fixed-image', 'State': {'Running': True},
                'Config': {'Labels': {'local-inference.ds41.kit.sha256': 'kit'}}}

    def test_only_exact_running_image_and_kit(self):
        validate(self.info(), 'fixed-image', 'kit', 'kirby')
        for image, kit in (('wrong', 'kit'), ('fixed-image', 'wrong')):
            with self.assertRaises(RuntimeError):
                validate(self.info(), image, kit, 'kirby')
        info = self.info()
        info['State']['Running'] = False
        with self.assertRaises(RuntimeError):
            validate(info, 'fixed-image', 'kit', 'kirby')


if __name__ == '__main__':
    unittest.main()
