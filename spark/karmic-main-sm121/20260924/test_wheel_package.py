import unittest

import prepare


class WheelPackageTests(unittest.TestCase):
    def test_installed_wheel_layout_is_not_git_layout(self):
        files = {
            'lmcache/core.py': ('100644', b'code'),
            'lmcache/lmcache_frontend/run_mp_server_with_frontend.sh': ('100755', b'script'),
            'lmcache/v1/distributed/bitmap_ops/README.md': ('120000', b'../../../../docs/bitmap.md'),
            'docs/bitmap.md': ('100644', b'document'),
        }
        actual = prepare.wheel_package(files)
        self.assertEqual(actual['core.py'], ('100644', b'code'))
        self.assertEqual(actual['lmcache_frontend/run_mp_server_with_frontend.sh'], ('100644', b'script'))
        self.assertEqual(actual['v1/distributed/bitmap_ops/README.md'], ('100644', b'document'))

    def test_unknown_link_fails_closed(self):
        with self.assertRaises(RuntimeError):
            prepare.wheel_package({'lmcache/unexpected': ('120000', b'/etc/passwd')})


if __name__ == '__main__':
    unittest.main()
