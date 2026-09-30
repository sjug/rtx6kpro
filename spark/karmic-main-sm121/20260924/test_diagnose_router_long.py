import unittest
from diagnose_router_long import stable_wrong


class StableWrongBranch(unittest.TestCase):
    def rows(self):
        return [dict(repeat=i, identical=True, cached_tokens=0, correct=False) for i in range(3)]

    def test_admitted(self):
        stable_wrong(self.rows())

    def test_incomplete(self):
        with self.assertRaises(RuntimeError):
            stable_wrong(self.rows()[:2])

    def test_confounds(self):
        for field, value in [('identical', False), ('cached_tokens', 1), ('correct', True), ('repeat', 8)]:
            with self.subTest(field=field):
                rows = self.rows()
                rows[1][field] = value
                with self.assertRaises(RuntimeError):
                    stable_wrong(rows)


if __name__ == '__main__':
    unittest.main()
