import unittest
from verify_dense_release_sass import verify


def asm(*instructions):
    return '\n'.join(f'/*{i * 16:04x}*/ {op} ;' for i, op in enumerate(instructions))


class ReleaseGateTests(unittest.TestCase):
    def test_before(self):
        self.assertEqual(verify(asm('LDSM.16.M88.4 R0, [R1]', 'MEMBAR.ALL.CTA',
                                   '@!P0 SYNCS.ARRIVE.TRANS64.A1T0 RZ, [R2], RZ'), 1), 1)

    def test_after_rejected(self):
        with self.assertRaises(RuntimeError):
            verify(asm('LDSM.16.M88.4 R0, [R1]', '@P0 SYNCS.ARRIVE.TRANS64.A1T0 RZ, [R2], RZ',
                       'MEMBAR.ALL.CTA'), 1)

    def test_load_after_fence_rejected(self):
        with self.assertRaises(RuntimeError):
            verify(asm('MEMBAR.ALL.CTA', 'LDS.U8 R0, [R1]',
                       '@P0 SYNCS.ARRIVE.TRANS64.A1T0 RZ, [R2], RZ'), 1)

    def test_predicated_fence_rejected(self):
        with self.assertRaises(RuntimeError):
            verify(asm('@P1 MEMBAR.ALL.CTA', '@P0 SYNCS.ARRIVE.TRANS64.A1T0 RZ, [R2], RZ'), 1)

    def test_empty_rejected(self):
        with self.assertRaises(RuntimeError):
            verify('')


if __name__ == '__main__':
    unittest.main()
