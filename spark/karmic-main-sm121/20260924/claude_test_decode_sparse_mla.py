"""CPU tests for claude_decode_sparse_mla.py on retained selection caches (read-only).
  python3 -m unittest claude_test_decode_sparse_mla
"""
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('claude_decode_mla_under_test', HERE / 'claude_decode_sparse_mla.py')
mla = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mla)
B1 = HERE / 'receipts/router-b1-tuning-20260925'
A1 = HERE / 'receipts/router-parent-a1-final-tuning-20260925'
NODES = ('dusty', 'toby', 'rusty', 'kirby')
BOOTS = [('engram-first', 80591, 'f569ae83.7f9dea26.a06d1e8a'), ('activation', 81418, '0596e570.0ce3fd8a.58578037'),
         ('moe-seams', 81811, '7dcc6601.e558f71a.43448b33'), ('capture', 81914, '65919df8.c6b1d76b.9b4777da'),
         ('A1', 81375, 'f98987c7.79a34e12.6f4cbb1c'), ('B1', 80927, '65919df8.c6b1d76b.9b4777da')]


def records(folder, node):
    return json.loads((folder / f'{node}.json').read_text())['records']


@unittest.skipUnless(all((B1 / f'{n}.json').exists() for n in NODES) and (A1 / 'dusty.json').exists(),
                     'retained B1/A1 selection snapshots absent')
class Retained(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b1 = {n: records(B1, n) for n in NODES}
        cls.decoded = {n: mla.decode(cls.b1[n]) for n in NODES}

    def test_every_sparse_mla_key_decodes(self):
        for node in NODES:
            found, undecoded = self.decoded[node]
            self.assertEqual(undecoded, [], node)
            self.assertEqual(len(found), sum(map(mla.is_sparse_mla, self.b1[node].values())))
            self.assertEqual({c for c, _, _ in found.values()},
                             {32, 80591, 80927, 81375, 81418, 81811, 81914}, node)
            per_capacity = {}
            for count, role, mode in found.values():
                per_capacity.setdefault(count, set()).add((role, mode))
            self.assertTrue(all(len(v) == 8 for v in per_capacity.values()))

    def test_b1_boot_measured_exactly_its_own_capacity(self):
        new = set(self.b1['dusty']) - set(records(A1, 'dusty'))
        found, _ = self.decoded['dusty']
        self.assertEqual({found[k][0] for k in new}, {80927})
        self.assertEqual(len(new), 8)

    def test_ranks_agree_on_every_regime(self):
        regimes = [mla.regimes(self.b1[n], self.decoded[n][0]) for n in NODES]
        self.assertTrue(all(r == regimes[0] for r in regimes))

    def test_regime_and_modal_response_are_one_to_one(self):
        regimes = mla.regimes(self.b1['dusty'], self.decoded['dusty'][0])
        relation = mla.relate(BOOTS, regimes)
        self.assertTrue(relation['consistent'], relation['violations'])
        self.assertEqual((relation['distinct_regimes'], relation['distinct_modals']), (5, 5))
        by = {r['boot']: r for r in relation['boots']}
        self.assertEqual(by['capture']['regime'], by['B1']['regime'])
        self.assertNotEqual(by['capture']['draft'], by['B1']['draft'])  # draft precision differs, modal does not
        a1, b1 = dict(((s, m) for s, m, _ in by['A1']['regime'])), dict(((s, m) for s, m, _ in by['B1']['regime']))
        self.assertEqual({s for s in a1 if a1[s] != b1[s]}, {'ratio2.extend', 'swa.decode'})

    def test_relation_detects_violations_and_missing(self):
        regimes = mla.regimes(self.b1['dusty'], self.decoded['dusty'][0])
        swapped = [('A1', 81375, 'x'), ('B1', 80927, 'x')]
        self.assertEqual(mla.relate(swapped, regimes)['violations'][0][2:], ('different regime', 'same modal'))
        same = [('B1', 80927, 'x'), ('capture', 81914, 'y')]
        self.assertEqual(mla.relate(same, regimes)['violations'][0][2:], ('same regime', 'different modal'))
        self.assertEqual(mla.relate([('none', 12345, 'z')], regimes)['missing'], ['none'])


class KeyShape(unittest.TestCase):
    def test_key_depends_on_blocks_stride_and_role(self):
        base = mla.key('ratio2', 'extend', 80927)
        self.assertNotEqual(base, mla.key('ratio2', 'extend', 80928))
        self.assertNotEqual(base, mla.key('ratio1', 'extend', 80927))
        self.assertNotEqual(base, mla.key('ratio2', 'decode', 80927))
        query, invocation = mla.query_and_invocation('ratio2', 'extend', 80927)
        self.assertEqual((query['indexed_page_size'], query['indexed_cache_shape'], query['swa_cache_stride']),
                         (128, [80927, 36864], [227840, 1]))
        self.assertIsNone(mla.query_and_invocation('draft', 'decode', 1)[1]['indexed_k_cache'])

    def test_main_regime_excludes_draft(self):
        cfg = lambda m: {'v41_compute_mode': m, 'v41_heads_per_block': 16}
        self.assertEqual(mla.main_regime({'swa.extend': cfg('bf16'), 'draft.extend': cfg('fp8')}),
                         mla.main_regime({'swa.extend': cfg('bf16'), 'draft.extend': cfg('bf16')}))


if __name__ == '__main__':
    unittest.main()
