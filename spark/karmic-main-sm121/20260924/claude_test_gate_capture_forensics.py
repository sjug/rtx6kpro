"""CPU tests for claude_gate_capture_forensics.py (synthetic data, no network).
  .venv-snapshot-cpu/bin/python -m unittest claude_test_gate_capture_forensics
"""
import importlib.util
from pathlib import Path
import unittest

import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('claude_gate_forensics', HERE / 'claude_gate_capture_forensics.py')
forensics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(forensics)


def case(seed=0, rows=256, n=384, k=1024):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, k, generator=g).to(torch.bfloat16)
    w = (torch.randn(n, k, generator=g) * 0.05).to(torch.bfloat16)
    logits = (x.double() @ w.double().T).float()
    return x, w, logits


class Forensics(unittest.TestCase):
    def test_clean_logits_have_no_affected_tile(self):
        x, w, logits = case()
        report = forensics.analyze(x, logits, w)
        self.assertEqual((report['affected_tiles'], report['row_fits'], report['all_rows_exact']), ([], [], False))

    def test_stale_next_stage_activation_chunk_is_recovered_exactly(self):
        x, w, logits = case(1)
        xs = x.double().clone()
        for r in (192, 193, 194):                    # kt 0, k 48..63 read from kt 2 (same stage)
            xs[r, 48:64] = x.double()[r, 176:192]
        bad = xs @ w.double().T
        logits[192:195, 320:384] = bad[192:195, 320:384].float()
        report = forensics.analyze(x, logits, w)
        (tile,) = report['affected_tiles']
        self.assertEqual((tile['m_tile'], tile['n_tile'], tile['rows']), (3, 5, [192, 193, 194]))
        self.assertTrue(report['all_rows_exact'])
        best = report['row_fits'][0]['best'][0]
        self.assertEqual((best['operand'], best['chunk'], best['k_tile'], best['k_offset_in_tile'], best['source']),
                         ('A', 16, 0, 48, 'kt+2'))

    def test_stale_weight_chunk_is_attributed_to_b(self):
        x, w, logits = case(2)
        ws = w.double().clone()
        ws[64:128, 320:336] = w.double()[64:128, 192:208]    # kt 5 chunk from kt 3 (previous occupant)
        bad = x.double() @ ws.T
        logits[0:64, 64:128] = bad[0:64, 64:128].float()
        report = forensics.analyze(x, logits, w)
        self.assertTrue(report['all_rows_exact'])
        best = report['row_fits'][0]['best'][0]
        self.assertEqual((best['operand'], best['k_tile'], best['source']), ('B', 5, 'kt-2'))

    def test_random_corruption_is_not_called_exact(self):
        x, w, logits = case(3)
        logits[10, 5:20] += torch.randn(15, generator=torch.Generator().manual_seed(9)) * 0.1
        report = forensics.analyze(x, logits, w)
        self.assertTrue(report['affected_tiles'])
        self.assertFalse(report['all_rows_exact'])


if __name__ == '__main__':
    unittest.main()
