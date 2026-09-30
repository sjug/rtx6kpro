from pathlib import Path
import unittest

from diagnose_router_remaining import commands


class RemainingCommands(unittest.TestCase):
    def test_original_context_envelope_preserved(self):
        rows = commands(Path('receipts/test'))
        self.assertEqual([r[0] for r in rows], ['context', 'conversations', 'final'])
        self.assertEqual(rows[0][2][-2:], ['--needle-lengths', '262000,500000,599936'])
        self.assertIn('--dual-needle', rows[0][2])


    def test_mixed_load_uses_historical_input(self):
        options = commands(Path('receipts/test'))[1][2]
        self.assertEqual(options[-2], '--mixed-needle-input')
        self.assertTrue(options[-1].endswith('admission-k7-1m-u80/needle-524288-input.json'))


    def test_no_benchmark_or_runtime_mutation_command(self):
        rows = commands(Path('receipts/test'))
        self.assertEqual({r[1] for r in rows}, {
            'qualify_upstream.py', 'claude_gate_conversations.py', 'probe_repeatability.py'})
