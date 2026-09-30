import unittest
from unittest.mock import Mock, patch
from cache_metrics import delta, total, finish


class CacheAccounting(unittest.TestCase):
    def test_prompt_logprobs_zero_queries(self):
        before = {'queries': 100, 'hits': 0, 'successes': 2, 'preemptions': 0}
        after = dict(before, successes=3)
        with patch('cache_metrics.snapshot', return_value=after):
            self.assertEqual(finish('unused', before, 385, Mock(), expected_queries=0), 0)

    def test_prompt_logprobs_rejects_foreign_cache_queries(self):
        before = {'queries': 100, 'hits': 0, 'successes': 2, 'preemptions': 0}
        with patch('cache_metrics.snapshot', return_value=dict(before, successes=3, queries=485)):
            with self.assertRaises(RuntimeError):
                finish('unused', before, 385, Mock(), expected_queries=0)

    def test_labels(self):
        self.assertEqual(total('x{engine="0"} 4\nx{engine="1"} 3\n', 'x'), 7)

    def test_cold(self):
        self.assertEqual(delta({'queries': 10, 'hits': 5, 'successes': 2, 'preemptions': 0},
                               {'queries': 110, 'hits': 5, 'successes': 3, 'preemptions': 0}, 100), 0)

    def test_foreign_request_rejected(self):
        with self.assertRaises(RuntimeError):
            delta({'queries': 0, 'hits': 0, 'successes': 0},
                  {'queries': 100, 'hits': 0, 'successes': 2}, 100)

    def test_reset_rejected(self):
        with self.assertRaises(RuntimeError):
            delta({'queries': 100, 'hits': 10, 'successes': 2, 'preemptions': 0},
                  {'queries': 200, 'hits': 0, 'successes': 3, 'preemptions': 0}, 100)

    def test_preemption_rejected(self):
        with self.assertRaises(RuntimeError):
            delta({'queries': 0, 'hits': 0, 'successes': 0, 'preemptions': 0},
                  {'queries': 100, 'hits': 0, 'successes': 1, 'preemptions': 1}, 100)
