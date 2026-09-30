import unittest

from qualify_original_needle import first_token_margin


class MarginTests(unittest.TestCase):
    def choice(self, values):
        return {'logprobs': {'content': [{'token': '739', 'logprob': -0.625,
                                         'top_logprobs': values}]}}

    def test_best_distinct_competitor(self):
        result = first_token_margin(self.choice([
            {'token': '739', 'logprob': -0.625},
            {'token': 'other', 'logprob': -2.0},
            {'token': '510', 'logprob': -0.75}]))
        self.assertEqual(result['competing_token'], '510')
        self.assertEqual(result['margin_nats'], 0.125)

    def test_missing_competitor_fails(self):
        with self.assertRaises(RuntimeError):
            first_token_margin(self.choice([{'token': '739', 'logprob': -0.625}]))

    def test_nonfinite_fails(self):
        with self.assertRaises(RuntimeError):
            first_token_margin(self.choice([{'token': '510', 'logprob': float('inf')}]))

    def test_tie_is_recorded_not_redefined_as_failure(self):
        result = first_token_margin(self.choice([{'token': '510', 'logprob': -0.625}]))
        self.assertEqual(result['margin_nats'], 0.0)


if __name__ == '__main__':
    unittest.main()
