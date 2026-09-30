import unittest

from analyze_activation_responses import correlate


class Correlation(unittest.TestCase):
    def data(self):
        responses = [{'length': 255, 'signature': 'normal' if epoch < 3 else 'different',
                      'started_unix': epoch * 10, 'ended_unix': epoch * 10 + 5}
                     for epoch in range(4)]
        traces = [{'epoch': epoch, 'rank': rank, 'rows': 255, 'wall': epoch * 10 + 1,
                   'names': ['<input>', '<positions>', 'layers.0:0', '<output>'],
                   'hashes': [[1, 2], [3, 4], [5, 6], [7, 8]]}
                  for epoch in range(4) for rank in range(4)]
        return responses, traces, {'divergent_steps': []}

    def test_modal_body_points_beyond_body(self):
        findings = correlate(*self.data())
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]['verdict'], 'model-body-digests-modal-investigate-post-model')

    def test_localized_body(self):
        responses, traces, analysis = self.data()
        analysis['divergent_steps'] = [{'epoch': 3, 'earliest_name': 'layers.0:0'}]
        self.assertEqual(correlate(responses, traces, analysis)[0]['verdict'], 'activation-divergence-observed')

    def test_missing_rank_or_ambiguous_time_is_inconclusive(self):
        responses, traces, analysis = self.data()
        for incomplete in (traces[:-1], traces + [traces[-4]]):
            self.assertEqual(correlate(responses, incomplete, analysis)[0]['verdict'],
                             'inconclusive-untraced-or-unmatched')


if __name__ == '__main__':
    unittest.main()
