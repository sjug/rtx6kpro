import unittest
from summarize_telemetry import summarize, timestamp


def sample(second, counter=10):
    return (f'2026-09-24T04:00:{second:02}Z\n2400 MHz, 0x0000000000000000, 60, 50.0 W\n'
            f'MemAvailable: 4194304 kB\nSwapFree: 1024 kB\nallocstall_normal {counter}\n')


class TelemetryTests(unittest.TestCase):
    def test_filters_and_counts(self):
        result = summarize(sample(0) + sample(1, 11) + sample(2, 13) + sample(3, 20),
                           timestamp('2026-09-24T04:00:01Z'), timestamp('2026-09-24T04:00:02Z'))
        self.assertEqual(result['sample_count'], 2)
        self.assertEqual(result['min_available_gib'], 4)
        self.assertEqual(result['vm_counter_deltas'], {'allocstall_normal': 2})

    def test_rejects_missing_and_reset_samples(self):
        start, end = timestamp('2026-09-24T04:00:00Z'), timestamp('2026-09-24T04:00:02Z')
        for text in ('', sample(0).replace('2400 MHz', 'N/A'), sample(0, 10) + sample(1, 0)):
            with self.assertRaises(ValueError):
                summarize(text, start, end)

    def test_requires_timezone(self):
        with self.assertRaises(ValueError):
            timestamp('2026-09-24T04:00:00')
