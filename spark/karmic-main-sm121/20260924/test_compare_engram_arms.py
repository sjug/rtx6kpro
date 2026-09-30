import unittest
from compare_engram_arms import launches


def event(time, name, job="1", extra=""):
    return f"{time} node=dusty pid=2 tid=3 job={job} {name} {extra}"


class LaunchPairs(unittest.TestCase):
    def test_named_pair_ignores_other_events(self):
        rows = launches([event(100, "kernel-launch-begin", extra="prepared=385"),
                         event(200, "unrelated"), event(5000100, "kernel-launch-end")])
        self.assertEqual(rows[0]["host_ms"], 5)
        self.assertEqual(rows[0]["prepared"], 385)

    def test_jobs_not_mixed(self):
        rows = launches([event(100, "kernel-launch-begin", "1", "prepared=1"),
                         event(200, "kernel-launch-begin", "2", "prepared=4"),
                         event(300, "kernel-launch-end", "1"),
                         event(400, "kernel-launch-end", "2")])
        self.assertEqual([r["prepared"] for r in rows], [1, 4])

    def test_reject_incomplete(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            launches([event(100, "kernel-launch-begin", extra="prepared=1")])

    def test_reject_end_without_start(self):
        with self.assertRaisesRegex(ValueError, "without begin"):
            launches([event(100, "kernel-launch-end")])

    def test_reject_nested(self):
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            launches([event(100, "kernel-launch-begin", extra="prepared=1")] * 2)

    def test_reject_backwards(self):
        with self.assertRaisesRegex(ValueError, "Nonmonotonic"):
            launches([event(200, "kernel-launch-begin", extra="prepared=1"),
                      event(100, "kernel-launch-end")])


if __name__ == "__main__":
    unittest.main()
