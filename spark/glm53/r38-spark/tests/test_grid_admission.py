"""An extra server request invalidates a matched-concurrency timing cell."""
import importlib.util
import json
from pathlib import Path
import unittest

path = Path(__file__).parents[1] / "qualification/compare-qwen-grids.py"
spec = importlib.util.spec_from_file_location("grid_compare", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Receipt:
    def __init__(self, extra_request=False):
        self.data = {"metadata": {"concurrency_levels": [1, 2, 4]}, "results": []}
        for c in (1, 2, 4):
            for m in (0, 16384, 32768, 65536, 131072):
                self.data["results"].append({
                    "concurrency": c, "context_tokens": m, "num_errors": 0,
                    "warmup_timed_out": False, "capacity_limited": False,
                    "underfilled": False, "aggregate_tps": 20.0,
                    "server_steps_per_s": 10.0, "server_accept_len_effective": 2.0,
                    "avg_running_reqs": c, "max_running_reqs": c,
                })
        if extra_request:
            self.data["results"][3]["avg_running_reqs"] = 2
            self.data["results"][3]["max_running_reqs"] = 2

    def read_text(self):
        return json.dumps(self.data)


class GridAdmission(unittest.TestCase):
    def test_matched_concurrency_is_valid(self):
        self.assertEqual(len(module.load(Receipt())[1]), 15)

    def test_extra_request_is_not_a_valid_c1_cell(self):
        with self.assertRaisesRegex(ValueError, "exceeds requested concurrency"):
            module.load(Receipt(extra_request=True))


if __name__ == "__main__":
    unittest.main()
