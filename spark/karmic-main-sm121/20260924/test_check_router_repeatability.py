import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from check_router_repeatability import validate, validate_raw


class ReceiptTests(unittest.TestCase):
    def fixture(self):
        records = [dict(cycle=c, length=n, signature='a', unexplained_ttft_ge_5s=False)
                   for c in range(2) for n in (254, 256, 258)]
        summary = [dict(length=n, trials=2, unique_signatures=1, nonmodal_trials=0,
                        signature_counts={'a': 2}) for n in (254, 256, 258)]
        return summary, records

    def test_clean(self):
        self.assertEqual(validate(*self.fixture(), cycles=2)['nonmodal_total'], 0)

    def test_event_count(self):
        summary, records = self.fixture()
        records[0]['signature'] = 'b'
        summary[0].update(unique_signatures=2, nonmodal_trials=1, signature_counts={'a': 1, 'b': 1})
        self.assertEqual(validate(summary, records, 2)['nonmodal_total'], 1)

    def test_reject_false_clean_summary(self):
        summary, records = self.fixture()
        records[0]['signature'] = 'b'
        with self.assertRaises(RuntimeError):
            validate(summary, records, 2)

    def test_reject_duplicate(self):
        summary, records = self.fixture()
        records[-1] = copy.deepcopy(records[0])
        with self.assertRaises(RuntimeError):
            validate(summary, records, 2)

    def test_reject_incomplete(self):
        summary, records = self.fixture()
        with self.assertRaises(RuntimeError):
            validate(summary, records[:-1], 2)

    def test_raw_signature_verification(self):
        choice = {'message': {'content': '739184'}, 'finish_reason': 'stop', 'logprobs': {}}
        signature = hashlib.sha256(json.dumps(choice, sort_keys=True).encode()).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp)
            (probe / 'cycle-000').mkdir()
            (probe / 'cycle-000/254-0.json').write_text(json.dumps({'response': {'choices': [choice]}}))
            records = [{'cycle': 0, 'length': 254, 'signature': signature}]
            validate_raw(probe, records)
            records[0]['signature'] = 'wrong'
            with self.assertRaises(RuntimeError):
                validate_raw(probe, records)


if __name__ == '__main__':
    unittest.main()
