import copy
import json
import unittest

import prepare_decision_reference as reference


class ReferenceTests(unittest.TestCase):
    def test_retained_b2_receipts_agree(self):
        result = reference.build_reference()
        self.assertEqual(len(result['regime']), 8)
        self.assertEqual(len(result['source_receipts']), 16)
        self.assertEqual(result['image_id'], reference.B2_IMAGE)

    def test_full_signature_detects_logprob_change(self):
        response = {'choices': [{'message': {'content': '510'}, 'finish_reason': 'stop',
                                'logprobs': {'content': [{'token': '510', 'logprob': -0.1}]}}]}
        changed = copy.deepcopy(response)
        changed['choices'][0]['logprobs']['content'][0]['logprob'] = -0.2
        self.assertNotEqual(reference.signature(response), reference.signature(changed))

    def test_missing_regime_slot_rejected(self):
        payload = json.loads((reference.ROOT / 'receipts/router-b1-tuning-20260925/dusty.json').read_text())
        from claude_decode_sparse_mla import key
        del payload['records'][key('ratio1', 'extend', 80927)]
        with self.assertRaises(ValueError):
            reference.exact_regime(payload, 80927)


if __name__ == '__main__':
    unittest.main()
