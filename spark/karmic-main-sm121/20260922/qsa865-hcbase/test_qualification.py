"""Local fail-closed checks for qualification clients, no serving requests."""
import importlib.util
from pathlib import Path
import unittest


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class QualificationTests(unittest.TestCase):
    def test_counting_accepts_only_complete_correct_prefix(self):
        check = load('probe-counting').check
        choice = {'finish_reason': 'length'}
        usage = {'completion_tokens': 400}
        self.assertTrue(check('2214, 2215', '2214, 2215, 2216', choice, usage, 400))
        for text in ('', '2214, 225', 'garbage'):
            self.assertFalse(check(text, '2214, 2215, 2216', choice, usage, 400))
        self.assertFalse(check('2214', '2214, 2215', {'finish_reason': 'stop'}, usage, 400))
        self.assertFalse(check('2214', '2214, 2215', choice, {'completion_tokens': 1}, 400))

    def test_replay_excludes_proxy_secrets_and_preserves_sampling(self):
        reconstruct = load('replay-private').reconstruct
        trace = {'virtual_key': 'not-a-real-key', 'selected_key': 'not-a-real-key',
                 'input_history': [{'role': 'user', 'content': 'test'}],
                 'params': {'temperature': 1, 'top_k': 20, 'max_completion_tokens': 32000,
                            'stream_options': {'include_usage': True}}, 'tools': []}
        result = reconstruct(trace)
        self.assertEqual(result['messages'], trace['input_history'])
        self.assertEqual(result['temperature'], 1)
        self.assertEqual(result['max_completion_tokens'], 32000)
        self.assertFalse(result['stream'])
        self.assertNotIn('virtual_key', result)
        self.assertNotIn('selected_key', result)
        self.assertNotIn('stream_options', result)


if __name__ == '__main__':
    unittest.main()
