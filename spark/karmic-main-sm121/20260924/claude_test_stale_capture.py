"""Stdlib tests for claude_stale_capture.py; node and HTTP calls are mocked, nothing leaves this host."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('claude_stale_capture', HERE / 'claude_stale_capture.py')
cap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cap)


def scrape(drafts, dtok, gen):
    return ('vllm:spec_decode_num_drafts_total{engine="0",model_name="m"} %s\n'
            'vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="m"} %s\n'
            'vllm:generation_tokens_total{engine="0",model_name="m"} %s\n' % (drafts, dtok, gen))


class Pure(unittest.TestCase):
    def test_local_preflight_identity(self):
        helper, pin, lock_sha, original, prime, original_sha, prime_sha = cap.local_identity()
        self.assertEqual(original_sha, '68a1a5493c155cadc1c6762976a8668a0394d1b8815482dbf34caf35d12995bc')
        self.assertNotEqual(original_sha, prime_sha)
        control, runs = cap.build_control(original_sha, 'T1')
        helper.parse_control(control)
        self.assertEqual(control['actions'], [{'run': 'T1-orig-first', 'mode': 'observe'}])
        self.assertEqual(control['coverage']['ring']['layers'], [[2, 3], [8, 9], [14, 15]])
        self.assertEqual(runs, ('T1-orig-first', 'T1-orig-repeat'))
        self.assertEqual(prime['retrieval_code'], '826493')

    def test_spec_deltas_require_single_verify(self):
        deltas = cap.spec_deltas(scrape(10, 70, 30), scrape(11, 77, 33))
        cap.check_spec(deltas, 'ok')
        for after in (scrape(12, 84, 36), scrape(11, 76, 33), scrape(11, 77, 34)):
            with self.assertRaises(RuntimeError):
                cap.check_spec(cap.spec_deltas(scrape(10, 70, 30), after), 'bad')

    def test_markers(self):
        log = '\n'.join([
            '(Worker_TP0 pid=1) [DS41-STALE-PROBE] control loaded, 1 actions',
            '(Worker_TP0 pid=1) [DS41-STALE-PROBE] observe /cache/ds41-stale-state/T1-orig-first-dusty.pt layers=90 omitted=0 coverage={}',
            'unrelated line'])
        loaded, observed, failures = cap.markers(log)
        self.assertEqual((len(loaded), dict(observed), failures), (1, {'T1-orig-first': 1}, []))
        _, _, failures = cap.markers('[DS41-STALE-PROBE] RuntimeError: uncaptured KV-cache layers')
        self.assertEqual(len(failures), 1)
        for marker in ('observe-failed run=x reason=bad geometry', 'control-rejected reason=bad input'):
            _, observed, failures = cap.markers('[DS41-STALE-PROBE] ' + marker)
            self.assertFalse(observed)
            self.assertEqual(len(failures), 1)

    def test_request_body_matches_probe(self):
        entry = {'messages': [{'role': 'user', 'content': 'x'}]}
        body = cap.request_body(entry)
        self.assertEqual({k: body[k] for k in ('temperature', 'max_tokens', 'logprobs', 'top_logprobs',
                                               'chat_template_kwargs')},
                         {'temperature': 0, 'max_tokens': 8, 'logprobs': True, 'top_logprobs': 20,
                          'chat_template_kwargs': {'thinking': False}})
        self.assertNotEqual(body['cache_salt'], cap.request_body(entry)['cache_salt'])
        probe = (HERE / 'probe_repeatability.py').read_text()
        self.assertIn("'max_tokens': a.max_tokens, 'logprobs': True, 'top_logprobs': 20,", probe)
        self.assertIn("'chat_template_kwargs': {'thinking': False}, 'cache_salt': uuid.uuid4().hex}", probe)


class RunOrdering(unittest.TestCase):
    """The run never sends a request before every rank has the control, and disarms on failure."""

    def setUp(self):
        self.calls = []
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def fake_ssh(self, fail_deploy_on=None, existing=''):
        def ssh(node, command, *, stdin=None, timeout=60):
            self.calls.append(('ssh', node, command))
            if command.startswith('podman inspect'):
                raise AssertionError('identity is mocked separately')
            if command.startswith('ls -1'):
                return existing
            if command.startswith('date -u'):
                return '2026-09-25T16:00:00Z\n'
            if 'mv -T' in command and 'cat >' in command:
                if node == fail_deploy_on:
                    raise RuntimeError('deploy failed')
                return ''
            if command.startswith('sha256sum'):
                return self.control_sha + '  x\n'
            if command.startswith('if [ -e'):
                return ''
            if command.startswith('podman logs'):
                return ''
            raise AssertionError(command)
        return ssh

    def run_driver(self, **kwargs):
        out = Path(self.tmp.name) / 'cap'
        real_build = cap.build_control

        def build(prompt_sha, stamp):
            control, runs = real_build(prompt_sha, stamp)
            payload = (json.dumps(control, indent=1, sort_keys=True) + '\n').encode()
            self.control_sha = __import__('hashlib').sha256(payload).hexdigest()
            return control, runs

        requests = []
        with mock.patch.object(cap, 'ssh', self.fake_ssh(**kwargs)), \
                mock.patch.object(cap, 'container_identity', lambda node, pin, kit: {'Id': node}), \
                mock.patch.object(cap, 'build_control', build), \
                mock.patch.object(cap, 'one_request', lambda label, *a: requests.append(label) or
                                  {'label': label, 'signature': label}), \
                mock.patch.object(cap, 'collect', lambda out, runs: None), \
                mock.patch('cache_metrics.idle_snapshot', lambda url: {}, create=True):
            import sys
            sys.path.insert(0, str(HERE))
            import cache_metrics  # noqa: F401
            with mock.patch.object(cache_metrics, 'idle_snapshot', lambda url: {}):
                try:
                    code = cap.run(type('A', (), {'out': str(out)})())
                except Exception as error:  # noqa: BLE001
                    code = error
        return code, requests, out

    def test_existing_control_blocks_before_any_request_or_deploy(self):
        code, requests, _ = self.run_driver(existing='/home/x/.cache/vllm-jj-ds41-tp4/ds41-stale-control.json')
        self.assertIsInstance(code, RuntimeError)
        self.assertEqual(requests, [])
        self.assertFalse(any('cat >' in c for _, _, c in self.calls))

    def test_partial_deploy_disarms_and_sends_nothing(self):
        code, requests, out = self.run_driver(fail_deploy_on='rusty')
        self.assertIsInstance(code, RuntimeError)
        self.assertEqual(requests, [])
        disarmed = json.loads((out / 'disarm.json').read_text())
        self.assertEqual(sorted(disarmed), ['dusty', 'rusty', 'toby'])

    def test_unarmed_rank_stops_after_first_prime(self):
        code, requests, out = self.run_driver()        # logs never show "control loaded"
        self.assertIsInstance(code, RuntimeError)
        self.assertEqual(requests, ['prime-0'])
        self.assertEqual(json.loads((out / 'report.json').read_text())['status'], 'failed')
        self.assertTrue((out / 'disarm.json').is_file())


if __name__ == '__main__':
    unittest.main()
