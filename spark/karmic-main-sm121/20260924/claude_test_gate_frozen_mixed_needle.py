"""Local regression tests for claude_gate_frozen_mixed_needle.py (no network beyond a loopback fake).

    .venv-snapshot-cpu/bin/python -m unittest claude_test_gate_frozen_mixed_needle
"""
import ast
import contextlib
import hashlib
import http.server
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
import urllib.error

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import claude_gate_frozen_mixed_needle as gate  # noqa: E402

EXPECTED = '739184, 482617'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def synthetic_input(path):
    content = (gate.ARCHIVE + '. Memorize this unique retrieval code: 739184.\nArchive:\n' + ' filler' * 400
               + '\nThe late retrieval code is 482617.\n' + ' filler' * 200 + '\nArchive ends.')
    raw = json.dumps({'messages': [{'role': 'user', 'content': content}], 'prompt_tokens': 524288,
                      'chat_template_kwargs': {'thinking': False}, 'expected': EXPECTED}).encode()
    path.write_bytes(raw)
    return raw, content


def logprobs(first, second, gap):
    return {'content': [{'token': first, 'logprob': -0.25, 'top_logprobs': [
        {'token': first, 'logprob': -0.25}, {'token': second, 'logprob': -0.25 - gap}]}]}


class Engine:
    """Loopback endpoint: counters, answers and per-trial overlap, with fault injection."""

    def __init__(self, *, wrong_trials=(), needle_hits=0, needle_fast=False, foreign=False, preempt=False,
                 needle_error=False, vary_logprobs=False, prompt_tokens=524288, no_logprobs=False):
        self.wrong_trials, self.needle_hits, self.needle_fast = set(wrong_trials), needle_hits, needle_fast
        self.foreign, self.preempt, self.needle_error = foreign, preempt, needle_error
        self.vary_logprobs, self.prompt_tokens, self.no_logprobs = vary_logprobs, prompt_tokens, no_logprobs
        self.lock = threading.Lock()
        self.queries = self.hits = self.successes = self.preemptions = self.running = self.waiting = 0
        self.bodies, self.needles, self.turns = [], 0, 0

    def metrics(self):
        with self.lock:
            values = (('vllm:prefix_cache_queries_total', self.queries), ('vllm:prefix_cache_hits_total', self.hits),
                      ('vllm:request_success_total', self.successes), ('vllm:num_preemptions_total', self.preemptions),
                      ('vllm:num_requests_running', self.running), ('vllm:num_requests_waiting', self.waiting))
        return ''.join(f'{name}{{engine="0"}} {float(value)}\n' for name, value in values)

    def chat(self, body):
        with self.lock:
            self.bodies.append(body)
        content = body['messages'][0]['content']
        if gate.ARCHIVE in content:
            return self.needle(body)
        tokens = len(json.dumps(body['messages']))
        with self.lock:
            self.queries += tokens
            self.running += 1
        time.sleep(0.2)
        code = int(re.search(r'Remember code (\d+)', content).group(1))
        answer = str(code + 1) if 'Add one' in body['messages'][-1]['content'] else str(code)
        with self.lock:
            self.running -= 1
            self.successes += 1
            if self.turns == 0 and self.foreign:
                self.successes += 1
                self.queries += 5
            if self.turns == 0 and self.preempt:
                self.preemptions += 1
            self.turns += 1
        return 200, {'choices': [{'message': {'role': 'assistant', 'content': answer}, 'finish_reason': 'stop'}],
                     'usage': {'prompt_tokens': tokens, 'completion_tokens': 2}}

    def needle(self, body):
        if self.needle_error:
            return 500, {'error': 'injected'}
        with self.lock:
            trial, target = self.needles, self.turns + 12
            self.needles += 1
            self.queries += self.prompt_tokens
            self.hits += self.needle_hits
            self.running += 1
        if not self.needle_fast:
            deadline = time.monotonic() + 20
            while self.turns < target and time.monotonic() < deadline:
                time.sleep(0.05)
            time.sleep(0.3)
        answer = '510c94b1bb4f42d2998093bd6b9c3d99, 739184' if trial in self.wrong_trials else EXPECTED
        gap = 1.25 + (0.125 * trial if self.vary_logprobs else 0)
        choice = {'message': {'role': 'assistant', 'content': answer}, 'finish_reason': 'stop',
                  'logprobs': None if self.no_logprobs else logprobs(answer[:3], '510' if answer[:3] != '510' else '739', gap)}
        with self.lock:
            self.running -= 1
            self.successes += 1
        return 200, {'choices': [choice], 'usage': {'prompt_tokens': self.prompt_tokens, 'completion_tokens': 7}}


def serve(engine):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            body = engine.metrics().encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            status, result = engine.chat(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            body = json.dumps(result).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class FrozenPayload(unittest.TestCase):
    """Against the real frozen input file."""

    def test_real_input_accepted_and_body_is_original_request_plus_salt(self):
        spec = gate.load_input(gate.INPUT)
        salt = 'a' * 32
        body = gate.needle_body(spec, salt)
        gate.check_body(body, spec)
        self.assertIs(body['messages'], spec['messages'])
        self.assertEqual(sha(body['messages'][0]['content'].encode()), gate.CONTENT_SHA256)
        self.assertEqual(gate.conv.ARCHIVE_ID.findall(body['messages'][0]['content']), [gate.ARCHIVE])
        self.assertEqual(body['cache_salt'], salt)
        other = gate.needle_body(spec, 'b' * 32)
        self.assertEqual({k: v for k, v in body.items() if k != 'cache_salt'},
                         {k: v for k, v in other.items() if k != 'cache_salt'})

    def test_fields_match_the_original_gate_request(self):
        tree = ast.parse((HERE / 'qualify_original_needle.py').read_text())
        literal = next(node for node in ast.walk(tree) if isinstance(node, ast.Dict)
                       and any(isinstance(k, ast.Constant) and k.value == 'cache_salt' for k in node.keys))
        original = {k.value: v for k, v in zip(literal.keys, literal.values)}
        body = gate.needle_body(gate.load_input(gate.INPUT), 'c' * 32)
        self.assertEqual(set(body), set(original))
        for key in ('temperature', 'max_tokens', 'logprobs', 'top_logprobs'):
            self.assertEqual(body[key], ast.literal_eval(original[key]), key)
        self.assertEqual(body['model'], ast.literal_eval(original['model']))

    def test_changed_inputs_refused(self):
        spec = gate.load_input(gate.INPUT)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.json'
            path.write_bytes(gate.INPUT.read_bytes() + b' ')
            with self.assertRaisesRegex(RuntimeError, 'file identity'):
                gate.load_input(path)
        altered = json.loads(json.dumps(spec))
        altered['messages'][0]['content'] = altered['messages'][0]['content'].replace('510c94b1', '00000000', 1)
        with self.assertRaisesRegex(RuntimeError, 'differ'):
            gate.check_body(gate.needle_body(altered, 'd' * 32), spec)
        for salt in ('', 'A' * 32, 'e' * 31):
            with self.assertRaises(ValueError):
                gate.needle_body(spec, salt)


class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.input = self.root / 'input.json'
        raw, self.content = synthetic_input(self.input)
        content_sha = sha(self.content.encode())
        self.patches = [mock.patch.object(gate, 'INPUT_SHA256', sha(raw)),
                        mock.patch.object(gate, 'CONTENT_SHA256', content_sha)]
        for patch in self.patches:
            patch.start()

    def tearDown(self):
        for patch in self.patches:
            patch.stop()
        self.tmp.cleanup()

    def run_gate(self, engine, trials=3, out='run'):
        server = serve(engine)
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                gate.main(['--base-url', f'http://127.0.0.1:{server.server_port}', '--input', str(self.input),
                           '--out', str(self.root / out), '--trials', str(trials)])
        finally:
            server.shutdown()
            server.server_close()
        return self.root / out, stdout.getvalue()

    def summary(self, out='run'):
        return json.loads((self.root / out / 'summary.json').read_text())

    def needle_bodies(self, engine):
        return [b for b in engine.bodies if gate.ARCHIVE in b['messages'][0]['content']]

    def test_pass_records_unchanged_payload_unique_salts_margins_and_evidence(self):
        engine = Engine()
        out, text = self.run_gate(engine)
        self.assertIn('FROZEN-MIXED-NEEDLE-PASS', text.splitlines())
        bodies = self.needle_bodies(engine)
        self.assertEqual(len(bodies), 3)
        for body in bodies:
            self.assertEqual(body['messages'], [{'role': 'user', 'content': self.content}])
            self.assertEqual((body['logprobs'], body['top_logprobs'], body['temperature']), (True, 20, 0))
        self.assertEqual(len({b['cache_salt'] for b in bodies}), 3)
        summary = self.summary()
        self.assertEqual([t['correct'] for t in summary['correctness']['trials']], [True] * 3)
        for index in range(3):
            row = json.loads((out / f'trial-{index}' / 'trial.json').read_text())
            self.assertEqual(row['first_token']['margin_nats'], 1.25)
            self.assertEqual(row['accounting']['aggregate_hits'], 0)
            self.assertIn('proven', row['accounting']['all_requests_cold'])
            self.assertEqual(row['accounting']['requests'], 13)
            self.assertTrue(json.loads((out / f'trial-{index}' / 'evidence.json').read_text())['overlap_observed'])
            self.assertEqual(row['scheduling']['turns_completed_before_long_end'], 12)
            self.assertTrue((out / f'trial-{index}' / 'frozen-needle.json').is_file())
        determinism = summary['determinism']
        self.assertTrue(determinism['all_trials_same_signature'])
        self.assertEqual(set(determinism['scheduling']), {'0', '1', '2'})
        self.assertIn('not gated', determinism['scope'])

    def test_signature_differences_are_reported_not_gated(self):
        _, text = self.run_gate(Engine(vary_logprobs=True))
        self.assertIn('FROZEN-MIXED-NEEDLE-PASS', text.splitlines())
        determinism = self.summary()['determinism']
        self.assertFalse(determinism['all_trials_same_signature'])
        self.assertEqual(len(determinism['signature_groups']), 3)
        self.assertEqual(determinism['first_token_margins_nats'], [1.25, 1.375, 1.5])

    def test_wrong_answer_fails_after_all_trials_are_recorded(self):
        with self.assertRaisesRegex(SystemExit, r'FROZEN-MIXED-NEEDLE-FAIL: wrong answer in trial\(s\) \[1\]'):
            self.run_gate(Engine(wrong_trials={1}))
        trials = self.summary()['correctness']['trials']
        self.assertEqual([t['correct'] for t in trials], [True, False, True])
        self.assertTrue(trials[1]['answer'].startswith('510c94b1'))

    def test_integrity_failures_stop(self):
        cases = [('coldness cannot be proven', dict(needle_hits=4096)),
                 ('overlap', dict(needle_fast=True)),
                 ('does not match its own requests', dict(foreign=True)),
                 ('Preemption', dict(preempt=True)),
                 ('token count changed', dict(prompt_tokens=524287)),
                 ('Missing needle logprobs', dict(no_logprobs=True))]
        for index, (message, options) in enumerate(cases):
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                self.run_gate(Engine(**options), out=f'run-{index}')
            self.assertEqual(self.summary(f'run-{index}')['correctness']['trials_completed'], 0)

    def test_http_error_recorded_and_raised(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.run_gate(Engine(needle_error=True))
        caught.exception.close()
        self.assertTrue((self.root / 'run/trial-0/frozen-needle-http-error.json').is_file())

    def test_refusals_before_any_request(self):
        engine = Engine()
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self.run_gate(engine, trials=1)
        (self.root / 'exists').mkdir()
        with self.assertRaises(FileExistsError):
            self.run_gate(engine, out='exists')
        self.assertEqual(engine.bodies, [])


if __name__ == '__main__':
    unittest.main()
