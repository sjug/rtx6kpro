"""Stdlib-only tests for claude_gate_conversations.py (no node, no live service).

End-to-end tests run the gate's real main() against an in-process loopback
fake of /v1/chat/completions and /metrics (127.0.0.1, ephemeral port). The fake
models the counters the gate reads (prefix-cache queries and hits, request
successes, preemptions, running and waiting gauges, with engine labels so the
summation is exercised) and can inject each failure the gate must catch.
The real cache_metrics.py of this task tree is used unchanged.
"""
import contextlib
import http.server
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("claude_gate_conversations", HERE / "claude_gate_conversations.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

NEEDLE_EXPECTED = "739184, 482617"


class FakeEngine:
    """Counters and answers of a single-model endpoint, with fault injection."""

    def __init__(self, *, align=1024, cache=True, wrong_add_one=False, needle_hits=0, foreign_success=False,
                 preempt=False, needle_fast=False, fail_label_text=None):
        self.align, self.cache = align, cache
        self.wrong_add_one, self.needle_hits = wrong_add_one, needle_hits
        self.foreign_success, self.preempt, self.needle_fast = foreign_success, preempt, needle_fast
        self.fail_label_text = fail_label_text
        self.lock = threading.Lock()
        self.queries = self.hits = self.successes = self.preemptions = 0
        self.running = self.waiting = 0
        self.cached = []
        self.turns_done = 0
        self.all_turns_done = threading.Event()

    @staticmethod
    def render(messages):
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"

    def answer(self, messages):
        first = messages[0]["content"]
        last = messages[-1]["content"]
        if "Archive identity" in first:
            return NEEDLE_EXPECTED
        if "access code 739184" in first:
            return "739184"
        code = int(re.search(r"Remember code (\d+)", first).group(1))
        if "Add one" in last:
            return str(code) if self.wrong_add_one and code == 630142 else str(code + 1)
        return str(code)

    def hit(self, prompt):
        if not self.cache:
            return 0
        best = max((len(os.path.commonprefix([prompt, c])) for c in self.cached), default=0)
        best = min(best, len(prompt) - 1)
        return best // self.align * self.align

    def metrics(self):
        with self.lock:
            q, h, s, pr, r, w = self.queries, self.hits, self.successes, self.preemptions, self.running, self.waiting
        # Split every value over two engine label sets to exercise label summation.
        lines = []
        for name, value in (("vllm:prefix_cache_queries_total", q), ("vllm:prefix_cache_hits_total", h),
                            ("vllm:request_success_total", s), ("vllm:num_preemptions_total", pr),
                            ("vllm:num_requests_running", r), ("vllm:num_requests_waiting", w)):
            half = value // 2
            lines.append(f'{name}{{engine="0",model_name="DeepSeek-V4.1-Flash"}} {float(half)}')
            lines.append(f'{name}{{engine="1",model_name="DeepSeek-V4.1-Flash"}} {float(value - half)}')
        return "\n".join(lines) + "\n"

    def chat(self, body):
        messages = body["messages"]
        prompt = self.render(messages)
        n = len(prompt)
        needle = "Archive identity" in messages[0]["content"]
        if self.fail_label_text and self.fail_label_text in messages[-1]["content"]:
            return 500, {"error": "injected"}
        with self.lock:
            h = self.needle_hits if needle else self.hit(prompt)
            self.queries += n
            self.hits += h
            self.running += 1
        if needle:
            if not self.needle_fast:
                self.all_turns_done.wait(20)
                time.sleep(0.3)
        else:
            time.sleep(0.3 if "Remember code" in messages[0]["content"] else 0.01)
        content = self.answer(messages)
        with self.lock:
            self.running -= 1
            self.successes += 1
            self.cached.append(prompt + content)
            if "Remember code" in messages[0]["content"]:
                if self.foreign_success and self.turns_done == 0:
                    self.successes += 1
                    self.queries += 10
                if self.preempt and self.turns_done == 0:
                    self.preemptions += 1
                self.turns_done += 1
                if self.turns_done == 12:
                    self.all_turns_done.set()
        return 200, {"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": n, "completion_tokens": 3, "total_tokens": n + 3}}


def serve(engine):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            body = engine.metrics().encode()
            self.send_response(200 if self.path == "/metrics" else 404)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            status, result = engine.chat(payload)
            body = json.dumps(result).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.needle = self.root / "needle.json"
        content = ("Archive identity " + "a" * 32 + ". Memorize this unique retrieval code: 739184.\nArchive:\n"
                   + " filler" * 700 + "\nThe late retrieval code is 482617.\n" + " filler" * 300
                   + "\nArchive ends. Reply with only the initial retrieval code and the late retrieval code, "
                     "in that order, separated by a comma and a space.")
        self.needle.write_text(json.dumps({"messages": [{"role": "user", "content": content}],
                                           "prompt_tokens": 7000, "expected": NEEDLE_EXPECTED}))

    def tearDown(self):
        self.tmp.cleanup()

    def run_gate(self, engine, *, needle=True):
        server = serve(engine)
        out = self.root / "run"
        argv = ["claude_gate_conversations.py", "--base-url", f"http://127.0.0.1:{server.server_port}",
                "--out", str(out)] + (["--mixed-needle-input", str(self.needle)] if needle else [])
        stdout = io.StringIO()
        try:
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(stdout):
                gate.main()
        finally:
            server.shutdown()
            server.server_close()
        return out, stdout.getvalue()

    def test_pass_with_mixed_needle(self):
        out, text = self.run_gate(FakeEngine())
        self.assertIn("CLAUDE-CONVERSATIONS-PASS", text.splitlines())
        summary = json.loads((out / "summary.json").read_text())
        serial = {row["label"]: row for row in summary["serial"]}
        self.assertEqual(serial["prefix-cold"]["hits"], 0)
        for label in ("prefix-repeat", "prefix-extension", "prefix-divergent"):
            self.assertGreaterEqual(serial[label]["ratio"], 0.95)
        aggregate = summary["concurrent"]["aggregate"]
        self.assertEqual((aggregate["requests"], aggregate["aggregate_hits"], aggregate["preemptions"]), (13, 0, 0))
        self.assertEqual(aggregate["needle_prefilled_tokens_at_least"], aggregate["needle_prompt_tokens"])
        self.assertIn("proven", aggregate["all_requests_cold"])
        self.assertIn("cannot attribute", aggregate["attribution"])
        evidence = json.loads((out / "mixed-evidence.json").read_text())
        self.assertTrue(evidence["overlap_observed"])
        self.assertGreater(evidence["samples_running_ge_2_while_long_active"], 0)
        self.assertEqual(len(evidence["turns_completed_before_long_end"]), 12)
        for label in ("prefix-cold", "concurrent-3-turn3", "mixed-long-needle"):
            self.assertTrue((out / f"{label}.json").is_file())
        for label in ("prefix-cold", "prefix-repeat", "prefix-extension", "prefix-divergent"):
            self.assertTrue((out / f"{label}-cache.json").is_file())
        sent = json.loads((out / "mixed-long-needle.json").read_text())["request"]["messages"][0]["content"]
        self.assertNotIn("a" * 32, sent)
        self.assertEqual(len(gate.ARCHIVE_ID.findall(sent)), 1)

    def test_pass_without_needle_reports_conversation_hits_only(self):
        out, text = self.run_gate(FakeEngine(align=16), needle=False)
        self.assertIn("CLAUDE-CONVERSATIONS-PASS", text.splitlines())
        aggregate = json.loads((out / "summary.json").read_text())["concurrent"]["aggregate"]
        self.assertGreater(aggregate["aggregate_hits"], 0)          # later turns reuse earlier turns
        self.assertNotIn("all_requests_cold", aggregate)
        self.assertFalse((out / "mixed-evidence.json").exists())

    def test_wrong_answer_fails(self):
        with self.assertRaisesRegex(RuntimeError, "Wrong or incomplete answer"):
            self.run_gate(FakeEngine(wrong_add_one=True))

    def test_cold_prefix_with_hits_fails(self):
        engine = FakeEngine()
        original = engine.hit                                            # first archive request reports a stale hit
        engine.hit = lambda prompt: 1024 if "Remember access code" in prompt and engine.queries == 0 else original(prompt)
        with self.assertRaisesRegex(RuntimeError, "Unique prefix was not cold"):
            self.run_gate(engine)

    def test_repeat_without_reuse_fails(self):
        with self.assertRaisesRegex(RuntimeError, "prefix-repeat did not reuse prefix"):
            self.run_gate(FakeEngine(cache=False))

    def test_needle_hits_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError, "coldness cannot be proven"):
            self.run_gate(FakeEngine(needle_hits=2048))

    def test_conversation_reuse_with_needle_fails_closed(self):
        # Legitimate turn-to-turn reuse makes the needle's coldness unattributable: fail, do not guess.
        with self.assertRaisesRegex(RuntimeError, "coldness cannot be proven"):
            self.run_gate(FakeEngine(align=16))

    def test_foreign_request_fails(self):
        with self.assertRaisesRegex(RuntimeError, "does not match its own requests"):
            self.run_gate(FakeEngine(foreign_success=True))

    def test_preemption_fails(self):
        with self.assertRaisesRegex(RuntimeError, "Preemption during the concurrent phase"):
            self.run_gate(FakeEngine(preempt=True))

    def test_no_overlap_fails(self):
        with self.assertRaisesRegex(RuntimeError, "overlap"):
            self.run_gate(FakeEngine(needle_fast=True))

    def test_http_error_recorded_and_raised(self):
        import urllib.error
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.run_gate(FakeEngine(fail_label_text="Repeat the access code alone."))
        caught.exception.close()
        self.assertTrue((self.root / "run" / "prefix-extension-http-error.json").is_file())

    def test_refuses_existing_out(self):
        (self.root / "run").mkdir()
        with self.assertRaises(FileExistsError):
            self.run_gate(FakeEngine())


class Rules(unittest.TestCase):
    def test_check_answer(self):
        ok = {"message": {"content": " 630141\n"}, "finish_reason": "stop"}
        self.assertEqual(gate.check_answer(ok, "630141").strip(), "630141")
        for bad in ({"message": {"content": "630141"}, "finish_reason": "length"},
                    {"message": {"content": "630141."}, "finish_reason": "stop"},
                    {"message": {"content": None}, "finish_reason": "stop"}):
            with self.assertRaises(RuntimeError):
                gate.check_answer(bad, "630141")

    def test_prefix_verdict(self):
        self.assertEqual(gate.prefix_verdict("prefix-cold", 0, 100), 0)
        with self.assertRaises(RuntimeError):
            gate.prefix_verdict("prefix-cold", 256, 1000)
        self.assertEqual(gate.prefix_verdict("prefix-repeat", 950, 1000), 0.95)
        with self.assertRaises(RuntimeError):
            gate.prefix_verdict("prefix-repeat", 949, 1000)
        with self.assertRaises(RuntimeError):
            gate.prefix_verdict("prefix-extension", 1001, 1000)

    def test_unique_archive_text(self):
        text = gate.unique_archive_text("Archive identity " + "0" * 32 + ". rest")
        self.assertNotIn("0" * 32, text)
        for bad in ("no identity", "Archive identity " + "0" * 32 + " Archive identity " + "1" * 32):
            with self.assertRaises(RuntimeError):
                gate.unique_archive_text(bad)

    def counters(self, **kw):
        base = {"queries": 0, "hits": 0, "successes": 0, "preemptions": 0}
        base.update(kw)
        return base

    def test_concurrent_verdict(self):
        turns = [50] * 12
        v = gate.concurrent_verdict(self.counters(), self.counters(queries=1600, successes=13), turns,
                                    needle_prompt_tokens=1000)
        self.assertEqual(v["needle_prefilled_tokens_at_least"], 1000)
        with self.assertRaisesRegex(RuntimeError, "coldness cannot be proven"):
            gate.concurrent_verdict(self.counters(), self.counters(queries=1600, hits=256, successes=13), turns,
                                    needle_prompt_tokens=1000)
        v = gate.concurrent_verdict(self.counters(), self.counters(queries=600, hits=256, successes=12), turns)
        self.assertEqual(v["aggregate_hits"], 256)
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            gate.concurrent_verdict(self.counters(), self.counters(queries=600, successes=13), turns)
        with self.assertRaisesRegex(RuntimeError, "Preemption"):
            gate.concurrent_verdict(self.counters(), self.counters(queries=600, successes=12, preemptions=1), turns)
        with self.assertRaisesRegex(RuntimeError, "Expected 12"):
            gate.concurrent_verdict(self.counters(), self.counters(queries=550, successes=11), [50] * 11)

    def test_mixed_evidence_rule(self):
        base = {"unix": 0.0, "running": 0, "waiting": 0}
        turns = [{"label": "concurrent-0-turn1", "started_unix": 1.0, "ended_unix": 2.0}]
        receipt = {"started_unix": 0.5, "elapsed_s": 5.0}
        shared = [{"unix": 1.5, "running": 2, "waiting": 0, "long_done": False}]
        alone = [{"unix": 1.5, "running": 1, "waiting": 0, "long_done": False}]
        self.assertTrue(gate.mixed_evidence(base, 0.5, base, shared, [], turns, receipt)["overlap_observed"])
        self.assertFalse(gate.mixed_evidence(base, 0.5, base, alone, [], turns, receipt)["overlap_observed"])
        self.assertFalse(gate.mixed_evidence(base, 0.5, base, shared, ["err"], turns, receipt)["overlap_observed"])
        late = [{"label": "concurrent-0-turn1", "started_unix": 1.0, "ended_unix": 6.0}]
        self.assertFalse(gate.mixed_evidence(base, 0.5, base, shared, [], late, receipt)["overlap_observed"])

    def test_wait_idle_times_out(self):
        clock = iter([0.0, 0.1, 31.0])
        busy = lambda url: {"running": 1, "waiting": 0}
        with self.assertRaisesRegex(RuntimeError, "not idle"):
            gate.wait_idle("x", busy, busy, sleep=lambda s: None, clock=lambda: next(clock))


class Provenance(unittest.TestCase):
    """The adaptation keeps the historical prompts, answers and thresholds."""

    def test_historical_strings_preserved(self):
        historical = gate.HISTORICAL.read_text()
        adapted = (HERE / "claude_gate_conversations.py").read_text()
        for fragment in (
            'f"Archive {identity}. Remember access code 739184.\\n"',
            '" archived material" * 16384 + "\\nReply only with the access code."',
            'user("Repeat the access code alone.")',
            'archive + "\\nUse digits only."',
            "code = str(630140 + index)",
            'f"Conversation {uuid.uuid4().hex}. Remember code {code}. Reply only with that code."',
            'user("What code did I give you? Reply only with the code.")',
            'user("Add one to that code and reply only with the integer.")',
            'str(int(code) + 1)',
            '"temperature": 0, "max_tokens": 64',
            '"chat_template_kwargs": {"thinking": False}',
            'answer.strip() != expected or choice["finish_reason"] != "stop"',
            '"overlap_observed": not errors and bool(shared) and bool(before_end)',
            'shared = [sample for sample in active if sample["running"] >= 2]',
        ):
            self.assertIn(fragment, historical, fragment)
            self.assertIn(fragment, adapted, fragment)
        self.assertIn("cached / total < 0.95", historical)
        self.assertEqual(gate.REUSE_MIN, 0.95)
        self.assertIn('r"Archive identity [0-9a-f]{32}"', historical)
        self.assertIn('r"Archive identity [0-9a-f]{32}"', adapted)


if __name__ == "__main__":
    unittest.main()
