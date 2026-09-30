"""Offline checks for probe-conversations.py. No sockets, nodes or model requests.

The probe is executed unchanged through runpy while urllib.request.urlopen is
replaced by an in-process fake that tracks in-flight requests and serves
Prometheus text for vllm:num_requests_running/waiting. Temporary receipts are
created beside this file, not on tmpfs.
"""

import contextlib
import io
import json
from pathlib import Path
import re
import runpy
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import urllib.request

ROOT = Path(__file__).resolve().parent
PROBE = ROOT / "probe-conversations.py"
LONG_TEXT = ("Archive identity 0123456789abcdef0123456789abcdef. Memorize this unique retrieval code: 739184.\n"
             "Archive:\n filler filler\nArchive ends. Reply with only the codes.")


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class FakeServer:
    def __init__(self, *, long_delay=2.5, turn_delay=0.3, running_bias=0,
                 omit_waiting=False, report_only_long=False, wrong_turn=None):
        self.lock = threading.Lock()
        self.inflight = 0
        self.long_inflight = 0
        self.seen = []
        self.long_delay = long_delay
        self.turn_delay = turn_delay
        self.running_bias = running_bias
        self.omit_waiting = omit_waiting
        self.report_only_long = report_only_long
        self.wrong_turn = wrong_turn

    def metrics(self):
        with self.lock:
            active = self.long_inflight if self.report_only_long else self.inflight
            running = min(active, 4) + self.running_bias
            waiting = max(active - 4, 0)
        lines = ["# HELP vllm:num_requests_running Number of requests in model execution batches.",
                 "# TYPE vllm:num_requests_running gauge",
                 f'vllm:num_requests_running{{engine="0",model_name="DeepSeek-V4.1-Flash"}} {float(running)}']
        if not self.omit_waiting:
            lines.append(f'vllm:num_requests_waiting{{engine="0",model_name="DeepSeek-V4.1-Flash"}} {float(waiting)}')
        return "\n".join(lines) + "\n"

    def chat(self, body):
        messages = body["messages"]
        first = messages[0]["content"]
        prompt_tokens = sum(len(m["content"]) for m in messages if isinstance(m.get("content"), str)) // 4 + 8
        is_long = first.startswith("Archive identity")
        with self.lock:
            self.inflight += 1
            self.long_inflight += int(is_long)
        try:
            cached = 0
            if first.startswith("Archive identity"):
                time.sleep(self.long_delay)
                answer = "739184, 482617"
            elif first.startswith("Archive "):
                archive = first.removesuffix("\nUse digits only.")
                with self.lock:
                    cached = int(prompt_tokens * 0.99) if archive in self.seen else 0
                    self.seen.append(archive)
                answer = "739184"
            else:
                time.sleep(self.turn_delay)
                match = re.search(r"Remember code (\d+)", first)
                assert match is not None
                code = int(match.group(1))
                turn = sum(1 for m in messages if m["role"] == "user")
                answer = str(code + 1 if turn == 3 else code)
                if self.wrong_turn == (code - 630140, turn):
                    answer = "999999"
            return {"choices": [{"finish_reason": "stop",
                                 "message": {"role": "assistant", "content": answer}}],
                    "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 2,
                              "prompt_tokens_details": {"cached_tokens": cached}}}
        finally:
            with self.lock:
                self.inflight -= 1
                self.long_inflight -= int(is_long)

    def urlopen(self, target, timeout=None):
        if isinstance(target, urllib.request.Request):
            assert isinstance(target.data, bytes)
            return Response(json.dumps(self.chat(json.loads(target.data))).encode())
        if target.endswith("/metrics"):
            return Response(self.metrics().encode())
        raise AssertionError(f"unexpected request {target}")


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT, prefix=".offline-probe-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.needle = self.root / "needle-input.json"
        self.needle.write_text(json.dumps({
            "messages": [{"role": "user", "content": LONG_TEXT}], "expected": "739184, 482617"}))

    def run_probe(self, server, *, mixed=True, fast_clock=False):
        out = self.root / "out"
        argv = ["probe-conversations.py", "--base-url", "http://fake", "--out", str(out)]
        if mixed:
            argv += ["--mixed-needle-input", str(self.needle)]
        stack = contextlib.ExitStack()
        stack.enter_context(patch.object(sys, "argv", argv))
        stack.enter_context(patch.object(urllib.request, "urlopen", server.urlopen))
        if fast_clock:
            clock = iter(range(0, 10**6, 5))
            stack.enter_context(patch.object(time, "monotonic", lambda: float(next(clock))))
            stack.enter_context(patch.object(time, "sleep", lambda _: None))
        stdout = io.StringIO()
        with stack, contextlib.redirect_stdout(stdout):
            try:
                runpy.run_path(str(PROBE), run_name="__main__")
            finally:
                self.stdout = stdout.getvalue()
        return out

    def test_mixed_overlap_and_queue_are_observed(self):
        out = self.run_probe(FakeServer())
        self.assertIn("DS41-CONVERSATIONS-PASS", self.stdout)
        evidence = json.loads((out / "mixed-evidence.json").read_text())
        self.assertEqual((evidence["baseline"]["running"], evidence["baseline"]["waiting"]), (0.0, 0.0))
        self.assertGreaterEqual(evidence["long_running_observed"]["running"], 1)
        self.assertTrue(evidence["overlap_observed"])
        self.assertTrue(evidence["queue_observed"])
        self.assertEqual(len(evidence["turns"]), 12)
        self.assertGreater(len(evidence["turns_completed_before_long_end"]), 0)
        self.assertTrue((out / "mixed-baseline.prom").is_file())
        self.assertTrue((out / "mixed-overlap-start.prom").is_file())
        self.assertIn("samples", json.loads((out / "mixed-samples.json").read_text()))

    def test_non_idle_baseline_refuses_before_submission(self):
        server = FakeServer(running_bias=1)
        with self.assertRaisesRegex(RuntimeError, "not idle"):
            self.run_probe(server, fast_clock=True)
        out = self.root / "out"
        self.assertTrue((out / "mixed-baseline.prom").is_file())
        self.assertFalse((out / "mixed-long-needle.json").exists())

    def test_missing_gauge_is_not_zero(self):
        with self.assertRaisesRegex(RuntimeError, "vllm:num_requests_waiting is missing"):
            self.run_probe(FakeServer(omit_waiting=True))

    def test_submission_alone_is_not_overlap(self):
        # Either refusal is valid; which one fires depends on whether a scrape lands mid-request.
        with self.assertRaisesRegex(RuntimeError, "Long request finished before overlap|overlap was not observed"):
            self.run_probe(FakeServer(long_delay=0.0))

    def test_unshared_engine_samples_fail_overlap(self):
        with self.assertRaisesRegex(RuntimeError, "overlap was not observed"):
            self.run_probe(FakeServer(report_only_long=True))
        evidence = json.loads((self.root / "out" / "mixed-evidence.json").read_text())
        self.assertFalse(evidence["overlap_observed"])
        self.assertEqual(evidence["samples_running_ge_2_while_long_active"], 0)

    def test_exact_answer_gate_unchanged(self):
        with self.assertRaisesRegex(RuntimeError, "Wrong or incomplete concurrent-2-turn3"):
            self.run_probe(FakeServer(wrong_turn=(2, 3)))

    def test_without_mixed_input_no_metrics_needed(self):
        server = FakeServer(omit_waiting=True)
        self.run_probe(server, mixed=False)
        self.assertIn("DS41-CONVERSATIONS-PASS", self.stdout)
        self.assertFalse((self.root / "out" / "mixed-evidence.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
