#!/usr/bin/env python3
"""Exact-answer prefix reuse, concurrent multi-turn and mixed-load gate (metrics cache accounting).

Adapted from spark/ds41/r38/probe-conversations.py. The historical file is not
changed; its SHA-256 is recorded in the receipt. Preserved verbatim: prompts,
expected answers, the exact-answer rule (strip() equality and finish_reason
"stop"), the 0.95 reuse threshold, the four concurrent three-turn
conversations, the fresh archive identity for the mixed-load needle and the
overlap evidence rule.

Cache accounting. The current launcher omits usage cached-token details
(`enable_prompt_tokens_details` defaults to False at vLLM 1794dcf1 and the
frozen launch arrays do not set it), so:

* Serial prefix tests (cold, exact repeat, conversation extension, divergent
  suffix) run one request at a time on an idle engine and use
  `cache_metrics.idle_snapshot` / `finish`: the counters must advance by
  exactly that request (queries == prompt_tokens, one success, no
  preemption). The hit count is what the scheduler recorded at admission
  (`KVCacheManager.record_prefix_cache_stats(request, num_new_local_computed_tokens)`,
  scheduler.py 1745), the same quantity usage `cached_tokens` would report.
  Request-boundary checkpoints are off for this model type at the pin
  (`VllmConfig.use_request_boundary_checkpoints` allow-list), so hits come
  from the block-hash path.
* The concurrent phase reports aggregate counters only. Aggregate counters
  cannot attribute a hit to a request. They can prove: the phase accounted
  exactly its own requests (successes, and queries == sum of prompt tokens),
  no preemption, and, because per-request hits are nonnegative, that an
  aggregate of zero means every request in the phase was cold. The mixed-load
  needle must be cold (historical rule), so with a needle the gate requires
  aggregate hits == 0 and fails closed otherwise, recording the only provable
  statement left (needle prefilled at least prompt_tokens - aggregate hits).
  Without a needle the conversation-phase hits are informational, as in the
  historical probe.

Engine idleness is awaited (up to 30 s) before every accounted window.

Usage, from the workstation against the serving head (no node actions):
  python3 claude_gate_conversations.py --out receipts/conversations-<tag> \\
      [--mixed-needle-input ../../ds41/r38/receipts/20260916/admission-k7-1m-u80/needle-524288-input.json]
Success marker: CLAUDE-CONVERSATIONS-PASS. Any failure raises.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

HERE = Path(__file__).resolve().parent
HISTORICAL = HERE.parents[1] / "ds41" / "r38" / "probe-conversations.py"
MODEL = "DeepSeek-V4.1-Flash"
REUSE_MIN = 0.95
IDLE_TIMEOUT_S = 30
CONVERSATIONS = 4
TURNS = 3
ARCHIVE_ID = re.compile(r"Archive identity [0-9a-f]{32}")


# --------------------------------------------------------------------------
# Pure rules (unit tested, no network)
# --------------------------------------------------------------------------

def user(text):
    return {"role": "user", "content": text}


def check_answer(choice, expected):
    """Exact answer after strip(), complete stop; raises otherwise (historical rule)."""
    answer = choice["message"].get("content") or ""
    if answer.strip() != expected or choice["finish_reason"] != "stop":
        raise RuntimeError(f"Wrong or incomplete answer: expected {expected!r}, got {choice!r}")
    return answer


def prefix_verdict(label, hits, prompt_tokens):
    """Serial rule: cold means zero hits; repeat, extension and divergent need >= 95 percent reuse."""
    if prompt_tokens <= 0 or hits < 0 or hits > prompt_tokens:
        raise RuntimeError(f"Invalid cache accounting for {label}: hits={hits} prompt_tokens={prompt_tokens}")
    ratio = hits / prompt_tokens
    if label == "prefix-cold":
        if hits:
            raise RuntimeError("Unique prefix was not cold")
    elif ratio < REUSE_MIN:
        raise RuntimeError(f"{label} did not reuse prefix: {hits}/{prompt_tokens} = {ratio:.3f}")
    return ratio


def unique_archive_text(content):
    text, replacements = ARCHIVE_ID.subn(f"Archive identity {uuid.uuid4().hex}", content)
    if replacements != 1:
        raise RuntimeError("Cannot give the mixed-load needle a unique prefix")
    return text


def concurrent_verdict(before, after, turn_prompt_tokens, *, needle_prompt_tokens=None):
    """What aggregate counters prove about the concurrent phase; raises on anything else."""
    if len(turn_prompt_tokens) != CONVERSATIONS * TURNS:
        raise RuntimeError(f"Expected {CONVERSATIONS * TURNS} conversation turns, got {len(turn_prompt_tokens)}")
    needle = needle_prompt_tokens is not None
    expected_requests = len(turn_prompt_tokens) + int(needle)
    expected_queries = sum(turn_prompt_tokens) + (needle_prompt_tokens or 0)
    queries = after["queries"] - before["queries"]
    hits = after["hits"] - before["hits"]
    completed = after["successes"] - before["successes"]
    preempted = after["preemptions"] - before["preemptions"]
    if completed != expected_requests or queries != expected_queries:
        raise RuntimeError("Concurrent phase accounting does not match its own requests: "
                           f"successes={completed}/{expected_requests} queries={queries}/{expected_queries}")
    if preempted:
        raise RuntimeError(f"Preemption during the concurrent phase ({preempted}); cache accounting ambiguous")
    if hits < 0 or hits > queries:
        raise RuntimeError("Invalid cache counter delta in the concurrent phase")
    verdict = {"requests": completed, "queries": queries, "aggregate_hits": hits, "preemptions": preempted,
               "attribution": "aggregate only; counters cannot attribute a hit to a request"}
    if needle:
        verdict["needle_prompt_tokens"] = needle_prompt_tokens
        verdict["needle_prefilled_tokens_at_least"] = needle_prompt_tokens - hits
        if hits:
            raise RuntimeError(
                f"Mixed-load needle coldness cannot be proven: aggregate hits={hits} > 0 and aggregate "
                f"counters cannot attribute them (needle prefilled at least {needle_prompt_tokens - hits} "
                f"of {needle_prompt_tokens} tokens); inspect concurrent-*.json prompt sizes")
        verdict["all_requests_cold"] = "proven: aggregate hits are zero and per-request hits are nonnegative"
    return verdict


def mixed_evidence(baseline, submitted, started, samples, errors, turns, long_receipt):
    """Historical overlap rule, unchanged."""
    long_end = long_receipt["started_unix"] + long_receipt["elapsed_s"]
    active = [sample for sample in samples if not sample["long_done"]]
    shared = [sample for sample in active if sample["running"] >= 2]
    before_end = [turn["label"] for turn in turns if turn["ended_unix"] < long_end]
    return {
        "baseline": {key: baseline[key] for key in ("unix", "running", "waiting")},
        "long_submitted_unix": submitted,
        "long_running_observed": {key: started[key] for key in ("unix", "running", "waiting")},
        "long_ended_unix": long_end,
        "turns": turns,
        "turns_completed_before_long_end": before_end,
        "sampler_errors": errors,
        "samples": len(samples),
        "samples_while_long_active": len(active),
        "samples_running_ge_2_while_long_active": len(shared),
        "max_running_while_long_active": max((s["running"] for s in active), default=None),
        "max_waiting_while_long_active": max((s["waiting"] for s in active), default=None),
        "queue_observed": any(s["waiting"] >= 1 for s in active),
        # Submission alone never counts: require an idle baseline, the long request
        # observed running, a concurrent engine sample and a turn finishing first.
        "overlap_observed": not errors and bool(shared) and bool(before_end),
    }


def wait_idle(base_url, snapshot, idle_snapshot, *, timeout_s=IDLE_TIMEOUT_S, sleep=time.sleep, clock=time.monotonic):
    """Poll until running == waiting == 0, then take the accounted idle snapshot."""
    deadline = clock() + timeout_s
    while True:
        sample = snapshot(base_url)
        if not sample["running"] and not sample["waiting"]:
            return idle_snapshot(base_url)
        if clock() >= deadline:
            raise RuntimeError(f"Engine not idle within {timeout_s}s: running={sample['running']} "
                               f"waiting={sample['waiting']}")
        sleep(0.5)


def settle(base_url, snapshot, before, expected_requests, *, timeout_s=IDLE_TIMEOUT_S, sleep=time.sleep,
           clock=time.monotonic):
    """Wait until the counters show every request of the phase and the engine is idle."""
    deadline = clock() + timeout_s
    while True:
        after = snapshot(base_url)
        if (after["successes"] - before["successes"] >= expected_requests
                and not after["running"] and not after["waiting"]):
            return after
        if clock() >= deadline:
            raise RuntimeError("Cache counters did not settle after the concurrent phase")
        sleep(1)


# --------------------------------------------------------------------------
# Live client (requests only against --base-url)
# --------------------------------------------------------------------------

class Client:
    def __init__(self, base_url, out):
        self.base_url, self.out = base_url, out
        self.lock = threading.Lock()
        self.turns = []

    def chat(self, label, messages, expected, *, timeout=3600):
        body = {"model": MODEL, "messages": messages, "temperature": 0, "max_tokens": 64,
                "chat_template_kwargs": {"thinking": False}}
        req = urllib.request.Request(self.base_url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        start = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            (self.out / f"{label}-http-error.json").write_text(json.dumps({
                "status": error.code, "body": error.read().decode(errors="replace"),
                "request": body, "started_unix": start}, indent=2) + "\n")
            raise
        elapsed = time.time() - start
        receipt = {"label": label, "started_unix": start, "elapsed_s": elapsed,
                   "expected": expected, "request": body, "response": result}
        (self.out / f"{label}.json").write_text(json.dumps(receipt, indent=2) + "\n")
        choice = result["choices"][0]
        answer = check_answer(choice, expected)
        prompt_tokens = result["usage"]["prompt_tokens"]
        with self.lock:
            self.turns.append({"label": label, "started_unix": start, "ended_unix": start + elapsed,
                               "prompt_tokens": prompt_tokens})
        details = result["usage"].get("prompt_tokens_details") or {}
        print(json.dumps({"label": label, "answer": answer, "elapsed_s": elapsed, "prompt_tokens": prompt_tokens,
                          "usage_cached_tokens": details.get("cached_tokens", "not reported")}), flush=True)
        return choice["message"], prompt_tokens, receipt


def serial_prefix_phase(client, metrics):
    identity = uuid.uuid4().hex
    archive = (f"Archive {identity}. Remember access code 739184.\n"
               + " archived material" * 16384 + "\nReply only with the access code.")
    base = [user(archive)]
    rows, first = [], None
    for label in ("prefix-cold", "prefix-repeat", "prefix-extension", "prefix-divergent"):
        if label in ("prefix-cold", "prefix-repeat"):
            messages = base
        elif label == "prefix-extension":
            messages = base + [first, user("Repeat the access code alone.")]
        else:
            messages = [user(archive + "\nUse digits only.")]
        before = wait_idle(client.base_url, metrics.snapshot, metrics.idle_snapshot)
        message, prompt_tokens, _ = client.chat(label, messages, "739184")
        if label == "prefix-cold":
            first = message
        hits = metrics.finish(client.base_url, before, prompt_tokens, client.out / f"{label}-cache.json")
        ratio = prefix_verdict(label, hits, prompt_tokens)
        rows.append({"label": label, "prompt_tokens": prompt_tokens, "hits": hits, "ratio": ratio})
        print(json.dumps({"serial": rows[-1]}), flush=True)
    return rows


def conversation(client, index):
    code = str(630140 + index)
    messages = [user(f"Conversation {uuid.uuid4().hex}. Remember code {code}. Reply only with that code.")]
    first, _, _ = client.chat(f"concurrent-{index}-turn1", messages, code)
    messages += [first, user("What code did I give you? Reply only with the code.")]
    second, _, _ = client.chat(f"concurrent-{index}-turn2", messages, code)
    messages += [second, user("Add one to that code and reply only with the integer.")]
    client.chat(f"concurrent-{index}-turn3", messages, str(int(code) + 1))


def sample_engine(stop, long_run, samples, errors, snapshot, base_url):
    while not stop.is_set():
        try:
            sample = snapshot(base_url)
        except Exception as error:  # Incomplete evidence is reported, never ignored.
            errors.append(repr(error))
            return
        samples.append({"unix": sample["unix"], "running": sample["running"],
                        "waiting": sample["waiting"], "long_done": long_run.done()})
        stop.wait(0.25)


def concurrent_phase(client, metrics, needle_path):
    out = client.out
    needle = json.loads(needle_path.read_text()) if needle_path is not None else None
    baseline = wait_idle(client.base_url, metrics.snapshot, metrics.idle_snapshot)
    (out / "concurrent-baseline.prom").write_text(baseline["raw"])
    long_run = sampler = started = submitted = None
    stop, samples, errors = threading.Event(), [], []
    with ThreadPoolExecutor(max_workers=CONVERSATIONS + 1) as pool:
        if needle is not None:
            text = unique_archive_text(needle["messages"][0]["content"])
            submitted = time.time()
            long_run = pool.submit(client.chat, "mixed-long-needle", [user(text)], needle["expected"])
            deadline = time.monotonic() + IDLE_TIMEOUT_S
            while True:
                if long_run.done():
                    long_run.result()
                    raise RuntimeError("Long request finished before overlap was established")
                started = metrics.snapshot(client.base_url)
                if started["running"] >= 1:
                    (out / "mixed-overlap-start.prom").write_text(started["raw"])
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError("No running long request observed")
                time.sleep(0.2)
            sampler = threading.Thread(target=sample_engine, daemon=True,
                                       args=(stop, long_run, samples, errors, metrics.snapshot, client.base_url))
            sampler.start()
        try:
            list(pool.map(lambda index: conversation(client, index), range(CONVERSATIONS)))
        finally:
            if sampler is not None:
                stop.set()
                sampler.join()
                (out / "mixed-samples.json").write_text(json.dumps(
                    {"samples": samples, "sampler_errors": errors}, indent=2) + "\n")
        long_tokens = long_receipt = None
        if long_run is not None:
            _, long_tokens, long_receipt = long_run.result()
    turns = [turn for turn in client.turns if re.fullmatch(r"concurrent-\d+-turn\d", turn["label"])]
    after = settle(client.base_url, metrics.snapshot, baseline, len(turns) + int(long_run is not None))
    (out / "concurrent-after.prom").write_text(after["raw"])
    result: dict = {"turns": turns}
    if long_run is not None:
        assert needle is not None and started is not None and long_receipt is not None
        evidence = mixed_evidence(baseline, submitted, started, samples, errors, turns, long_receipt)
        (out / "mixed-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
        print(json.dumps({"mixed_evidence": {k: v for k, v in evidence.items() if k != "turns"}}), flush=True)
        result["mixed"] = {"prompt_tokens": long_tokens, "input_prompt_tokens": needle.get("prompt_tokens"),
                           "expected": needle["expected"], "overlap_observed": evidence["overlap_observed"]}
    # Accounting first: it fails closed on foreign requests, preemption or an unprovable cold needle.
    result["aggregate"] = concurrent_verdict(baseline, after, [t["prompt_tokens"] for t in turns],
                                             needle_prompt_tokens=long_tokens)
    print(json.dumps({"concurrent": result["aggregate"]}), flush=True)
    if long_run is not None and not result["mixed"]["overlap_observed"]:
        raise RuntimeError("Mixed-load overlap was not observed; see mixed-evidence.json")
    return result


def main():
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("--base-url", default="http://dusty:8000")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mixed-needle-input", type=Path,
                   help="Replay this needle with a fresh archive ID while four conversations run")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(HERE))
    import cache_metrics as metrics  # the task tree's accounting helper, imported unchanged

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

    summary = {"base_url": a.base_url, "reuse_min": REUSE_MIN,
               "gate_sha256": digest(Path(__file__)), "cache_metrics_sha256": digest(HERE / "cache_metrics.py"),
               "historical_probe": str(HISTORICAL), "historical_probe_sha256": digest(HISTORICAL),
               "mixed_needle_input": str(a.mixed_needle_input) if a.mixed_needle_input else None,
               "mixed_needle_input_sha256": digest(a.mixed_needle_input) if a.mixed_needle_input else None}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    client = Client(a.base_url, a.out)
    summary["serial"] = serial_prefix_phase(client, metrics)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    summary["concurrent"] = concurrent_phase(client, metrics, a.mixed_needle_input)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("CLAUDE-CONVERSATIONS-PASS", flush=True)


if __name__ == "__main__":
    main()
