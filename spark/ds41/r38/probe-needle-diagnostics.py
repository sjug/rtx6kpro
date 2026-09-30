#!/usr/bin/env python3
"""Diagnostic long-context retrieval probes for the DS4.1 524K dual-needle failure.

Provenance: added 2026-09-16 after qualify.py (sha256 108adc19dc90a6e1...) failed its
unchanged 524,288-token dual-needle gate (receipts/20260916/admission-k7-1m-u80/).
This script neither replaces nor relaxes that gate. Every cold condition rebuilds
qualify.py's prompt exactly, except for the one change its CONDITIONS entry states.
Wrong answers are recorded, not raised, so a series can compare conditions; any
server error stops the series. Complete inputs, requests and responses are saved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import secrets
import time
import urllib.error
import urllib.request
import uuid

MODEL = "DeepSeek-V4.1-Flash"
PREFIX = "Archive identity {identity}. Memorize this unique retrieval code: 739184.\nArchive:\n"
LATE = "\nThe late retrieval code is 482617.\n"
# Conditions: (dual, suffix, expected, change relative to qualify.py).
CONDITIONS = {
    "dual-original": (True, "\nArchive ends. Reply with only the initial retrieval code and the late "
                      "retrieval code, in that order, separated by a comma and a space.",
                      "739184, 482617", "none: qualify.py --dual-needle wording, fresh identity"),
    "single-original": (False, "\nArchive ends. Reply with only the retrieval code stated before the archive.",
                        "739184", "none: qualify.py single-needle wording, fresh identity"),
    "dual-stated-before": (True, "\nArchive ends. Reply with only the retrieval code stated before the "
                           "archive and the late retrieval code, in that order, separated by a comma and a space.",
                           "739184, 482617",
                           "suffix only: 'initial retrieval code' replaced by the single-needle phrase "
                           "'retrieval code stated before the archive'"),
}
# Identity variants: (base condition, identity factory, cold, change). The failing
# identity starts with the digit token "510"; "fixed-failing" shares blocks with the
# original prompt and is therefore never cold, while "digit-lead" is a fresh identity
# with the same leading shape.
FAILING_IDENTITY = "510c94b1bb4f42d2998093bd6b9c3d99"


def digit_lead():
    return f"{secrets.randbelow(900) + 100}{uuid.uuid4().hex[3:]}"


IDENTITIES = {
    "dual-original-fixed-failing-identity": ("dual-original", lambda: FAILING_IDENTITY, False,
                                             "identity only: the failing 524K identity, reused verbatim"),
    "dual-original-digit-lead-identity": ("dual-original", digit_lead, True,
                                          "identity only: fresh, forced to begin with three decimal digits"),
    "dual-stated-before-digit-lead-identity": ("dual-stated-before", digit_lead, True,
                                               "revised suffix plus a fresh identity forced to begin with "
                                               "three decimal digits"),
}
for name, (base, _, _, change) in IDENTITIES.items():
    CONDITIONS[name] = CONDITIONS[base][:3] + (change,)


p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--base-url", default="http://dusty:8000")
p.add_argument("--out", type=Path, required=True)
p.add_argument("--length", type=int, default=524288)
p.add_argument("--timeout", type=int, default=3600)
p.add_argument("--top-logprobs", type=int, default=5, help="Alternatives per position on logprob replays")
p.add_argument("--replay-logprobs", action="store_true",
               help="After each cold step, replay its exact input warm with top-5 logprobs")
p.add_argument("steps", nargs="+",
               help="LABEL=CONDITION for a cold probe; LABEL=[cold-]replay:INPUT.json for an exact replay; "
                    "LABEL=[cold-]resuffix:CONDITION:INPUT.json to replace only the final dual instruction. "
                    "Replays use top-5 logprobs; cold- adds a unique cache_salt so no prefix is reused")
a = p.parse_args()
steps = []
for step in a.steps:
    label, _, target = step.partition("=")
    kind = target.split(":", 1)[0]
    valid = target in CONDITIONS or kind in ("replay", "cold-replay") or (
        kind in ("resuffix", "cold-resuffix") and target.split(":", 2)[1:2] != []
        and target.split(":", 2)[1] in CONDITIONS and len(target.split(":", 2)) == 3)
    if not re.fullmatch(r"[a-z0-9-]+", label) or not valid:
        p.error(f"Bad step {step!r}")
    steps.append((label, target))
if not 256 <= a.length <= 1048512:
    p.error("Length must leave room for 64 output tokens within native context")
a.out.mkdir(parents=True, exist_ok=False)


class ServerError(RuntimeError):
    pass


def post(path, payload, timeout, label):
    req = urllib.request.Request(a.base_url + path, json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response), start, time.time() - start
    except urllib.error.HTTPError as error:
        (a.out / f"{label}-http-error.json").write_text(json.dumps({
            "path": path, "status": error.code, "body": error.read().decode(errors="replace"),
            "request_without_messages": {k: v for k, v in payload.items() if k != "messages"},
            "started_unix": start, "elapsed_s": time.time() - start,
        }, indent=2) + "\n")
        if error.code >= 500:
            raise ServerError(f"{label}: HTTP {error.code}") from error
        return None, start, time.time() - start


def gauges():
    with urllib.request.urlopen(a.base_url + "/metrics", timeout=10) as response:
        text = response.read().decode()
    values = {}
    for name in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
        found = re.findall(rf"^{re.escape(name)}(?:\{{[^}}\n]*\}})?\s+([0-9.eE+-]+)$", text, re.M)
        if not found:
            raise RuntimeError(f"Metric {name} is missing from /metrics")
        values[name.split("_")[-1]] = sum(map(float, found))
    return values


def wait_idle():
    deadline = time.monotonic() + 60
    while True:
        values = gauges()
        if values["running"] == 0 and values["waiting"] == 0:
            return {"unix": time.time(), **values}
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Server not idle before probe: {values}")
        time.sleep(1)


def count(messages, label):
    result, _, _ = post("/tokenize", {"model": MODEL, "messages": messages,
                                      "chat_template_kwargs": {"thinking": False},
                                      "add_generation_prompt": True}, 120, label)
    if result is None:
        raise RuntimeError(f"{label}: tokenization rejected")
    return result.get("count", len(result.get("tokens", [])))


def build(label, condition):
    """Mirror qualify.py's exact-length loop for one condition."""
    dual, suffix, expected, change = CONDITIONS[condition]
    identity = IDENTITIES[condition][1]() if condition in IDENTITIES else uuid.uuid4().hex
    prefix = PREFIX.format(identity=identity)
    filler = a.length - 100
    for _ in range(6):
        before_late = prefix + " filler" * (filler * 7 // 10)
        if dual:
            text = before_late + LATE + " filler" * (filler - filler * 7 // 10) + suffix
        else:
            text = prefix + " filler" * filler + suffix
        messages = [{"role": "user", "content": text}]
        tokens = count(messages, label)
        if tokens == a.length:
            break
        filler += a.length - tokens
    else:
        raise RuntimeError(f"{label}: cannot construct exact-length needle")
    late = count([{"role": "user", "content": before_late}], label) if dual else None
    if dual and not 0.6 * a.length < late < 0.8 * a.length:
        raise RuntimeError(f"{label}: late needle outside the intended depth range")
    return {"messages": messages, "prompt_tokens": a.length, "chat_template_kwargs": {"thinking": False},
            "temperature": 0, "max_tokens": 64, "expected": expected, "condition": condition,
            "change_from_qualify": change, "late_marker_prefix_tokens_including_template": late,
            "identity": identity, "cold_required": condition not in IDENTITIES or IDENTITIES[condition][2]}


def run(label, spec, cold, logprobs, salt=None):
    body = {"model": MODEL, "messages": spec["messages"], "chat_template_kwargs": {"thinking": False},
            "temperature": 0, "max_tokens": 64}
    if salt:
        body["cache_salt"] = salt
    if logprobs:
        body.update(logprobs=True, top_logprobs=a.top_logprobs)
    idle = wait_idle()
    result, start, elapsed = post("/v1/chat/completions", body, a.timeout, label)
    record = {"label": label, "cold_required": cold, "condition": spec.get("condition"),
              "input_sha256": hashlib.sha256(spec["messages"][0]["content"].encode()).hexdigest(),
              "source": spec.get("source"), "source_sha256": spec.get("source_sha256"),
              "change_from_source": spec.get("change_from_source"),
              "request_without_messages": {k: v for k, v in body.items() if k != "messages"},
              "idle_before": idle, "started_unix": start, "elapsed_s": elapsed,
              "expected": spec["expected"], "response": result}
    (a.out / f"{label}.json").write_text(json.dumps(record, indent=2) + "\n")
    if result is None:
        summary = {"label": label, "rejected": True}
    else:
        choice = result["choices"][0]
        answer = (choice["message"].get("content") or "").strip()
        usage = result["usage"]
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        summary = {"label": label, "condition": spec.get("condition"), "answer": answer,
                   "expected": spec["expected"], "exact_match": answer == spec["expected"],
                   "finish_reason": choice["finish_reason"], "prompt_tokens": usage["prompt_tokens"],
                   "completion_tokens": usage["completion_tokens"], "cached_tokens": cached,
                   "cold_valid": cached == 0 if cold else None, "elapsed_s": round(elapsed, 2)}
        content = (choice.get("logprobs") or {}).get("content") or []
        if content:
            summary["top_logprobs"] = [
                [token["token"]] + [[alt["token"], round(alt["logprob"], 4)] for alt in token["top_logprobs"][:5]]
                for token in content[:12]]
    print(json.dumps(summary), flush=True)
    return summary


manifest = {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "base_url": a.base_url, "length": a.length, "steps": a.steps,
            "replay_logprobs": a.replay_logprobs, "started_unix": time.time(),
            "conditions": {name: {"dual": c[0], "suffix": c[1], "expected": c[2], "change": c[3]}
                           for name, c in CONDITIONS.items()}}
(a.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
summaries = []
try:
    for label, target in steps:
        kind, _, rest = target.partition(":")
        if kind in ("replay", "cold-replay", "resuffix", "cold-resuffix"):
            condition, path = rest.split(":", 1) if kind.endswith("resuffix") else (None, rest)
            source = json.loads(Path(path).read_text())
            text = source["messages"][0]["content"]
            spec = {"messages": source["messages"], "expected": source["expected"],
                    "condition": source.get("condition"), "source": path,
                    "source_sha256": hashlib.sha256(text.encode()).hexdigest()}
            if condition:
                original = CONDITIONS["dual-original"][1]
                if not text.endswith(original) or text.count(original) != 1:
                    raise RuntimeError(f"{label}: source does not end with the original dual instruction")
                spec.update(messages=[{"role": "user", "content": text[:-len(original)] + CONDITIONS[condition][1]}],
                            expected=CONDITIONS[condition][2], condition=condition,
                            change_from_source="final instruction only: " + CONDITIONS[condition][3])
                (a.out / f"{label}-input.json").write_text(json.dumps(spec) + "\n")
            cold = kind.startswith("cold-")
            summaries.append(run(label, spec, cold=cold, logprobs=True,
                                 salt=secrets.token_urlsafe(32) if cold else None))
            continue
        spec = build(label, target)
        (a.out / f"{label}-input.json").write_text(json.dumps(spec) + "\n")
        summaries.append(run(label, spec, cold=spec["cold_required"], logprobs=False))
        if a.replay_logprobs:
            summaries.append(run(f"{label}-warm-logprobs", spec, cold=False, logprobs=True))
finally:
    (a.out / "summary.json").write_text(json.dumps({"finished_unix": time.time(),
                                                    "summaries": summaries}, indent=2) + "\n")
print("DS41-NEEDLE-DIAGNOSTICS-COMPLETE", flush=True)
