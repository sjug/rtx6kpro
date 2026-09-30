#!/usr/bin/env python3
"""Minimal reproducer: identical greedy requests give different logits past 512 tokens.

Provenance: written 2026-09-16 for the DS4.1 R38 TP4 nondeterminism investigation.
It needs only an OpenAI-compatible vLLM endpoint serving DeepSeek-V4.1-Flash. For
each prompt length it builds one prompt of exactly that many tokens (via /tokenize),
sends N identical requests at temperature 0 with logprobs, and gives each request a
unique cache_salt so no prefix cache is reused. It reports, per output position,
the largest absolute difference between the reported top-K logprobs of any two
runs. Zero everywhere means bitwise-identical reported logprobs.

Expected on the affected runtime: identical logprobs at lengths of about 512
tokens and below, differences of order 1 logprob from about 514 tokens on, growing
with length. Complete responses are written to --out for review.
"""
import argparse
import json
from pathlib import Path
import secrets
import time
import urllib.request
import uuid

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--base-url", default="http://dusty:8000")
p.add_argument("--model", default="DeepSeek-V4.1-Flash")
p.add_argument("--lengths", default="256,512,1024,4096")
p.add_argument("--repeats", type=int, default=4)
p.add_argument("--max-tokens", type=int, default=8)
p.add_argument("--top-logprobs", type=int, default=20)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)


def post(path, body, timeout=3600):
    req = urllib.request.Request(a.base_url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def build(length):
    """One user message of exactly `length` prompt tokens, unique per call."""
    head = f"Archive identity {uuid.uuid4().hex}. Memorize this unique retrieval code: 739184.\nArchive:\n"
    tail = "\nArchive ends. Reply with only the retrieval code stated before the archive."
    filler = max(length - 100, 1)
    for _ in range(8):
        messages = [{"role": "user", "content": head + " filler" * filler + tail}]
        count = post("/tokenize", {"model": a.model, "messages": messages, "add_generation_prompt": True,
                                   "chat_template_kwargs": {"thinking": False}}, 120)["count"]
        if count == length:
            return messages
        filler += length - count
    raise SystemExit(f"cannot build a {length}-token prompt")


report = {}
for length in [int(x) for x in a.lengths.split(",")]:
    messages = build(length)
    runs = []
    for i in range(a.repeats):
        body = {"model": a.model, "messages": messages, "temperature": 0, "max_tokens": a.max_tokens,
                "logprobs": True, "top_logprobs": a.top_logprobs, "cache_salt": secrets.token_urlsafe(32),
                "chat_template_kwargs": {"thinking": False}}
        result = post("/v1/chat/completions", body)
        (a.out / f"len{length}-run{i}.json").write_text(json.dumps({"request": body, "response": result}, indent=1) + "\n")
        usage = result["usage"]
        if usage["prompt_tokens"] != length or usage.get("prompt_tokens_details", {}).get("cached_tokens") != 0:
            raise SystemExit(f"run {i} at {length}: unexpected usage {usage}")
        runs.append([{alt["token"]: alt["logprob"] for alt in tok["top_logprobs"]}
                     for tok in result["choices"][0]["logprobs"]["content"]])
    positions = min(len(r) for r in runs)
    per_position = []
    for k in range(positions):
        worst = 0.0
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                shared = set(runs[i][k]) & set(runs[j][k])
                worst = max([worst] + [abs(runs[i][k][t] - runs[j][k][t]) for t in shared])
        per_position.append(round(worst, 4))
    report[length] = {"repeats": a.repeats, "positions": positions, "max_abs_logprob_diff_per_position": per_position,
                      "bitwise_identical": all(d == 0.0 for d in per_position)}
    print(json.dumps({"length": length, **report[length]}), flush=True)
(a.out / "report.json").write_text(json.dumps({"base_url": a.base_url, "model": a.model, "unix": time.time(),
                                               "report": report}, indent=1) + "\n")
