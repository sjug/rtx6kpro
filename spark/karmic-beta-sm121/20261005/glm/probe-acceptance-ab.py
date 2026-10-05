#!/usr/bin/env python3
"""Same-boot A/B of MTP acceptance for one request field against the server default.

Default arm A is explicit top_p=1.0; ARM_FIELD/ARM_VALUE (env) select another field,
e.g. ARM_FIELD=chat_template_kwargs ARM_VALUE='{"reasoning_effort":"max"}'.

Sequential single requests (c1), identical prompts, ignore_eos, alternating arms so
drift affects both equally. Acceptance comes from the server's spec-decode counters,
so the endpoint must be otherwise idle.
"""
import json
import os
import re
import sys
import time
import urllib.request

BASE = "http://sparky:8000"
OUT = sys.argv[1]
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
TOKENS = 1024
FIELD = os.environ.get("ARM_FIELD", "top_p")
VALUE = json.loads(os.environ.get("ARM_VALUE", "1.0"))
LABEL = f"{FIELD}={json.dumps(VALUE)}"
# ARM_BODY (env, JSON object) replaces the single field with several merged fields.
BODY = json.loads(os.environ["ARM_BODY"]) if os.environ.get("ARM_BODY") else {FIELD: VALUE}
if os.environ.get("ARM_BODY"):
    LABEL = "body=" + json.dumps(BODY, sort_keys=True)
TOKENS = int(os.environ.get("ARM_TOKENS", TOKENS))
PROMPTS = [
    "Write a detailed technical explanation of how a B-tree handles insertion and node splits.",
    "Tell a long story about a lighthouse keeper who discovers a message in a bottle.",
    "Explain step by step how to derive the quadratic formula, with commentary on each step.",
]

if os.environ.get("BENCH_PROMPT") == "1":
    # The standard grid's decode request (llm_decode_bench.GENERATION_PROMPT, context 0).
    sys.path.insert(0, "/home/jugs/git/llm-inference-bench")
    from llm_decode_bench import GENERATION_PROMPT
    PROMPTS = [GENERATION_PROMPT]


def counters():
    text = urllib.request.urlopen(f"{BASE}/metrics", timeout=30).read().decode()
    def get(name):
        return sum(float(v) for v in re.findall(r"^vllm:" + name + r"\{[^\n]*\} ([\d.e+-]+)$", text, re.M))
    busy = get("num_requests_running") + get("num_requests_waiting")
    return get("spec_decode_num_drafts_total"), get("spec_decode_num_accepted_tokens_total"), busy


def run(prompt, arm):
    body = {"model": "GLM-5.3-Flash", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": TOKENS, "ignore_eos": True}
    if arm is not None:
        body.update(BODY)
    d0, a0, busy = counters()
    if busy:
        sys.exit("endpoint not idle; acceptance counters would be contaminated")
    start = time.monotonic()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    resp = json.load(urllib.request.urlopen(req, timeout=600))
    elapsed = time.monotonic() - start
    d1, a1, _ = counters()
    drafts, accepted = d1 - d0, a1 - a0
    return {"arm": LABEL if arm is not None else "server-default",
            "completion_tokens": resp["usage"]["completion_tokens"],
            "drafts": drafts, "accepted": accepted,
            "accept_len": round(1 + accepted / drafts, 4) if drafts else None,
            "tok_per_s": round(resp["usage"]["completion_tokens"] / elapsed, 2)}


rows = []
with open(OUT, "a") as f:
    for r in range(ROUNDS):
        for prompt in PROMPTS:
            arms = [True, None] if r % 2 == 0 else [None, True]
            for arm in arms:
                row = run(prompt, arm) | {"round": r, "prompt": PROMPTS.index(prompt)}
                rows.append(row)
                f.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
for arm in (LABEL, "server-default"):
    sel = [x for x in rows if x["arm"] == arm]
    d = sum(x["drafts"] for x in sel); a = sum(x["accepted"] for x in sel)
    t = sum(x["completion_tokens"] for x in sel)
    print(f"SUMMARY arm={arm} requests={len(sel)} accept_len={1 + a / d:.4f} "
          f"mean_tok_per_s={sum(x['tok_per_s'] for x in sel) / len(sel):.2f} tokens={t:.0f}", flush=True)
