#!/usr/bin/env python3
"""Fresh-prefill timing probe for the GLM NCCL transport confirmation.

Each request starts with a unique nonce, so no prefix-cache block can match.
Cache hits are also verified from the Prometheus counters around every request.
"""
import json
import os
import re
import sys
import time
import urllib.request
import uuid

BASE = os.environ.get("BASE_URL", "http://sparky:8000")
MODEL = os.environ.get("MODEL", "GLM-5.3-Flash")
OUT = sys.argv[1]
PLAN = [int(x) for x in (sys.argv[2:] or ["8192", "8192", "131072", "131072"])]


def metric(name):
    text = urllib.request.urlopen(f"{BASE}/metrics", timeout=30).read().decode()
    vals = re.findall(r"^vllm:" + name + r"\{[^\n]*\} ([\d.e+-]+)$", text, re.M)
    return sum(float(v) for v in vals)


def request(target):
    nonce = uuid.uuid4().hex
    # " filler" is one token; the header and nonce add a few more.
    prompt = f"Session {nonce}. Reply with OK.\n" + " filler" * (target - 24)
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": 1,
                       "temperature": 0}).encode()
    req = urllib.request.Request(f"{BASE}/v1/completions", body,
                                 {"Content-Type": "application/json"})
    hits0 = metric("prefix_cache_hits_total")
    start = time.monotonic()
    resp = json.load(urllib.request.urlopen(req, timeout=1800))
    elapsed = time.monotonic() - start
    hits = metric("prefix_cache_hits_total") - hits0
    tokens = resp["usage"]["prompt_tokens"]
    return {"target": target, "prompt_tokens": tokens, "elapsed_seconds": round(elapsed, 2),
            "tok_per_s": round(tokens / elapsed, 1), "cache_hits": hits,
            "finish_reason": resp["choices"][0]["finish_reason"],
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


with open(OUT, "a") as f:
    for target in PLAN:
        row = request(target)
        print(json.dumps(row), flush=True)
        f.write(json.dumps(row) + "\n")
        if row["cache_hits"]:
            sys.exit(f"prefix cache hit on a fresh request: {row}")
