#!/usr/bin/env python3
"""Laguna qualification probes for the II image.

Probe A: max_tokens=8 chat (q_len != 0 mod K+1 truncation step).
Probe B: prompt sized to prefill-chunk + small tail (boundary step).
Both were engine-killers before the q_len guard. Gate 1: engine alive, both
complete. Gate 2 (run once per graph mode): outputs at temp 0 recorded to
JSON; FULL_AND_PIECEWISE output must equal PIECEWISE output token-exactly.
Usage: laguna-ii-qual.py --url http://dusty:8000 --out laguna-ii-<mode>.json
"""
import argparse, json, urllib.request, sys

def post(url, path, obj, timeout=900):
    req = urllib.request.Request(url + path, json.dumps(obj).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def toks(url, model, messages):
    return post(url, "/tokenize", {"model": model, "messages": messages,
                                   "add_generation_prompt": True})["count"]

def run(url, model, messages, max_tokens):
    d = post(url, "/v1/chat/completions",
             {"model": model, "messages": messages,
              "max_tokens": max_tokens, "temperature": 0})
    c = d["choices"][0]
    return {"content": c["message"].get("content"),
            "reasoning": c["message"].get("reasoning")
                         or c["message"].get("reasoning_content") or "",
            "finish": c["finish_reason"],
            "completion_tokens": d["usage"]["completion_tokens"],
            "prompt_tokens": d["usage"]["prompt_tokens"]}

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--target-b", type=int, default=8200)
a = ap.parse_args()

model = post(a.url, "/v1/models", None) if False else None
models = json.load(urllib.request.urlopen(a.url + "/v1/models", timeout=30))
model = models["data"][0]["id"]
print("model:", model, flush=True)

results = {}
msgs_a = [{"role": "user", "content": "List the prime numbers below 30 in ascending order, comma separated."}]
results["repro_a"] = run(a.url, model, msgs_a, 8)
print("repro A:", results["repro_a"], flush=True)

unit = "Component %d emits a heartbeat every third cycle and logs the drift. "
tail_q = "How many components are described above? Reply with just the number."
n = 500
while True:
    msgs_b = [{"role": "user", "content":
               "".join(unit % i for i in range(n)) + tail_q}]
    c = toks(a.url, model, msgs_b)
    if c >= a.target_b or n > 700: break
    n += max(1, (a.target_b - c) // 17)
print(f"repro B prompt: {c} tokens ({n} components)", flush=True)
results["repro_b"] = run(a.url, model, msgs_b, 400)
rb = dict(results["repro_b"]); rb["reasoning"] = rb["reasoning"][:80] + "..."
print("repro B:", rb, flush=True)

with open(a.out, "w") as f:
    json.dump(results, f, indent=1)
print("ENGINE ALIVE, both reproducers complete ->", a.out, flush=True)
