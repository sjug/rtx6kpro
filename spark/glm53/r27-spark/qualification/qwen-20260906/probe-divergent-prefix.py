#!/usr/bin/env python3
"""Shared prefix, different suffix: does the second request reuse the first's
prefix blocks? Reads the server prefix-cache counters around each request."""
import json, re, sys, time, urllib.request
base, model, label = sys.argv[1], sys.argv[2], sys.argv[3]
def counters():
    raw = urllib.request.urlopen(base + "/metrics", timeout=30).read().decode()
    out = {}
    for line in raw.splitlines():
        m = re.match(r"^vllm:prefix_cache_(queries|hits)_total(\{[^}]*\})? (\S+)$", line)
        if m: out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
    return out
words = ["alpha","bravo","charlie","delta","echo","foxtrot","golf","hotel","india","juliet","kilo","lima"]
shared = "Shared system context for the divergent-prefix probe. " + " ".join(words[(i*7)%12] + str(i%97) for i in range(24000))
tails = [" Question A: reply with the single word DONE.", " Question B: reply with the word FINISHED only.", " Question C: reply with OK only."]
for i, tail in enumerate(tails):
    before = counters()
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": shared + tail}], "max_tokens": 8, "temperature": 0, "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.monotonic(); ttft = None; usage = None; text = ""
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            ev = json.loads(line[5:])
            if ev.get("usage"): usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {}).get("content")
                if d:
                    if ttft is None: ttft = time.monotonic() - t0
                    text += d
    total = time.monotonic() - t0; after = counters()
    rec = {"label": label, "request": i, "ttft_s": round(ttft, 3) if ttft else None, "total_s": round(total, 3), "prompt_tokens": usage["prompt_tokens"] if usage else None,
           "prefix_queries_delta": after["queries"] - before["queries"], "prefix_hits_delta": after["hits"] - before["hits"], "answer": text.strip()[:40]}
    print(json.dumps(rec), flush=True)
