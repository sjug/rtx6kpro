#!/usr/bin/env python3
"""Focused decode-rate cells for env A/B: cc in {1,2,4}, 300-token ignore_eos
generations at temp 0, 3 repeats per cell, report best-of aggregate tok/s.
Same-boot comparisons only; not a substitute for the full bench harness."""
import argparse, concurrent.futures, json, time, urllib.request

def one(url, model, tokens, seed):
    body = {"model": model, "messages": [{"role": "user", "content":
            f"Task {seed}: write a long, detailed essay about distributed systems."}],
            "max_tokens": tokens, "temperature": 0, "ignore_eos": True}
    req = urllib.request.Request(url + "/v1/chat/completions",
                                 json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=600))
    return d["usage"]["completion_tokens"], time.time() - t0

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--label", required=True)
ap.add_argument("--tokens", type=int, default=300)
a = ap.parse_args()
model = json.load(urllib.request.urlopen(a.url + "/v1/models", timeout=30))["data"][0]["id"]

one(a.url, model, 64, 999)  # warmup
results = {}
for cc in (1, 2, 4):
    best = 0.0
    for rep in range(3):
        with concurrent.futures.ThreadPoolExecutor(cc) as ex:
            t0 = time.time()
            outs = list(ex.map(lambda i: one(a.url, model, a.tokens, cc*100+rep*10+i), range(cc)))
        wall = time.time() - t0
        agg = sum(o[0] for o in outs) / wall
        best = max(best, agg)
    results[f"cc{cc}"] = round(best, 1)
    print(f"[{a.label}] cc{cc}: {best:.1f} tok/s aggregate "
          f"({best/cc:.1f}/user)", flush=True)
print(json.dumps({a.label: results}))
