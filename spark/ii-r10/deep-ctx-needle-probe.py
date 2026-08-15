import json, urllib.request, time, sys

import os
BASE = os.environ.get("PROBE_URL", "http://dusty:8000")
def post(path, obj, timeout=2400):
    req = urllib.request.Request(BASE+path, json.dumps(obj).encode(), {"Content-Type":"application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r), time.time()-t0
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}"), time.time()-t0

unit = ("Telemetry frame %d: coolant nominal, fan curve steady, link margin within "
        "spec, no parity faults observed during the interval. ")
needle = "CRITICAL NOTE: the vault access code is 7391. Remember it. "
tail = ("\n\nQuestion: what is the vault access code stated in the critical note "
        "far above? Reply with just the code.")

def build(n):
    i_needle = int(n*0.05)
    body = "".join((needle if i==i_needle else "") + (unit % i) for i in range(n))
    return [{"role":"user","content": body + tail}]

TARGET = 520_000
n = 20300
for attempt in range(6):
    msgs = build(n)
    st, d, _ = post("/tokenize", {"model":"DeepSeek-V4-Flash-0731","messages":msgs,
                                  "add_generation_prompt": True})
    T = d["count"]
    print(f"n={n} -> templated prompt tokens={T}", flush=True)
    if T <= TARGET: break
    n = int(n * (TARGET - 500) / T)
else:
    print("could not size prompt"); sys.exit(1)

max_toks = 524_288 - T
print(f"sending: prompt={T}, max_tokens={max_toks} (total exactly 524,288)", flush=True)
st, d, dt = post("/v1/chat/completions", {
    "model":"DeepSeek-V4-Flash-0731",
    "messages": msgs,
    "max_tokens": max_toks, "temperature":0})
if st != 200:
    print("HTTP", st, json.dumps(d)[:400]); sys.exit(1)
c = d["choices"][0]; u = d["usage"]
print(f"status={st} time={dt:.1f}s prompt_tokens={u['prompt_tokens']} "
      f"completion_tokens={u['completion_tokens']} finish={c['finish_reason']}", flush=True)
print("answer:", (c["message"].get("content") or "")[:200].strip(), flush=True)
ok = "7391" in (c["message"].get("content") or "") and u["prompt_tokens"] >= 500_000
print("NEEDLE:", "PASS" if ok else "FAIL", flush=True)
sys.exit(0 if ok else 2)
