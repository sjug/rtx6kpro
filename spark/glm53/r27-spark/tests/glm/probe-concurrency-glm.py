#!/usr/bin/env python3
"""GLM concurrency reproducers under the unchanged (auto) checkpoint policy:
identical burst vs distinct, fresh / multi-turn extension / verbatim repeat,
and the repeat head-of-line probe. Mirrors the Qwen r27 probes."""
import concurrent.futures, json, os, threading, time, urllib.request, uuid
base = os.environ.get('BASE_URL', 'http://sparky:8000'); model = os.environ.get('MODEL', 'GLM-5.3-Flash')

def chat(messages, max_tokens=200):
    body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t = time.monotonic()
    with urllib.request.urlopen(req, timeout=900) as r: p = json.load(r)
    return time.monotonic() - t, p["usage"]["completion_tokens"], p["choices"][0]["message"].get("content") or ""

def run(label, msgs_list):
    t = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(len(msgs_list)) as ex: res = list(ex.map(chat, msgs_list))
    wall = time.monotonic() - t; toks = sum(c for _, c, _ in res)
    print(f"{label}: n={len(msgs_list)} wall={wall:.1f}s agg_tok/s={toks/wall:.1f} per_req_s={[round(d,1) for d,_,_ in res]}", flush=True)
    return res

print("== identical burst vs distinct ==", flush=True)
distinct = [[{"role": "user", "content": f"Write a long essay number {i} about the history of railway signalling in a different country for each number."}] for i in range(4)]
same = [[{"role": "user", "content": "Write a long essay about the history of railway signalling in Switzerland."}]] * 4
run("warm c1", distinct[:1]); run("c4 distinct", distinct); run("c4 identical", same); run("c4 distinct again", distinct)

print("== fresh / extension / verbatim-repeat ==", flush=True)
fresh = lambda: [[{"role": "user", "content": f"Essay {uuid.uuid4().hex}: describe the geology of a volcanic island in detail."}] for _ in range(4)]
a = fresh(); r = run("A: 4 fresh distinct", a)
ext = [m + [{"role": "assistant", "content": c}, {"role": "user", "content": "Continue with the island's climate."}] for m, (_, _, c) in zip(a, r)]
run("B: 4 extensions of A (multi-turn)", ext); run("C: 4 fresh distinct after B", fresh()); run("D: repeat A verbatim", a); run("E: 4 fresh distinct after D", fresh())

print("== repeat head-of-line ==", flush=True)
def stream(label, prompt, max_tokens, out):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True, "stream": True, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.monotonic(); ttft = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            if not line.startswith(b"data:"): continue
            data = line[5:].strip()
            if data == b"[DONE]": continue
            chunk = json.loads(data)
            if "error" in chunk: raise RuntimeError(chunk["error"])
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                if ttft is None and (delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")):
                    ttft = time.monotonic() - t0
    if ttft is None: raise RuntimeError(f"{label}: no generated text in stream")
    out[label] = (ttft, time.monotonic() - t0)
seed = f"Essay {uuid.uuid4().hex}: describe the geology of a volcanic island in detail."
o = {}; stream("seed", seed, 60, o)
res = {}; th = []
def start(label, prompt, n, delay):
    time.sleep(delay); t = threading.Thread(target=stream, args=(label, prompt, n, res)); t.start(); th.append(t)
for i in range(2): start(f"long{i}", f"Essay {uuid.uuid4().hex}: history of canals.", 600, 0)
start("repeat", seed, 60, 2.0)
for i in range(2): start(f"fresh{i}", f"Essay {uuid.uuid4().hex}: history of bridges.", 60, 3.0)
time.sleep(4); [t.join() for t in th]
assert set(res) == {"long0", "long1", "repeat", "fresh0", "fresh1"}, f"incomplete streaming probe: {res}"
for k, (ttft, tot) in res.items(): print(f"{k:8s} ttft={ttft:6.1f}s total={tot:6.1f}s", flush=True)
