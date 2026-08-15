#!/usr/bin/env python3
"""Deterministic probe for the vLLM #298 class: a K+1-token prompt scheduled
alongside uniform decode must not replay a FULL decode CUDA graph over prompt
state. Method: hold BG decode streams open, fire exact-width prompts at temp 0,
and inspect every answer for corruption signatures (raw special tokens,
replacement characters, foreign-script runs, pathological repetition) - the
same watchdog class upstream used to validate the fix. Exact-match against a
solo reference is NOT the verdict: DSpark probabilistic drafting makes this
profile nondeterministic even at temp 0 (verified 2026-08-14), so divergence
is expected; corruption signatures are not. Upstream measured ~1/160 corrupt
at C4 on the broken build.

Usage: probe-298-dispatch-race.py --url http://rusty:8000 --model NAME \
         --width 6 --trials 200 [--bg 3]
  width = 1 + speculative depth (DS4 K5 -> 6, GLM MTP3 -> 4).
"""
import argparse, json, threading, time, urllib.request, sys

def post(url, obj, timeout=600):
    req = urllib.request.Request(url + "/v1/chat/completions",
                                 json.dumps(obj).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def tokenize_count(url, model, messages):
    req = urllib.request.Request(url + "/tokenize",
        json.dumps({"model": model, "messages": messages,
                    "add_generation_prompt": True}).encode(),
        {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["count"]

def make_probe_messages(url, model, width, seed):
    # Binary-search filler words until the templated prompt is exactly `width`
    # tokens. Falls back to nearest achievable if template floor > width.
    base = f"q{seed}"
    best = None
    for extra in range(0, 64):
        msgs = [{"role": "user", "content": base + " x" * extra}]
        n = tokenize_count(url, model, msgs)
        if n == width:
            return msgs, n
        if best is None or abs(n - width) < abs(best[1] - width):
            best = (msgs, n)
        if n > width and extra == 0:
            break
    return best

def bg_stream(url, model, stop, idx):
    while not stop.is_set():
        try:
            post(url, {"model": model, "messages": [
                {"role": "user", "content": f"Count slowly from {idx*1000} upward, one number per line."}],
                "max_tokens": 800, "temperature": 0, "ignore_eos": True}, timeout=300)
        except Exception:
            time.sleep(1)

def answer(url, model, msgs, max_tokens):
    d = post(url, {"model": model, "messages": msgs,
                   "max_tokens": max_tokens, "temperature": 0})
    c = d["choices"][0]["message"]
    return (c.get("reasoning_content") or c.get("reasoning") or "") + "|" + (c.get("content") or "")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--bg", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=48)
    a = ap.parse_args()

    msgs, n = make_probe_messages(a.url, a.model, a.width, seed=7)
    print(f"probe prompt: {n} templated tokens (target {a.width})", flush=True)


    stop = threading.Event()
    threads = [threading.Thread(target=bg_stream, args=(a.url, a.model, stop, i),
                                daemon=True) for i in range(a.bg)]
    for t in threads: t.start()
    time.sleep(3)

    import re
    def corruption(text):
        hits = []
        if "\ufffd" in text: hits.append("replacement-char")
        if re.search(r"<\|[a-z_]+\|>", text): hits.append("raw-special-token")
        if re.search(r"[\u4e00-\u9fff\u0400-\u04ff\u0600-\u06ff]{3,}", text):
            hits.append("foreign-script-run")
        words = text.split()
        for n in range(0, max(0, len(words) - 8), 8):
            gram = " ".join(words[n:n+8])
            if len(gram) > 20 and text.count(gram) > 5:
                hits.append("8gram-repetition"); break
        return hits

    corrupt = 0
    try:
        for i in range(a.trials):
            got = answer(a.url, a.model, msgs, a.max_tokens)
            hits = corruption(got)
            if hits:
                corrupt += 1
                print(f"trial {i}: CORRUPTION {hits}\n  got: {got[:160]!r}",
                      flush=True)
            if (i + 1) % 50 == 0:
                print(f"{i+1}/{a.trials} trials, {corrupt} corrupt", flush=True)
    finally:
        stop.set()

    print(f"RESULT: {corrupt}/{a.trials} corruption-signature hits under "
          f"{a.bg}-stream load (width {a.width})", flush=True)
    sys.exit(0 if corrupt == 0 else 2)

if __name__ == "__main__":
    main()
