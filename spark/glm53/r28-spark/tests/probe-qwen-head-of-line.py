#!/usr/bin/env python3
"""R27 request ordering, with nonempty-content TTFT and propagated failures.

This corrects the old substring-based SSE timestamp, so compare the matched
R28 policy arms directly; historical R27 timings retain their old method.
"""
if not __debug__:
    raise RuntimeError("R28 verification requires Python assertions enabled")

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import time
import urllib.request
import uuid
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--policy", required=True, choices=("aligned", "auto"))
p.add_argument("--receipt-file", type=Path, required=True)
a = p.parse_args()
model = "Qwen3.8-Flash-Next-NVFP4-4p89"


def chat(label, prompt, budget):
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": budget, "temperature": 0, "ignore_eos": True,
               "stream": True, "stream_options": {"include_usage": True},
               "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request("http://dusty:8000/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    started = time.monotonic()
    first = None
    text = ""
    usage = None
    finish = None
    with urllib.request.urlopen(req, timeout=120) as response:
        for line in response:
            if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                continue
            event = json.loads(line[5:])
            if event.get("error"):
                raise RuntimeError(event["error"])
            choices = event.get("choices") or []
            if choices:
                content = choices[0].get("delta", {}).get("content") or ""
                if content:
                    if first is None:
                        first = time.monotonic() - started
                    text += content
                finish = choices[0].get("finish_reason") or finish
            usage = event.get("usage") or usage
    if first is None or not usage or not text.strip() or finish != "length":
        raise RuntimeError(f"Invalid stream {label}: {first=} {usage=} {finish=}")
    if usage["completion_tokens"] != budget:
        raise RuntimeError(f"Incomplete stream {label}: {usage}")
    if text.count("!") > len(text) / 4:
        raise RuntimeError(f"Collapsed output: {label}")
    return {"label": label, "ttft_s": first, "total_s": time.monotonic() - started,
            "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "content_preview": text[:160], "usage": usage, "finish_reason": finish}


seed = f"Essay {uuid.uuid4().hex}: describe the geology of a volcanic island in detail."
a.receipt_file.parent.mkdir(parents=True, exist_ok=True)
with a.receipt_file.open("x") as output:
    def emit(result):
        result.update(policy=a.policy, timestamp=datetime.datetime.now().astimezone().isoformat())
        line = json.dumps(result, sort_keys=True)
        print(line, flush=True)
        output.write(line + "\n")
        output.flush()

    emit(chat("seed", seed, 60))
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(chat, f"long{i}", f"Essay {uuid.uuid4().hex}: history of canals.", 600) for i in range(2)]
        time.sleep(2)
        futures.append(pool.submit(chat, "repeat", seed, 60))
        for i in range(2):
            time.sleep(3)
            futures.append(pool.submit(chat, f"fresh{i}", f"Essay {uuid.uuid4().hex}: history of bridges.", 60))
        results = [f.result() for f in futures]
    for result in results:
        emit(result)
    emit({"kind": "summary", "streams_complete": True,
          "fresh_max_ttft_s": max(r["ttft_s"] for r in results if r["label"].startswith("fresh")),
          "repeat_ttft_s": next(r["ttft_s"] for r in results if r["label"] == "repeat")})
