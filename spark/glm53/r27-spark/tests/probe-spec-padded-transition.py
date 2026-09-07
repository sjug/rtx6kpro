#!/usr/bin/env python3
"""Exercise uniform-capture to padded-replay transitions under MTP and watch
for persistent speculative acceptance collapse (vLLM PR 667).

Each phase runs ``rounds`` rounds of ``concurrency`` simultaneous greedy chat
completions. Speculative counters are read from ``/metrics`` before and after
every phase, so the acceptance length is the server's own accounting for that
phase alone. Coverage is PREDICTED from the launch geometry, not observed: a phase whose
decode token count equals a captured size and a whole number of spec windows
should replay a uniform graph; otherwise it should pad to the next captured
size and replay that graph with padded request rows, the transition PR 667
fixes. The receipt labels this as predicted coverage; runtime confirmation
needs engine-side evidence the server does not expose.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import json
import re
import time
import urllib.request
from pathlib import Path

PROMPTS = [
    "List 60 distinct European cities, one per line, each followed by a short "
    "clause about a landmark there. Do not stop early.",
    "Write a detailed, numbered 40-step procedure for assembling a bicycle from "
    "parts, with one sentence per step.",
    "Explain TCP congestion control in a long essay of at least 500 words, "
    "covering slow start, congestion avoidance, fast retransmit, and fast recovery.",
    "Produce a table of the first 50 prime numbers with a one-line fact about "
    "each number, one row per line.",
]

COUNTER_RE = re.compile(
    r"^vllm:spec_decode_num_(drafts|draft_tokens|accepted_tokens)_total(\{[^}]*\})? (\S+)$"
)


def read_counters(base_url: str) -> dict[str, float]:
    with urllib.request.urlopen(base_url + "/metrics", timeout=30) as response:
        raw = response.read().decode()
    values: dict[str, float] = {}
    for line in raw.splitlines():
        match = COUNTER_RE.match(line)
        if match:
            values[match.group(1)] = values.get(match.group(1), 0.0) + float(match.group(3))
    return values


def chat(base_url: str, model: str, prompt: str, max_tokens: int, timeout: int) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    request = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    finished = time.monotonic()
    choice = payload["choices"][0]
    return {
        "start": started,
        "end": finished,
        "completion_tokens": payload["usage"]["completion_tokens"],
        "finish_reason": choice["finish_reason"],
        "content_sha": __import__("hashlib").sha256(
            (choice["message"].get("content") or "").encode()
        ).hexdigest()[:16],
    }


def coverage(concurrency: int, spec_tokens: int, capture_sizes: list[int]) -> dict:
    window = spec_tokens + 1
    tokens = concurrency * window
    larger = [size for size in capture_sizes if size >= tokens]
    if not larger:
        return {"tokens": tokens, "graph": None, "mode": "eager (above max capture size)"}
    graph = min(larger)
    if graph == tokens:
        return {"tokens": tokens, "graph": graph, "mode": "uniform"}
    return {
        "tokens": tokens,
        "graph": graph,
        "mode": f"padded ({graph // window - concurrency} padded request rows)",
    }


def max_overlap(samples: list[dict]) -> int:
    events = sorted(
        [(s["start"], 1) for s in samples] + [(s["end"], -1) for s in samples]
    )
    current = best = 0
    for _, delta in events:
        current += delta
        best = max(best, current)
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://dusty:8000")
    parser.add_argument("--model", default="Qwen3.8-Flash-Next-NVFP4-4p89")
    parser.add_argument("--phases", default="1,3,1,2,4,1")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--spec-tokens", type=int, default=3)
    parser.add_argument("--capture-sizes", default="1,2,4,8,16,24,32")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--collapse-threshold", type=float, default=1.2)
    parser.add_argument("--receipt-file", type=Path, required=True)
    args = parser.parse_args()

    phases = [int(value) for value in args.phases.split(",")]
    capture_sizes = sorted(int(value) for value in args.capture_sizes.split(","))
    args.receipt_file.parent.mkdir(parents=True, exist_ok=True)
    args.receipt_file.write_text("", encoding="utf-8")

    def emit(record: dict) -> None:
        with args.receipt_file.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")

    results = []
    padded_seen = False
    for index, concurrency in enumerate(phases):
        cover = coverage(concurrency, args.spec_tokens, capture_sizes)
        padded_seen |= cover["mode"].startswith("padded")  # predicted, see coverage_basis
        before = read_counters(args.base_url)
        samples: list[dict] = []
        for round_index in range(args.rounds):
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [
                    pool.submit(
                        chat,
                        args.base_url,
                        args.model,
                        PROMPTS[(round_index * concurrency + slot) % len(PROMPTS)],
                        args.max_tokens,
                        args.timeout,
                    )
                    for slot in range(concurrency)
                ]
                samples.extend(future.result() for future in futures)
        after = read_counters(args.base_url)
        drafts = after.get("drafts", 0.0) - before.get("drafts", 0.0)
        accepted = after.get("accepted_tokens", 0.0) - before.get("accepted_tokens", 0.0)
        draft_tokens = after.get("draft_tokens", 0.0) - before.get("draft_tokens", 0.0)
        acceptance_length = 1.0 + accepted / drafts if drafts else None
        record = {
            "kind": "phase",
            "index": index,
            "concurrency": concurrency,
            "coverage": cover,
            "rounds": args.rounds,
            "requests": len(samples),
            "max_overlap": max_overlap(samples),
            "completion_tokens": sum(s["completion_tokens"] for s in samples),
            "finish_reasons": sorted({s["finish_reason"] for s in samples}),
            "drafts": drafts,
            "draft_tokens": draft_tokens,
            "accepted_tokens": accepted,
            "acceptance_length": acceptance_length,
            "timestamp": datetime.datetime.now().astimezone().isoformat(),
        }
        results.append(record)
        emit(record)
        print(
            f"phase {index} c{concurrency}: {cover['mode']} tokens={cover['tokens']} "
            f"graph={cover['graph']} overlap={record['max_overlap']} "
            f"acceptance_length={acceptance_length}",
            flush=True,
        )

    lengths = [r["acceptance_length"] for r in results if r["acceptance_length"] is not None]
    verdict: dict = {"kind": "summary", "phases": phases, "padded_transition_predicted": padded_seen, "coverage_basis": "predicted from concurrency, spec window, and capture sizes; not confirmed from engine runtime evidence"}
    if not lengths:
        verdict["valid"] = False
        verdict["failure"] = "no speculative counters observed"
    else:
        collapsed = [r for r in results if r["acceptance_length"] is not None and r["acceptance_length"] < args.collapse_threshold]
        overlap_ok = all(r["max_overlap"] == r["concurrency"] for r in results)
        verdict["valid"] = not collapsed and overlap_ok and padded_seen
        verdict["collapsed_phases"] = [r["index"] for r in collapsed]
        verdict["overlap_ok"] = overlap_ok
        verdict["min_acceptance_length"] = min(lengths)
        verdict["max_acceptance_length"] = max(lengths)
        verdict["final_phase_acceptance_length"] = lengths[-1]
    emit(verdict)
    print(json.dumps(verdict, sort_keys=True), flush=True)
    if not verdict["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
