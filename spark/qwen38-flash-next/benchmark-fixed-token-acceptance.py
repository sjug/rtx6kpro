#!/usr/bin/env python3
"""Compare MTP acceptance using one frozen, pre-rendered token corpus."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path


PROMPTS = [
    """Design a thread-safe bounded work queue in Python. Explain the invariants,
shutdown behavior, timeout handling, and why condition predicates need loops.
Include concise pseudocode and one worked producer-consumer example.""",
    """A distributed inference service has four workers. Each request needs six
units of verifier work and each worker completes nine units per second. Analyze
the steady-state capacity, identify the bottleneck, and show every calculation.""",
    """Review a CUDA kernel optimization that removes initialization of padded
activation-scale storage. Describe the correctness proof that would be required,
the poison tests to run, and the performance evidence needed before deployment.""",
    """Write a detailed incident analysis for a service that remained reachable
over HTTP while its inference engine stopped making progress. Separate symptoms,
root cause, contributing factors, mitigations, and durable corrective actions.""",
]


def get_json(url: str, timeout: int = 30) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def post_json(url: str, payload: dict, timeout: int = 1800) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    if result.get("error"):
        raise AssertionError(result)
    return result


def metrics_text(base_url: str) -> str:
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=30) as response:
        return response.read().decode()


def metric_sum(text: str, name: str) -> float:
    pattern = re.compile(rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([^\s]+)$")
    return sum(
        float(match.group(1))
        for line in text.splitlines()
        if (match := pattern.match(line))
    )


def metric_positions(text: str) -> dict[str, float]:
    pattern = re.compile(
        r'^vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*'
        r'position="([^"]+)"[^}]*\}\s+([^\s]+)$'
    )
    return {
        match.group(1): float(match.group(2))
        for line in text.splitlines()
        if (match := pattern.match(line))
    }


def resolve_model(base_url: str, requested: str | None) -> str:
    if requested:
        return requested
    models = [entry["id"] for entry in get_json(f"{base_url}/v1/models")["data"]]
    if len(models) != 1:
        raise AssertionError(f"expected one advertised model, got {models}")
    return models[0]


def build_corpus(base_url: str, model: str) -> dict:
    entries = []
    for index, prompt in enumerate(PROMPTS):
        result = post_json(
            f"{base_url}/tokenize",
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "chat_template_kwargs": {"enable_thinking": True},
                "add_generation_prompt": True,
            },
        )
        input_ids = result.get("tokens") or []
        if not input_ids:
            raise AssertionError(f"prompt {index} tokenized to an empty sequence")
        entries.append(
            {
                "id": f"prompt-{index}",
                "input_ids": input_ids,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            }
        )
    return {"format": 1, "prompts": entries}


def corpus_digest(corpus: dict) -> str:
    encoded = json.dumps(corpus, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_request(
    base_url: str,
    model: str,
    entry: dict,
    max_tokens: int,
    wave: int,
) -> dict:
    started = time.monotonic()
    result = post_json(
        f"{base_url}/v1/completions",
        {
            "model": model,
            "prompt": entry["input_ids"],
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": 0,
            "ignore_eos": True,
            "return_token_ids": True,
        },
    )
    elapsed = time.monotonic() - started
    choice = result["choices"][0]
    output_ids = choice.get("token_ids") or []
    if len(output_ids) != max_tokens:
        raise AssertionError(
            f"{entry['id']} wave {wave}: expected {max_tokens} output tokens, "
            f"got {len(output_ids)}"
        )
    return {
        "elapsed_seconds": round(elapsed, 6),
        "output_sha256": hashlib.sha256(
            json.dumps(output_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "output_tokens": len(output_ids),
        "prompt_id": entry["id"],
        "wave": wave,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model")
    parser.add_argument("--corpus-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--waves", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()
    if args.waves < 1 or args.max_tokens < 1:
        parser.error("--waves and --max-tokens must be positive")

    base_url = args.base_url.rstrip("/")
    model = resolve_model(base_url, args.model)
    if args.corpus_file.exists():
        corpus = json.loads(args.corpus_file.read_text())
    else:
        corpus = build_corpus(base_url, model)
        args.corpus_file.parent.mkdir(parents=True, exist_ok=True)
        args.corpus_file.write_text(json.dumps(corpus, indent=2) + "\n")
    entries = corpus.get("prompts") or []
    if len(entries) != 4 or any(not entry.get("input_ids") for entry in entries):
        raise AssertionError("the frozen corpus must contain four tokenized prompts")

    # Compile and populate any lazy path before taking the counter baseline.
    post_json(
        f"{base_url}/v1/completions",
        {
            "model": model,
            "prompt": entries[0]["input_ids"],
            "max_tokens": 16,
            "temperature": 0,
            "seed": 0,
            "ignore_eos": True,
        },
    )
    before_text = metrics_text(base_url)
    results = []
    for wave in range(args.waves):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(
                    run_request,
                    base_url,
                    model,
                    entry,
                    args.max_tokens,
                    wave,
                )
                for entry in entries
            ]
            results.extend(future.result() for future in futures)
    after_text = metrics_text(base_url)

    names = {
        "drafts": "vllm:spec_decode_num_drafts_total",
        "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
        "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
    }
    deltas = {
        key: metric_sum(after_text, name) - metric_sum(before_text, name)
        for key, name in names.items()
    }
    before_pos = metric_positions(before_text)
    after_pos = metric_positions(after_text)
    deltas["accepted_per_position"] = {
        position: after_pos.get(position, 0.0) - before_pos.get(position, 0.0)
        for position in sorted(set(before_pos) | set(after_pos))
    }
    output_tokens = sum(item["output_tokens"] for item in results)
    engine_steps = deltas["drafts"] + max(
        0.0, output_tokens - (deltas["accepted_tokens"] + deltas["drafts"])
    )
    hashes_by_prompt: dict[str, set[str]] = {}
    for item in results:
        hashes_by_prompt.setdefault(item["prompt_id"], set()).add(
            item["output_sha256"]
        )
    output_repeatable = all(len(hashes) == 1 for hashes in hashes_by_prompt.values())

    receipt = {
        "accept_length": output_tokens / engine_steps,
        "corpus_sha256": corpus_digest(corpus),
        "counter_deltas": deltas,
        "engine_steps": engine_steps,
        "max_tokens": args.max_tokens,
        "model": model,
        "output_repeatable_across_waves": output_repeatable,
        "output_sha256_by_prompt": {
            prompt: sorted(hashes) for prompt, hashes in hashes_by_prompt.items()
        },
        "output_tokens": output_tokens,
        "requests": results,
        "valid": True,
        "waves": args.waves,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
