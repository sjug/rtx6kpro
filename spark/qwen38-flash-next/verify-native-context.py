#!/usr/bin/env python3
"""Verify exact-length Qwen3.8 context admission with a retrieval needle."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path


def post_json(url: str, payload: dict, timeout: int) -> tuple[dict, float]:
    body = json.dumps(payload, separators=(",", ":")).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
        status = response.status
    elapsed = time.monotonic() - started
    if status != 200 or result.get("error"):
        raise RuntimeError(f"request failed: status={status} body={result}")
    return result, elapsed


def resolve_model(base_url: str, requested: str | None, timeout: int) -> str:
    if requested:
        return requested
    request = urllib.request.Request(f"{base_url.rstrip('/')}/v1/models")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    models = [entry["id"] for entry in result.get("data", [])]
    if len(models) != 1:
        raise AssertionError(
            f"--model is required when the endpoint advertises {len(models)} models: "
            f"{models}"
        )
    return models[0]


def emit(receipt: dict, receipt_file: Path) -> None:
    line = json.dumps(receipt, sort_keys=True)
    print(line, flush=True)
    with receipt_file.open("a", encoding="utf-8") as output:
        output.write(line + "\n")


def tokenize(base_url: str, model: str, text: str, timeout: int) -> list[int]:
    result, _ = post_json(
        f"{base_url.rstrip('/')}/tokenize",
        {"model": model, "prompt": text, "add_special_tokens": False},
        timeout,
    )
    tokens = result.get("tokens") or []
    if not tokens:
        raise AssertionError(f"tokenizer returned no tokens for {text!r}: {result}")
    return tokens


def tokenize_chat(
    base_url: str, model: str, messages: list[dict], timeout: int
) -> list[int]:
    result, _ = post_json(
        f"{base_url.rstrip('/')}/tokenize",
        {
            "model": model,
            "messages": messages,
            "chat_template_kwargs": {"enable_thinking": False},
            "add_generation_prompt": True,
        },
        timeout,
    )
    tokens = result.get("tokens") or []
    if not tokens:
        raise AssertionError(f"chat tokenizer returned no tokens: {result}")
    return tokens


def exact_needle_prompt(
    base_url: str,
    model: str,
    length: int,
    needle: str,
    timeout: int,
) -> tuple[list[dict], int]:
    prefix_text = (
        "This is a long archival retrieval test. Memorize the unique code in "
        f"the next sentence. The unique retrieval code is {needle}.\n\n"
        "Archive begins:\n"
    )
    needle_tokens = tokenize(base_url, model, needle, timeout)
    suffix_text = (
        "\nArchive ends. What was the unique retrieval code stated before the "
        "archive? Reply with only that code."
    )
    base_messages = [{"role": "user", "content": prefix_text + suffix_text}]
    base_tokens = tokenize_chat(base_url, model, base_messages, timeout)
    filler_count = length - len(base_tokens)
    if filler_count < 0:
        raise ValueError(
            f"requested length {length} is shorter than the probe framing "
            f"({len(base_tokens)} tokens)"
        )
    for _ in range(4):
        messages = [
            {
                "role": "user",
                "content": prefix_text + " filler" * filler_count + suffix_text,
            }
        ]
        prompt_tokens = tokenize_chat(base_url, model, messages, timeout)
        delta = length - len(prompt_tokens)
        if delta == 0:
            needle_token_offset = next(
                (
                    offset
                    for offset in range(len(prompt_tokens) - len(needle_tokens) + 1)
                    if prompt_tokens[offset : offset + len(needle_tokens)]
                    == needle_tokens
                ),
                None,
            )
            if needle_token_offset is None:
                raise AssertionError("needle token sequence is absent from the prompt")
            return messages, needle_token_offset
        filler_count += delta
        if filler_count < 0:
            break
    raise AssertionError(
        f"could not construct exact {length}-token chat prompt; "
        f"last count was {len(prompt_tokens)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--model",
        help="served model name; auto-detected only when exactly one is advertised",
    )
    parser.add_argument("--needle", default="739184")
    parser.add_argument("--lengths", type=int, nargs="*", default=[131072, 262000])
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--receipt-file", type=Path, required=True)
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")
    model = resolve_model(base_url, args.model, min(args.timeout, 30))
    args.receipt_file.parent.mkdir(parents=True, exist_ok=True)
    args.receipt_file.write_text("", encoding="utf-8")

    print("SHORT_START", flush=True)
    short, short_elapsed = post_json(
        f"{base_url}/v1/chat/completions",
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "What is 5 + 6? Give the final answer as one numeral.",
                }
            ],
            "max_tokens": 128,
            "temperature": 0,
        },
        args.timeout,
    )
    choice = short["choices"][0]
    message = choice["message"]
    short_receipt = {
        "kind": "short",
        "elapsed_seconds": round(short_elapsed, 3),
        "finish_reason": choice.get("finish_reason"),
        "content": message.get("content"),
        "reasoning_content": message.get("reasoning_content")
        or message.get("reasoning"),
        "usage": short.get("usage"),
    }
    if choice.get("finish_reason") != "stop" or "11" not in (
        message.get("content") or ""
    ):
        raise AssertionError(f"short correctness failure: {short_receipt}")
    emit(short_receipt, args.receipt_file)

    for length in args.lengths:
        print(f"CONTEXT_START length={length}", flush=True)
        messages, needle_token_offset = exact_needle_prompt(
            base_url, model, length, args.needle, args.timeout
        )
        result, elapsed = post_json(
            f"{base_url}/v1/chat/completions",
            {
                "model": model,
                "messages": messages,
                "chat_template_kwargs": {"enable_thinking": False},
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "return_token_ids": True,
            },
            args.timeout,
        )
        choice = result["choices"][0]
        output_ids = choice.get("token_ids") or []
        message = choice.get("message") or {}
        text = message.get("content") or ""
        receipt = {
            "kind": "context_needle",
            "requested_input_tokens": length,
            "elapsed_seconds": round(elapsed, 3),
            "needle": args.needle,
            "needle_token_offset": needle_token_offset,
            "output_ids": output_ids,
            "text": text,
            "finish_reason": choice.get("finish_reason"),
            "reasoning": message.get("reasoning_content")
            or message.get("reasoning")
            or "",
            "prompt_tokens": result.get("usage", {}).get("prompt_tokens"),
            "usage": result.get("usage", {}),
            "cached_tokens": (result.get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
            "completion_tokens": result.get("usage", {}).get("completion_tokens"),
        }
        failure = None
        if receipt["prompt_tokens"] != length:
            failure = "prompt_token_mismatch"
        elif not output_ids or not text.strip():
            failure = "empty_generation"
        elif 0 in output_ids or text.strip() == "!":
            failure = "collapsed_token_0_generation"
        elif args.needle not in text:
            failure = "needle_retrieval_failed"
        receipt["valid"] = failure is None
        if failure is not None:
            receipt["failure"] = failure
        emit(receipt, args.receipt_file)
        if failure is not None:
            raise AssertionError(f"{failure}: {receipt}")

    emit(
        {
            "kind": "summary",
            "lengths": args.lengths,
            "model": model,
            "needle": args.needle,
            "valid": True,
        },
        args.receipt_file,
    )
    print("NATIVE_CONTEXT_NEEDLE_VERIFY_PASS", flush=True)


if __name__ == "__main__":
    main()
