#!/usr/bin/env python3
"""Fail-closed, repeated semantic admission for Qwen3.8 Flash Next serving."""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import struct
import urllib.request
import zlib
from pathlib import Path
from typing import TextIO


THINKING_SAMPLING = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
}


def post_json(url: str, payload: dict, timeout: int = 180) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
        status = response.status
    if status != 200 or result.get("error"):
        raise AssertionError(f"request failed: status={status} body={result}")
    return result


def resolve_model(base_url: str, requested: str | None) -> str:
    if requested:
        return requested
    with urllib.request.urlopen(
        f"{base_url.rstrip('/')}/v1/models", timeout=30
    ) as response:
        result = json.load(response)
    models = [entry["id"] for entry in result.get("data", [])]
    if len(models) != 1:
        raise AssertionError(
            f"--model is required when the endpoint advertises {len(models)} models: "
            f"{models}"
        )
    return models[0]


def emit(receipt: dict, receipt_file: TextIO) -> None:
    line = json.dumps(receipt, sort_keys=True)
    print(line, flush=True)
    receipt_file.write(line + "\n")
    receipt_file.flush()


def thinking_payload(model: str, effort: str, messages: list[dict]) -> dict:
    return {
        "model": model,
        "messages": messages,
        "chat_template_kwargs": {
            "enable_thinking": True,
            "preserve_thinking": True,
        },
        "reasoning_effort": effort,
        "max_tokens": 512,
        **THINKING_SAMPLING,
    }


def reasoning_text(message: dict) -> str:
    return message.get("reasoning_content") or message.get("reasoning") or ""


def check_reasoning_efforts(
    endpoint: str, model: str, repetition: int, receipt_file: TextIO
) -> None:
    prompt = (
        "A shop has 17 boxes with 23 bolts in each box. It uses 58 bolts. "
        "How many bolts remain? Explain briefly, then end with exactly "
        "FINAL: 333"
    )
    for effort in ("low", "medium", "xhigh"):
        result = post_json(
            endpoint,
            thinking_payload(model, effort, [{"role": "user", "content": prompt}]),
        )
        choice = result["choices"][0]
        message = choice["message"]
        reasoning = reasoning_text(message)
        answer = message.get("content") or ""
        if choice.get("finish_reason") != "stop":
            raise AssertionError(f"{effort}: did not stop: {choice}")
        if len(reasoning.strip()) < 20:
            raise AssertionError(f"{effort}: missing meaningful reasoning: {message}")
        if not answer.rstrip().endswith("FINAL: 333"):
            raise AssertionError(f"{effort}: wrong answer: {answer!r}")
        emit(
            {
                "answer": answer,
                "completion_tokens": result.get("usage", {}).get("completion_tokens"),
                "effort": effort,
                "kind": "reasoning",
                "reasoning_chars": len(reasoning),
                "repetition": repetition,
                "valid": True,
            },
            receipt_file,
        )


def check_non_thinking(
    endpoint: str, model: str, repetition: int, receipt_file: TextIO
) -> None:
    result = post_json(
        endpoint,
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Calculate 17 times 23 minus 58. Reply with exactly FINAL: 333"
                    ),
                }
            ],
            "chat_template_kwargs": {"enable_thinking": False},
            "max_tokens": 64,
            "temperature": 0,
        },
    )
    choice = result["choices"][0]
    message = choice["message"]
    answer = message.get("content") or ""
    reasoning = reasoning_text(message)
    if choice.get("finish_reason") != "stop":
        raise AssertionError(f"non-thinking: did not stop: {choice}")
    if reasoning.strip():
        raise AssertionError(f"non-thinking: unexpected reasoning: {reasoning!r}")
    if answer.strip() != "FINAL: 333":
        raise AssertionError(f"non-thinking: wrong answer: {answer!r}")
    emit(
        {
            "answer": answer,
            "kind": "non_thinking",
            "reasoning_chars": 0,
            "repetition": repetition,
            "valid": True,
        },
        receipt_file,
    )


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", binascii.crc32(kind + data) & 0xFFFFFFFF)
    )


def red_blue_png_data_url() -> str:
    width, height = 128, 64
    row = b"\xff\x00\x00" * (width // 2) + b"\x00\x00\xff" * (width // 2)
    pixels = b"".join(b"\x00" + row for _ in range(height))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + png_chunk(b"IDAT", zlib.compress(pixels, level=9))
        + png_chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode()


def check_image(
    endpoint: str, model: str, repetition: int, receipt_file: TextIO
) -> None:
    content = [
        {
            "type": "text",
            "text": (
                "Identify the solid color on the left and the solid color on the "
                "right. End with exactly FINAL: red, blue"
            ),
        },
        {"type": "image_url", "image_url": {"url": red_blue_png_data_url()}},
    ]
    result = post_json(
        endpoint,
        thinking_payload(model, "low", [{"role": "user", "content": content}]),
    )
    choice = result["choices"][0]
    answer = choice["message"].get("content") or ""
    if choice.get("finish_reason") != "stop":
        raise AssertionError(f"image: did not stop: {choice}")
    if not answer.rstrip().endswith("FINAL: red, blue"):
        raise AssertionError(f"image: incorrect answer: {answer!r}")
    emit(
        {
            "answer": answer,
            "kind": "image",
            "repetition": repetition,
            "valid": True,
        },
        receipt_file,
    )


def check_tool_round_trip(
    endpoint: str, model: str, repetition: int, receipt_file: TextIO
) -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "multiply",
                "description": "Multiply two integers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    messages: list[dict] = [
        {
            "role": "user",
            "content": (
                "Use the multiply tool to calculate 17 times 23. After the tool "
                "result, answer with exactly FINAL: 391"
            ),
        }
    ]
    first_payload = thinking_payload(model, "low", messages)
    first_payload["tools"] = tools
    first_payload["tool_choice"] = "required"
    first = post_json(endpoint, first_payload)
    assistant = first["choices"][0]["message"]
    tool_calls = assistant.get("tool_calls") or []
    if len(tool_calls) != 1:
        raise AssertionError(f"tool: expected one call: {first['choices'][0]}")
    call = tool_calls[0]
    if call.get("function", {}).get("name") != "multiply":
        raise AssertionError(f"tool: wrong function: {call}")
    arguments = json.loads(call["function"]["arguments"])
    if arguments != {"a": 17, "b": 23}:
        raise AssertionError(f"tool: wrong arguments: {arguments}")

    messages.append(assistant)
    messages.append(
        {
            "role": "tool",
            "tool_call_id": call["id"],
            "name": "multiply",
            "content": "391",
        }
    )
    second_payload = thinking_payload(model, "low", messages)
    second_payload["tools"] = tools
    second_payload["tool_choice"] = "auto"
    second = post_json(endpoint, second_payload)
    second_choice = second["choices"][0]
    answer = second_choice["message"].get("content") or ""
    if second_choice.get("finish_reason") != "stop":
        raise AssertionError(f"tool result: did not stop: {second_choice}")
    if answer.strip() != "FINAL: 391":
        raise AssertionError(f"tool result: wrong final answer: {answer!r}")
    emit(
        {
            "answer": answer,
            "arguments": arguments,
            "function": "multiply",
            "kind": "tool_round_trip",
            "repetition": repetition,
            "valid": True,
        },
        receipt_file,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--model",
        help="served model name; auto-detected only when exactly one is advertised",
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--receipt-file", type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    endpoint_root = args.base_url.rstrip("/")
    model = resolve_model(endpoint_root, args.model)
    endpoint = f"{endpoint_root}/v1/chat/completions"
    args.receipt_file.parent.mkdir(parents=True, exist_ok=True)

    with args.receipt_file.open("w", encoding="utf-8") as receipt_file:
        for repetition in range(1, args.runs + 1):
            check_reasoning_efforts(endpoint, model, repetition, receipt_file)
            check_non_thinking(endpoint, model, repetition, receipt_file)
            check_image(endpoint, model, repetition, receipt_file)
            check_tool_round_trip(endpoint, model, repetition, receipt_file)
        emit(
            {
                "kind": "summary",
                "model": model,
                "runs": args.runs,
                "valid": True,
            },
            receipt_file,
        )
    print("QWEN38_SEMANTIC_ADMISSION_PASS", flush=True)


if __name__ == "__main__":
    main()
