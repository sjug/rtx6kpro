#!/usr/bin/env python3
"""Repeated semantic, reasoning, vision, and tool admission for GLM-5.3."""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import struct
import urllib.request
import zlib
from pathlib import Path


def post_json(url: str, payload: dict, timeout: int = 300) -> dict:
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


def emit(receipt: dict, output) -> None:
    line = json.dumps(receipt, sort_keys=True)
    print(line, flush=True)
    output.write(line + "\n")
    output.flush()


def payload(model: str, effort: str, messages: list[dict]) -> dict:
    return {
        "model": model,
        "messages": messages,
        "reasoning_effort": effort,
        "temperature": 0,
        "max_completion_tokens": 768,
    }


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", binascii.crc32(kind + data) & 0xFFFFFFFF)
    )


def red_blue_png() -> str:
    width, height = 128, 64
    row = b"\xff\x00\x00" * 64 + b"\x00\x00\xff" * 64
    pixels = b"".join(b"\x00" + row for _ in range(height))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + png_chunk(b"IDAT", zlib.compress(pixels, level=9))
        + png_chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode()


def reasoning_gate(endpoint: str, model: str, effort: str) -> dict:
    result = post_json(
        endpoint,
        payload(
            model,
            effort,
            [
                {
                    "role": "user",
                    "content": (
                        "Calculate 17 times 23 minus 58. Explain briefly and end "
                        "with exactly FINAL: 333"
                    ),
                }
            ],
        ),
    )
    choice = result["choices"][0]
    message = choice["message"]
    answer = message.get("content") or ""
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    if choice.get("finish_reason") != "stop":
        raise AssertionError(f"{effort}: did not stop: {choice}")
    if len(reasoning.strip()) < 5:
        raise AssertionError(f"{effort}: missing reasoning: {message}")
    if not answer.rstrip().endswith("FINAL: 333"):
        raise AssertionError(f"{effort}: wrong answer: {answer!r}")
    return {
        "answer": answer,
        "completion_tokens": result.get("usage", {}).get("completion_tokens"),
        "effort": effort,
        "kind": "reasoning",
        "reasoning_chars": len(reasoning),
        "valid": True,
    }


def vision_gate(endpoint: str, model: str) -> dict:
    result = post_json(
        endpoint,
        payload(
            model,
            "low",
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Name the solid color on the left and the solid "
                                "color on the right. End exactly FINAL: red, blue"
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": red_blue_png()}},
                    ],
                }
            ],
        ),
    )
    choice = result["choices"][0]
    answer = choice["message"].get("content") or ""
    if choice.get("finish_reason") != "stop" or not answer.rstrip().endswith(
        "FINAL: red, blue"
    ):
        raise AssertionError(f"vision failure: {choice}")
    return {"answer": answer, "kind": "vision", "valid": True}


def tool_gate(endpoint: str, model: str) -> dict:
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
                "Use multiply for 17 times 23. After its result, answer exactly "
                "FINAL: 391"
            ),
        }
    ]
    first_payload = payload(model, "low", messages)
    first_payload.update({"tools": tools, "tool_choice": "required"})
    first = post_json(endpoint, first_payload)
    assistant = first["choices"][0]["message"]
    calls = assistant.get("tool_calls") or []
    if len(calls) != 1:
        raise AssertionError(f"expected one tool call: {first['choices'][0]}")
    call = calls[0]
    arguments = json.loads(call["function"]["arguments"])
    if call["function"]["name"] != "multiply" or arguments != {"a": 17, "b": 23}:
        raise AssertionError(f"wrong tool call: {call}")
    messages.extend(
        [
            assistant,
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "name": "multiply",
                "content": "391",
            },
        ]
    )
    second_payload = payload(model, "low", messages)
    second_payload.update({"tools": tools, "tool_choice": "auto"})
    second = post_json(endpoint, second_payload)
    choice = second["choices"][0]
    answer = choice["message"].get("content") or ""
    if choice.get("finish_reason") != "stop" or answer.strip() != "FINAL: 391":
        raise AssertionError(f"tool round trip failed: {choice}")
    return {
        "answer": answer,
        "arguments": arguments,
        "function": "multiply",
        "kind": "tool_round_trip",
        "valid": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://sparky:8000")
    parser.add_argument("--model", default="GLM-5.3-Flash")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--receipt-file", type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    endpoint = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    args.receipt_file.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt_file.open("w", encoding="utf-8") as output:
        for repetition in range(1, args.runs + 1):
            for effort in ("low", "high", "max"):
                receipt = reasoning_gate(endpoint, args.model, effort)
                receipt["repetition"] = repetition
                emit(receipt, output)
            for gate in (vision_gate, tool_gate):
                receipt = gate(endpoint, args.model)
                receipt["repetition"] = repetition
                emit(receipt, output)
        emit(
            {
                "kind": "summary",
                "model": args.model,
                "runs": args.runs,
                "valid": True,
            },
            output,
        )
    print("GLM53_SEMANTIC_ADMISSION_PASS", flush=True)


if __name__ == "__main__":
    main()
