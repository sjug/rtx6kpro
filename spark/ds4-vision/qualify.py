#!/usr/bin/env python3
"""Bounded meaningful-output admission before any Vision performance timing."""
import argparse
import base64
import binascii
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import struct
import time
import urllib.request
import zlib

p = argparse.ArgumentParser()
p.add_argument("--base-url", default="http://rusty:8000")
p.add_argument("--out", type=Path, required=True)
p.add_argument("--long", action="store_true")
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
model = "DeepSeek-V4-Flash-Vision-Exp"


def post(path, payload, timeout=180):
    req = urllib.request.Request(a.base_url + path, json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        result = json.load(response)
    if result.get("error"):
        raise RuntimeError(result)
    return result, time.monotonic() - start


def chat(messages, **kwargs):
    return post("/v1/chat/completions", {
        "model": model, "messages": messages, "max_tokens": 1536,
        "temperature": 0, **kwargs,
    })


def save(label, result, elapsed, expected=None):
    receipt = {"label": label, "elapsed_s": elapsed, "response": result}
    (a.out / f"{label}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    choice = result["choices"][0]
    answer = choice["message"].get("content") or ""
    if choice["finish_reason"] != "stop" or not answer.strip() or answer.strip() == "!":
        raise RuntimeError(f"Incomplete or meaningless {label}: {choice}")
    if expected is not None and answer.strip().lower() != expected.lower():
        raise RuntimeError(f"Wrong {label} answer: {answer!r}, expected {expected!r}")
    print(json.dumps({"label": label, "valid": True, "answer": answer,
                      "elapsed_s": elapsed, "usage": result.get("usage")}), flush=True)
    return choice["message"]


def user(text):
    return [{"role": "user", "content": text}]


def png():
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(
            ">I", binascii.crc32(kind + data) & 0xffffffff)
    width, height = 512, 256
    row = b"\xff\x00\x00" * 256 + b"\x00\x00\xff" * 256
    raw = b"".join(b"\0" + row for _ in range(height))
    image = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
             + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(image).decode()


vision_messages = [{"role": "user", "content": [
    {"type": "text", "text": "Name the solid colors on the left and right of this image, in that order. Reply with only the two color names separated by a comma."},
    {"type": "image_url", "image_url": {"url": png()}},
]}]

for rep in range(3):
    save(f"default-max-arithmetic-{rep}", *chat(user(
        "Calculate 17 times 23 minus 58. Reply with only the final integer.")), "333")
    save(f"nonthinking-{rep}", *chat(user("What is 19 times 7? Reply with only the integer."),
         chat_template_kwargs={"thinking": False}, max_tokens=64), "133")
    assistant = save(f"vision-{rep}", *chat(vision_messages), "red, blue")
    save(f"vision-continuation-{rep}", *chat(vision_messages + [assistant] + user(
        "Which color was on the right? Reply with only the color name.")), "blue")

tools = [{"type": "function", "function": {
    "name": "multiply", "description": "Multiply two integers.",
    "parameters": {"type": "object", "properties": {
        "a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"], "additionalProperties": False}}}]
messages = user("Use the multiply tool to calculate 17 times 23, then reply with only the result.")
result, elapsed = chat(messages, tools=tools, tool_choice="required")
(a.out / "tool-call.json").write_text(json.dumps(result, indent=2) + "\n")
assistant = result["choices"][0]["message"]
calls = assistant.get("tool_calls") or []
if len(calls) != 1 or calls[0]["function"]["name"] != "multiply":
    raise RuntimeError(f"Incorrect tool call: {result}")
call = calls[0]
if json.loads(call["function"]["arguments"]) != {"a": 17, "b": 23}:
    raise RuntimeError(f"Incorrect tool arguments: {call}")
messages += [assistant, {"role": "tool", "tool_call_id": call["id"], "content": "391"}]
save("tool-round-trip", *chat(messages, tools=tools, tool_choice="auto"), "391")

mixed = [vision_messages, user("Reply with only the result of 23 plus 19."),
         vision_messages, user("Reply with only the result of 21 times 4.")]
with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(chat, mixed))
for i, ((result, elapsed), expected) in enumerate(zip(results, ["red, blue", "42", "red, blue", "84"])):
    save(f"concurrent-{i}", result, elapsed, expected)

if a.long:
    for length in (16384, 524000):
        prefix = "Memorize this unique retrieval code: 739184.\nArchive:\n"
        suffix = "\nArchive ends. Reply with only the retrieval code stated before the archive."
        filler = length - 100
        for _ in range(6):
            messages = user(prefix + " filler" * filler + suffix)
            tokenized, _ = post("/tokenize", {"model": model, "messages": messages,
                "chat_template_kwargs": {"thinking": False}, "add_generation_prompt": True}, 60)
            count = tokenized.get("count", len(tokenized.get("tokens", [])))
            if count == length:
                break
            filler += length - count
        else:
            raise RuntimeError("Cannot construct exact-length needle")
        result, elapsed = post("/v1/chat/completions", {
            "model": model, "messages": messages,
            "chat_template_kwargs": {"thinking": False}, "temperature": 0, "max_tokens": 64,
        }, 1800)
        save(f"needle-{length}", result, elapsed, "739184")
        if result["usage"]["prompt_tokens"] != length:
            raise RuntimeError("Needle token accounting mismatch")
print("VISION-QUALIFICATION-PASS", flush=True)
