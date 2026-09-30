#!/usr/bin/env python3
"""September 24 upstream-profile admission; R38 probes with metrics cache accounting.

Prompt and correctness assertions are unchanged. This copy avoids requiring a
launcher flag solely for optional API usage metadata. Original SHA256:
108adc19dc90a6e19ad1cb0f5d66ef2bdc7344c3a6fa611e67edf52c6bcf3d53.
"""
from cache_metrics import idle_snapshot, finish
import argparse
import base64
import binascii
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import struct
import time
import urllib.error
import urllib.request
import uuid
import zlib

p = argparse.ArgumentParser()
p.add_argument("--base-url", default="http://dusty:8000")
p.add_argument("--out", type=Path, required=True)
p.add_argument("--long", action="store_true")
p.add_argument("--needle-lengths", default="16384,131000",
               help="Exact input lengths for --long; each must fit the separately admitted server envelope")
p.add_argument("--needle-timeout", type=int, default=3600)
p.add_argument("--dual-needle", action="store_true", help="Also retrieve a second code near 70 percent depth")
a = p.parse_args()
needle_lengths = [int(value) for value in a.needle_lengths.split(",")]
if not needle_lengths or any(value < 256 or value > 1048512 for value in needle_lengths):
    p.error("Needle lengths must leave room for 64 output tokens within native context")
a.out.mkdir(parents=True, exist_ok=True)
model = "DeepSeek-V4.1-Flash"


def post(path, payload, timeout=180):
    req = urllib.request.Request(a.base_url + path, json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        (a.out / f"http-error-{uuid.uuid4().hex}.json").write_text(json.dumps({
            "path": path, "status": error.code, "body": error.read().decode(errors="replace"),
            "request": payload, "elapsed_s": time.monotonic() - start,
        }, indent=2) + "\n")
        raise
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


save("cold-first-completion", *chat(user("Calculate 17 times 23 minus 58. Reply with only the final integer."),
    chat_template_kwargs={"thinking": False}, max_tokens=64), "333")

vision_messages = [{"role": "user", "content": [
    {"type": "text", "text": "Name the solid colors on the left and right of this image, in that order. Reply with only the two color names separated by a comma."},
    {"type": "image_url", "image_url": {"url": png()}},
]}]

for rep in range(3):
    save(f"default-thinking-arithmetic-{rep}", *chat(user(
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
    for length in needle_lengths:
        prefix = f"Archive identity {uuid.uuid4().hex}. Memorize this unique retrieval code: 739184.\nArchive:\n"
        suffix = "\nArchive ends. Reply with only the retrieval code stated before the archive."
        expected = "739184"
        if a.dual_needle:
            suffix = "\nArchive ends. Reply with only the initial retrieval code and the late retrieval code, in that order, separated by a comma and a space."
            expected = "739184, 482617"
        filler = length - 100
        for _ in range(6):
            before_late = prefix + " filler" * (filler * 7 // 10)
            if a.dual_needle:
                text = (before_late + "\nThe late retrieval code is 482617.\n"
                        + " filler" * (filler - filler * 7 // 10) + suffix)
            else:
                text = prefix + " filler" * filler + suffix
            messages = user(text)
            tokenized, _ = post("/tokenize", {"model": model, "messages": messages,
                "chat_template_kwargs": {"thinking": False}, "add_generation_prompt": True}, 60)
            count = tokenized.get("count", len(tokenized.get("tokens", [])))
            if count == length:
                break
            filler += length - count
        else:
            raise RuntimeError("Cannot construct exact-length needle")
        late_position = None
        if a.dual_needle:
            position, _ = post("/tokenize", {"model": model, "messages": user(before_late),
                "chat_template_kwargs": {"thinking": False}, "add_generation_prompt": True}, 60)
            late_position = position.get("count", len(position.get("tokens", [])))
            if not 0.6 * length < late_position < 0.8 * length:
                raise RuntimeError("Late needle is outside the intended depth range")
        (a.out / f"needle-{length}-input.json").write_text(json.dumps({
            "messages": messages, "prompt_tokens": length,
            "chat_template_kwargs": {"thinking": False}, "temperature": 0,
            "max_tokens": 64,
            "expected": expected, "late_marker_prefix_tokens_including_template": late_position,
        }) + "\n")
        before = idle_snapshot(a.base_url)
        result, elapsed = post("/v1/chat/completions", {
            "model": model, "messages": messages,
            "chat_template_kwargs": {"thinking": False}, "temperature": 0, "max_tokens": 64,
        }, a.needle_timeout)
        save(f"needle-{length}", result, elapsed, expected)
        if result["usage"]["prompt_tokens"] != length:
            raise RuntimeError("Needle token accounting mismatch")
        cached = finish(a.base_url, before, length, a.out / f"needle-{length}-cache-metrics.json")
        if cached != 0:
            raise RuntimeError(f"Cold needle has missing or nonzero cached-token accounting: {cached}")
print("DS41-QUALIFICATION-PASS", flush=True)
