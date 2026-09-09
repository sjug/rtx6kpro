#!/usr/bin/env python3
"""Exact GLM retrieval with its default thinking contract, not Qwen's disable flag."""
if not __debug__:
    raise RuntimeError("R28 verification requires Python assertions enabled")

import argparse
import importlib.util
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--base-url", required=True)
parser.add_argument("--model", default="GLM-5.3-Flash")
parser.add_argument("--receipt-file", type=Path, required=True)
args = parser.parse_args()
helper = Path(__file__).resolve().parents[3] / "qwen38-flash-next/verify-native-context.py"
spec = importlib.util.spec_from_file_location("needle", helper)
needle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(needle)
code = "739526"
with args.receipt_file.open("x") as receipt:
    for length in (2048, 2049, 262000, 1048000):
        messages, offset = needle.exact_needle_prompt(args.base_url, args.model, length, code, 1800)
        print(f"CONTEXT_START length={length}", flush=True)
        request = {"model": args.model, "messages": messages, "temperature": 0,
                   "max_tokens": 128, "return_token_ids": True}
        response, elapsed = needle.post_json(args.base_url + "/v1/chat/completions", request, 1800)
        choice = response["choices"][0]
        message = choice["message"]
        valid = (response["usage"]["prompt_tokens"] == length
                 and choice["finish_reason"] == "stop"
                 and (message.get("content") or "").strip() == code)
        row = {"kind": "context_needle", "prompt_tokens": length, "needle": code,
               "needle_token_offset": offset, "elapsed_seconds": elapsed,
               "finish_reason": choice["finish_reason"], "text": message.get("content"),
               "valid": valid, "request": request, "response": response}
        receipt.write(json.dumps(row) + "\n")
        receipt.flush()
        print(json.dumps({k: v for k, v in row.items() if k not in ("request", "response")}), flush=True)
        if not valid:
            raise RuntimeError(f"GLM exact retrieval failed at {length}")
print("GLM_NATIVE_CONTEXT_PASS", flush=True)
