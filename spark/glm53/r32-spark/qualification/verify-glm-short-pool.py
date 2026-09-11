#!/usr/bin/env python3
"""Live semantic coverage around complete and incomplete short GLM pool tails.

This supplements the exact kernel-oracle build gates; it is not an isolated
test of the index-expansion operator or an exact-output comparison with R29.
"""
if not __debug__:
    raise RuntimeError("R32 verification requires Python assertions enabled")

import argparse
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="GLM-5.3-Flash")
    parser.add_argument("--receipt-file", type=Path, required=True)
    args = parser.parse_args()
    helper = Path(__file__).resolve().parents[3] / "qwen38-flash-next/verify-native-context.py"
    spec = importlib.util.spec_from_file_location("needle", helper)
    needle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(needle)
    with args.receipt_file.open("x") as receipt:
        for length in (128, 129, 2047, 2048, 2049, 2050, 2051):
            code = "739526"
            messages, offset = needle.exact_needle_prompt(
                args.base_url, args.model, length, code, 300,
            )
            request = {
                "model": args.model, "messages": messages, "temperature": 0,
                "max_tokens": 128, "return_token_ids": True,
            }
            response, elapsed = needle.post_json(
                args.base_url + "/v1/chat/completions", request, 300,
            )
            choice = response["choices"][0]
            valid = (
                response["usage"]["prompt_tokens"] == length
                and choice["finish_reason"] == "stop"
                and (choice["message"].get("content") or "").strip() == code
            )
            row = {
                "kind": "short_pool_semantic", "prompt_tokens": length,
                "pool_size": 4, "tail_tokens": length % 4,
                "needle": code, "needle_token_offset": offset,
                "elapsed_seconds": elapsed, "valid": valid,
                "request": request, "response": response,
            }
            receipt.write(json.dumps(row) + "\n")
            receipt.flush()
            print(json.dumps({k: v for k, v in row.items() if k not in ("request", "response")}), flush=True)
            if not valid:
                raise RuntimeError(f"GLM short-pool semantic check failed at {length}")
    print("GLM_R32_SHORT_POOL_SEMANTIC_PASS cases=7", flush=True)


if __name__ == "__main__":
    main()
