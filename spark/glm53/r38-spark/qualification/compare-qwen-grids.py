#!/usr/bin/env python3
"""Compare complete standard grids, without modifying either raw receipt."""
if not __debug__:
    raise RuntimeError("R38 verification requires Python assertions enabled")

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import geometric_mean


def load(path):
    data = json.loads(path.read_text())
    levels = data['metadata']['concurrency_levels']
    if not levels or len(set(levels)) != len(levels) or not set(levels) <= {1, 2, 4}:
        raise ValueError(f"Unexpected concurrency levels: {path}")
    expected = {(c, m) for c in levels for m in (0, 16384, 32768, 65536, 131072)}
    cells = {(r["concurrency"], r["context_tokens"]): r for r in data["results"]}
    if set(cells) != expected or len(data["results"]) != len(expected):
        raise ValueError(f"Incomplete or duplicate grid: {path}")
    for key, row in cells.items():
        for flag in ("num_errors", "warmup_timed_out", "capacity_limited", "underfilled"):
            if row[flag]:
                raise ValueError(f"{path}: {key}: {flag}={row[flag]}")
        for metric in ("avg_running_reqs", "max_running_reqs"):
            value = row[metric]
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {metric}: {path}: {key}")
            if value > row["concurrency"]:
                raise ValueError(f"{path}: {key}: {metric}={value} exceeds requested concurrency")
        for metric in ("aggregate_tps", "server_steps_per_s", "server_accept_len_effective"):
            if not math.isfinite(row[metric]) or row[metric] <= 0:
                raise ValueError(f"Invalid {metric}: {path}: {key}")
    return data, cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--concurrency", type=int, nargs='+', choices=(1, 2, 4))
    args = parser.parse_args()
    baseline, a = load(args.baseline)
    candidate, b = load(args.candidate)
    levels = args.concurrency or [1, 2, 4]
    expected = {(c, m) for c in levels for m in (0, 16384, 32768, 65536, 131072)}
    a = {k: v for k, v in a.items() if k[0] in levels}
    b = {k: v for k, v in b.items() if k[0] in levels}
    if set(a) != expected or set(b) != expected:
        raise ValueError('Both inputs must contain every requested comparison cell')
    for key in ("version", "decode_mode", "duration_per_test", "decode_warmup_seconds",
                "context_lengths", "ignore_eos", "max_tokens",
                "chat_template_kwargs", "temperature", "prefill_mode"):
        if baseline["metadata"][key] != candidate["metadata"][key]:
            raise ValueError(f"Protocol drift: {key}")
    for key in ("checkpoint_revision", "recurrent_checkpoint_policy", "harness_sha256"):
        if not baseline["run_metadata"].get(key) or baseline["run_metadata"][key] != candidate["run_metadata"].get(key):
            raise ValueError(f"Identity/protocol drift: {key}")
    aliases = {"Qwen3.8-Flash-Next", "Qwen3.8-Flash-Next-NVFP4-4p89"}
    names = {baseline["metadata"]["model"], candidate["metadata"]["model"]}
    if len(names) > 1 and not (names <= aliases and baseline["run_metadata"]["checkpoint_revision"] == "c374e7e24b54f6cb0017d0c2e6d26823d2f2fb5d"):
        raise ValueError("Unrecognized model alias change")
    report = {"comparison": "R32 aligned versus R38 aligned; boot identities are recorded in the input metadata",
              "model_aliases": sorted(names), "repeatable_gain_proven": False, "grids_valid": True, "inputs": {},
              "summary": [], "cells": [], "prefill": []}
    for label, path in (("r32", args.baseline), ("r38", args.candidate)):
        report["inputs"][label] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    for c in levels:
        row = {"concurrency": c}
        for metric in ("server_steps_per_s", "aggregate_tps", "server_accept_len_effective"):
            av = geometric_mean(v[metric] for (cc, _), v in a.items() if cc == c)
            bv = geometric_mean(v[metric] for (cc, _), v in b.items() if cc == c)
            row[metric] = {"r32": av, "r38": bv, "change_pct": (bv / av - 1) * 100}
        report["summary"].append(row)
    for key in sorted(a):
        report["cells"].append({"concurrency": key[0], "context_tokens": key[1],
            **{metric: {"r32": a[key][metric], "r38": b[key][metric],
                        "change_pct": (b[key][metric] / a[key][metric] - 1) * 100}
               for metric in ("server_steps_per_s", "aggregate_tps", "server_accept_len_effective")}})
    for context in sorted(baseline["prefill"], key=int):
        av = baseline["prefill"][context]["tok_per_sec"]
        bv = candidate["prefill"][context]["tok_per_sec"]
        if not all(math.isfinite(v) and v > 0 for v in (av, bv)):
            raise ValueError(f"Invalid prefill timing at {context}")
        report["prefill"].append({"context_tokens": int(context), "r32": av, "r38": bv,
                                  "change_pct": (bv / av - 1) * 100})
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
