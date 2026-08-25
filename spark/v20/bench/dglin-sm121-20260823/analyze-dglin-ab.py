#!/usr/bin/env python3
"""Summarize the balanced r18p DGLIN serving comparison."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def _geomean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"geomean requires positive values, got {values}")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def _load(path: Path) -> dict:
    with path.open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"expected object in {path}")
    return value


def _decode_index(run: dict) -> dict[tuple[int, int], dict]:
    return {
        (int(row["concurrency"]), int(row["context_tokens"])): row
        for row in run["results"]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()

    pairs = []
    for pair in (1, 2, 3):
        candidates = list(args.results.glob(f"pair{pair}-*-arm*.json"))
        by_arm = {}
        for path in candidates:
            if path.name.endswith(".resume.json"):
                continue
            arm = "a" if "-arma.json" in path.name else "b"
            by_arm[arm] = (path, _load(path))
        if set(by_arm) != {"a", "b"}:
            raise FileNotFoundError(
                f"pair {pair} needs one final A and B file, found {by_arm}"
            )
        pairs.append((pair, by_arm["a"], by_arm["b"]))

    decode_ratios = {
        "raw_tps": [],
        "steps_per_s": [],
        "accept_length": [],
    }
    decode_by_concurrency = {
        1: {name: [] for name in decode_ratios},
        4: {name: [] for name in decode_ratios},
    }
    prefill_ratios: dict[int, list[float]] = {}

    print("# DGLIN balanced serving comparison")
    print()
    print("B/A ratios above 1.0 favor DGLIN.")
    print()
    print("## Decode cells")
    print()
    print("| pair | C | context | raw B/A | steps B/A | accept-length B/A |")
    print("|---:|---:|---:|---:|---:|---:|")
    for pair, (_, arm_a), (_, arm_b) in pairs:
        a_index = _decode_index(arm_a)
        b_index = _decode_index(arm_b)
        if set(a_index) != set(b_index):
            raise AssertionError(f"pair {pair} has unmatched decode cells")
        for key in sorted(a_index):
            concurrency, context = key
            a = a_index[key]
            b = b_index[key]
            ratios = {
                "raw_tps": float(b["aggregate_tps"]) / float(a["aggregate_tps"]),
                "steps_per_s": float(b["server_steps_per_s"])
                / float(a["server_steps_per_s"]),
                "accept_length": float(b["server_spec_accept_length"])
                / float(a["server_spec_accept_length"]),
            }
            for name, ratio in ratios.items():
                decode_ratios[name].append(ratio)
                decode_by_concurrency[concurrency][name].append(ratio)
            print(
                f"| {pair} | {concurrency} | {context} | "
                f"{ratios['raw_tps']:.4f} | {ratios['steps_per_s']:.4f} | "
                f"{ratios['accept_length']:.4f} |"
            )

    print()
    print("## Decode geomeans")
    print()
    print("| scope | raw B/A | steps B/A | accept-length B/A |")
    print("|---|---:|---:|---:|")
    for scope, values in [
        ("all", decode_ratios),
        ("C=1", decode_by_concurrency[1]),
        ("C=4", decode_by_concurrency[4]),
    ]:
        print(
            f"| {scope} | {_geomean(values['raw_tps']):.4f} | "
            f"{_geomean(values['steps_per_s']):.4f} | "
            f"{_geomean(values['accept_length']):.4f} |"
        )

    print()
    print("## Prefill")
    print()
    print("| pair | tokens | client tok/s B/A |")
    print("|---:|---:|---:|")
    for pair, (_, arm_a), (_, arm_b) in pairs:
        if set(arm_a["prefill"]) != set(arm_b["prefill"]):
            raise AssertionError(f"pair {pair} has unmatched prefill cells")
        for tokens_text in sorted(arm_a["prefill"], key=int):
            tokens = int(tokens_text)
            a = float(arm_a["prefill"][tokens_text]["client_tok_per_sec"])
            b = float(arm_b["prefill"][tokens_text]["client_tok_per_sec"])
            ratio = b / a
            prefill_ratios.setdefault(tokens, []).append(ratio)
            print(f"| {pair} | {tokens} | {ratio:.4f} |")

    print()
    print("## Prefill geomeans")
    print()
    print("| tokens | client tok/s B/A |")
    print("|---:|---:|")
    for tokens, ratios in sorted(prefill_ratios.items()):
        print(f"| {tokens} | {_geomean(ratios):.4f} |")
    print(
        f"| all | {_geomean([ratio for values in prefill_ratios.values() for ratio in values]):.4f} |"
    )


if __name__ == "__main__":
    main()
