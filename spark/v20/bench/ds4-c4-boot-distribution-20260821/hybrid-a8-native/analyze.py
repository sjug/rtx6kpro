#!/usr/bin/env python3
"""Compare the batch-selective hybrid with its contemporary A8 controls."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONTEXTS = (0, 16384, 32768, 65536, 131072)


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def load(path: Path) -> dict:
    data = json.loads(path.read_text())
    result = {}
    for concurrency in (1, 4):
        rows = {
            int(row["context_tokens"]): row
            for row in data["results"]
            if int(row["concurrency"]) == concurrency
        }
        if set(rows) != set(CONTEXTS):
            raise ValueError(f"{path}: c{concurrency} {sorted(rows)}")
        result[concurrency] = {
            "steps": geomean(
                [float(rows[context]["server_steps_per_s"]) for context in CONTEXTS]
            ),
            "tokens": geomean(
                [float(rows[context]["aggregate_tps"]) for context in CONTEXTS]
            ),
            "accept": geomean(
                [
                    float(rows[context]["server_spec_accept_length"])
                    for context in CONTEXTS
                ]
            ),
            "by_context": {
                context: float(rows[context]["server_steps_per_s"])
                for context in CONTEXTS
            },
        }
    result["prefill64k"] = float(data["prefill"]["65536"]["client_tok_per_sec"])
    return result


def median_sweeps(label: str) -> dict:
    sweeps = [load(ROOT / "results" / f"{label}-sweep{sweep}.json") for sweep in (1, 2)]
    result = {}
    for concurrency in (1, 4):
        result[concurrency] = {
            metric: statistics.median(sweep[concurrency][metric] for sweep in sweeps)
            for metric in ("steps", "tokens", "accept")
        } | {
            "by_context": {
                context: statistics.median(
                    sweep[concurrency]["by_context"][context] for sweep in sweeps
                )
                for context in CONTEXTS
            }
        }
    result["prefill64k"] = statistics.median(sweep["prefill64k"] for sweep in sweeps)
    return result


def median_arms(arms: tuple[dict, ...]) -> dict:
    result = {}
    for concurrency in (1, 4):
        result[concurrency] = {
            metric: statistics.median(arm[concurrency][metric] for arm in arms)
            for metric in ("steps", "tokens", "accept")
        } | {
            "by_context": {
                context: statistics.median(
                    arm[concurrency]["by_context"][context] for arm in arms
                )
                for context in CONTEXTS
            }
        }
    result["prefill64k"] = statistics.median(arm["prefill64k"] for arm in arms)
    return result


def pct(candidate: float, baseline: float) -> float:
    return (candidate / baseline - 1.0) * 100.0


def main() -> None:
    opening = median_sweeps("a8-opening")
    hybrid = median_sweeps("hybrid")
    closing = median_sweeps("a8-closing")
    a8 = median_arms((opening, closing))

    print("arm concurrency steps_geo tokens_geo accept_len prefill64k")
    for label, arm in (
        ("a8-opening", opening),
        ("hybrid", hybrid),
        ("a8-closing", closing),
        ("a8-median", a8),
    ):
        for concurrency in (1, 4):
            print(
                f"{label} {concurrency} {arm[concurrency]['steps']:.3f} "
                f"{arm[concurrency]['tokens']:.3f} "
                f"{arm[concurrency]['accept']:.3f} {arm['prefill64k']:.1f}"
            )

    print("\nhybrid versus median A8")
    print(f"64K prefill: {pct(hybrid['prefill64k'], a8['prefill64k']):+.2f}%")
    for concurrency in (1, 4):
        print(
            f"c{concurrency}: "
            f"steps={pct(hybrid[concurrency]['steps'], a8[concurrency]['steps']):+.2f}% "
            f"tokens={pct(hybrid[concurrency]['tokens'], a8[concurrency]['tokens']):+.2f}% "
            f"accept={pct(hybrid[concurrency]['accept'], a8[concurrency]['accept']):+.2f}%"
        )
        for context in CONTEXTS:
            print(
                f"  {context}: steps="
                f"{pct(hybrid[concurrency]['by_context'][context], a8[concurrency]['by_context'][context]):+.2f}%"
            )

    print("\nA8 closing versus opening drift")
    print(f"64K prefill: {pct(closing['prefill64k'], opening['prefill64k']):+.2f}%")
    for concurrency in (1, 4):
        print(
            f"c{concurrency}: "
            f"steps={pct(closing[concurrency]['steps'], opening[concurrency]['steps']):+.2f}% "
            f"tokens={pct(closing[concurrency]['tokens'], opening[concurrency]['tokens']):+.2f}% "
            f"accept={pct(closing[concurrency]['accept'], opening[concurrency]['accept']):+.2f}%"
        )


if __name__ == "__main__":
    main()
