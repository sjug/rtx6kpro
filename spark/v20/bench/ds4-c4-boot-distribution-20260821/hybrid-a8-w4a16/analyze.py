#!/usr/bin/env python3
"""Compare source-native W4A16 with contemporary A8 and packed W4A16."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PHASE5 = ROOT.parent / "hybrid-a8-native"
PHASE2 = ROOT.parent / "a8-a16-crossover"
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


def median_paths(paths: list[Path]) -> dict:
    arms = [load(path) for path in paths]
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


def describe_delta(label: str, candidate: dict, baseline: dict) -> None:
    print(f"\nsource-native W4A16 versus {label}")
    print(f"64K prefill: {pct(candidate['prefill64k'], baseline['prefill64k']):+.2f}%")
    for concurrency in (1, 4):
        print(
            f"c{concurrency}: "
            f"steps={pct(candidate[concurrency]['steps'], baseline[concurrency]['steps']):+.2f}% "
            f"tokens={pct(candidate[concurrency]['tokens'], baseline[concurrency]['tokens']):+.2f}% "
            f"accept={pct(candidate[concurrency]['accept'], baseline[concurrency]['accept']):+.2f}%"
        )
        for context in CONTEXTS:
            print(
                f"  {context}: steps="
                f"{pct(candidate[concurrency]['by_context'][context], baseline[concurrency]['by_context'][context]):+.2f}%"
            )


def main() -> None:
    a8 = median_paths(
        [
            PHASE5 / "results" / f"a8-{position}-sweep{sweep}.json"
            for position in ("opening", "closing")
            for sweep in (1, 2)
        ]
    )
    packed_w4a16 = median_paths(
        [PHASE2 / "results" / f"a16-candidate-sweep{sweep}.json" for sweep in (1, 2)]
    )
    source_w4a16 = median_paths(
        [
            ROOT / "results" / f"source-native-w4a16-sweep{sweep}.json"
            for sweep in (1, 2)
        ]
    )

    print("arm concurrency steps_geo tokens_geo accept_len prefill64k")
    for label, arm in (
        ("a8", a8),
        ("packed-w4a16", packed_w4a16),
        ("source-w4a16", source_w4a16),
    ):
        for concurrency in (1, 4):
            print(
                f"{label} {concurrency} {arm[concurrency]['steps']:.3f} "
                f"{arm[concurrency]['tokens']:.3f} "
                f"{arm[concurrency]['accept']:.3f} {arm['prefill64k']:.1f}"
            )

    describe_delta("A8", source_w4a16, a8)
    describe_delta("packed W4A16", source_w4a16, packed_w4a16)


if __name__ == "__main__":
    main()
