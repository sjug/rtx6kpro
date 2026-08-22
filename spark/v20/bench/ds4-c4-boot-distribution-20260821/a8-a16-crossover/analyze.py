#!/usr/bin/env python3
"""Summarize the r18p A8/A16/A8 crossover."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONTEXTS = (0, 16384, 32768, 65536, 131072)
LABELS = ("a8-control-1", "a16-candidate", "a8-control-2")


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def load(label: str, sweep: int) -> dict:
    with (ROOT / "results" / f"{label}-sweep{sweep}.json").open(encoding="utf-8") as stream:
        data = json.load(stream)
    result = {}
    for concurrency in (1, 4):
        rows = {
            int(row["context_tokens"]): row
            for row in data["results"]
            if int(row["concurrency"]) == concurrency
        }
        if set(rows) != set(CONTEXTS):
            raise ValueError(f"{label} sweep {sweep} c{concurrency}: {sorted(rows)}")
        result[concurrency] = {
            "steps": geomean([float(rows[context]["server_steps_per_s"]) for context in CONTEXTS]),
            "tokens": geomean([float(rows[context]["aggregate_tps"]) for context in CONTEXTS]),
            "accept": geomean(
                [float(rows[context]["server_spec_accept_length"]) for context in CONTEXTS]
            ),
            "by_context": {
                context: float(rows[context]["server_steps_per_s"]) for context in CONTEXTS
            },
        }
    result["prefill64k"] = float(data["prefill"]["65536"]["client_tok_per_sec"])
    return result


def median_arm(label: str) -> dict:
    sweeps = [load(label, sweep) for sweep in (1, 2)]
    return {
        concurrency: {
            metric: statistics.median(sweep[concurrency][metric] for sweep in sweeps)
            for metric in ("steps", "tokens", "accept")
        }
        | {
            "by_context": {
                context: statistics.median(
                    sweep[concurrency]["by_context"][context] for sweep in sweeps
                )
                for context in CONTEXTS
            }
        }
        for concurrency in (1, 4)
    } | {"prefill64k": statistics.median(sweep["prefill64k"] for sweep in sweeps)}


def pct(candidate: float, baseline: float) -> float:
    return (candidate / baseline - 1.0) * 100.0


def main() -> None:
    arms = {label: median_arm(label) for label in LABELS}
    controls = {
        concurrency: {
            metric: statistics.median(
                arms[label][concurrency][metric] for label in ("a8-control-1", "a8-control-2")
            )
            for metric in ("steps", "tokens", "accept")
        }
        | {
            "by_context": {
                context: statistics.median(
                    arms[label][concurrency]["by_context"][context]
                    for label in ("a8-control-1", "a8-control-2")
                )
                for context in CONTEXTS
            }
        }
        for concurrency in (1, 4)
    } | {
        "prefill64k": statistics.median(
            arms[label]["prefill64k"] for label in ("a8-control-1", "a8-control-2")
        )
    }

    print("label concurrency steps_geo tokens_geo accept_len prefill64k")
    for label in LABELS:
        for concurrency in (1, 4):
            arm = arms[label][concurrency]
            print(
                f"{label} {concurrency} {arm['steps']:.3f} {arm['tokens']:.3f} "
                f"{arm['accept']:.3f} {arms[label]['prefill64k']:.1f}"
            )

    print("\nA16 versus median A8 controls")
    print(
        "64K prefill: "
        f"{pct(arms['a16-candidate']['prefill64k'], controls['prefill64k']):+.2f}%"
    )
    for concurrency in (1, 4):
        candidate = arms["a16-candidate"][concurrency]
        control = controls[concurrency]
        print(
            f"c{concurrency}: steps={pct(candidate['steps'], control['steps']):+.2f}% "
            f"tokens={pct(candidate['tokens'], control['tokens']):+.2f}% "
            f"accept={pct(candidate['accept'], control['accept']):+.2f}%"
        )
        for context in CONTEXTS:
            print(
                f"  {context}: steps="
                f"{pct(candidate['by_context'][context], control['by_context'][context]):+.2f}%"
            )

    print("\nA8 control 2 versus A8 control 1")
    for concurrency in (1, 4):
        first = arms["a8-control-1"][concurrency]
        second = arms["a8-control-2"][concurrency]
        print(
            f"c{concurrency}: steps={pct(second['steps'], first['steps']):+.2f}% "
            f"tokens={pct(second['tokens'], first['tokens']):+.2f}% "
            f"accept={pct(second['accept'], first['accept']):+.2f}%"
        )
    print(
        "64K prefill: "
        f"{pct(arms['a8-control-2']['prefill64k'], arms['a8-control-1']['prefill64k']):+.2f}%"
    )


if __name__ == "__main__":
    main()
