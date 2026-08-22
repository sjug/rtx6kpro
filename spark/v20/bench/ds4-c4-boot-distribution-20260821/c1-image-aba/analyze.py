#!/usr/bin/env python3
"""Summarize the contemporary r18p/r34/r18p c1 comparison."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STUDY_ROOT = ROOT.parent
CONTEXTS = (0, 16384, 32768, 65536, 131072)


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    rows = {
        int(row["context_tokens"]): row
        for row in data["results"]
        if int(row["concurrency"]) == 1
    }
    if set(rows) != set(CONTEXTS):
        raise ValueError(f"{path}: {sorted(rows)}")
    return {
        "steps": geomean(
            [float(rows[context]["server_steps_per_s"]) for context in CONTEXTS]
        ),
        "tokens": geomean(
            [float(rows[context]["aggregate_tps"]) for context in CONTEXTS]
        ),
        "accept": geomean(
            [float(rows[context]["server_spec_accept_length"]) for context in CONTEXTS]
        ),
        "by_context": {
            context: float(rows[context]["server_steps_per_s"])
            for context in CONTEXTS
        },
        "prefill64k": float(data["prefill"]["65536"]["client_tok_per_sec"]),
    }


def median_sweeps(paths: tuple[Path, Path]) -> dict:
    sweeps = [load(path) for path in paths]
    return {
        metric: statistics.median(sweep[metric] for sweep in sweeps)
        for metric in ("steps", "tokens", "accept", "prefill64k")
    } | {
        "by_context": {
            context: statistics.median(
                sweep["by_context"][context] for sweep in sweeps
            )
            for context in CONTEXTS
        }
    }


def pct(candidate: float, baseline: float) -> float:
    return (candidate / baseline - 1.0) * 100.0


def main() -> None:
    arms = {
        "r18p-opening": median_sweeps(
            tuple(
                STUDY_ROOT
                / "a8-a16-crossover"
                / "results"
                / f"a8-control-2-sweep{sweep}.json"
                for sweep in (1, 2)
            )
        ),
        "r34": median_sweeps(
            tuple(ROOT / "results" / f"r34-sweep{sweep}.json" for sweep in (1, 2))
        ),
        "r18p-closing": median_sweeps(
            tuple(ROOT / "results" / f"r18p-sweep{sweep}.json" for sweep in (1, 2))
        ),
    }
    control = {
        metric: statistics.median(
            arms[label][metric] for label in ("r18p-opening", "r18p-closing")
        )
        for metric in ("steps", "tokens", "accept", "prefill64k")
    } | {
        "by_context": {
            context: statistics.median(
                arms[label]["by_context"][context]
                for label in ("r18p-opening", "r18p-closing")
            )
            for context in CONTEXTS
        }
    }

    print("label steps_geo tokens_geo accept_len prefill64k")
    for label, arm in arms.items():
        print(
            f"{label} {arm['steps']:.3f} {arm['tokens']:.3f} "
            f"{arm['accept']:.3f} {arm['prefill64k']:.1f}"
        )

    print("\nr18p median controls versus r34")
    print(
        f"steps={pct(control['steps'], arms['r34']['steps']):+.2f}% "
        f"tokens={pct(control['tokens'], arms['r34']['tokens']):+.2f}% "
        f"accept={pct(control['accept'], arms['r34']['accept']):+.2f}% "
        f"prefill64k={pct(control['prefill64k'], arms['r34']['prefill64k']):+.2f}%"
    )
    for context in CONTEXTS:
        print(
            f"  {context}: steps="
            f"{pct(control['by_context'][context], arms['r34']['by_context'][context]):+.2f}%"
        )

    print("\nr18p closing versus opening control")
    print(
        f"steps={pct(arms['r18p-closing']['steps'], arms['r18p-opening']['steps']):+.2f}% "
        f"tokens={pct(arms['r18p-closing']['tokens'], arms['r18p-opening']['tokens']):+.2f}% "
        f"accept={pct(arms['r18p-closing']['accept'], arms['r18p-opening']['accept']):+.2f}% "
        f"prefill64k={pct(arms['r18p-closing']['prefill64k'], arms['r18p-opening']['prefill64k']):+.2f}%"
    )


if __name__ == "__main__":
    main()
