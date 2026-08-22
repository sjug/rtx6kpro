#!/usr/bin/env python3
"""Compare native NVFP4 with the bracketing r18p A8 controls."""

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


def median_sweeps(paths: tuple[Path, Path]) -> dict:
    sweeps = [load(path) for path in paths]
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
    result["prefill64k"] = statistics.median(
        sweep["prefill64k"] for sweep in sweeps
    )
    return result


def pct(candidate: float, baseline: float) -> float:
    return (candidate / baseline - 1.0) * 100.0


def main() -> None:
    a8_controls = [
        median_sweeps(
            tuple(
                STUDY_ROOT
                / "a8-a16-crossover"
                / "results"
                / f"{label}-sweep{sweep}.json"
                for sweep in (1, 2)
            )
        )
        for label in ("a8-control-1", "a8-control-2")
    ]
    a8 = {
        concurrency: {
            metric: statistics.median(
                control[concurrency][metric] for control in a8_controls
            )
            for metric in ("steps", "tokens", "accept")
        }
        | {
            "by_context": {
                context: statistics.median(
                    control[concurrency]["by_context"][context]
                    for control in a8_controls
                )
                for context in CONTEXTS
            }
        }
        for concurrency in (1, 4)
    } | {
        "prefill64k": statistics.median(
            control["prefill64k"] for control in a8_controls
        )
    }
    native = median_sweeps(
        tuple(ROOT / "results" / f"native-sweep{sweep}.json" for sweep in (1, 2))
    )

    print("mode concurrency steps_geo tokens_geo accept_len prefill64k")
    for label, arm in (("a8", a8), ("native", native)):
        for concurrency in (1, 4):
            print(
                f"{label} {concurrency} {arm[concurrency]['steps']:.3f} "
                f"{arm[concurrency]['tokens']:.3f} "
                f"{arm[concurrency]['accept']:.3f} {arm['prefill64k']:.1f}"
            )

    print("\nnative versus A8")
    print(f"64K prefill: {pct(native['prefill64k'], a8['prefill64k']):+.2f}%")
    for concurrency in (1, 4):
        print(
            f"c{concurrency}: steps={pct(native[concurrency]['steps'], a8[concurrency]['steps']):+.2f}% "
            f"tokens={pct(native[concurrency]['tokens'], a8[concurrency]['tokens']):+.2f}% "
            f"accept={pct(native[concurrency]['accept'], a8[concurrency]['accept']):+.2f}%"
        )
        for context in CONTEXTS:
            print(
                f"  {context}: steps="
                f"{pct(native[concurrency]['by_context'][context], a8[concurrency]['by_context'][context]):+.2f}%"
            )


if __name__ == "__main__":
    main()
