#!/usr/bin/env python3
"""Summarize the DS4 c4 engine-start distribution study."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONTEXTS = (0, 16384, 32768, 65536, 131072)


def geomean(values: list[float]) -> float:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError(f"invalid geometric-mean inputs: {values}")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    rows = {int(row["context_tokens"]): row for row in data["results"]}
    if set(rows) != set(CONTEXTS):
        raise ValueError(f"{path}: unexpected contexts {sorted(rows)}")
    return {
        "steps_geo": geomean([float(rows[ctx]["server_steps_per_s"]) for ctx in CONTEXTS]),
        "tokens_geo": geomean([float(rows[ctx]["aggregate_tps"]) for ctx in CONTEXTS]),
        "accept_geo": geomean([float(rows[ctx]["server_spec_accept_length"]) for ctx in CONTEXTS]),
        "steps_64k": float(rows[65536]["server_steps_per_s"]),
        "tokens_64k": float(rows[65536]["aggregate_tps"]),
        "prefill_64k": float(data["prefill"]["65536"]["client_tok_per_sec"]),
    }


def load_contexts(path: Path) -> dict[int, dict[str, float]]:
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    rows = {int(row["context_tokens"]): row for row in data["results"]}
    if set(rows) != set(CONTEXTS):
        raise ValueError(f"{path}: unexpected contexts {sorted(rows)}")
    return {
        context: {
            "steps": float(row["server_steps_per_s"]),
            "tokens": float(row["aggregate_tps"]),
            "accept": float(row["server_spec_accept_length"]),
        }
        for context, row in rows.items()
    }


def pct(candidate: float, baseline: float) -> float:
    return (candidate / baseline - 1.0) * 100.0


def main() -> None:
    boot_rows: dict[str, list[dict]] = {"r18p": [], "r34": []}
    print("arm boot steps_geo tokens_geo accept_len prefill64k")
    for arm in ("r18p", "r34"):
        for boot in (1, 2, 3):
            sweeps = [load(ROOT / "results" / f"{arm}-boot{boot}-sweep{sweep}.json") for sweep in (1, 2)]
            row = {
                key: statistics.median(sample[key] for sample in sweeps)
                for key in sweeps[0]
            }
            boot_rows[arm].append(row)
            print(
                f"{arm} {boot} {row['steps_geo']:.3f} {row['tokens_geo']:.3f} "
                f"{row['accept_geo']:.3f} {row['prefill_64k']:.1f}"
            )

    print("\npaired r18p-minus-r34 deltas")
    for boot, (r18p, r34) in enumerate(zip(boot_rows["r18p"], boot_rows["r34"]), start=1):
        print(
            f"pair{boot}: steps={pct(r18p['steps_geo'], r34['steps_geo']):+.2f}% "
            f"tokens={pct(r18p['tokens_geo'], r34['tokens_geo']):+.2f}% "
            f"accept={pct(r18p['accept_geo'], r34['accept_geo']):+.2f}% "
            f"prefill64k={pct(r18p['prefill_64k'], r34['prefill_64k']):+.2f}%"
        )

    print("\narm medians across boots")
    medians = {}
    for arm in ("r18p", "r34"):
        medians[arm] = {
            key: statistics.median(row[key] for row in boot_rows[arm])
            for key in boot_rows[arm][0]
        }
        spread = (
            min(row["steps_geo"] for row in boot_rows[arm]),
            max(row["steps_geo"] for row in boot_rows[arm]),
        )
        print(
            f"{arm}: steps_geo={medians[arm]['steps_geo']:.3f} "
            f"range={spread[0]:.3f}..{spread[1]:.3f} "
            f"tokens_geo={medians[arm]['tokens_geo']:.3f} "
            f"accept_len={medians[arm]['accept_geo']:.3f} "
            f"prefill64k={medians[arm]['prefill_64k']:.1f}"
        )

    print(
        "r18p median delta: "
        f"steps={pct(medians['r18p']['steps_geo'], medians['r34']['steps_geo']):+.2f}% "
        f"tokens={pct(medians['r18p']['tokens_geo'], medians['r34']['tokens_geo']):+.2f}% "
        f"accept={pct(medians['r18p']['accept_geo'], medians['r34']['accept_geo']):+.2f}% "
        f"prefill64k={pct(medians['r18p']['prefill_64k'], medians['r34']['prefill_64k']):+.2f}%"
    )

    print("\ncontext medians across boot medians")
    context_medians: dict[str, dict[int, dict[str, float]]] = {}
    for arm in ("r18p", "r34"):
        context_medians[arm] = {}
        for context in CONTEXTS:
            boot_values = []
            for boot in (1, 2, 3):
                sweeps = [
                    load_contexts(ROOT / "results" / f"{arm}-boot{boot}-sweep{sweep}.json")
                    for sweep in (1, 2)
                ]
                boot_values.append(
                    {
                        metric: statistics.median(sweep[context][metric] for sweep in sweeps)
                        for metric in ("steps", "tokens", "accept")
                    }
                )
            context_medians[arm][context] = {
                metric: statistics.median(boot[metric] for boot in boot_values)
                for metric in ("steps", "tokens", "accept")
            }

    print("context r18p_steps r34_steps delta_steps delta_tokens delta_accept")
    for context in CONTEXTS:
        r18p = context_medians["r18p"][context]
        r34 = context_medians["r34"][context]
        print(
            f"{context} {r18p['steps']:.3f} {r34['steps']:.3f} "
            f"{pct(r18p['steps'], r34['steps']):+.2f}% "
            f"{pct(r18p['tokens'], r34['tokens']):+.2f}% "
            f"{pct(r18p['accept'], r34['accept']):+.2f}%"
        )


if __name__ == "__main__":
    main()
