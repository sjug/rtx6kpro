#!/usr/bin/env python3
"""Aggregate nccl-matrix cell JSONs into per-size rankings and a verdict.
Baseline for 'gain vs production' = r33 + NCCL_PROTO=LL,Simple + default
channels (what the serving runners pin today)."""
import json, sys
from pathlib import Path

d = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
cells = {}
for f in sorted(d.glob("*.json")):
    j = json.load(open(f))
    cells[f.stem] = {(r["op"], r["size"]): r["us"] for r in j["results"]}

if not cells:
    sys.exit("no cell JSONs found")

keys = sorted({k for v in cells.values() for k in v})
BASELINE = "r33-pLL-Simple-cdefault"

print(f"{'op':<11}{'size':>8}  best-cell (us)                        "
      f"baseline(us)  gain")
for op, size in keys:
    ranked = sorted(((v[(op, size)], name) for name, v in cells.items()
                     if (op, size) in v))
    best_us, best_name = ranked[0]
    base = cells.get(BASELINE, {}).get((op, size))
    gain = f"{base/best_us:5.2f}x" if base else "  n/a"
    print(f"{op:<11}{size>>10:>6}KB  {best_name:<34}{best_us:>8.1f} "
          f"{base if base else 0:>10.1f}  {gain}")

# Per-image best config over the decode-critical AR band (32-256KB), by
# geomean of latency.
import math
BAND = [(op, s) for op, s in keys if op == "all_reduce" and 32*1024 <= s <= 256*1024]
print("\ndecode-band (32-256KB allreduce) geomean latency by config:")
scored = []
for name, v in cells.items():
    if all(k in v for k in BAND):
        g = math.exp(sum(math.log(v[k]) for k in BAND) / len(BAND))
        scored.append((g, name))
for g, name in sorted(scored)[:10]:
    print(f"  {g:7.1f}us  {name}")
