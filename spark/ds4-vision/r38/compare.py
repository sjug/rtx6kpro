"""Compare DSv4 grids without inventing metadata absent from the old receipt."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import geometric_mean

p = argparse.ArgumentParser()
p.add_argument("baseline", type=Path)
p.add_argument("candidate", type=Path)
a = p.parse_args()
if hashlib.sha256(a.baseline.read_bytes()).hexdigest() != "11634125bd26c28eaeb9af052c0b405cb502b665a3129f5bcf6fed5dabd6a4ee":
    raise RuntimeError("Not the immutable qualified R32 Vision receipt")
x, y = (json.loads(path.read_text()) for path in (a.baseline, a.candidate))
for key in ("version", "model", "decode_mode", "duration_per_test", "context_lengths",
            "concurrency_levels", "chat_template_kwargs", "temperature", "prefill_mode",
            "max_tokens", "ignore_eos", "max_total_tokens", "decode_warmup_seconds"):
    if x["metadata"][key] != y["metadata"][key]:
        raise RuntimeError(f"Protocol drift: {key}")
for d, expected in ((x, "74e53e710bef141f6f68e722582569f9c6aa388bce405ad6f1423566a2300c9c"),
                    (y, "ea031e1d3d051033f077fc986bf6f8fce04cf9ab52483d5a719ba13114567fc5")):
    if d["run_metadata"]["image_id"] != expected:
        raise RuntimeError("Wrong image arm")
    if d["run_metadata"]["checkpoint_revision"] != "6821d6ad3681a4b137b066b76094fa82ebd0a380":
        raise RuntimeError("Checkpoint mismatch")
    rows = d["results"]
    expected_cells = {(c, m) for c in (1, 2, 4) for m in (0, 16384, 32768, 65536, 131072)}
    if len(rows) != 15 or {(r["concurrency"], r["context_tokens"]) for r in rows} != expected_cells:
        raise RuntimeError("Incomplete grid")
    for row in rows:
        if any(row[f] for f in ("num_errors", "warmup_timed_out", "capacity_limited", "underfilled")):
            raise RuntimeError("Flagged grid cell")
        for field in ("aggregate_tps", "server_steps_per_s", "server_accept_len_effective"):
            if not math.isfinite(row[field]) or row[field] <= 0:
                raise RuntimeError("Invalid measured value")
summary = []
for c in (1, 2, 4):
    row = {"concurrency": c}
    for metric in ("aggregate_tps", "server_steps_per_s", "server_accept_len_effective"):
        before, after = (geometric_mean(r[metric] for r in d["results"] if r["concurrency"] == c) for d in (x, y))
        row[metric] = {"r32": before, "r38": after, "change_pct": 100 * (after / before - 1)}
    summary.append(row)
prefill = []
for context in sorted(x["prefill"], key=int):
    before, after = (d["prefill"][context]["tok_per_sec"] for d in (x, y))
    if not all(math.isfinite(v) and v > 0 for v in (before, after)):
        raise RuntimeError("Invalid prefill measurement")
    prefill.append({"context_tokens": int(context), "r32": before, "r38": after,
                    "change_pct": 100 * (after / before - 1)})
print(json.dumps({"scope": "Historical qualified R32 versus current R38, same checkpoint and profile; not a repeatability study",
                  "baseline": str(a.baseline), "candidate": str(a.candidate),
                  "candidate_sha256": hashlib.sha256(a.candidate.read_bytes()).hexdigest(),
                  "summary": summary, "prefill": prefill}, indent=2, allow_nan=False))
