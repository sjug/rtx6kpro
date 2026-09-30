#!/opt/venv/bin/python
"""Validity and stability oracle for the B12X DSA indexer row top-k (DS4.1, k=512).

Provenance: written 2026-09-17 for the DS4.1 R38 TP4 nondeterminism attribution.
Runs the production kernel `b12x.attention.dsa_indexer.tiled_topk.run_row_topk`
on synthetic fp32 rows (widened from bf16, as the production prepare kernel does)
and checks every call against a PyTorch reference:
  validity   every selected score >= the k-th largest score, no duplicates,
             indices in range, reported values equal gathered scores, and the
             selected value multiset equals torch.topk's (B12X's own criterion);
  stability  whether repeated identical calls return the same index set.
A validity failure would indicate a kernel defect. Set churn with validity
intact is tie-break order dependence. Prints one JSON document to stdout.
"""
import json
import sys
import time

import torch
from b12x.attention.dsa_indexer.tiled_topk import _resolve_smem_candidate_capacity, run_row_topk

dev = torch.device("cuda")
ROWS, TOPK, CALLS = 8, 512, 25
assert _resolve_smem_candidate_capacity(topk=TOPK) == 1024


def scenario(name, width, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    if name == "fp32_random_no_ties":
        rows = torch.rand((ROWS, width), generator=g, dtype=torch.float32)
    elif name == "bf16_random":  # realistic: bf16 scores over a wide row, natural ties
        rows = (torch.randn((ROWS, width), generator=g) * 2.0).to(torch.bfloat16).float()
    elif name == "bf16_tie_mass_below_buffer":  # ~600 ties straddle rank 512 (< 1024 buffer)
        rows = torch.empty((ROWS, width))
        for r in range(ROWS):
            row = torch.cat([torch.full((200,), 1.5), torch.full((600,), 1.25), torch.rand(width - 800, generator=g) * 0.5])
            rows[r] = row[torch.randperm(width, generator=g)]
        rows = rows.to(torch.bfloat16).float()
    elif name == "bf16_tie_mass_overflow":  # 4000 ties > 1024 buffer -> exact fallback path
        rows = torch.empty((ROWS, width))
        for r in range(ROWS):
            row = torch.cat([torch.full((200,), 1.5), torch.full((4000,), 1.25), torch.rand(width - 4200, generator=g) * 0.5])
            rows[r] = row[torch.randperm(width, generator=g)]
        rows = rows.to(torch.bfloat16).float()
    elif name == "all_equal":
        rows = torch.full((ROWS, width), 0.5)
    else:
        raise ValueError(name)
    return rows.contiguous().to(dev)


def run(name, width, seed=0):
    row_logits = scenario(name, width, seed)
    lengths = torch.full((ROWS,), width, dtype=torch.int32, device=dev)
    out_idx = torch.empty((ROWS, TOPK), dtype=torch.int32, device=dev)
    out_val = torch.empty((ROWS, TOPK), dtype=torch.float32, device=dev)
    ref = torch.topk(row_logits, TOPK, dim=1, largest=True, sorted=True)
    kth = ref.values[:, -1]
    ref_sorted = ref.values.sort(dim=1).values
    sets, failures, elapsed = [], [], []
    for _ in range(CALLS):
        start = time.perf_counter()
        run_row_topk(row_logits=row_logits, lengths=lengths, topk=TOPK, output_values=out_val, output_indices=out_idx)
        torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - start)
        idx, val = out_idx.clone(), out_val.clone()
        li = idx.long()
        if not bool(((li >= 0) & (li < width)).all()):
            failures.append("index out of range or sentinel")
        gathered = torch.gather(row_logits, 1, li.clamp(0, width - 1))
        if not torch.equal(gathered, val):
            failures.append("reported values differ from gathered scores")
        for r in range(ROWS):
            if len(set(li[r].tolist())) != TOPK:
                failures.append(f"row {r}: duplicate index")
        if not bool((gathered >= kth[:, None]).all()):
            failures.append("selected entry below the k-th reference score")
        if not torch.equal(val.sort(dim=1).values, ref_sorted):
            failures.append("selected value multiset differs from torch.topk")
        sets.append([frozenset(li[r].tolist()) for r in range(ROWS)])
    rows = []
    for r in range(ROWS):
        distinct = {s[r] for s in sets}
        churn = max(len(sets[0][r] ^ s[r]) for s in sets)
        n_tied = int((row_logits[r] == kth[r]).sum())
        n_above = int((row_logits[r] > kth[r]).sum())
        rows.append({"distinct_sets": len(distinct), "max_symmetric_difference": churn,
                     "candidates_equal_to_kth": n_tied, "candidates_above_kth": n_above,
                     "boundary_slots": TOPK - n_above})
    return {"scenario": name, "width": width, "calls": CALLS, "validity_failures": sorted(set(failures)),
            "valid_every_call": not failures, "rows": rows,
            "any_set_churn": any(x["distinct_sets"] > 1 for x in rows),
            "kernel_ms_median": round(1000 * sorted(elapsed)[len(elapsed) // 2], 3)}


results = []
for name in ("fp32_random_no_ties", "bf16_random", "bf16_tie_mass_below_buffer", "bf16_tie_mass_overflow", "all_equal"):
    for width in (4096, 32768, 524288):
        if name == "bf16_tie_mass_overflow" and width < 8192:
            width = 8192  # the 4,200-entry tie mass needs a wider row
        results.append(run(name, width))
        print(json.dumps({k: v for k, v in results[-1].items() if k != "rows"}), file=sys.stderr, flush=True)
print(json.dumps({"device": torch.cuda.get_device_name(), "torch": torch.__version__, "results": results}, indent=1))
