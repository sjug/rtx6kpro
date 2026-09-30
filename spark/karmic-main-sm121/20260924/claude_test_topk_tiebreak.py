#!/usr/bin/env python3
"""GPU regression for the deterministic DSA top-k tie-break (claude-dsa-topk-tiebreak.patch).

Contract under test, against the stable reference (descending value, ascending
logical index, i.e. torch.sort(descending=True, stable=True)):

  1. Exact index-set equality with the reference for every scenario, including
     exact-key ties straddling rank k, tie masses that overflow the candidate
     buffer (1024 for k=512, 8192 otherwise), rows shorter than k, empty rows,
     -inf entries, and every supported k.
  2. Bitwise-identical `indices` and `values` across repeated calls, order
     included (the patched kernel emits ascending virtual index).
  3. Selected values equal the gathered logits (no score perturbation).
  4. CUDA graph capture and replay reproduce the same output.
  5. Mapped output (output_gather_table) resolves ties on logical position and
     emits the mapped index of those positions.
  6. Supertile fold via run_tiled_topk with carries equals the global reference,
     including ties that fall between carried and fresh candidates.
  7. extent_splits > 1 pseudo-rows equal per-slice references.

Run inside the candidate image (GPU attached). `--stock-module PATH` loads a
different tiled_topk.py (for example the pinned original) and reports failures
instead of raising, to demonstrate the pre-patch behaviour.
"""
import argparse
import importlib.util
import json
import sys
import time

import zlib

import torch

INF = float("inf")


def load_module(path):
    if path is None:
        from b12x.attention.dsa_indexer import tiled_topk as module
        return module
    # Package-qualified name so the module's relative imports (`from .x import`)
    # resolve against the installed b12x package; the package must be importable.
    import b12x.attention.dsa_indexer  # noqa: F401  (registers the parent package)
    name = "b12x.attention.dsa_indexer.tiled_topk_under_test"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def case_seed(width, k, kind):
    """Deterministic per-case seed (Python's str hash is salted per process)."""
    return zlib.crc32(f"{width}:{k}:{kind}".encode()) & 0xFFFF


def stable_reference(logits, lengths, k):
    """(rows, k) int64, ascending, -1 for slots beyond a short row's valid extent."""
    rows, width = logits.shape
    pos = torch.arange(width, device=logits.device)
    masked = logits.masked_fill(pos[None, :] >= lengths[:, None], -INF)
    order = torch.argsort(masked, dim=1, descending=True, stable=True)
    take = min(k, width)
    order = order[:, :take]
    valid = order < lengths[:, None]
    ref = torch.full((rows, k), -1, device=logits.device, dtype=torch.int64)
    ref[:, :take] = torch.where(valid, order, torch.full_like(order, -1))
    return ref.sort(dim=1).values


def make_scores(kind, rows, width, k, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((rows, width), generator=g).to(device)
    if kind == "fp32_random":
        return x
    if kind == "bf16":
        return x.bfloat16().float()
    if kind == "equal":
        return torch.full((rows, width), 0.5, device=device)
    if kind == "ties_at_rank":
        # ~ (k + 90) exact ties above the bulk: ties straddle rank k, inside the buffer.
        s = x.bfloat16().float() - 10.0
        n = min(width, k + 90)
        idx = torch.randperm(width, generator=g)[:n].to(device)
        s[:, idx] = 3.0
        return s
    if kind == "ties_overflow":
        # Tie mass wider than the candidate buffer (1024 for k=512, 8192 otherwise).
        s = x.bfloat16().float() - 10.0
        n = min(width, (1024 if k == 512 else 8192) + 700)
        idx = torch.randperm(width, generator=g)[:n].to(device)
        s[:, idx] = 3.0
        return s
    if kind == "neg_inf_mix":
        s = x.bfloat16().float()
        s[:, ::3] = -INF
        return s
    if kind == "two_levels":
        s = torch.full((rows, width), -1.0, device=device)
        s[:, 1::2] = 2.0
        return s
    raise ValueError(kind)


def lengths_for(width, k, rows, device):
    base = [width, max(0, width - 31), 0, min(width, k), min(width, k + 1), min(width, 513)]
    values = [base[i % len(base)] for i in range(rows)]
    return torch.tensor(values, device=device, dtype=torch.int32)


class Report:
    def __init__(self, strict):
        self.strict = strict
        self.cases = 0
        self.failures = []

    def check(self, ok, label):
        self.cases += 1
        if not ok:
            self.failures.append(label)
            if self.strict:
                raise RuntimeError("CLAUDE-TOPK-TIEBREAK-FAIL " + label)
            print(json.dumps({"fail": label}), flush=True)


def run_row_cases(mod, report, device, widths, ks, kinds, repeats, canonical):
    rows = 6
    for width in widths:
        for k in ks:
            if width < 17:
                continue
            lengths = lengths_for(width, k, rows, device)
            for kind in kinds:
                scores = make_scores(kind, rows, width, k, seed=case_seed(width, k, kind), device=device)
                ref = stable_reference(scores, lengths, k)
                values = torch.empty((rows, k), device=device, dtype=torch.float32)
                indices = torch.empty((rows, k), device=device, dtype=torch.int32)
                label = f"row width={width} k={k} kind={kind}"

                def call():
                    mod.run_row_topk(row_logits=scores, lengths=lengths, topk=k,
                                     output_values=values, output_indices=indices)

                first_i = first_v = None
                for r in range(repeats):
                    call()
                    torch.cuda.synchronize()
                    got = indices.long()
                    report.check(torch.equal(got.sort(dim=1).values, ref), label + f" set repeat={r}")
                    valid = indices >= 0
                    gathered = scores.gather(1, got.clamp_min(0))
                    report.check(torch.equal(values[valid], gathered[valid]), label + f" values repeat={r}")
                    if canonical:
                        long_rows = lengths > k
                        if long_rows.any():
                            seq = got[long_rows]
                            report.check(bool((seq[:, 1:] > seq[:, :-1]).all()), label + f" ascending repeat={r}")
                    if first_i is None:
                        first_i, first_v = indices.clone(), values.clone()
                    else:
                        report.check(torch.equal(indices, first_i) and torch.equal(values, first_v),
                                     label + f" bitwise repeat={r}")
                if kind == "fp32_random":
                    tk = torch.topk(scores.masked_fill(
                        torch.arange(width, device=device)[None] >= lengths[:, None], -INF), min(k, width), dim=1).indices
                    tk = torch.where(tk < lengths[:, None], tk, torch.full_like(tk, -1))
                    padded = torch.full((rows, k), -1, device=device, dtype=torch.int64)
                    padded[:, :tk.shape[1]] = tk
                    report.check(torch.equal(padded.sort(dim=1).values, ref), label + " torch.topk agrees (no ties)")
                # Graph capture and replay.
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    call()
                for _ in range(3):
                    indices.fill_(-99)
                    values.fill_(999.0)
                    graph.replay()
                    torch.cuda.synchronize()
                    report.check(torch.equal(indices, first_i) and torch.equal(values, first_v), label + " graph replay")
                print(json.dumps({"case": label}), flush=True)


def run_mapped_cases(mod, report, device, repeats):
    rows = 9
    g = torch.Generator(device="cpu").manual_seed(7)
    # (4096, 512/1024) is the generic shape; (16384, 512) is the live DS4.1
    # mapped selection (max_candidates = candidate_topk_blocks * 8, topk 512).
    for width, k in ((4096, 512), (4096, 1024), (16384, 512)):
        for kind in ("equal", "ties_at_rank", "bf16"):
            scores = make_scores(kind, rows, width, k, seed=11 + k, device=device)
            lengths = lengths_for(width, k, rows, device)
            gather = torch.stack([torch.randperm(width, generator=g) for _ in range(rows)]).to(device=device, dtype=torch.int32)
            ref_pos = stable_reference(scores, lengths, k)
            ref_mapped = torch.where(ref_pos >= 0, gather.long().gather(1, ref_pos.clamp_min(0)), ref_pos)
            values = torch.empty((rows, k), device=device, dtype=torch.float32)
            indices = torch.empty((rows, k), device=device, dtype=torch.int32)
            label = f"mapped width={width} k={k} kind={kind}"
            first = None
            for r in range(repeats):
                mod.run_row_topk(row_logits=scores, lengths=lengths, topk=k, output_values=values,
                                 output_indices=indices, output_gather_table=gather)
                torch.cuda.synchronize()
                got = indices.long()
                # Positions are recoverable: mapped value -> position via inverse permutation.
                inverse = torch.empty_like(gather, dtype=torch.int64)
                inverse.scatter_(1, gather.long(), torch.arange(width, device=device).expand(rows, width))
                pos = torch.where(got >= 0, inverse.gather(1, got.clamp_min(0)), got)
                report.check(torch.equal(pos.sort(dim=1).values, ref_pos), label + f" positions repeat={r}")
                report.check(torch.equal(got.sort(dim=1).values, ref_mapped.sort(dim=1).values), label + f" mapped repeat={r}")
                if first is None:
                    first = indices.clone()
                else:
                    report.check(torch.equal(indices, first), label + f" bitwise repeat={r}")
            print(json.dumps({"case": label}), flush=True)


def run_fold_cases(mod, report, device, repeats):
    """Mirror _impl.py's supertile streaming fold with block_q=1, block_k=256."""
    block_k = 256
    rows = 4
    for width, supertile in ((4096, 1024), (32768, 4096), (524288, 32768)):
        num_k_tiles = width // block_k
        supertile_tiles = supertile // block_k
        num_chunks = (num_k_tiles + supertile_tiles - 1) // supertile_tiles
        for k in (512, 2048):
            for kind in ("equal", "ties_at_rank", "bf16", "two_levels"):
                scores = make_scores(kind, rows, width, k, seed=3 + width % 977, device=device)
                lengths = torch.tensor([width, width - 300, min(width, 700), 0], device=device, dtype=torch.int32)
                ref = stable_reference(scores, lengths, k)
                k_start = torch.zeros((rows,), device=device, dtype=torch.int32)
                carry_v = torch.empty((2, rows, k), device=device, dtype=torch.float32)
                carry_i = torch.empty((2, rows, k), device=device, dtype=torch.int32)
                out_v = torch.empty((rows, k), device=device, dtype=torch.float32)
                out_i = torch.empty((rows, k), device=device, dtype=torch.int32)
                label = f"fold width={width} supertile={supertile} k={k} kind={kind}"

                def fold():
                    for chunk_idx in range(num_chunks):
                        begin = chunk_idx * supertile_tiles
                        end = min(begin + supertile_tiles, num_k_tiles)
                        chunk_tiles = end - begin
                        chunk_start = begin * block_k
                        chunk_rows = chunk_tiles * block_k
                        tile_logits = scores[:, chunk_start:chunk_start + chunk_rows].contiguous().reshape(-1)
                        is_first = chunk_idx == 0
                        is_last = chunk_idx == num_chunks - 1
                        mod.run_tiled_topk(
                            tile_logits=tile_logits, k_start=k_start, lengths=lengths, topk=k,
                            block_q=1, block_k=block_k,
                            output_values=out_v if is_last else carry_v[chunk_idx % 2],
                            output_indices=out_i if is_last else carry_i[chunk_idx % 2],
                            num_k_tiles=chunk_tiles, input_index_offset=chunk_start,
                            input_extent=chunk_rows, output_index_offset=chunk_start,
                            carry_values=None if is_first else carry_v[(chunk_idx - 1) % 2],
                            carry_indices=None if is_first else carry_i[(chunk_idx - 1) % 2],
                            is_first=is_first,
                        )

                first = None
                for r in range(repeats):
                    fold()
                    torch.cuda.synchronize()
                    got = out_i.long()
                    report.check(torch.equal(got.sort(dim=1).values, ref), label + f" set repeat={r}")
                    valid = out_i >= 0
                    report.check(torch.equal(out_v[valid], scores.gather(1, got.clamp_min(0))[valid]), label + f" values repeat={r}")
                    if first is None:
                        first = out_i.clone()
                    else:
                        report.check(torch.equal(out_i, first), label + f" bitwise repeat={r}")
                print(json.dumps({"case": label}), flush=True)


def run_extent_split_cases(mod, report, device, repeats):
    block_k = 256
    rows, width, splits = 3, 8192, 2
    extent = width // splits
    for k in (512, 1024):
        for kind in ("equal", "ties_at_rank"):
            scores = make_scores(kind, rows, width, k, seed=41, device=device)
            lengths = torch.tensor([width, width - 100, extent + 5], device=device, dtype=torch.int32)
            k_start = torch.zeros((rows,), device=device, dtype=torch.int32)
            out_v = torch.empty((rows * splits, k), device=device, dtype=torch.float32)
            out_i = torch.empty((rows * splits, k), device=device, dtype=torch.int32)
            label = f"extent_splits k={k} kind={kind}"
            first = None
            for r in range(repeats):
                mod.run_tiled_topk(
                    tile_logits=scores.contiguous().reshape(-1), k_start=k_start, lengths=lengths, topk=k,
                    block_q=1, block_k=block_k, output_values=out_v, output_indices=out_i,
                    num_k_tiles=width // block_k, input_extent=extent, extent_splits=splits,
                    output_row_stride=splits,
                )
                torch.cuda.synchronize()
                for s in range(splits):
                    lo, hi = s * extent, (s + 1) * extent
                    slice_len = (lengths - lo).clamp(0, extent)
                    ref = stable_reference(scores[:, lo:hi], slice_len, k)
                    ref = torch.where(ref >= 0, ref + lo, ref)
                    got = out_i[s::splits].long()
                    report.check(torch.equal(got.sort(dim=1).values, ref), label + f" slice={s} repeat={r}")
                if first is None:
                    first = out_i.clone()
                else:
                    report.check(torch.equal(out_i, first), label + f" bitwise repeat={r}")
            print(json.dumps({"case": label}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stock-module", default=None, help="Path to a tiled_topk.py to test instead (reports, no raise)")
    p.add_argument("--repeats", type=int, default=8)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    device = torch.device("cuda")
    mod = load_module(a.stock_module)
    strict = a.stock_module is None
    report = Report(strict)
    # 16384 = live DS4.1 candidate width (k=512); 75000 = block-stage width at
    # 600000 context (ceil(600000 / 8) blocks of 8, k=2048, fp32 block logits).
    widths = (17, 511, 512, 513, 514, 1023, 1025, 4096, 16384, 32768) if a.quick else (17, 511, 512, 513, 514, 1023, 1024, 1025, 2049, 4096, 16384, 32768, 75000, 131072, 524288)
    ks = (512,) if a.quick else (512, 1024, 2048)
    kinds = ("fp32_random", "bf16", "equal", "ties_at_rank", "ties_overflow", "neg_inf_mix", "two_levels")
    started = time.monotonic()
    run_row_cases(mod, report, device, widths, ks, kinds, a.repeats, canonical=strict)
    run_mapped_cases(mod, report, device, a.repeats)
    run_fold_cases(mod, report, device, a.repeats)
    run_extent_split_cases(mod, report, device, a.repeats)
    summary = {"module": a.stock_module or "installed b12x", "checks": report.cases,
               "failures": len(report.failures), "seconds": round(time.monotonic() - started, 1)}
    print(json.dumps(summary), flush=True)
    if report.failures:
        print("CLAUDE-TOPK-TIEBREAK-REPORT failures=%d (expected for the stock kernel)" % len(report.failures))
        sys.exit(0 if not strict else 1)
    print(f"CLAUDE-TOPK-TIEBREAK-PASS checks={report.cases}", flush=True)


if __name__ == "__main__":
    main()
