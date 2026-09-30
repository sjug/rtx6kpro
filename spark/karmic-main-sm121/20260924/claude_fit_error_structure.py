#!/usr/bin/env python3
"""Offline structure fit for one faulty CTA block of the Engram projection GEMM.

CPU only, one Torch thread, float64 partial products. Reads trace dictionaries
written by trace_attention_ops.py (torch.save of {'operators': [...]}) or raw
tensors. Never touches a device, a remote path, or a command.

Trace mode (row 'engram_projection'; keys under row['outputs']):
  python3 claude_fit_error_structure.py trace --trace RUN-node.pt \
      --observed-key projected_kv --reference-key dense_replay_synchronized \
      [--reference-trace OTHER.pt] [--block ROW_TILE,N_TILE | --list-blocks]
      [--tile-m 64 --tile-n 128 --k-tile 128 --stages 3 --tol 2 --top 8]

  observed/reference keys: projected_kv, dense_replay, dense_replay_synchronized.
  With --reference-trace the reference key is read from that trace and every
  operand (quantized values, compact and physical scales, weight prefix, weight
  physical scales) must be bitwise identical between the two traces.
  A block is one CTA work tile of the swapped (tile_m, tile_n) tactic:
  activation rows [ROW_TILE*tile_m, +tile_m), output columns [N_TILE*tile_n,
  +tile_n). Only blocks inside the captured weight prefix can be fitted.

Tensor mode keeps the earlier interface (--x-values ... --reference).
Replace mode needs only outputs: `replace --capture first-mismatch.pt` tests whether
a faulty block equals another tile's fragment (misdirected store) or its sum.

Hypotheses, scored by relative residual on the block's differing elements:
  H1  one 32-wide group g scaled by (2^j - 1), j in -12..12 without 0
      (j = 1 doubled group, j = -1 halved group); plus "missing group" (x -1)
      and a free least-squares factor
  H2  activation values of one k tile taken from a stale stage (tile t - s),
      H2b values and scales stale, H5 activation scales only stale
  H3  weight values of one k tile stale, H3b values and scales, H5b scales only
  H4  a whole activation row replaced by another row of the same block
  H6  per-row table: best single stale k block and best scaled group per row,
      to separate warp-uniform from lane-level staleness
  H7  leftover accumulator registers: output = bf16(Y + G) with G the same
      fragment of another work tile (from the reference output), or Y doubled
  H9  first k tile or first k block contribution missing or doubled
  H10 block replaced by another tile's fragment (a misdirected or lost store)
  H11 forward refill: one k tile (or one 32-wide k block of it) computed from
      the data of k tile t + stages, both operands and scales
  H12 forward refill of the final `stages` k tiles from the NEXT work tile
      (linear index + --grid), every mixture of A, SFA, B, SFB refilled;
      needs the next tile's weight rows in weight_prefix
  H8  early pass of a stage's first wait: k tile s replaced by the previous
      launch's last-tile data left in that stage (needs weight rows of the
      candidate n' tiles; pass the full weight as weight_prefix to cover all)
Residual near the rounding floor explains the fault; near 1.0 does not.
"""
from __future__ import annotations

import argparse
import math
import re
import sys

import numpy as np
import torch

torch.set_num_threads(1)

GROUP = 32


def _load_any(path: str):
    if path.endswith(".npy"):
        return torch.from_numpy(np.load(path))
    return torch.load(path, map_location="cpu", weights_only=False)


def _tensor(path: str) -> torch.Tensor:
    obj = _load_any(path)
    if isinstance(obj, dict) and len(obj) == 1:
        obj = next(iter(obj.values()))
    if not isinstance(obj, torch.Tensor):
        raise SystemExit(f"{path}: expected one tensor, got {type(obj).__name__}")
    return obj.detach().cpu()


def _fp8(t: torch.Tensor) -> torch.Tensor:
    t = t.detach().cpu()
    if t.dtype == torch.uint8:
        t = t.view(torch.float8_e4m3fn)
    if t.dtype != torch.float8_e4m3fn:
        raise SystemExit(f"values must be float8_e4m3fn or uint8, got {t.dtype}")
    return t.to(torch.float64)


def _e8m0(t: torch.Tensor) -> torch.Tensor:
    t = t.detach().cpu()
    if t.dtype == torch.float8_e8m0fnu:
        t = t.view(torch.uint8)
    if t.dtype != torch.uint8:
        raise SystemExit(f"scales must be float8_e8m0fnu or uint8, got {t.dtype}")
    return torch.ldexp(torch.ones(t.shape, dtype=torch.float64), t.to(torch.int64) - 127)


def physical_scales_to_compact(scale_mma: torch.Tensor, rows: int, chunks: int) -> torch.Tensor:
    """Decode the b12x physical MXFP8 scale plane to compact (rows, chunks) uint8.

    Quantizer mapping (wo_mxfp8.py): row32 = r % 32, row4 = (r // 32) % 4,
    tile_m = r // 128, k4 = c % 4, tile_k = c // 4. The logical view is the
    physical (1, Tm, Tk, 32, 4, 4) buffer permuted to (32, 4, Tm, 4, Tk, 1).
    """
    t = scale_mma.detach().cpu()
    if t.dtype == torch.float8_e8m0fnu:
        t = t.view(torch.uint8)
    if t.ndim != 6:
        raise SystemExit(f"physical scale plane must be 6-D, got {tuple(t.shape)}")
    if t.shape[0] == 1 and tuple(t.shape[3:]) == (32, 4, 4):
        t = t.permute(3, 4, 1, 5, 2, 0)  # contiguous physical order -> logical order
    if tuple(t.shape[:2]) != (32, 4) or t.shape[3] != 4 or t.shape[5] != 1:
        raise SystemExit(f"unrecognised scale plane layout {tuple(t.shape)}")
    r = torch.arange(rows)
    c = torch.arange(chunks)
    row32, row4, tile_m = r % 32, (r // 32) % 4, r // 128
    k4, tile_k = c % 4, c // 4
    return t[row32[:, None], row4[:, None], tile_m[:, None], k4[None, :], tile_k[None, :], 0].contiguous()


def _projection_row(trace) -> dict:
    if isinstance(trace, dict) and "operators" in trace:
        rows = [r for r in trace["operators"] if r.get("name") == "engram_projection"]
        if len(rows) != 1:
            raise SystemExit(f"trace holds {len(rows)} engram_projection rows, need exactly one")
        return rows[0]
    if isinstance(trace, dict) and "outputs" in trace:
        return trace
    raise SystemExit("unrecognised trace structure")


def _operands(row: dict) -> dict:
    out = row["outputs"]
    need = ("quantized_values", "quantized_scale_rows", "quantized_scale_mma", "weight_prefix", "weight_scale_mma")
    missing = [k for k in need if k not in out]
    if missing:
        raise SystemExit(f"trace row lacks {missing}; the dense_replay capture is required for weights")
    return {k: out[k].detach().cpu() for k in need}


def _assert_identical(a: dict, b: dict) -> None:
    for k in a:
        x, y = a[k].view(torch.uint8) if a[k].dtype != torch.uint8 else a[k], b[k].view(torch.uint8) if b[k].dtype != torch.uint8 else b[k]
        if x.shape != y.shape or not torch.equal(x.contiguous(), y.contiguous()):
            raise SystemExit(f"operand {k} differs between the two traces; cross-trace comparison refused")
    print("operands identical across traces: " + ", ".join(a))


def _tactic(row: dict, args) -> tuple[int, int, bool]:
    cfg = row.get("dense_config", "")
    m = re.search(r"tile_m=(\d+).*?tile_n=(\d+)", cfg)
    swap = "swap_ab=True" in cfg
    tile_m = args.tile_m or (int(m.group(1)) if m else 64)
    tile_n = args.tile_n or (int(m.group(2)) if m else 128)
    return tile_m, tile_n, swap


def _partials(x, sx, w, sw) -> torch.Tensor:
    m, k = x.shape
    n = w.shape[0]
    g = k // GROUP
    xg = x.view(m, g, GROUP) * sx.view(m, g, 1)
    wg = w.view(n, g, GROUP) * sw.view(n, g, 1)
    return torch.einsum("mgk,ngk->gmn", xg, wg)


def _rel(residual, err) -> float:
    d = float(torch.linalg.vector_norm(err))
    return float(torch.linalg.vector_norm(residual)) / d if d else math.inf


def fit(x, sx, w, sw, obs, ref, *, k_tile: int, stages: int, tol: float, top: int, label: str,
        full_ref=None, block_origin=None, tile_m=None, tile_n=None,
        x_full=None, sx_full=None, w_full=None, sw_full=None, grid=0) -> None:
    """x (R,K) f64 dequant-free fp8 values; sx (R,G); w (C,K); sw (C,G); obs/ref (R,C)."""
    R, K = x.shape
    C = w.shape[0]
    G = K // GROUP
    ulp = torch.ldexp(torch.ones_like(ref), torch.floor(torch.log2(ref.abs().clamp_min(1e-30))) - 7)
    err = obs - ref
    mask = err.abs() > tol * ulp
    if not bool(mask.any()):
        print(f"{label}: no element differs beyond tolerance")
        return
    rows = torch.nonzero(mask.any(dim=1)).flatten()
    cols = torch.nonzero(mask.any(dim=0)).flatten()
    print(f"{label}: differing rows {rows.tolist()} cols {cols.tolist()[:12]}{'...' if cols.numel() > 12 else ''} "
          f"elements {int(mask.sum())} maxabs {float(err.abs().max()):.4g}")
    P = _partials(x[rows], sx[rows], w[cols], sw[cols])
    E = err[rows][:, cols]
    Y = P.sum(dim=0)
    print(f"  reference reconstruction max|sum(P)-ref| = {float((Y - ref[rows][:, cols]).abs().max()):.4g}")
    tg = k_tile // GROUP
    results: list[tuple[float, str]] = []

    for g in range(G):
        pg = P[g]
        denom = float((pg * pg).sum())
        if denom == 0:
            continue
        coef = float((E * pg).sum()) / denom
        results.append((_rel(E - coef * pg, E), f"H1 group {g} (k {g*GROUP}..{g*GROUP+GROUP-1}) free factor {coef:+.4g}"))
        results.append((_rel(E + pg, E), f"H1 group {g} missing (factor -1)"))
        for j in range(-12, 13):
            if j == 0:
                continue
            factor = 2.0 ** j - 1.0
            tag = "doubled" if j == 1 else "halved" if j == -1 else f"scale x2^{j}"
            results.append((_rel(E - factor * pg, E), f"H1 group {g} {tag} (factor {factor:+.4g})"))

    Xr, SXr, Wc, SWc = x[rows].view(rows.numel(), G, GROUP), sx[rows], w[cols].view(cols.numel(), G, GROUP), sw[cols]
    for t in range(G // tg):
        gs = list(range(t * tg, (t + 1) * tg))
        base = P[gs].sum(dim=0)
        for s in range(1, stages + 1):
            if t - s < 0:
                continue
            src = [g - s * tg for g in gs]
            for name, xs, sxs, ws, sws in (
                ("H2 stale activation values", Xr[:, src], SXr[:, gs], Wc[:, gs], SWc[:, gs]),
                ("H2b stale activation values+scales", Xr[:, src], SXr[:, src], Wc[:, gs], SWc[:, gs]),
                ("H5 stale activation scales only", Xr[:, gs], SXr[:, src], Wc[:, gs], SWc[:, gs]),
                ("H3 stale weight values", Xr[:, gs], SXr[:, gs], Wc[:, src], SWc[:, gs]),
                ("H3b stale weight values+scales", Xr[:, gs], SXr[:, gs], Wc[:, src], SWc[:, src]),
                ("H5b stale weight scales only", Xr[:, gs], SXr[:, gs], Wc[:, gs], SWc[:, src]),
            ):
                stale = torch.einsum("mgk,ngk->mn", xs * sxs.unsqueeze(-1), ws * sws.unsqueeze(-1))
                results.append((_rel(E - (stale - base), E), f"{name}: k tile {t} from tile {t-s} (shift {s})"))

    # H6: per-row structure. Each affected row is fitted alone against one stale
    # 32-wide k block (activation values, shift 1..stages) and against one scaled
    # group. Rows of one warp fragment agreeing on the same block and shift
    # indicate a warp-uniform stale read; disagreeing rows indicate lane-level
    # (per-thread) staleness, which the whole-warp models above cannot express.
    print("  per-row fits (H6): row | best stale k block (shift) residual | best scaled group residual")
    Xr_all = x[rows].view(rows.numel(), G, GROUP)
    for i, r in enumerate(rows.tolist()):
        e = E[i]
        best_stale = (math.inf, None)
        for g in range(G):
            for s in range(1, stages + 1):
                gsrc = g - s * tg
                if gsrc < 0:
                    continue
                stale = (Xr_all[i, gsrc] * sx[rows[i], gsrc]) @ (Wc[:, g] * SWc[:, g].unsqueeze(-1)).T
                model = stale - P[g, i]
                res = _rel(e - model, e)
                if res < best_stale[0]:
                    best_stale = (res, f"block {g} from {gsrc} (shift {s})")
        best_scaled = (math.inf, None)
        for g in range(G):
            pg = P[g, i]
            denom = float((pg * pg).sum())
            if denom == 0:
                continue
            coef = float((e * pg).sum()) / denom
            res = _rel(e - coef * pg, e)
            if res < best_scaled[0]:
                best_scaled = (res, f"group {g} factor {coef:+.3g}")
        print(f"    row {r:4d} | {best_stale[1]} {best_stale[0]:.3f} | {best_scaled[1]} {best_scaled[0]:.3f}")

    # H9: the first k tile's contribution lost or doubled (write-after-write
    # between a late zeroing of the accumulators and the first MMA's result, or
    # a first MMA that did not read C): E ~= -sum P over k tile 0, or over k
    # block 0 only, or +sum (doubled).
    for label, gsel in (("k tile 0", list(range(tg))), ("k block 0", [0])):
        pk = P[gsel].sum(dim=0)
        results.append((_rel(E + pk, E), f"H9 {label} contribution missing"))
        results.append((_rel(E - pk, E), f"H9 {label} contribution doubled"))

    # H11: forward refill. A stage released before the warp's shared loads were
    # performed can be overwritten by the producer's next fill of that stage:
    # k tile t is then computed from k tile t + S data (both operands and both
    # scales, same tile). E ~= P(t + S) - P(t) for one k tile, S = stages.
    # (For t >= ktiles - S the refill is the next work tile's data, which needs
    # that tile's weight rows; not modelled here.)
    for t in range(G // tg - stages):
        gs = list(range(t * tg, (t + 1) * tg)); gn = [g + stages * tg for g in gs]
        results.append((_rel(E - (P[gn].sum(dim=0) - P[gs].sum(dim=0)), E), f"H11 k tile {t} computed from k tile {t + stages} (forward refill)"))
        for b in range(tg):
            results.append((_rel(E - (P[gn[b]] - P[gs[b]]), E), f"H11 k block {t}.{b} computed from k tile {t + stages} block {b}"))

    # H12: forward refill from the NEXT work tile for the final `stages` k tiles.
    # Stage s holds k tiles congruent to s (mod stages); after the CTA releases
    # k tile t >= ktiles - stages, the producer refills that stage with the next
    # work tile's k tile t - (ktiles - stages). The next tile of a persistent CTA
    # is linear index + grid (m fastest). The four TMA transactions (A values,
    # A scales, B values, B scales) land separately, so each component is either
    # current or refilled: 15 non-trivial mixtures per k tile. Under swap, A is
    # the activation tile (rows of m') and B the weight tile (rows of n').
    if full_ref is not None and block_origin is not None and x_full is not None and grid:
        Mfull = x_full.shape[0]; Nw = w_full.shape[0]
        mtiles = -(-Mfull // tile_m); ntiles = -(-full_ref.shape[1] // tile_n)
        r0, c0 = block_origin
        cur = (r0 // tile_m) + mtiles * (c0 // tile_n)
        nxt = cur + grid
        if nxt < mtiles * ntiles:
            mt2, nt2 = nxt % mtiles, nxt // mtiles
            lr = rows + (r0 - (r0 // tile_m) * tile_m); lc = cols + (c0 - (c0 // tile_n) * tile_n)
            rr = mt2 * tile_m + lr; cc = nt2 * tile_n + lc
            if int(cc.max()) < Nw:
                valid = rr < Mfull
                ktiles = G // tg
                Xn = torch.zeros(rows.numel(), G, GROUP, dtype=torch.float64); SXn = torch.ones(rows.numel(), G, dtype=torch.float64)
                if bool(valid.any()):
                    Xn[valid] = x_full[rr[valid]].view(int(valid.sum()), G, GROUP); SXn[valid] = sx_full[rr[valid]]
                Wn = w_full[cc].view(cols.numel(), G, GROUP); SWn = sw_full[cc]
                Xc = x[rows].view(rows.numel(), G, GROUP); SXc = sx[rows]; Wc_ = w[cols].view(cols.numel(), G, GROUP); SWc_ = sw[cols]
                cand = []
                for t in range(ktiles - stages, ktiles):
                    gs = list(range(t * tg, (t + 1) * tg)); gn = [g - (ktiles - stages) * tg for g in gs]
                    Pcur = P[gs].sum(dim=0)
                    for bits in range(1, 16):
                        xa = Xn[:, gn] if bits & 1 else Xc[:, gs]
                        sa = SXn[:, gn] if bits & 2 else SXc[:, gs]
                        wb = Wn[:, gn] if bits & 4 else Wc_[:, gs]
                        sb = SWn[:, gn] if bits & 8 else SWc_[:, gs]
                        model = torch.einsum("mgk,ngk->mn", xa * sa.unsqueeze(-1), wb * sb.unsqueeze(-1)) - Pcur
                        parts = "+".join(n_ for n_, b in (("A", 1), ("SFA", 2), ("B", 4), ("SFB", 8)) if bits & b)
                        cand.append((_rel(E - model, E), f"H12 k tile {t} refilled from next tile ({mt2},{nt2}) k tile {t - (ktiles - stages)} [{parts}]"))
                cand.sort(key=lambda item: item[0])
                results.extend(cand[:4])
                # per-k-block granularity: operand fragments are loaded per k block,
                # scale bytes once per stage, so a single overtaken fragment load
                # affects one 32-wide k block of one operand (scales stage-wide).
                candb = []
                for t in range(ktiles - stages, ktiles):
                    gs = list(range(t * tg, (t + 1) * tg)); gn = [g - (ktiles - stages) * tg for g in gs]
                    for b in range(tg):
                        for parts, ab, bb, sfr in (("A", 1, 0, 0), ("B", 0, 1, 0), ("A+B", 1, 1, 0), ("A+B+SFA+SFB", 1, 1, 1)):
                            xa = Xn[:, gn[b]] if ab else Xc[:, gs[b]]; wb = Wn[:, gn[b]] if bb else Wc_[:, gs[b]]
                            sa = SXn[:, gn[b]] if sfr else SXc[:, gs[b]]; sb = SWn[:, gn[b]] if sfr else SWc_[:, gs[b]]
                            model = torch.einsum("mk,nk->mn", xa * sa.unsqueeze(-1), wb * sb.unsqueeze(-1)) - P[gs[b]]
                            candb.append((_rel(E - model, E), f"H12 k block {t}.{b} refilled from next tile ({mt2},{nt2}) k tile {t - (ktiles - stages)} block {b} [{parts}]"))
                candb.sort(key=lambda item: item[0])
                results.extend(candb[:2])
            else:
                print(f"  H12 skipped: next tile ({mt2},{nt2}) needs weight rows up to {int(cc.max())}, have {Nw}")
        else:
            print("  H12 skipped: this CTA has no next work tile under grid", grid)

    Yall = _partials(x, sx, w[cols], sw[cols]).sum(dim=0)
    for i, r in enumerate(rows.tolist()):
        best = min(((_rel(E[i] - (Yall[r2] - Yall[r]), E[i]), r2) for r2 in range(R) if r2 != r), default=None)
        if best is not None:
            results.append((best[0], f"H4 row {r} replaced by row {best[1]} (per-row)"))

    # H7: leftover accumulator contents. If the accumulator registers of this
    # fragment were not zeroed at tile start, the output is bf16(Y + G) with G
    # the previous occupant of those registers. Candidates for G: the same
    # fragment (block-local rows/cols) of every other work tile of this launch,
    # taken from the reference output. Also test G = Y (doubling) and G = -Y.
    if full_ref is not None and block_origin is not None:
        r0, c0 = block_origin
        Mfull, Nfull = full_ref.shape
        Yb = Y
        def bf16(t):
            return t.to(torch.bfloat16).to(torch.float64)
        base = bf16(Yb)
        cand = []
        lr = (rows + r0 - (r0 // tile_m) * tile_m)  # block-local rows -> local index within a tile
        lc = (cols + c0 - (c0 // tile_n) * tile_n)
        for mt in range(-(-Mfull // tile_m)):
            for nt in range(-(-Nfull // tile_n)):
                rr = mt * tile_m + lr
                cc = nt * tile_n + lc
                if int(rr.max()) >= Mfull or int(cc.max()) >= Nfull:
                    continue
                Gt = full_ref[rr][:, cc]
                model = bf16(Yb + Gt) - base
                cand.append((_rel(E - model, E), f"H7 leftover accumulators = fragment of tile ({mt},{nt})"))
        cand.sort(key=lambda item: item[0])
        results.extend(cand[:3])
        results.append((_rel(E - (bf16(2 * Yb) - base), E), "H7 leftover accumulators = same fragment (doubled)"))
        # H10: replacement, not addition: the block holds another tile's values.
        cand2 = []
        for mt in range(-(-Mfull // tile_m)):
            for nt in range(-(-Nfull // tile_n)):
                rr = mt * tile_m + lr
                cc = nt * tile_n + lc
                if int(rr.max()) >= Mfull or int(cc.max()) >= Nfull or (mt, nt) == (r0 // tile_m, c0 // tile_n):
                    continue
                cand2.append((_rel(E - (full_ref[rr][:, cc] - base), E), f"H10 block replaced by fragment of tile ({mt},{nt})"))
        cand2.sort(key=lambda item: item[0])
        results.extend(cand2[:2])
        results.append((_rel(E - (bf16(torch.zeros_like(Yb)) - base), E), "H7 accumulators lost (output ~ 0)"))

    # H8: early pass of a stage's first wait. If a warp passed the wait on stage s
    # before that stage held this tile's data, it consumed whatever the previous
    # occupant left there: for the same persistent kernel in isolation, the
    # previous launch's last tile (m', n') at its k tile 45 + s (stage s always
    # holds k tiles congruent to s mod 3 with 48 k tiles per tile). The output
    # error is then P_garbage(m', n', 45 + s) - P_current(t = s) for the warp's
    # fragment, with rows beyond M zero-filled by TMA. Candidates: every m' and
    # every n' inside the captured weight rows, s in 0..stages-1.
    if full_ref is not None and block_origin is not None and x_full is not None:
        Mfull = x_full.shape[0]
        Nw = w_full.shape[0]
        r0, c0 = block_origin
        lr = rows + (r0 - (r0 // tile_m) * tile_m)
        lc = cols + (c0 - (c0 // tile_n) * tile_n)
        ktiles = G // tg
        cand = []
        for s_ in range(stages):
            kprev = ktiles - stages + s_
            if kprev < 0 or s_ >= ktiles:
                continue
            gs_cur = list(range(s_ * tg, (s_ + 1) * tg))
            gs_prev = list(range(kprev * tg, (kprev + 1) * tg))
            Pcur = P[gs_cur].sum(dim=0)
            for mt in range(-(-Mfull // tile_m)):
                rr = mt * tile_m + lr
                valid = rr < Mfull
                xa = torch.zeros(rows.numel(), tg, GROUP, dtype=torch.float64)
                sa = torch.ones(rows.numel(), tg, dtype=torch.float64)
                if bool(valid.any()):
                    xa[valid] = x_full[rr[valid]].view(int(valid.sum()), G, GROUP)[:, gs_prev]
                    sa[valid] = sx_full[rr[valid]][:, gs_prev]
                for nt in range(Nw // tile_n):
                    cc = nt * tile_n + lc
                    wb = w_full[cc].view(cols.numel(), G, GROUP)[:, gs_prev]
                    sb = sw_full[cc][:, gs_prev]
                    Gp = torch.einsum("mgk,ngk->mn", xa * sa.unsqueeze(-1), wb * sb.unsqueeze(-1))
                    cand.append((_rel(E - (Gp - Pcur), E),
                                 f"H8 stage {s_} early pass: k tile {s_} from previous tile ({mt},{nt}) k tile {kprev}"))
        cand.sort(key=lambda item: item[0])
        results.extend(cand[:3])

    results.sort(key=lambda item: item[0])
    floor = float(torch.linalg.vector_norm(tol * ulp[rows][:, cols])) / float(torch.linalg.vector_norm(E))
    print(f"  rounding floor ~ {floor:.4f}; best {top} hypotheses (relative residual):")
    for res, name in results[:top]:
        print(f"    {res:8.4f}  {name}")


def run_trace(args) -> int:
    trace = _load_any(args.trace)
    row = _projection_row(trace)
    ops = _operands(row)
    if args.reference_trace:
        other = _projection_row(_load_any(args.reference_trace))
        _assert_identical(ops, _operands(other))
        ref_src = other["outputs"]
    else:
        ref_src = row["outputs"]
    if args.observed_key not in row["outputs"] or args.reference_key not in ref_src:
        raise SystemExit(f"keys available: observed {sorted(row['outputs'])}, reference {sorted(ref_src)}")
    obs = row["outputs"][args.observed_key].detach().cpu().to(torch.float64)
    ref = ref_src[args.reference_key].detach().cpu().to(torch.float64)
    obs, ref = obs.reshape(obs.shape[0], -1), ref.reshape(ref.shape[0], -1)
    tile_m, tile_n, swap = _tactic(row, args)
    print(f"tactic tile_m={tile_m} tile_n={tile_n} swap_ab={swap} programs={row.get('dense_programs')}")

    xq = _fp8(ops["quantized_values"])
    M, K = xq.shape
    G = K // GROUP
    sx_c = ops["quantized_scale_rows"].reshape(M, G)
    sx_p = physical_scales_to_compact(ops["quantized_scale_mma"], M, G)
    if not torch.equal(sx_c.view(torch.uint8).contiguous(), sx_p):
        raise SystemExit("activation compact scales differ from the physical plane decode; refusing to fit")
    sx = _e8m0(sx_c)
    wq = _fp8(ops["weight_prefix"])
    Nw = wq.shape[0]
    sw = _e8m0(physical_scales_to_compact(ops["weight_scale_mma"], Nw, G))

    ulp = torch.ldexp(torch.ones_like(ref), torch.floor(torch.log2(ref.abs().clamp_min(1e-30))) - 7)
    mask = (obs - ref).abs() > args.tol * ulp
    blocks = {}
    for r, c in torch.nonzero(mask).tolist():
        blocks.setdefault((r // tile_m, c // tile_n), 0)
        blocks[(r // tile_m, c // tile_n)] += 1
    if not blocks:
        print("observed equals reference within tolerance; nothing to fit")
        return 0
    print("faulty CTA blocks (row_tile, n_tile): elements -> " + ", ".join(f"({a},{b}): {n}" for (a, b), n in sorted(blocks.items())))
    if args.list_blocks:
        return 0
    if args.block:
        rt, nt = (int(v) for v in args.block.split(","))
    else:
        rt, nt = sorted(blocks)[0]
        print(f"no --block given; fitting the first faulty block ({rt},{nt}) only")
    if (rt, nt) not in blocks:
        raise SystemExit(f"block ({rt},{nt}) has no differing element")
    r0, r1 = rt * tile_m, min((rt + 1) * tile_m, M)
    c0, c1 = nt * tile_n, min((nt + 1) * tile_n, obs.shape[1])
    if c1 > Nw:
        raise SystemExit(f"block columns {c0}..{c1-1} exceed the captured weight prefix ({Nw} rows); cannot fit")
    fit(xq[r0:r1], sx[r0:r1], wq[c0:c1], sw[c0:c1], obs[r0:r1, c0:c1], ref[r0:r1, c0:c1],
        k_tile=args.k_tile, stages=args.stages, tol=args.tol, top=args.top,
        label=f"block ({rt},{nt}) rows {r0}..{r1-1} cols {c0}..{c1-1}",
        full_ref=ref, block_origin=(r0, c0), tile_m=tile_m, tile_n=tile_n,
        x_full=xq, sx_full=sx, w_full=wq, sw_full=sw, grid=args.grid)
    return 0


def run_replace(args) -> int:
    """Replacement check on outputs only: does a faulty block equal another tile's fragment?

    Reads a torch.save file holding {'output': obs, 'reference': ref} (the probe's
    first-mismatch capture) or two tensor files. For every faulty (row_tile, n_tile)
    block it reports the closest other tile whose same-local fragment (in the
    reference) matches the observed block, and the closest one under an
    additive model. Exact-or-near replacement points at a misdirected store,
    which the allocator's reuse of the output buffer would otherwise hide.
    """
    if args.capture:
        d = _load_any(args.capture)
        obs, ref = d["output"], d["reference"]
    else:
        obs, ref = _tensor(args.observed), _tensor(args.reference)
    obs = obs.detach().cpu().to(torch.float64).reshape(obs.shape[0], -1)
    ref = ref.detach().cpu().to(torch.float64).reshape(ref.shape[0], -1)
    M, N = ref.shape
    tm, tn = args.tile_m, args.tile_n
    ulp = torch.ldexp(torch.ones_like(ref), torch.floor(torch.log2(ref.abs().clamp_min(1e-30))) - 7)
    mask = (obs - ref).abs() > args.tol * ulp
    blocks = sorted({(r // tm, c // tn) for r, c in torch.nonzero(mask).tolist()})
    if not blocks:
        print("no differing element"); return 0
    print(f"faulty blocks: {blocks}")
    for (rt, nt) in blocks:
        r0, r1 = rt * tm, min((rt + 1) * tm, M); c0, c1 = nt * tn, min((nt + 1) * tn, N)
        sub = mask[r0:r1, c0:c1]
        rows = torch.nonzero(sub.any(dim=1)).flatten(); cols = torch.nonzero(sub.any(dim=0)).flatten()
        Eo = obs[r0:r1, c0:c1][rows][:, cols]; Er = ref[r0:r1, c0:c1][rows][:, cols]
        E = Eo - Er
        best_rep, best_add = [], []
        for mt in range(-(-M // tm)):
            for nt2 in range(-(-N // tn)):
                if (mt, nt2) == (rt, nt): continue
                rr = mt * tm + rows; cc = nt2 * tn + cols
                if int(rr.max()) >= M or int(cc.max()) >= N: continue
                G = ref[rr][:, cc]
                best_rep.append((_rel(Eo - G, Eo), (mt, nt2)))
                best_add.append((_rel(E - G, E), (mt, nt2)))
        best_rep.sort(); best_add.sort()
        print(f"block ({rt},{nt}) rows {rows.tolist()[:6]}.. cols {cols.tolist()[:6]}.. elements {int(sub.sum())} maxabs {float(E.abs().max()):.4g}")
        print(f"  replacement (observed == other tile's fragment): " + ", ".join(f"{t} {r:.3f}" for r, t in best_rep[:3]))
        print(f"  additive   (observed - reference == other fragment): " + ", ".join(f"{t} {r:.3f}" for r, t in best_add[:3]))
        print(f"  self checks: observed==0 {_rel(Eo, Eo) if False else float(torch.linalg.vector_norm(Eo)/max(1e-30, float(torch.linalg.vector_norm(Er)))):.3f} (|obs|/|ref|), doubled {_rel(Eo - 2*Er, Eo):.3f}")
    return 0


def run_tensors(args) -> int:
    x = _fp8(_tensor(args.x_values)); sx = _e8m0(_tensor(args.x_scales))
    w = _fp8(_tensor(args.w_values)); sw = _e8m0(_tensor(args.w_scales))
    obs = _tensor(args.observed).to(torch.float64); ref = _tensor(args.reference).to(torch.float64)
    m, k = x.shape; n = w.shape[0]
    if sx.shape != (m, k // GROUP) or sw.shape != (n, k // GROUP) or w.shape[1] != k:
        raise SystemExit("operand shapes disagree")
    fit(x, sx, w, sw, obs[:m, :n], ref[:m, :n], k_tile=args.k_tile, stages=args.stages, tol=args.tol, top=args.top, label="tensors")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--k-tile", type=int, default=128)
    common.add_argument("--stages", type=int, default=3)
    common.add_argument("--tol", type=float, default=2.0)
    common.add_argument("--top", type=int, default=8)
    t = sub.add_parser("trace", parents=[common])
    t.add_argument("--trace", required=True)
    t.add_argument("--reference-trace")
    t.add_argument("--observed-key", default="projected_kv")
    t.add_argument("--reference-key", default="dense_replay_synchronized")
    t.add_argument("--block", help="ROW_TILE,N_TILE of the single CTA block to fit")
    t.add_argument("--list-blocks", action="store_true")
    t.add_argument("--tile-m", type=int)
    t.add_argument("--tile-n", type=int)
    t.add_argument("--grid", type=int, default=48, help="persistent CTA count; next tile = linear + grid (0 disables H12)")
    q = sub.add_parser("replace", parents=[common])
    q.add_argument("--capture", help="torch.save file with {'output', 'reference'} (probe first-mismatch)")
    q.add_argument("--observed"); q.add_argument("--reference")
    q.add_argument("--tile-m", type=int, default=64); q.add_argument("--tile-n", type=int, default=128)
    r = sub.add_parser("tensors", parents=[common])
    for name in ("x-values", "x-scales", "w-values", "w-scales", "observed", "reference"):
        r.add_argument(f"--{name}", required=True)
    args = ap.parse_args()
    if args.mode == "replace":
        return run_replace(args)
    return run_trace(args) if args.mode == "trace" else run_tensors(args)


if __name__ == "__main__":
    sys.exit(main())
