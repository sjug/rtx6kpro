#!/usr/bin/env python3
"""Fail-closed regression harness for the shared block-FP8 dense GEMM kernel.

Runs, in the image under test and on one idle GPU, every `gemm.block_fp8_linear`
winner recorded in a tuning receipt (default: the combined qualified boot),
each as its own prepared program (the program depends on the capacity), with
synthetic seeded operands, and checks for every case:

  finite    every output element finite (full output, pre-filled with NaN)
  bitwise   every launch identical to the first (SHA-256 of the full output)
  layout    the consumed operand planes match the pinned layouts: shapes of
            values / scale_rows / scale_mma, the scale_mma <-> scale_rows
            mapping on the sampled rows and columns, weight values verbatim,
            weight block scales expanded row/32-chunk, zero K padding
  accuracy  SAMPLED, not a full oracle: for recorded row indices (full width)
            and column indices (full height) spanning first, last, interior
            and the tile boundaries of the winning tactic, a CPU float64
            product of the consumed quantized activations (read back from the
            binding after the run) and the packed weights, both dequantized
            with the E8M0 scales the kernel read (scale_mma), must satisfy
            |out - ref| <= 2^-7 * |ref| + 2^-7 * rms(ref)

Any exception, non-finite value, launch-to-launch difference, layout mismatch
or accuracy breach fails the run. A JSON receipt records environment, the
installed dense source digest, every program key, every case, the sampled
indices and the cpasync contract outcome.

Coverage from receipts/combined-tuning.json at pin a7d7d29b: 152 programs
over K,N in {576x5120, 1280x4096, 1280x8192, 5120x512, 5120x1152, 5120x1792,
6144x25600, 15360x5120} and capacities {1..8, 12, 16, 20, 24, 28, 32, 128, 256, 384, 512,
8192}, including split-K 2 and 4 decode tactics (M <= 8), swapped and
unswapped tiles, tile_k 64 and 128, unroll on and off.

cpasync is not a legal load path for this recipe. `--cpasync-attempt` proves
that on the installed source three ways and fails on anything else:
  1. the recipe knob domain (`knob_values(dense_query(query))["load_path"]`)
     equals ("tma",);
  2. `DenseGemmKernel.can_implement` with the exact operand options the
     contract passes (E4M3, E8M0, 32-wide, bf16 out, the winner's tile,
     cluster (1,1), N, Kphys, batch 1, k/k/n majors) returns False for
     load_path="cpasync" and True for "tma" (positive control);
  3. configuring a cpasync override raises exactly
     ValueError("dense GEMM load_path is outside its recipe/caller domain").

Preparation mirrors the production Engram layer (vLLM 1794dcf1
deepseek_v4_1/b12x_layers.py): the prepare call binds real scratch, source and
output on the prepared state and the prime launch runs `state.run_binding`.

Environment: the code-generation snapshot (split-K atomic path) is read at
import, so the launcher environment is applied before b12x is imported.
`--launch-env NODE` takes it from launch_contract.render(NODE) (compile-relevant
keys only); `--env K=V` overrides. Split-K 4 winners require
B12X_DENSE_SPLITK_TURBO=1, as in serving; that path reduces through BF16
atomics, so a bitwise failure confined to split-K cases is a finding about
that path, not about the fence.

Invocation inside the image (GPU, idle host):
  python3 claude_dense_regression.py --receipt receipts/combined-tuning.json \
      --launch-env dusty --repeats 64 --cpasync-attempt \
      --report receipts/dense-regression-<image>.json
Subsets: --only 6144x25600 --capacities 256,384,512 ; --rows-per-capacity 2
CPU self-test of the checking logic (no GPU, no b12x): --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOL_REL = 2.0 ** -7
CAPACITIES_DEFAULT = "all"
COMPILE_WORKERS_DEFAULT = 20          # task-tree policy (probe_engram_dense_replay.py, probe_block32_tactic.py)
BOUNDARY_BUDGET = 24                  # tile boundaries sampled per axis (each contributes b-1 and b)
ALLOWED_ENV_PREFIXES = ("B12X_", "CUTE_", "CUTLASS_")
ALLOWED_ENV_NAMES = {"CC", "CXX", "CUDA_HOME", "CUDA_PATH", "CUDA_TOOLKIT_PATH", "CUDACXX",
                     "NVCC_APPEND_FLAGS", "NVCC_PREPEND_FLAGS"}
CPASYNC_DOMAIN_ERROR = "dense GEMM load_path is outside its recipe/caller domain"
SCALE_VEC = 32
SCALE_ROW_TILE = 128                  # MXFP8_SCALE_ROW_TILE (wo_mxfp8.py)
SCALE_K_TILE = 4                      # MXFP8_SCALE_K_TILE, in 32-wide chunks (one 128-wide k tile)


def _sha(t) -> str:
    import torch
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def load_recon():
    spec = importlib.util.spec_from_file_location("claude_tuning_key_recon", HERE / "claude_tuning_key_recon.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def enumerate_winners(receipt_path: Path, *, turbo: bool, extra_ks=(576, 15360)):
    """(K, N, capacity, assignment dict) for every block-FP8 linear record in the receipt."""
    recon = load_recon()
    receipt = json.loads(receipt_path.read_text())
    records = receipt["records"]
    hits = []
    matched = set()
    for cap in recon.CAPACITIES:
        for k in sorted({*recon.KS, *extra_ks}):
            for n in recon.NS:
                key = recon.record_key(cap, k, n, turbo=turbo)
                rec = records.get(key)
                if rec is not None:
                    hits.append((k, n, cap, dict(rec["assignment"])))
                    matched.add(key)
    dense = {key for key, rec in records.items() if "split_k_slices" in rec["assignment"]}
    if matched != dense:
        raise RuntimeError(f"Dense receipt coverage mismatch: {len(dense - matched)} unmatched, "
                           f"{len(matched - dense)} unexpected; check shapes and codegen")
    return receipt["identity"], hits


# --------------------------------------------------------------------------
# Pinned layouts (b12x a7d7d29b). The GPU run verifies the consumed planes
# against these facts; the CPU self-test verifies the readers against a CPU
# port of the pinned packing expressions and the test file checks the pinned
# source still contains those expressions.
#   physical_k(K)      = align_up(K, 128)                       (_shared/block_fp8.py)
#   x_q.values         (M, Kphys) e4m3                          (_block_fp8_linear_x_q_from_scratch)
#   x_q.scale_rows     (1, M, Kphys/32) e8m0
#   x_q.scale_mma      physical (1, Tm, Tk, 32, 4, 4).permute(3, 4, 1, 5, 2, 0)
#                      -> (32, 4, Tm, 4, Tk, 1): [r%32, (r//32)%4, r//128, c%4, c//4, 0]
#   weight.values      (N, Kphys) e4m3 verbatim, K padding zero (pack_block_fp8_linear_weight_mxfp8)
#   weight.scale_rows  (1, N, Kphys/32) = block scale (N/32, Kphys/32) expanded 32 rows x 1 chunk
#   weight.scale_mma   same physical packing as x_q (pack_mxfp8_scales_for_dense_gemm)
#   the GEMM consumes values.view(M, Kphys, 1) + scale_mma for both operands   (_preparation._run)
# --------------------------------------------------------------------------

def physical_k(logical_k: int) -> int:
    return (logical_k + 127) // 128 * 128


def expected_shapes(m: int, kphys: int):
    tm, tk = math.ceil(m / SCALE_ROW_TILE), kphys // 128
    return {"values": (m, kphys), "scale_rows": (1, m, kphys // SCALE_VEC), "scale_mma": (32, 4, tm, 4, tk, 1)}


def logical_scales_from_mma(scale_mma, rows) -> "torch.Tensor":
    """(len(rows), Kphys/32) uint8 read through the exact index mapping the kernel's view implies."""
    import torch
    u8 = scale_mma.view(torch.uint8)
    out = []
    for r in rows:
        plane = u8[r % 32, (r // 32) % 4, r // 128, :, :, 0]      # (4 k4, Tk)
        out.append(plane.permute(1, 0).reshape(-1))                  # c = tk*4 + k4
    return torch.stack(out).cpu()


def mirror_pack_scales_cpu(scale_rows_u8) -> "torch.Tensor":
    """CPU port of pinned pack_mxfp8_scales_for_dense_gemm (num_groups=1) for the self-test."""
    import torch
    m, sf_k = scale_rows_u8.shape
    m_tiles, k_tiles = math.ceil(m / SCALE_ROW_TILE), math.ceil(sf_k / SCALE_K_TILE)
    padded = torch.full((1, m_tiles * SCALE_ROW_TILE, k_tiles * SCALE_K_TILE), 127, dtype=torch.uint8)
    padded[:, :m, :sf_k] = scale_rows_u8
    physical = padded.view(1, m_tiles, 4, 32, k_tiles, 4).permute(0, 1, 4, 3, 2, 5).contiguous()
    return physical.permute(3, 4, 1, 5, 2, 0)


def mirror_expand_block_scales_cpu(block_u8, n: int, kphys: int) -> "torch.Tensor":
    """CPU port of pinned _expand_block_scales_to_mxfp8_rows for 32x32 blocks, num_groups=1."""
    m_tiles, k_tiles = math.ceil(n / 32), math.ceil(kphys / 32)
    assert tuple(block_u8.shape) == (m_tiles, k_tiles), block_u8.shape
    return (block_u8.reshape(1, m_tiles, k_tiles)[:, :, None, :, None]
            .expand(1, m_tiles, 32, k_tiles, 1).reshape(1, m_tiles * 32, k_tiles)[:, :n, : kphys // 32].contiguous())


# --------------------------------------------------------------------------
# Sampling and reference
# --------------------------------------------------------------------------

def sample_indices(extent: int, tiles, budget: int = BOUNDARY_BUDGET) -> list[int]:
    """first, last, middle and (b-1, b) around tile boundaries; boundaries thinned evenly to the budget."""
    picks = {0, extent - 1, extent // 2}
    boundaries = sorted({b for t in tiles if t > 0 for b in range(t, extent, t)})
    if len(boundaries) > budget:
        boundaries = [boundaries[(i * len(boundaries)) // budget] for i in range(budget)]
    for b in boundaries:
        picks.update((b - 1, b))
    return sorted(p for p in picks if 0 <= p < extent)


def sampling_for(m: int, n: int, assignment: dict) -> dict:
    tiles = (int(assignment["tile_m"]), int(assignment["tile_n"]), SCALE_ROW_TILE)
    return {"rows": sample_indices(m, tiles), "cols": sample_indices(n, tiles),
            "tiles": list(tiles), "boundary_budget": BOUNDARY_BUDGET}


def dequant_rows_f64(values_u8_or_f8, scales_u8) -> "torch.Tensor":
    """values (R, Kphys) e4m3 * 2^(scale-127) per 32-chunk -> float64 on CPU."""
    import torch
    v = values_u8_or_f8.cpu().to(torch.float64)
    e = (scales_u8.cpu().to(torch.int64) - 127).repeat_interleave(SCALE_VEC, dim=1)
    return torch.ldexp(v, e)


def accuracy_check(out, ref):
    """Return (ok, max_ratio). out and ref are float64 tensors of equal shape."""
    import torch
    err = (out - ref).abs()
    rms = float(torch.sqrt((ref * ref).mean())) if ref.numel() else 0.0
    tol = TOL_REL * ref.abs() + TOL_REL * rms
    ratio = err / tol.clamp_min(1e-30)
    return bool((ratio <= 1.0).all()), float(ratio.max()) if ratio.numel() else 0.0


def sampled_reference(xq, xs_rows, wq, ws_rows, rows, cols, *, chunk=2048):
    """CPU float64 references: (len(rows), N) for the sampled rows and (M, len(cols)) for the sampled columns.

    xq (M, Kphys) e4m3, xs_rows (M, Kphys/32) uint8, wq (N, Kphys) e4m3, ws_rows (N, Kphys/32) uint8,
    all on any device; everything is moved to the CPU in chunks.
    """
    import torch
    m, n = xq.shape[0], wq.shape[0]
    xr = dequant_rows_f64(xq[rows], xs_rows[rows])                        # (R, Kphys)
    ref_rows = torch.empty((len(rows), n), dtype=torch.float64)
    for j in range(0, n, chunk):
        wc = dequant_rows_f64(wq[j:j + chunk], ws_rows[j:j + chunk])     # (chunk, Kphys)
        ref_rows[:, j:j + chunk] = xr @ wc.T
    wcols = dequant_rows_f64(wq[cols], ws_rows[cols])                     # (C, Kphys)
    ref_cols = torch.empty((m, len(cols)), dtype=torch.float64)
    for i in range(0, m, chunk):
        xc = dequant_rows_f64(xq[i:i + chunk], xs_rows[i:i + chunk])
        ref_cols[i:i + chunk] = xc @ wcols.T
    return ref_rows, ref_cols


# --------------------------------------------------------------------------
# GPU run
# --------------------------------------------------------------------------

def apply_env(args) -> dict:
    applied = {}
    if args.launch_env:
        sys.path.insert(0, str(HERE))
        from launch_contract import render  # user's launcher, read only
        env = render(args.launch_env)["env"]
        for name, value in env.items():
            if name in ALLOWED_ENV_NAMES or name.startswith(ALLOWED_ENV_PREFIXES):
                os.environ[name] = str(value)
                applied[name] = str(value)
    for item in args.env:
        name, _, value = item.partition("=")
        os.environ[name] = value
        applied[name] = value
    return applied


def rows_for(capacity: int, per_capacity: int):
    rows = [capacity]
    if per_capacity >= 2 and capacity > 1:
        rows.append(max(1, capacity - 1))
    if per_capacity >= 3 and capacity > 2:
        rows.append(max(1, capacity // 2))
    return sorted(set(rows), reverse=True)


def verify_packed_weight(packed, w_e4m3, s_u8, k: int, n: int, cols) -> dict:
    """Consumed weight planes against the pinned packing and the harness inputs (sampled columns)."""
    import torch
    kp = physical_k(k)
    w = packed.weight
    got = {name: tuple(getattr(w, name).shape) for name in ("values", "scale_rows", "scale_mma")}
    exp = expected_shapes(n, kp)
    if got != exp:
        raise RuntimeError(f"packed weight shapes {got} differ from pinned layout {exp}")
    if w.values.dtype != torch.float8_e4m3fn or w.scale_rows.dtype != torch.float8_e8m0fnu:
        raise RuntimeError("packed weight dtypes differ from pinned layout")
    vals = w.values[cols].cpu()
    if not torch.equal(vals[:, :k].view(torch.uint8), w_e4m3[cols].view(torch.uint8)):
        raise RuntimeError("packed weight values are not the verbatim E4M3 input on sampled columns")
    if kp > k and int(vals[:, k:].view(torch.uint8).ne(0).sum()) != 0:
        raise RuntimeError("packed weight K padding is not zero on sampled columns")
    s_phys = torch.full((n // 32, kp // 32), 127, dtype=torch.uint8)     # pinned pack pads block scales with 127
    s_phys[:, : k // 32] = s_u8.cpu()
    expanded = mirror_expand_block_scales_cpu(s_phys, n, kp)[0]         # (N, Kphys/32) uint8
    rows_u8 = w.scale_rows.view(torch.uint8)[0][cols].cpu()
    if not torch.equal(rows_u8, expanded[cols]):
        raise RuntimeError("packed weight scale_rows differ from the 32x32 block scale expansion")
    mma_u8 = logical_scales_from_mma(w.scale_mma, cols)
    if not torch.equal(mma_u8, rows_u8):
        raise RuntimeError("packed weight scale_mma does not map to scale_rows on sampled columns")
    return {"shapes": {k_: list(v) for k_, v in got.items()}, "columns_checked": len(cols), "k_padding": kp - k}


def verify_binding(binding, rows_count: int, k: int, rows) -> dict:
    """Consumed activation planes: pinned shapes and scale_mma <-> scale_rows on sampled rows."""
    import torch
    kp = physical_k(k)
    x = binding.x_q
    got = {name: tuple(getattr(x, name).shape) for name in ("values", "scale_rows", "scale_mma")}
    exp = expected_shapes(rows_count, kp)
    if got != exp:
        raise RuntimeError(f"binding x_q shapes {got} differ from pinned layout {exp}")
    if x.values.dtype != torch.float8_e4m3fn or x.scale_rows.dtype != torch.float8_e8m0fnu:
        raise RuntimeError("binding x_q dtypes differ from pinned layout")
    rows_u8 = x.scale_rows.view(torch.uint8)[0][rows].cpu()
    mma_u8 = logical_scales_from_mma(x.scale_mma, rows)
    if not torch.equal(mma_u8, rows_u8):
        raise RuntimeError("binding x_q scale_mma does not map to scale_rows on sampled rows")
    pad_finite = True
    if kp > k:
        pad_finite = bool(torch.isfinite(x.values[:, k:].to(torch.float32)).all())
    return {"shapes": {k_: list(v) for k_, v in got.items()}, "rows_checked": len(rows), "padding_finite": pad_finite}


def assert_cpasync_rejected(plan, assignment: dict, caps_factory) -> dict:
    """Three exact checks on the installed source; anything else raises (fail-closed)."""
    import b12x._lib.dense_gemm as dense
    from b12x.gemm import DenseGemmConfig
    from b12x.gemm._tuning import knob_values, operand_options
    from b12x.gemm.block_fp8_linear._tuning import dense_query
    dq = dense_query(plan.query)
    domain = tuple(knob_values(dq)["load_path"])
    if domain != ("tma",):
        raise RuntimeError(f"recipe load_path domain is {domain}, expected ('tma',)")
    opts = operand_options(dq)
    def predicate(load_path):
        return bool(dense.DenseGemmKernel.can_implement(
            dense.get_cutlass_dtype(opts["ab_dtype"]), dense.get_cutlass_dtype(opts["sf_dtype"]), opts["sf_vec_size"],
            dense.get_cutlass_dtype(dq.output_dtype), (assignment["tile_m"], assignment["tile_n"]), (1, 1),
            dq.out_features, dq.in_features, dq.batch, "k", "k", "n",
            load_path=load_path, swap_ab=assignment["swap_ab"], block_fp8=dq.recipe == "block_fp8"))
    if predicate("cpasync"):
        raise RuntimeError("DenseGemmKernel.can_implement admits cpasync for the E4M3/E8M0/32 recipe")
    if not predicate("tma"):
        raise RuntimeError("positive control failed: can_implement rejects the winner's own tma tile")
    cfg = DenseGemmConfig(**{**assignment, "load_path": "cpasync"})
    try:
        from b12x.gemm import block_fp8_linear as bfl
        bfl.plan(caps_factory(), override=cfg).memory_requirements()     # contract.configure -> validate_config
    except ValueError as exc:
        if str(exc) != CPASYNC_DOMAIN_ERROR:
            raise RuntimeError(f"cpasync override rejected with an unexpected message: {exc!r}") from exc
    else:
        raise RuntimeError("cpasync override was accepted by the contract")
    return {"domain": list(domain), "operand_options": {k_: str(v) for k_, v in opts.items()},
            "can_implement_cpasync": False, "can_implement_tma": True, "override_error": CPASYNC_DOMAIN_ERROR,
            "recipe": dq.recipe, "n": dq.out_features, "kphys": dq.in_features}


def run_gpu(args) -> int:
    applied = apply_env(args)
    turbo = os.environ.get("B12X_DENSE_SPLITK_TURBO", "0") == "1"
    import torch
    from b12x.gemm import DenseGemmConfig, block_fp8_linear as bfl
    from b12x.preparation import PreparationSession, PreparedCall
    from b12x.preparation.types import require_prepared
    from b12x._lib.compile_plan import program_keys
    import b12x._lib.dense_gemm as dense_module

    torch.manual_seed(args.seed)
    device = torch.device("cuda", 0)
    identity, winners = enumerate_winners(Path(args.receipt), turbo=turbo)
    if args.only:
        wanted = {tuple(int(v) for v in s.lower().split("x")) for s in args.only}
        winners = [w for w in winners if (w[0], w[1]) in wanted]
    if args.capacities != "all":
        caps_wanted = {int(v) for v in args.capacities.split(",")}
        winners = [w for w in winners if w[2] in caps_wanted]
    if not winners:
        raise SystemExit("no winners selected")
    report = {
        "receipt": str(args.receipt), "receipt_sha256": hashlib.sha256(Path(args.receipt).read_bytes()).hexdigest(),
        "identity": identity, "device": torch.cuda.get_device_name(0), "env_applied": applied,
        "split_k_turbo": turbo, "repeats": args.repeats, "seed": args.seed, "compile_workers": args.compile_workers,
        "dense_source_sha256": hashlib.sha256(Path(dense_module.__file__).read_bytes()).hexdigest(),
        "dense_source_path": dense_module.__file__, "tolerance_rel": TOL_REL, "boundary_budget": BOUNDARY_BUDGET,
        "reference_scope": "sampled rows (full width) and sampled columns (full height); not a full oracle",
        "cases": [], "weights": {}, "cpasync": None,
    }

    def write_report():
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n")

    def caps_for(k, n, cap):
        return bfl.Caps(device=device, max_tokens=cap, in_features=k, out_features=n, block_size=(32, 32))

    # Weights per shape, shared by all capacities of that shape; inputs kept for the layout check.
    weights, inputs = {}, {}
    for k, n, _, _ in winners:
        if (k, n) in weights:
            continue
        g = torch.Generator(device="cpu").manual_seed(args.seed ^ (k * 1000003 + n))
        w = (torch.randn((n, k), generator=g) * 0.3).clamp(-400, 400).to(torch.float8_e4m3fn)
        s = torch.randint(118, 131, (n // 32, k // 32), generator=g, dtype=torch.uint8)
        inputs[(k, n)] = (w, s)
        weights[(k, n)] = bfl.pack_weight(w.to(device), s.to(device), block_size=(32, 32))
    plans = [(k, n, cap, a, bfl.plan(caps_for(k, n, cap), override=DenseGemmConfig(**a))) for k, n, cap, a in winners]

    if args.cpasync_attempt:
        k, n, cap, assignment, plan = max(plans, key=lambda p: (p[0] * p[1], p[2]))
        try:
            report["cpasync"] = {"status": "rejected", "based_on": {"k": k, "n": n, "capacity": cap, "config": assignment},
                                 **assert_cpasync_rejected(plan, assignment, lambda: caps_for(k, n, cap))}
            print("cpasync: domain ('tma',), can_implement(cpasync)=False, can_implement(tma)=True, override ->",
                  CPASYNC_DOMAIN_ERROR, flush=True)
        except Exception as exc:  # noqa: BLE001
            report["cpasync"] = {"status": "FAIL", "error": repr(exc)[:400]}
            write_report()
            print(f"CLAUDE-DENSE-REGRESSION-FAIL cpasync contract check: {exc!r}", flush=True)
            return 1

    scratches = {}

    def make_prepare(entry):
        k, n, cap, _, plan = entry

        def prepare(state):                      # mirrors deepseek_v4_1/b12x_layers.py make_call
            q = state.query
            source = torch.empty((q.max_tokens, q.in_features), device=state.device, dtype=getattr(torch, q.source_dtype))
            output = torch.empty((q.max_tokens, q.out_features, 1), device=state.device, dtype=getattr(torch, q.output_dtype))
            scratch = [torch.empty(spec.shape, dtype=spec.dtype, device=state.device) for spec in state.scratch.scratch_specs()]
            scratches[id(plan)] = scratch
            binding = state.bind(scratch=scratch, source=source, packed_weight=weights[(k, n)], output=output)
            return PreparedCall(run=lambda: state.run_binding(binding), produce=lambda: source.normal_(std=0.25),
                                owners=(weights[(k, n)],))
        return prepare

    t0 = time.time()
    failures = 0
    with PreparationSession(device=device, autotune=False, compile_workers=args.compile_workers) as session:
        session.prepare(tuple(p.request(name=f"dense-regression-{k}x{n}-m{cap}", prepare_call=make_prepare((k, n, cap, a, p)))
                              for k, n, cap, a, p in plans))
        session.freeze()
        report["prepare_seconds"] = time.time() - t0
        for k, n, cap, assignment, plan in plans:
            state = require_prepared(plan, "gemm.block_fp8_linear")
            keys = [{"dialect": p.dialect, "key": p.key, "name": p.name} for p in program_keys(state.dense.gemm)]
            packed = weights[(k, n)]
            scratch = scratches[id(plan)]
            for rows_count in rows_for(cap, args.rows_per_capacity):
                sampling = sampling_for(rows_count, n, assignment)
                case = {"k": k, "n": n, "capacity": cap, "rows": rows_count, "config": assignment, "programs": keys,
                        "sampled_rows": sampling["rows"], "sampled_cols": sampling["cols"], "sampling_tiles": sampling["tiles"]}
                try:
                    if (k, n) not in report["weights"]:
                        w_in, s_in = inputs[(k, n)]
                        report["weights"][f"{k}x{n}"] = verify_packed_weight(packed, w_in, s_in, k, n, sampling["cols"])
                    g = torch.Generator(device="cpu").manual_seed(args.seed ^ (rows_count * 7919 + k * 31 + n))
                    source = (torch.randn((rows_count, k), generator=g) * 0.7).to(torch.bfloat16).to(device)
                    output = torch.empty((rows_count, n), dtype=torch.bfloat16, device=device)
                    binding = bfl.bind(plan, scratch=scratch, source=source, packed_weight=packed, output=output.view(rows_count, n, 1))
                    first = first_out = None
                    for i in range(args.repeats):
                        output.fill_(float("nan"))
                        bfl.run(binding=binding)
                        torch.cuda.synchronize()
                        if not bool(torch.isfinite(output).all()):
                            raise RuntimeError(f"non-finite output at repeat {i}")
                        digest = _sha(output)
                        if first is None:
                            first, first_out = digest, output.clone()
                        elif digest != first:
                            raise RuntimeError(f"bitwise mismatch at repeat {i}: {digest[:12]} != {first[:12]}")
                    case["layout"] = verify_binding(binding, rows_count, k, sampling["rows"])
                    xs = binding.x_q.scale_rows.view(torch.uint8)[0]
                    ws = packed.weight.scale_rows.view(torch.uint8)[0]
                    kp = physical_k(k)
                    xq = binding.x_q.values
                    if not case["layout"]["padding_finite"]:
                        # zero weight padding times a non-finite activation pad would poison the reference only
                        case["reference_k"] = k
                        xq, xs = xq[:, :k], xs[:, : k // SCALE_VEC]
                        wq, wsr = packed.weight.values[:, :k], ws[:, : k // SCALE_VEC]
                    else:
                        case["reference_k"] = kp
                        wq, wsr = packed.weight.values, ws
                    ref_rows, ref_cols = sampled_reference(xq, xs, wq, wsr, sampling["rows"], sampling["cols"])
                    out64 = first_out.to(torch.float64).cpu()
                    ok_r, ratio_r = accuracy_check(out64[sampling["rows"]], ref_rows)
                    ok_c, ratio_c = accuracy_check(out64[:, sampling["cols"]], ref_cols)
                    case.update({"sha256": first, "accuracy_ratio_rows": ratio_r, "accuracy_ratio_cols": ratio_c,
                                 "reference_elements": int(ref_rows.numel() + ref_cols.numel()),
                                 "status": "PASS" if (ok_r and ok_c) else "FAIL-ACCURACY"})
                    if not (ok_r and ok_c):
                        failures += 1
                except Exception as exc:  # noqa: BLE001
                    case.update({"status": "FAIL", "error": repr(exc)[:400]})
                    failures += 1
                report["cases"].append(case)
                print(f"{case['status']:14s} K={k} N={n} cap={cap} rows={rows_count} tile=({assignment['tile_m']},{assignment['tile_n']}) "
                      f"k{assignment['tile_k']} swap={assignment['swap_ab']} splitk={assignment['split_k_slices']} "
                      f"unroll={assignment['large_m_unroll']} ratio_rows={case.get('accuracy_ratio_rows', float('nan')):.3f} "
                      f"ratio_cols={case.get('accuracy_ratio_cols', float('nan')):.3f} "
                      f"sampled={len(sampling['rows'])}r/{len(sampling['cols'])}c", flush=True)
                write_report()
    report["failures"] = failures
    report["seconds"] = time.time() - t0
    write_report()
    if failures:
        print(f"CLAUDE-DENSE-REGRESSION-FAIL failures={failures} cases={len(report['cases'])}", flush=True)
        return 1
    print(f"CLAUDE-DENSE-REGRESSION-PASS cases={len(report['cases'])} programs={len(plans)} repeats={args.repeats} "
          f"cpasync={'checked' if args.cpasync_attempt else 'not-requested'}", flush=True)
    return 0


# --------------------------------------------------------------------------
# CPU self-test (no GPU, no b12x)
# --------------------------------------------------------------------------

class _Obj:
    pass


def fake_rows(values_e4m3, scale_rows_u8):
    """MXFP8Rows-shaped object built with the CPU port of the pinned packing."""
    import torch
    o = _Obj()
    o.values = values_e4m3
    o.scale_rows = scale_rows_u8.reshape(1, *scale_rows_u8.shape).view(torch.float8_e8m0fnu)
    o.scale_mma = mirror_pack_scales_cpu(scale_rows_u8).view(torch.float8_e8m0fnu)
    return o


def self_test() -> int:
    import torch
    torch.manual_seed(0)
    identity, winners = enumerate_winners(HERE / "receipts" / "combined-tuning.json", turbo=True)
    shapes = sorted({(k, n) for k, n, _, _ in winners})
    assert len(winners) == 133, len(winners)
    assert (576, 5120) in shapes and (6144, 25600) in shapes
    assert any(a["split_k_slices"] == 4 for *_, a in winners) and any(a["swap_ab"] for *_, a in winners)
    # pinned layouts on a K=576 (Kphys=640) shape through the CPU mirror of the packing
    m, k, n = 37, 576, 256
    kp = physical_k(k)
    xq = torch.zeros((m, kp), dtype=torch.float8_e4m3fn); xq[:, :k] = (torch.randn(m, k) * 0.5).to(torch.float8_e4m3fn)
    sx = torch.randint(120, 130, (m, kp // 32), dtype=torch.uint8)
    w = (torch.randn(n, k) * 0.3).to(torch.float8_e4m3fn); s = torch.randint(118, 131, (n // 32, k // 32), dtype=torch.uint8)
    wq = torch.zeros((n, kp), dtype=torch.float8_e4m3fn); wq[:, :k] = w
    s_phys = torch.full((n // 32, kp // 32), 127, dtype=torch.uint8); s_phys[:, : k // 32] = s
    sw = mirror_expand_block_scales_cpu(s_phys, n, kp)[0]
    x, wgt = fake_rows(xq, sx), fake_rows(wq, sw)
    assert {nm: tuple(getattr(x, nm).shape) for nm in ("values", "scale_rows", "scale_mma")} == expected_shapes(m, kp)
    assert {nm: tuple(getattr(wgt, nm).shape) for nm in ("values", "scale_rows", "scale_mma")} == expected_shapes(n, kp)
    rows, cols = sample_indices(m, (16, 64, 128)), sample_indices(n, (64, 128, 128))
    assert torch.equal(logical_scales_from_mma(x.scale_mma, rows), sx[rows])
    assert torch.equal(logical_scales_from_mma(wgt.scale_mma, cols), sw[cols])
    packed = _Obj(); packed.weight = wgt
    verify_packed_weight(packed, w, s, k, n, cols)
    b = _Obj(); b.x_q = x
    assert verify_binding(b, m, k, rows)["padding_finite"]
    # sampled reference against a full float64 oracle and a fake bf16 kernel
    full = dequant_rows_f64(xq, sx) @ dequant_rows_f64(wq, sw).T
    ref_rows, ref_cols = sampled_reference(xq, sx, wq, sw, rows, cols, chunk=100)
    assert torch.equal(ref_rows, full[rows]) and torch.equal(ref_cols, full[:, cols])
    fake = full.to(torch.float32).to(torch.bfloat16).to(torch.float64)
    ok, ratio = accuracy_check(fake[rows], ref_rows); assert ok and ratio <= 1.0, ratio
    ok, ratio = accuracy_check(fake[:, cols], ref_cols); assert ok and ratio <= 1.0, ratio
    bad = fake.clone(); bad[rows[1], cols[2]] += 0.5 * abs(float(full[rows[1], cols[2]])) + 1.0
    assert not accuracy_check(bad[rows], ref_rows)[0] and not accuracy_check(bad[:, cols], ref_cols)[0]
    assert _sha(fake.to(torch.bfloat16)) != _sha(bad.to(torch.bfloat16))
    # sampling spans first/last/interior and boundaries, and the budget bounds the size
    r = sample_indices(8192, (64, 128, 128))
    assert r[0] == 0 and r[-1] == 8191 and 4096 in r and 63 in r and 64 in r and len(r) <= 3 + 2 * BOUNDARY_BUDGET
    assert sample_indices(1, (64, 128, 128)) == [0] and sample_indices(3, (64, 128, 128)) == [0, 1, 2]
    assert rows_for(256, 3) == [256, 255, 128] and rows_for(1, 3) == [1]
    print("CLAUDE-DENSE-REGRESSION-SELFTEST-PASS winners=133 shapes=%d" % len(shapes))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipt", default=str(HERE / "receipts" / "combined-tuning.json"))
    ap.add_argument("--launch-env", help="apply launch_contract.render(NODE)['env'] compile-relevant keys before importing b12x")
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE override, repeatable")
    ap.add_argument("--repeats", type=int, default=64)
    ap.add_argument("--rows-per-capacity", type=int, default=2, help="1: capacity; 2: +capacity-1; 3: +capacity//2")
    ap.add_argument("--only", action="append", default=[], help="KxN shape filter, repeatable, e.g. 6144x25600")
    ap.add_argument("--capacities", default=CAPACITIES_DEFAULT, help="comma list or 'all'")
    ap.add_argument("--compile-workers", type=int, default=COMPILE_WORKERS_DEFAULT)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--cpasync-attempt", action="store_true", help="prove the contract rejects cpasync for this recipe")
    ap.add_argument("--report", default=str(HERE / "receipts" / "dense-regression.json"))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if args.repeats < 64:
        raise SystemExit("at least 64 identical launches are required")
    if Path(args.report).exists():
        raise SystemExit(f"refusing to overwrite {args.report}")
    return run_gpu(args)


if __name__ == "__main__":
    sys.exit(main())
