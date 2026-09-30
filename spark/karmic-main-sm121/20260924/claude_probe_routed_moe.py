#!/usr/bin/env python3
"""Isolated bitwise repeatability of the B12X routed MoE at DS4.1 geometry, on one boot's frozen plans.

DIAGNOSTIC ONLY. It mirrors the serving call chain at vLLM 1794dcf1
(vllm/model_executor/layers/fused_moe/b12x.py) and B12X a7d7d29b:

  plan_weights(fp4_e8m0_k32, w31; A8 silu bf16) -> prepare_weights
  -> plan_execution(ExecutionCapacity(max_tokens=8192, top_k=6,
                    warmup_token_counts=COUNTS, route_num_experts=0),
                    RoutingSpec(apply_router_weight_on_input=False),
                    invocation={"tuning_route_pattern": TUNING_WORKLOAD_VERSION})
  -> plan.request(prepare_calls, benchmark_calls) -> PreparationSession
  -> fused_moe.bind(plan, scratch=workspace2 bytes, a, experts, topk_weights,
                    topk_ids, output=fused_out) -> fused_moe.run

Production resolution: `variant_for` uses an exact planned variant or else the
8192 capacity variant. The DS4.1 fixed counts are capture sizes and planned
decode counts (all <= 32; the target is not torch.compile'd), so every
prefill of 33..8192 rows, including 128, 254, 255 and 256, runs the
capacity variant. The probe refuses a probed row count that would resolve
differently.

Exact frozen selections (no escape hatches):
  * --tuning-receipt is the boot's selection-cache file itself. Its name must
    be the digest of its identity, its identity must equal this process's
    `cache_identity(namespace, device)`, and it must live in
    `_cute_compile_cache_dir()/preparation`, so the compiled programs are that
    boot's as well.
  * The session runs with autotune=False and cache_only=True against a
    private copy of that file (the original is never opened for writing): a
    missing selection or compiled program raises instead of defaulting,
    measuring or compiling. Every variant must report source "cached".
    (autotune=False alone is warm-up-only in this B12X: it assigns defaults
    without consulting the cache.)
  * Serving switches B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1 and B12X_MOE_FORCE_A8=1
    are required; every B12X_* variable is printed.

Serving memory layout: one uint8 arena per row count, as WorkspaceManager
leases it for the MoE: fused_out (rows x 5120 bf16) at offset 0, then
workspace2 at the next 256-byte boundary with 2*ceil(sum(plan scratch)/2)
bytes (B12xExperts.workspace_shapes).

Repeats: output is NaN-filled before each run (unwritten rows fail), optional
scratch poison 0/127/255, optional side-stream load that runs concurrently
with every MoE launch. The side operands are initialized on the main stream
and the side stream waits for them once before its first use. Comparison is
incremental against repeat 0; memory stays bounded for any repeat count; the
first failing output, the reference and the inputs are saved once.

Weights and hidden states are synthetic and fixed (seeded); routing is B12X's
make_routing_ids workload or a --routes-file capture. This checks bitwise
repeatability of the frozen production plan, not accuracy: a repeatable but
wrong kernel passes.

Exit: 0 PASS, 1 FAIL (any nonrepeatable or non-finite repeat), 2 REFUSED.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import torch

REQUIRED_ENV = {"B12X_DYNAMIC_DETERMINISTIC_OUTPUT": "1", "B12X_MOE_FORCE_A8": "1"}
# DS4.1 serving capture sizes (upstream-launch.json) plus the 8192 prefill chunk.
DEFAULT_COUNTS = "1,2,3,4,5,6,7,8,12,16,20,24,28,32,8192"
POISON = (0, 127, 255)


class Refused(SystemExit):
    def __init__(self, message):
        print(f"CLAUDE-ROUTED-MOE-PROBE-REFUSED: {message}", flush=True)
        super().__init__(2)


def require(condition, message):
    if not condition:
        raise Refused(message)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tuning-receipt", required=True,
                   help="The boot's selection-cache file (<compile cache>/preparation/<digest>.json)")
    p.add_argument("--rows", default="128,254,255,256", help="Live row counts to repeat")
    p.add_argument("--repeats", type=int, default=1000)
    p.add_argument("--counts", default=DEFAULT_COUNTS, help="Serving warmup_token_counts")
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--num-experts", type=int, default=384)
    p.add_argument("--hidden", type=int, default=5120)
    p.add_argument("--intermediate", type=int, default=576, help="Per-rank intermediate (2304 / TP4)")
    p.add_argument("--topk", type=int, default=6)
    p.add_argument("--swiglu-limit", type=float, default=10.0,
                   help="Pinned DS4.1 config.json text_config.swiglu_limit")
    p.add_argument("--ids-dtype", choices=("int32", "int64"), default="int32",
                   help="Router output dtype (fused_topk_bias defaults to int32 on the no-EP path)")
    p.add_argument("--routing", default="shared_40", help="B12X routing workload name")
    p.add_argument("--routes-file", type=Path, help="torch.save'd {'ids': [rows,topk], 'weights': [rows,topk]"
                   " (, 'hidden': [rows,H])}; overrides --routing for its row count")
    p.add_argument("--poison", action="store_true", help="Fill workspace2 with 0/127/255 between repeats")
    p.add_argument("--side-stream-load", action="store_true",
                   help="Run bf16 matmuls on a second stream concurrently with every MoE launch")
    p.add_argument("--side-shape", default="256,5120,1152", help="M,K,N of the side matmul")
    p.add_argument("--side-iters", type=int, default=16)
    p.add_argument("--stop-after", type=int, default=0, help="Stop a row count after N failures (0: never)")
    p.add_argument("--out-prefix", default="claude-routed-moe", help="Failure tensors: <prefix>-first-failure-m<rows>.pt")
    p.add_argument("--seed", type=int, default=20260925)
    args = p.parse_args(argv)
    require(args.repeats >= 2, "need at least two repeats")
    return args


# ---- pure helpers (CPU-tested) -------------------------------------------------------------

def round_up(value, multiple):
    return -(-int(value) // multiple) * multiple


def arena_layout(rows, hidden, scratch_nbytes):
    """Offsets of fused_out and workspace2 in the serving MoE lease (bytes)."""
    output_bytes = rows * hidden * 2                        # bf16 fused_out, spec 0
    workspace2_bytes = 2 * max(1, -(-int(scratch_nbytes) // 2))   # bf16 elements, ceil
    offset = round_up(output_bytes, 256)
    return {"output_bytes": output_bytes, "workspace2_offset": offset,
            "workspace2_bytes": workspace2_bytes, "total": offset + workspace2_bytes}


def parse_counts(text, max_tokens):
    counts = tuple(sorted({int(max_tokens), *(int(c) for c in text.split(","))}))
    require(all(0 < c <= max_tokens for c in counts), "counts must lie in 1..max_tokens")
    return counts


def resolves_to_capacity(rows, counts):
    return rows not in counts and 0 < rows <= max(counts)


def check_selection_file(path, digest):
    """Load the boot's selection cache; its name must be the digest of its identity."""
    payload = json.loads(Path(path).read_text())
    require(isinstance(payload, dict) and set(payload) == {"identity", "records"},
            "tuning receipt is not a B12X selection-cache file")
    require(Path(path).stem == digest(payload["identity"]),
            "tuning receipt file name is not the digest of its identity")
    return payload


class Comparator:
    """Bitwise comparison against repeat 0 with bounded state."""

    def __init__(self):
        self.reference = None
        self.repeats = self.mismatches = self.nonfinite = 0
        self.first_failure = None
        self.failure_output = None
        self.changed_rows = None
        self.max_abs = 0.0

    def update(self, repeat, output):
        self.repeats += 1
        finite = bool(torch.isfinite(output).all())
        if self.reference is None:
            self.reference = output.clone()
            self.changed_rows = torch.zeros(output.shape[0], dtype=torch.bool, device=output.device)
            if not finite:
                self._fail(repeat, output, nonfinite=True)
            return finite
        same = finite and torch.equal(output, self.reference)
        if not same:
            if finite:
                diff = (output.float() - self.reference.float()).abs()
                self.max_abs = max(self.max_abs, float(diff.max()))
                self.changed_rows |= diff.amax(dim=1) > 0
            self._fail(repeat, output, nonfinite=not finite)
        return same

    def _fail(self, repeat, output, nonfinite):
        self.mismatches += 1
        self.nonfinite += int(nonfinite)
        if self.first_failure is None:
            self.first_failure = repeat
            self.failure_output = output.clone()

    def summary(self):
        rows = [] if self.changed_rows is None else torch.nonzero(self.changed_rows).flatten().tolist()
        return {"repeats": self.repeats, "mismatches": self.mismatches, "nonfinite": self.nonfinite,
                "first_failure_repeat": self.first_failure, "max_abs_diff": self.max_abs,
                "changed_row_count": len(rows), "changed_rows": rows[:64]}


def make_side_load(shape, device, generator):
    """Operands built on the current stream; the side stream waits for them once."""
    m, k, n = (int(v) for v in shape.split(","))
    side = torch.cuda.Stream(device=device)
    lhs = torch.randn((m, k), dtype=torch.bfloat16, device=device, generator=generator)
    rhs = torch.randn((k, n), dtype=torch.bfloat16, device=device, generator=generator)
    side.wait_stream(torch.cuda.current_stream(device))
    return side, lhs, rhs


# ---- B12X (GPU) ----------------------------------------------------------------------------

def check_env():
    for name, value in REQUIRED_ENV.items():
        require(os.environ.get(name) == value, f"{name} must be {value!r}, got {os.environ.get(name)!r}")
    print(json.dumps({"b12x_env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("B12X_")}}),
          flush=True)


def synthetic_experts(a, device, generator):
    """Random MXFP4 (e2m1 nibbles) experts with e8m0 block scales, W31 layout, serving dtypes."""
    from b12x.moe import fused_moe

    E, H, I = a.num_experts, a.hidden, a.intermediate
    require(H % 32 == 0 and I % 32 == 0, "hidden and intermediate must be multiples of 32")
    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=device, generator=generator)
    w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=device, generator=generator)
    w13_scale = torch.randint(124, 127, (E, 2 * I, H // 32), dtype=torch.uint8, device=device, generator=generator)
    w2_scale = torch.randint(122, 125, (E, H, I // 32), dtype=torch.uint8, device=device, generator=generator)
    ones = torch.ones(E, dtype=torch.float32, device=device)
    weight_plan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(format=fused_moe.PackedSourceFormat("fp4_e8m0_k32"),
                                      w13_layout=fused_moe.W13Layout("w31")),
        activation=fused_moe.ActivationSpec(mode=fused_moe.ActivationMode.A8, nonlinearity="silu",
                                            io_dtype=torch.bfloat16, swiglu_limit=a.swiglu_limit),
        geometry=fused_moe.MoEGeometry(num_experts=E, hidden_size=H, intermediate_size=I),
    )
    return fused_moe.prepare_weights(
        plan=weight_plan,
        weights=fused_moe.PackedWeights(
            w13=w13, w2=w2, w13_block_scales=w13_scale, w2_block_scales=w2_scale,
            w13_global_scales=ones, w2_global_scales=ones.clone(),
            input_scale=ones.clone(), intermediate_scale=ones.clone(), immutable_input_scales=True,
        ),
    )


def prepare_call_factory(state, *, tokens, topk, prepared, num_experts, device):
    """Mirror of vLLM `_PreparedMoECall` (priming tensors only; never used for the probe runs)."""
    from b12x.moe.fused_moe.workloads import make_tuning_routes
    from b12x.preparation import PreparedCall

    scratch = tuple(torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
                    for spec in state.scratch.scratch_specs())
    hidden = torch.empty((tokens, int(prepared.hidden_size)), dtype=torch.bfloat16, device=device)
    generator = torch.Generator(device=device).manual_seed(42)
    source = torch.empty_like(hidden).normal_(mean=0.0, std=0.125, generator=generator)
    output = torch.empty(hidden.shape, dtype=torch.bfloat16, device=device)
    route_ids = make_tuning_routes(tokens, topk, num_experts, device=device)
    rows_idx = torch.arange(tokens, dtype=torch.float32, device=device).unsqueeze(1)
    cols_idx = torch.arange(topk, dtype=torch.float32, device=device).unsqueeze(0)
    route_weights = torch.softmax(rows_idx * 0.03125 + cols_idx * 0.125, dim=-1).contiguous()
    ids = torch.empty_like(route_ids[0])
    weights = torch.empty_like(route_weights)

    def reset():
        output.zero_()
        for buffer in scratch:
            buffer.zero_()

    def produce(pattern=0):
        hidden.copy_(source)
        ids.copy_(route_ids[pattern])
        weights.copy_(route_weights)

    def restore():
        reset()
        produce()

    binding = state.bind(scratch=scratch, a=hidden, experts=prepared, topk_weights=weights,
                         topk_ids=ids, output=output, input_scales_static=True)
    return PreparedCall(run=binding.run, output=output, produce=produce, reset=reset, restore=restore,
                        capture_safe=False,
                        benchmark_producers=tuple((lambda p=p: produce(p)) for p in range(route_ids.shape[0])))


def make_inputs(a, rows, device, generator):
    from b12x.moe.fused_moe.workloads import make_routing_ids

    ids_dtype = getattr(torch, a.ids_dtype)
    hidden = torch.empty((rows, a.hidden), dtype=torch.bfloat16, device=device).normal_(
        mean=0.0, std=0.125, generator=generator)
    if a.routes_file is not None:
        routes = torch.load(a.routes_file, map_location=device)
        ids, weights = routes["ids"], routes["weights"]
        require(tuple(ids.shape) == (rows, a.topk), f"routes file is not [{rows},{a.topk}]")
        if "hidden" in routes:
            hidden = routes["hidden"].to(device=device, dtype=torch.bfloat16).contiguous()
    else:
        ids = make_routing_ids(rows, a.topk, a.num_experts, workload=a.routing, device=device).reshape(rows, a.topk)
        logits = torch.randn((rows, a.topk), dtype=torch.float32, device=device, generator=generator)
        weights = torch.softmax(logits, dim=-1)
    ids = ids.to(ids_dtype).contiguous()
    weights = weights.to(torch.float32).contiguous()
    require(bool(((ids >= 0) & (ids < a.num_experts)).all()), "route ids out of range")
    ordered, _ = ids.sort(dim=1)
    require(bool((ordered[:, 1:] != ordered[:, :-1]).all()), "duplicate expert within a token's top-k")
    require(tuple(hidden.shape) == (rows, a.hidden), "hidden shape differs")
    return hidden, ids, weights


def main(argv=None):
    a = parse_args(argv)
    check_env()
    from b12x._lib.compiler import _cute_compile_cache_dir
    from b12x.moe import fused_moe
    from b12x.moe.fused_moe._preparation import variant_for
    from b12x.moe.fused_moe.workloads import TUNING_WORKLOAD_VERSION
    from b12x.preparation import PreparationSession
    from b12x.preparation._cache import cache_identity, digest
    from b12x.preparation.types import FrozenMapping, require_prepared

    device = torch.device("cuda", torch.cuda.current_device())
    receipt_path = Path(a.tuning_receipt).resolve()
    payload = check_selection_file(receipt_path, digest)
    identity = payload["identity"]
    namespace = identity["namespace"]
    live = cache_identity(namespace, device.index)
    compile_dir = Path(_cute_compile_cache_dir()).resolve()
    print(json.dumps({"receipt": str(receipt_path), "records": len(payload["records"]),
                      "receipt_identity": identity, "live_identity": live,
                      "compile_cache_dir": str(compile_dir)}), flush=True)
    require(live == identity, "live selection-cache identity differs from the receipt")
    require(receipt_path.parent == compile_dir / "preparation",
            f"receipt is not in this process's compile cache ({compile_dir / 'preparation'})")
    require(int(namespace.get("tensor_parallel", 0)) == 4, "receipt namespace is not the TP4 deployment")
    counts = parse_counts(a.counts, a.max_tokens)
    rows_list = [int(r) for r in a.rows.split(",")]
    for rows in rows_list:
        require(resolves_to_capacity(rows, counts),
                f"rows={rows} would not use the capacity variant as serving prefill does")

    generator = torch.Generator(device=device).manual_seed(a.seed)
    prepared = synthetic_experts(a, device, generator)
    plan = fused_moe.plan_execution(
        experts=prepared,
        capacity=fused_moe.ExecutionCapacity(max_tokens=max(counts), top_k=a.topk,
                                             warmup_token_counts=counts, route_num_experts=0),
        routing=fused_moe.RoutingSpec(apply_router_weight_on_input=False),
        invocation=FrozenMapping({"tuning_route_pattern": TUNING_WORKLOAD_VERSION}),
    )
    require(hasattr(plan, "token_counts") and tuple(plan.token_counts) == counts, "composite plan counts differ")

    def calls_for(count):
        return lambda state: prepare_call_factory(state, tokens=count, topk=a.topk, prepared=prepared,
                                                  num_experts=a.num_experts, device=device)

    request = plan.request(name="claude-routed-moe-probe",
                           prepare_calls={c: calls_for(c) for c in counts},
                           benchmark_calls={c: calls_for(c) for c in counts})
    failed = False
    with tempfile.TemporaryDirectory(prefix="claude-routed-moe-selection-") as private:
        copy = Path(private) / receipt_path.name
        shutil.copyfile(receipt_path, copy)
        require(copy.read_bytes() == receipt_path.read_bytes(), "private selection copy differs")
        started = time.monotonic()
        with PreparationSession(device=device, autotune=False, cache_only=True, cache_dir=private,
                                namespace=namespace, compile_workers=0) as session:
            try:
                session.prepare((request,))
            except LookupError as error:
                raise Refused(f"frozen selection or compiled program missing: {error}")
            session.freeze()
            print(json.dumps({"prepared_seconds": round(time.monotonic() - started, 1)}), flush=True)
            sources = {}
            for count, child in sorted(plan.variants.items()):
                selection = None if child.prepared is None else child.prepared.selection
                source = None if selection is None else selection.source
                sources[count] = source
                config = None if selection is None else selection.config
                payload = config.to_dict() if hasattr(config, "to_dict") else repr(config)
                print(json.dumps({"variant": count, "source": source, "config": payload}), flush=True)
            require(all(s == "cached" for s in sources.values()),
                    f"variants without a cached selection: {[c for c, s in sources.items() if s != 'cached']}")

            root = require_prepared(plan, "moe.decode", device)
            scratch_nbytes = sum(spec.nbytes for spec in plan.scratch_specs())
            side = lhs = rhs = None
            if a.side_stream_load:
                side, lhs, rhs = make_side_load(a.side_shape, device, generator)
            for rows in rows_list:
                state = variant_for(root.variants, rows)
                require(state is variant_for(root.variants, max(counts)),
                        f"rows={rows} did not resolve to the capacity variant")
                backend = getattr(plan.variants[max(counts)].prepared.selection.config, "backend", None)
                layout = arena_layout(rows, a.hidden, scratch_nbytes)
                arena = torch.empty(layout["total"], dtype=torch.uint8, device=device)
                output = arena[:layout["output_bytes"]].view(torch.bfloat16).view(rows, a.hidden)
                workspace2 = arena[layout["workspace2_offset"]:layout["total"]]
                hidden, ids, weights = make_inputs(a, rows, device, generator)
                binding = fused_moe.bind(plan, scratch=workspace2, a=hidden, experts=prepared,
                                         topk_weights=weights, topk_ids=ids, output=output,
                                         input_scales_static=True)
                compare = Comparator()
                started = time.monotonic()
                for repeat in range(a.repeats):
                    if a.poison:
                        workspace2.fill_(POISON[repeat % 3])
                    output.fill_(float("nan"))
                    if side is not None:
                        side.wait_stream(torch.cuda.current_stream(device))   # after this repeat's fills
                        with torch.cuda.stream(side):
                            for _ in range(a.side_iters):
                                torch.matmul(lhs, rhs)
                    fused_moe.run(binding=binding)
                    torch.cuda.synchronize(device)
                    compare.update(repeat, output)
                    if a.stop_after and compare.mismatches >= a.stop_after:
                        break
                result = {"rows": rows, "backend": backend, "layout": layout, "routing":
                          str(a.routes_file) if a.routes_file else a.routing, "ids_dtype": a.ids_dtype,
                          "poison": a.poison, "side_stream_load": a.side_stream_load,
                          "seconds": round(time.monotonic() - started, 1),
                          "reference_nonzero": bool(compare.reference.float().abs().max() > 0),
                          **compare.summary()}
                if compare.first_failure is not None:
                    path = Path(f"{a.out_prefix}-first-failure-m{rows}.pt")
                    torch.save({"reference": compare.reference.cpu(), "failure": compare.failure_output.cpu(),
                                "repeat": compare.first_failure, "hidden": hidden.cpu(), "ids": ids.cpu(),
                                "weights": weights.cpu(), "args": vars(a) | {"routes_file": str(a.routes_file)}},
                               path)
                    result["saved"] = str(path.resolve())
                print(json.dumps(result), flush=True)
                require(result["reference_nonzero"], f"rows={rows}: degenerate all-zero output")
                failed |= compare.mismatches > 0
                del arena, output, workspace2, binding, compare
    print("CLAUDE-ROUTED-MOE-PROBE-" + ("FAIL" if failed else "PASS"), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
