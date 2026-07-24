"""Layer-level CUDA-event microbenchmark: B12X mapped one-grid hybrid MoE decode
(grid48, SM121) vs the serial two-tier decode path, GLM-5.2 TP4 per-rank shapes.

Shapes: m=4, hidden=6144, intermediate=512, 64 NVFP4 kept + 192 NF3 experts,
top_k=8, silu, bf16, tiles (64,256,64,256).

All weights/scales/activations are zeros (no denormal artifacts). Routing is
fixed at 2 kept + 6 NF3 experts per token (uniform-routing expectation for a
64/256 kept tier), 32 distinct routes.

Timing: primary numbers are CUDA-graph replays (matches vLLM production, which
runs decode inside CUDA graphs; removes Python launch overhead). Eager
per-call event timings are reported as a cross-check.
"""

import json
import statistics
import sys

import torch


def log(msg):
    print(f"[bench] {msg}", flush=True)


H, I, NV, NF, TOPK, M = 6144, 512, 64, 192, 8, 4
TILES = (64, 256, 64, 256)
WARMUP, ITERS = 50, 200


def timed_events(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        stops[i].record()
    torch.cuda.synchronize()
    us = [starts[i].elapsed_time(stops[i]) * 1000.0 for i in range(iters)]
    # batch cross-check: one event pair around the whole loop
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    batch_us = s.elapsed_time(e) * 1000.0 / iters
    return {
        "mean_us": statistics.fmean(us),
        "median_us": statistics.median(us),
        "min_us": min(us),
        "batch_us": batch_us,
    }


def main():
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    sms = int(props.multi_processor_count)
    max_shared = int(getattr(props, "shared_memory_per_block_optin", 101_376))
    log(
        f"device={props.name} cap={props.major}.{props.minor} sms={sms} "
        f"shared_optin={max_shared}"
    )
    assert (props.major, props.minor) == (12, 1), "expected SM121"
    assert sms == 48, "expected 48 SMs"

    import b12x

    log(f"b12x version={getattr(b12x, '__version__', 'unknown')}")
    from b12x.moe.fused.w4a16.host import make_w4a16_packed_buffers
    from b12x.moe.fused.w4a16.kernel import (
        compile_w4a16_fused_moe,
        compile_w4a16_hybrid_mapped_grid48,
        run_w4a16_moe,
    )
    from b12x.moe.fused.w4a16.prepare import (
        PreparedNF3MoeWeights,
        W4A16PackedWeights,
        _make_workspace,
    )

    # ---- mapped grid48 launch --------------------------------------------
    log("compiling mapped grid48 launch...")
    launch = compile_w4a16_hybrid_mapped_grid48(
        size_m=M,
        hidden_size=H,
        intermediate_size=I,
        nv_num_experts=NV,
        nf_num_experts=NF,
        top_k=TOPK,
        activation="silu",
        element_dtype="bf16",
        fast_math=True,
        sms=sms,
        max_shared_mem=max_shared,
        force_tile_config=TILES,
    )
    assert int(launch.grid_x) == 48, launch.grid_x
    assert int(launch.workspace_words) == 194, launch.workspace_words
    assert int(launch.shared_memory_bytes) == 45_184, launch.shared_memory_bytes
    log(
        f"mapped launch ok: grid_x={launch.grid_x} "
        f"workspace_words={launch.workspace_words} "
        f"shared={launch.shared_memory_bytes} "
        f"scratch={launch.scratch_elements} regs={launch.registers_per_thread}"
    )

    # ---- fabricated weights (zeros; flat sizes match tensor_contracts) ----
    z = lambda shape, dt: torch.zeros(shape, dtype=dt, device=dev)
    i32, u8, f32 = torch.int32, torch.uint8, torch.float32
    prep_kept = W4A16PackedWeights(
        w13=z((NV, H // 16, (2 * I // 64) * 128), i32),  # 50,331,648 i32
        w13_scale=z((NV, H // 16, 2 * I), u8),  # 25,165,824 B
        w13_global_scale=torch.ones((NV,), dtype=f32, device=dev),
        w2=z((NV, I // 16, (H // 64) * 128), i32),  # 25,165,824 i32
        w2_scale=z((NV, I // 16, H), u8),  # 12,582,912 B
        w2_global_scale=torch.ones((NV,), dtype=f32, device=dev),
        workspace=_make_workspace(dev),
        hidden_size=H,
        intermediate_size=I,
        num_experts=NV,
        is_gated=True,
        params_dtype=torch.bfloat16,
        source_format="modelopt_nvfp4",
        w13_layout="w13",
        weight_layout="packed",
        scale_format="e4m3_k16",
    )
    nf3_units_w13 = (H // 16) * (2 * I // 2)  # 196,608 units * 3 words
    nf3_units_w2 = (I // 16) * (H // 2)  # 98,304 units * 3 words
    nf3_global = torch.full((NF,), 2.0**116, dtype=f32, device=dev)
    prep_nf3 = PreparedNF3MoeWeights(
        w13=z((NF, nf3_units_w13 * 3), i32),  # 113,246,208 i32
        w13_scale=z((NF, H // 32, 2 * I), u8),  # 37,748,736 B
        w13_global_scale=nf3_global,
        w2=z((NF, nf3_units_w2 * 3), i32),  # 56,623,104 i32
        w2_scale=z((NF, I // 32, H), u8),  # 18,874,368 B
        w2_global_scale=nf3_global.clone(),
        workspace=_make_workspace(dev),
        hidden_size=H,
        intermediate_size=I,
        num_experts=NF,
        is_gated=True,
        params_dtype=torch.bfloat16,
        fc1_tile_n=TILES[1],
        fc2_tile_n=TILES[3],
    )

    def mapped_views(p):
        return (
            p.w13.view(i32).view(-1),
            p.w2.view(i32).view(-1),
            p.w13_scale.view(u8).view(i32).view(-1),
            p.w2_scale.view(u8).view(i32).view(-1),
            p.w13_global_scale.view(-1),
            p.w2_global_scale.view(-1),
        )

    weight_views = (*mapped_views(prep_kept), *mapped_views(prep_nf3))

    # ---- routing / activations -------------------------------------------
    x = torch.zeros((M, H), dtype=torch.bfloat16, device=dev)
    # 2 kept + 6 NF3 per token, 32 distinct global expert ids in [0, 256)
    ids = [
        [2 * t, 2 * t + 1] + [64 + 6 * t + j for j in range(6)] for t in range(M)
    ]
    grid_ids = torch.tensor(ids, dtype=i32, device=dev).contiguous()
    assert grid_ids.data_ptr() % 16 == 0
    weights = torch.full((M, TOPK), 1.0 / TOPK, dtype=f32, device=dev)
    # tier_local_map exactly as _combined_tier_local_descriptors encodes it:
    # kept ids 0..63 -> local 0..63; nf3 ids 64..255 -> 0x100 | (id - 64)
    g = torch.arange(256, dtype=i32)
    tier_map = torch.where(g < NV, g, (g - NV) | 0x100).to(dev).contiguous()

    # mapped scratch (dedicated; vLLM borrows equivalent-shape buffers)
    fc1_bf16 = torch.empty((32, 1024), dtype=torch.bfloat16, device=dev)
    activated_bf16 = torch.empty((32, 512), dtype=torch.bfloat16, device=dev)
    out_mapped = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
    fc1_c_tmp = torch.empty((int(launch.scratch_elements),), dtype=f32, device=dev)
    fc2_c_tmp = torch.empty((int(launch.scratch_elements),), dtype=f32, device=dev)
    # Zeroed ONCE: the kernel's grid barrier is self-resetting (count word is
    # reset by the last arriver; sense word increments monotonically). vLLM
    # zeroes this workspace once when arming the path and then replays it in
    # CUDA graphs indefinitely.
    ws_mapped = torch.zeros((int(launch.workspace_words),), dtype=i32, device=dev)

    def run_mapped():
        torch.ops.b12x.w4a16_hybrid_mapped_grid188_launch(
            x,
            *weight_views,
            grid_ids,
            tier_map,
            fc1_bf16,
            activated_bf16,
            out_mapped,
            weights,
            fc1_c_tmp,
            fc2_c_tmp,
            ws_mapped,
            M,
            int(launch.size_m),
            H,
            I,
            NV,
            NF,
            TOPK,
            "silu",
            "bf16",
            True,
            sms,
            max_shared,
            *TILES,
            int(launch.grid_x),
            int(torch.cuda.current_stream(dev).cuda_stream),
        )

    log("mapped: first eager call...")
    run_mapped()
    torch.cuda.synchronize()
    log("mapped: eager call ok")

    # ---- serial two-tier path (mirrors vllm _run_tier decode) ------------
    log("compiling serial TC-decode launches...")
    common = dict(
        hidden_size=H,
        intermediate_size=I,
        top_k=TOPK,
        activation="silu",
        apply_router_weight_on_input=False,
        element_dtype="bf16",
        fast_math=True,
        sms=sms,
        max_shared_mem=max_shared,
        force_tile_config=TILES,
    )
    decode_kept = compile_w4a16_fused_moe(
        size_m=8,
        zero_fc2_output=False,
        moe_block_size=8,
        max_m_blocks=8 * TOPK,
        direct_topk_routes=True,
        tc_decode_fused_sum=True,
        num_experts=NV,
        weight_layout="packed",
        scale_format="e4m3_k16",
        **common,
    )
    decode_nf3 = compile_w4a16_fused_moe(
        size_m=8,
        zero_fc2_output=False,
        moe_block_size=8,
        max_m_blocks=8 * TOPK,
        direct_topk_routes=True,
        tc_decode_fused_sum=True,
        num_experts=NF,
        weight_layout="nf3_2p1",
        scale_format="e4m3_k32",
        **common,
    )
    assert (int(decode_kept.fc1_tile_n), int(decode_kept.fc2_tile_n)) == (256, 256)
    assert (int(decode_nf3.fc1_tile_n), int(decode_nf3.fc2_tile_n)) == (256, 256)
    log("serial launches compiled")

    buffers = make_w4a16_packed_buffers(
        prep_kept,
        m=8,
        topk=TOPK,
        dtype=torch.bfloat16,
        device=dev,
        route_num_experts=NV + NF,
    )
    out_kept = buffers.output[:M]
    out_nf3 = torch.empty_like(out_kept)
    emap_kept = torch.full((NV + NF,), -1, dtype=i32, device=dev)
    emap_kept[:NV] = torch.arange(NV, dtype=i32, device=dev)
    emap_nf3 = torch.full((NV + NF,), -1, dtype=i32, device=dev)
    emap_nf3[NV:] = torch.arange(NF, dtype=i32, device=dev)

    def run_tier(prepared, fused, emap, out):
        tier_ids = emap[grid_ids.long()].to(i32).contiguous()
        return run_w4a16_moe(
            x,
            prepared,
            weights,
            tier_ids,
            activation="silu",
            intermediate_cache13=buffers.intermediate_cache13,
            intermediate_cache2=buffers.intermediate_cache2,
            output=out,
            fc1_c_tmp=buffers.fc1_c_tmp,
            fc2_c_tmp=buffers.fc2_c_tmp,
            packed_route_indices=buffers.packed_route_indices,
            block_expert_ids=buffers.block_expert_ids,
            packed_route_count=buffers.packed_route_count,
            expert_offsets=buffers.expert_offsets,
            expert_map=None,
            fused_launch=fused,
        )

    def run_serial_kept():
        return run_tier(prep_kept, decode_kept, emap_kept, out_kept)

    def run_serial_nf3():
        return run_tier(prep_nf3, decode_nf3, emap_nf3, out_nf3)

    def run_serial():
        a = run_serial_kept()
        b = run_serial_nf3()
        return a + b

    log("serial: first eager call...")
    run_serial()
    torch.cuda.synchronize()
    log("serial: eager call ok")

    results = {}

    # ---- eager timings (include Python launch overhead) -------------------
    log("timing eager mapped...")
    results["mapped_eager"] = timed_events(run_mapped)
    log(f"  {results['mapped_eager']}")
    log("timing eager serial...")
    results["serial_eager"] = timed_events(run_serial)
    log(f"  {results['serial_eager']}")

    # ---- CUDA graph timings (production-like; primary numbers) ------------
    graphs = {}

    def capture(name, fn):
        try:
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                fn()
            graphs[name] = gr
            log(f"graph '{name}' captured")
            return True
        except Exception as exc:
            log(f"graph '{name}' capture FAILED: {type(exc).__name__}: {exc}")
            return False

    capture("mapped", run_mapped)
    capture("serial", run_serial)
    capture("serial_kept", run_serial_kept)
    capture("serial_nf3", run_serial_nf3)
    capture("serial_add", lambda: out_kept + out_nf3)

    for name, gr in graphs.items():
        log(f"timing graph {name}...")
        results[f"{name}_graph"] = timed_events(gr.replay)
        log(f"  {results[f'{name}_graph']}")

    # ---- summary ----------------------------------------------------------
    def pick(key):
        r = results.get(key)
        return None if r is None else round(r["median_us"], 2)

    mapped = results.get("mapped_graph") or results["mapped_eager"]
    serial = results.get("serial_graph") or results["serial_eager"]
    summary = {
        "device": props.name,
        "capability": f"{props.major}.{props.minor}",
        "sms": sms,
        "max_shared_mem": max_shared,
        "timing_source": "cuda_graph_replay"
        if "mapped_graph" in results and "serial_graph" in results
        else "eager",
        "mapped_us_mean": round(mapped["mean_us"], 2),
        "mapped_us_median": round(mapped["median_us"], 2),
        "serial_us_mean": round(serial["mean_us"], 2),
        "serial_us_median": round(serial["median_us"], 2),
        "serial_over_mapped": round(serial["median_us"] / mapped["median_us"], 3),
        "serial_components_us_median": {
            "kept_tier(gather+launch)": pick("serial_kept_graph"),
            "nf3_tier(gather+launch)": pick("serial_nf3_graph"),
            "combine_add": pick("serial_add_graph"),
        },
        "eager_mapped_us_median": pick("mapped_eager"),
        "eager_serial_us_median": pick("serial_eager"),
        "iterations": ITERS,
        "warmup": WARMUP,
        "notes": (
            "zeros weights/scales/activations; routing 2 kept + 6 nf3 per token"
            " (32 distinct routes); serial = vllm decode sequence (per-tier"
            " global->local id gather + TC-decode fused launch per tier +"
            " bf16 output add), no separate topk-sum kernel (fused into FC2);"
            " mapped workspace zeroed once (self-resetting grid barrier, as in"
            " vllm production); graph timings are CUDA-graph replays like"
            " production decode."
        ),
        "all_results_us": {
            k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in results.items()
        },
    }
    print("BENCH_JSON " + json.dumps(summary), flush=True)
    print(
        f"\nGB10 grid48 mapped: {summary['mapped_us_median']} us median | "
        f"serial two-tier: {summary['serial_us_median']} us median | "
        f"ratio serial/mapped: {summary['serial_over_mapped']}x",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback = __import__("traceback")
        traceback.print_exc()
        sys.exit(1)
