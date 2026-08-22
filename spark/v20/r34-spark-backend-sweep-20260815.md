# GG r34-spark: DS4-0731 backend config sweep (dusty/kirby TP2, 2026-08-15)

First serving exercise of the r34-spark image
(`gilded-gnosis-v20-r34-spark-sm121-vllmc3ffb74-b12xcd3ce19-fi1ac6942-cu132-20260814`),
same launch contract as the r33 restore on this pair: dspark MODE, K5
probabilistic, MAX_MODEL_LEN 524288, GPU_MEM 0.87, NCCL LL,Simple + c4
channel pin, dusty=head 10.11.1.1 / kirby=worker 10.11.1.2. One boot per
arm; 15-cell decode sweep (ctx 0/16k/32k/64k/128k x cc 1/2/4) via
llm-inference-bench with the new MTP acceptance normalization
(`server_steps_per_s` = engine forward passes/s, acceptance divided out).
JSONs in `bench/benchmark_results-ds4-0731-r34spark-*.json`.

## Results (geomean over 15 cells)

| arm | raw tok/s | vs prod | steps/s | vs prod | acc len | KV pool | boot |
|---|---|---|---|---|---|---|---|
| b12x-a8 (prod default) | 64.1 | - | 22.24 | - | 2.89 | 1,013,458 | 352s |
| b12x-a16 | 64.3 | +0.3% | 22.27 | +0.1% | 2.89 | 978,282 | 307s |
| b12x-a8-dglin | 64.2 | +0.2% | 21.83 | -1.8% | 2.95 | 869,401 | 286s |
| lucifer-cutlass (autotune OFF) | 58.2 | -9.3% | 19.58 | -11.9% | 2.98 | 871,519 | 220s |
| lucifer-default | 54.1 | -15.6% | 18.63 | -16.2% | 2.91 | 1,101,222 | 220s |
| b12x-a8 + INDEXER_BACKEND=native | 64.4 | +0.5% | 21.58 | -3.0% | 2.99 | 1,024,105 | 242s |

Concurrency shape (steps/s geomean): b12x-a8 wins cc1 (15.44 vs a16
14.19); b12x-a16 wins cc4 (33.20 vs 31.92, +4%). Crossover consistent
with a8's lighter per-token dequant vs a16's better batched arithmetic.

## Findings

1. **Production default confirmed.** b12x-a8 stays the right call: best
   cc1, within noise of a16 overall, biggest-KV-pool tier. Nothing in
   r34 changes the backend choice.
2. **Acceptance normalization earns its keep.** Raw geomeans of the four
   b12x-family arms are indistinguishable (+-0.5%) while steps/s exposes
   real spreads (dglin -1.8%, native indexer -3.0%) that raw numbers
   masked through acceptance luck (2.89-2.99 across boots).
3. **b12x indexer is worth ~3%** at equal raw throughput
   (INDEXER_BACKEND=native costs -3.0% steps/s; its higher measured
   acceptance that boot hid the cost in raw tok/s).
4. **lucifer-cutlass does NOT boot on GB10 with FlashInfer autotune on:**
   `trtllm::fused_moe::gemm1` tactic profiling fails hard with "Failed to
   initialize the TMA descriptor" during kernel warmup (20 tactics
   already skipped as unsupported; first profiled tactic kills the
   engine). With ENABLE_FLASHINFER_AUTOTUNE=0 it serves correctly at
   -11.9% steps/s vs production. Upstream-worthy: SM121 needs a TMA-free
   tactic filter or a capability guard in the autotune path.
5. **lucifer-default is the slowest arm (-16.2%)** but posts the largest
   KV pool (1,101,222) - the b12x workspaces cost ~90k tokens of pool.
6. **KV pool per-boot band reconfirmed**: 869k-1,101k across arms on the
   same hosts/window; config-driven, plus known UMA free-memory
   sensitivity. 524k single-session fits in every arm.

Caveats: one boot and one 30s-per-cell measurement per arm; DSpark
probabilistic profile, so acceptance (and raw tok/s) vary run to run -
steps/s comparisons are the durable numbers. No r33 steps/s
back-comparison exists (the field ships with the 2026-08-14 bench
update); raw cc1/0ctx 40.5 tok/s sits inside the historical r33 boot
band around the 43.7 contract figure.

Serving qualification of the image remains governed by
`r34-spark-qualification-plan.md` (this sweep is performance evidence,
not the qual).
