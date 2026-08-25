# DS4 DGLIN investigation on SM121

Date: 2026-08-24 EDT

Status: complete. The exact-dispatch discriminator and all six balanced serving
boots passed. At the end of this experiment, the final DGLIN arm was stopped so
dusty could build and validate the rebased SM121 image.

## Test boundary

- Host: dusty, one NVIDIA GB10, SM121, 48 SMs.
- Image: `infernal-invocation-r18p-spark-sm121-vllmf560085-b12x07cdf45-fi1ac6942-cu133-torch213-20260820`.
- Image ID: `445f9ac3196dd10a47d0d90441ba43a8417903ba029f59a1a9c7fbef8ecfa4a1`.
- B12X launcher selection: `B12xFp8BlockScaledMMKernel`.
- Minimal standalone `VllmConfig` auto selection:
  `CutlassFp8BlockScaledMMKernel`.
- Actual `b12x-a8-dglin` serving selection with the DS4 model config:
  `DeepGemmFp8BlockScaledMMKernel`.
- The first harness invoked vLLM's `init_fp8_linear_kernel` and each selected
  kernel's `apply_weights`, but its minimal config lacked the DS4 model config.
  DeepGEMM therefore rejected that context and vLLM selected CUTLASS. The table
  below is a valid B12X-versus-CUTLASS control, not a DGLIN result.
- Both arms use dynamic 1x128 activation quantization and 128x128 block-FP8
  weights. All outputs were finite. B12X versus CUTLASS relative L2 was
  0.0364 to 0.0381, so this is a performance discriminator, not a claim of
  bit-identical arithmetic.

## Exact-shape CUTLASS control

`CUTLASS speedup` is B12X median latency divided by CUTLASS median latency.
Values above 1.0 favor CUTLASS. They must not be attributed to DGLIN serving,
which selects DeepGEMM.

| projection | M | B12X us | CUTLASS us | CUTLASS speedup |
|---|---:|---:|---:|---:|
| fused_wqa_wkv | 6 | 39.7 | 52.0 | 0.762x |
| fused_wqa_wkv | 12 | 98.0 | 52.3 | 1.873x |
| fused_wqa_wkv | 18 | 97.2 | 52.1 | 1.865x |
| fused_wqa_wkv | 24 | 95.6 | 51.0 | 1.874x |
| fused_wqa_wkv | 4096 | 582.2 | 497.4 | 1.171x |
| fused_wqa_wkv | 8192 | 2223.3 | 994.5 | 2.236x |
| wq_b | 6 | 38.9 | 53.5 | 0.727x |
| wq_b | 12 | 103.7 | 53.2 | 1.949x |
| wq_b | 18 | 96.9 | 54.4 | 1.782x |
| wq_b | 24 | 95.5 | 53.1 | 1.797x |
| wq_b | 4096 | 1035.7 | 1216.5 | 0.851x |
| wq_b | 8192 | 2180.8 | 2193.6 | 0.994x |
| wo_b | 6 | 42.7 | 54.7 | 0.780x |
| wo_b | 12 | 96.6 | 54.0 | 1.790x |
| wo_b | 18 | 96.9 | 53.5 | 1.812x |
| wo_b | 24 | 96.0 | 52.1 | 1.842x |
| wo_b | 4096 | 1303.3 | 1086.5 | 1.200x |
| wo_b | 8192 | 5620.6 | 2105.8 | 2.669x |
| shared_gate_up | 6 | 39.0 | 52.8 | 0.738x |
| shared_gate_up | 12 | 96.6 | 52.6 | 1.835x |
| shared_gate_up | 18 | 96.1 | 52.4 | 1.836x |
| shared_gate_up | 24 | 95.3 | 51.1 | 1.864x |
| shared_gate_up | 4096 | 687.9 | 617.0 | 1.115x |
| shared_gate_up | 8192 | 2686.5 | 1277.2 | 2.103x |
| shared_down | 6 | 37.4 | 51.7 | 0.724x |
| shared_down | 12 | 95.9 | 51.8 | 1.851x |
| shared_down | 18 | 96.1 | 52.5 | 1.828x |
| shared_down | 24 | 95.2 | 51.2 | 1.861x |
| shared_down | 4096 | 300.6 | 331.9 | 0.906x |
| shared_down | 8192 | 609.7 | 625.6 | 0.975x |

## Interpretation of the CUTLASS control

The discriminator predicts a real crossover, not a universal backend win.

- At M=6, the B12X dense kernel is 22 to 38 percent faster across all five
  projections. This is the K5 concurrency-one verify shape.
- At M=12, 18, and 24, CUTLASS is 1.78x to 1.95x faster across all five
  projections. These are the K5 concurrency-two through concurrency-four
  verify shapes.
- At large M, CUTLASS wins three of five exact projections and is especially
  strong at M=8192 for `fused_wqa_wkv`, `wo_b`, and `shared_gate_up`.
  B12X remains faster for `shared_down` and at M=4096 for `wq_b`.

This establishes a strong backend crossover for CUTLASS but does not predict
the DGLIN result. The corrected discriminator must construct the cached DS4
model configuration, assert that auto selection is DeepGEMM, and then compare
that selected implementation with B12X. End-to-end serving remains the
decision gate because projection frequency, fusion ownership, communication,
speculative acceptance, and graph replay are outside the microbenchmark.

## Exact-shape DeepGEMM discriminator

The corrected harness constructed the pinned DS4 `ModelConfig` and failed
closed unless auto selection resolved to
`DeepGemmFp8BlockScaledMMKernel`. `DeepGEMM speedup` is B12X median latency
divided by DeepGEMM median latency. Values above 1.0 favor DeepGEMM.

| projection | M | B12X us | DeepGEMM us | DeepGEMM speedup |
|---|---:|---:|---:|---:|
| fused_wqa_wkv | 6 | 38.44 | 27.78 | 1.384x |
| fused_wqa_wkv | 12 | 94.29 | 27.34 | 3.449x |
| fused_wqa_wkv | 18 | 94.15 | 33.10 | 2.844x |
| fused_wqa_wkv | 24 | 91.92 | 32.67 | 2.814x |
| fused_wqa_wkv | 4096 | 583.96 | 505.59 | 1.155x |
| fused_wqa_wkv | 8192 | 2135.58 | 1020.69 | 2.092x |
| wq_b | 6 | 39.26 | 28.98 | 1.355x |
| wq_b | 12 | 100.76 | 28.10 | 3.586x |
| wq_b | 18 | 93.79 | 29.91 | 3.136x |
| wq_b | 24 | 92.87 | 33.82 | 2.746x |
| wq_b | 4096 | 1046.10 | 1182.21 | 0.885x |
| wq_b | 8192 | 2190.46 | 2326.55 | 0.942x |
| wo_b | 6 | 42.11 | 28.40 | 1.483x |
| wo_b | 12 | 95.57 | 28.23 | 3.385x |
| wo_b | 18 | 94.53 | 37.06 | 2.551x |
| wo_b | 24 | 93.05 | 35.12 | 2.649x |
| wo_b | 4096 | 1167.68 | 1029.68 | 1.134x |
| wo_b | 8192 | 5596.87 | 2114.10 | 2.647x |
| shared_gate_up | 6 | 38.60 | 28.16 | 1.371x |
| shared_gate_up | 12 | 94.59 | 27.73 | 3.411x |
| shared_gate_up | 18 | 94.67 | 33.22 | 2.850x |
| shared_gate_up | 24 | 93.45 | 33.01 | 2.831x |
| shared_gate_up | 4096 | 769.47 | 639.65 | 1.203x |
| shared_gate_up | 8192 | 2583.59 | 1227.42 | 2.105x |
| shared_down | 6 | 37.44 | 27.46 | 1.364x |
| shared_down | 12 | 94.64 | 27.59 | 3.431x |
| shared_down | 18 | 94.74 | 27.63 | 3.429x |
| shared_down | 24 | 93.83 | 27.59 | 3.401x |
| shared_down | 4096 | 285.74 | 300.56 | 0.951x |
| shared_down | 8192 | 612.16 | 610.99 | 1.002x |

DeepGEMM wins every measured decode-shape projection at M=6, 12, 18, and 24.
The speedup is 1.35x to 1.48x at M=6 and 2.55x to 3.59x at M=12 through 24.
At prefill M, DeepGEMM wins most projections, while B12X remains faster for
`wq_b` and approximately ties `shared_down`. All 30 output pairs were finite;
13 were exact and the maximum relative L2 difference was 0.00286.

## Balanced serving result

The serving sequence used three adjacent pairs in balanced order: stock/DGLIN,
DGLIN/stock, stock/DGLIN. Each pair used one deterministic prompt seed shared
by both arms; the seed differed across pairs. DGLIN selected DeepGEMM on both
ranks in every DGLIN boot. All 60 decode cells and 30 prefill scouts completed
without errors, queueing, warmup timeout, or capacity skips.

The geometric-mean DGLIN/stock deltas across all three pairs were:

| metric | all cells | cc1 | cc4 |
|---|---:|---:|---:|
| aggregate decode tok/s | +0.81% | +0.99% | +0.62% |
| MTP-normalized steps/s | +0.47% | +0.49% | +0.44% |
| effective acceptance length | +0.34% | +0.50% | +0.18% |

Prefill improved in all 15 paired measurements from 8K through 128K. The
overall geometric-mean gain was +1.49%; pair-level gains were +1.13%, +1.78%,
and +1.55%. By context, the gains were +1.37% at 8K, +1.03% at 16K, +2.27%
at 32K, +1.43% at 64K, and +1.33% at 128K.

The direction is repeatable but small. DGLIN is a measured improvement over
stock r18p on this SM121 TP2 pair, primarily for prefill. It does not explain
or materially change the prior high-concurrency decode question: normalized
cc4 steps improved only +0.44%, well below the historical boot-to-boot spread.
The microbenchmark shows that DeepGEMM is much faster for each isolated dense
projection, but the small end-to-end gain shows that those projections account
for only a limited fraction of DS4 serving time.

## Paired-FC2 prerequisite

The proposed B12X PR #223 restoration was tested separately on the same GB10
against the r18p container environment. Three tests passed, including complete
256-expert 148/108 and 192/64 large-M cases. The compiled mixed-Trellis launch
used the paired FC2 schedule. See `pr223-gpu-tests.log`.

## Files

- `benchmark_ds4_dense_fp8.py`: exact-dispatch B12X-versus-DeepGEMM
  microbenchmark with the pinned DS4 model config.
- `dense-fp8-cutlass-control.jsonl`: the initial B12X-versus-CUTLASS control.
- `dense-fp8-deepgemm.jsonl`: actual DGLIN kernel selections, timings,
  numerical comparisons, and final result object.
- `run-dglin-ab.sh`: balanced six-boot serving experiment.
- `analyze-dglin-ab.py`: paired per-cell and geometric-mean analysis.
- `results/`: six final benchmark JSON files and per-boot receipts.
- `pr223-gpu-tests.log`: SM121 test receipt for the paired-FC2 restoration.
