# DRAFT — do not file

Repository: `local-inference-lab/b12x`

Proposed title: `[Perf][SM121][EXL3] Removing paired FC2 regresses mixed-Trellis large-M latency by 20-26%`

## Summary

The BTX bound mixed-Trellis implementation in the Infernal Invocation r18 composition is approximately 20-26% slower at large M than both the earlier Gilded Gnosis r34 implementation and an r18 source variant with paired FC2 restored. The regression is general across the tested 148 K3 / 108 K4 and 192 K3 / 64 K4 expert splits. Decode-sized launches remain unchanged.

This is a single-GPU kernel result with no NCCL or distributed runtime involved. The same-harness r34, stock-r18, and paired-r18 comparison isolates the missing paired-FC2 execution as the mechanism and predicts closure of the residual 7.9-9.8% serving-prefill gap after the separate NCCL progress failure is mitigated.

## Source identities and hardware

- GPU: NVIDIA GB10, SM121, 48 SMs
- r34 runtime: CUDA 13.2, PyTorch 2.12.0, legacy B12X integration `cd3ce19`
- r18 runtime: CUDA 13.3, PyTorch 2.13.0, bound B12X integration `75787c7a7431b3bea414d2ebf5f2b8671b23eb33`
- Paired-r18 source: stock r18 plus the attached two-file `paired-fc2-restoration.patch`
- Geometry: hidden 6144, intermediate 512, top-k 8, 256 materialized experts
- Tier splits: 148 K3 / 108 K4 and 192 K3 / 64 K4, MCG codebook
- Tile: `(128, 128, 32, 512)`
- Decode plan: capacity 32, block M 8
- Prefill plan: capacity 3072, block M 32

Checkpoint inspection found one routed layer at 206/50 and the remaining 74 routed layers at 148/108 for gate, up, and down. There are no projection-mixed experts in this checkpoint. The 192/64 arm is retained as a second complete-expert-population control, not as a claim about the checkpoint.

## Same-harness results

The table reports median microseconds. Stock-r18 values are the average of an unmodified/paired/unmodified B/A/B sequence. All arms use the same production-faithful harness and deterministic inputs.

| Geometry | M | r34 legacy | r18 stock | r18 paired | Stock vs r34 | Paired vs r34 |
|:---------|--:|-----------:|----------:|-----------:|---------------:|----------------:|
| 148/108 | 33 | 3,441.7 | 4,225.9 | 3,403.4 | +22.8% | -1.1% |
| 148/108 | 221 | 5,615.8 | 6,756.8 | 5,639.7 | +20.3% | +0.4% |
| 148/108 | 222 | 5,652.7 | 6,769.0 | 5,659.1 | +19.8% | +0.1% |
| 148/108 | 3072 | 19,298.4 | 23,747.3 | 19,414.1 | +23.1% | +0.6% |
| 192/64 | 33 | 3,263.7 | 4,112.0 | 3,274.0 | +26.0% | +0.3% |
| 192/64 | 221 | 5,366.9 | 6,583.1 | 5,318.7 | +22.7% | -0.9% |
| 192/64 | 222 | 5,417.9 | 6,589.2 | 5,342.5 | +21.6% | -1.4% |
| 192/64 | 3072 | 19,159.4 | 23,705.1 | 19,283.3 | +23.7% | +0.6% |

Paired r18 is within 1.4% of r34 at every measured large-M cell. At M <= 32, paired and stock r18 remain within 0.91%. Every paired-r18 output is finite and bit-identical to its stock-r18 reference; the r34 output hashes also match at every same-harness large-M cell.

The M=3072 stock-r18 minus r34 difference is approximately 4.45 ms per routed layer, or 0.33 seconds across 75 routed layers. That matches the remaining approximately 0.32-second serving-prefill deficit after the exact-four NCCL mitigation.

## Why the earlier 192/64 parity result was invalid evidence

An earlier comparison reported 32.15 ms on r34 and 32.21 ms on r18 using each revision's `benchmarks/benchmark_mixed_trellis.py`. That instrument materializes only 16 experts per tier, pads storage to 192/64, forces exactly six K3 and two K4 routes per token, and times CUDA-graph replay. It does not exercise the same workload as the attached harness, which materializes all 256 experts and routes across the complete expert population through each generation's public execution API.

The 32.15/32.21 ms row therefore cannot be used as a control for the production-faithful results above. It was a harness false negative that incorrectly exonerated mixed-Trellis.

## Reproducer

The attached `microbench_glm_exl3_mixed_trellis.py` synthesizes deterministic K3/K4 weights and routing, uses each installed B12X generation through its public API, and needs neither checkpoint weights nor a cluster.

Run the same file in r34 and stock r18:

```console
python3 microbench_glm_exl3_mixed_trellis.py \
  --label IMAGE_LABEL \
  --tier0-experts 148 \
  --tier1-experts 108 \
  --m 1 4 16 32 33 221 222 3072 \
  --warmup 10 \
  --repeats 50
```

Repeat with `--tier0-experts 192 --tier1-experts 64`. Apply `paired-fc2-restoration.patch` to the exact r18 composed B12X tree and run the same two arms with the stock-r18 tensors supplied through `--reference-output-dir`.

## Source-level cause

Commit `b4d6c7593c9fae46f4cb6f7a645e0e10fc9c1faa` ("Remove standalone Trellis execution paths") removes paired-M8 FC2 execution. Before that change, block-32/64 prefill created two M8 route subtiles and `_run_tile_m8_pair` advanced both through one MMA pipeline, sharing each compressed B fragment and its Trellis decode. Afterward, the same `schedule_route_block_factor=2` contract calls the shared `_run_tile` primitive twice, loading and decoding the same FC2 weight tile separately for each routed M8 block.

The historical parent of `b4d6c75` is not an executable performance control because its caller and kernel body have an unrelated ABI mismatch. Instead, the attached source variant restores only the paired execution contract on the exact r18 tree. That within-r18 A/B is bit-identical, leaves decode flat, and recovers r34 large-M parity at both expert splits.

PR #238 is not a fix for this result. It changes bounded W4A16 route-output reduction and explicitly leaves Trellis layouts unchanged; its isolated result is a scratch-capacity feature rather than a paired-FC2 latency optimization.

## Requested change and acceptance criteria

Please restore paired FC2, or an equivalent implementation that shares compressed FC2 weight loading and Trellis decoding across the two scheduled M8 route blocks, in the BTX mixed-Trellis path.

Acceptance criteria on SM121:

- r34 legacy latency parity at both 148/108 and 192/64 for M = 33, 221, 222, and 3072;
- stock-r18 parity at M <= 32;
- finite, bit-identical outputs against stock r18;
- compile metadata within the admitted register, shared-memory, spill, and occupancy limits; and
- a production-faithful full-expert-population benchmark rather than the 16-materialized-expert upstream benchmark.
