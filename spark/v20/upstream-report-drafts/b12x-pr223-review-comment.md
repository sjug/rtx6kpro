# DRAFT: review comment for B12X PR #223

Commit `b4d6c7593c9fae46f4cb6f7a645e0e10fc9c1faa` in this PR removes the paired-M8 FC2 pipeline. On the complete-population mixed-Trellis workload, this produces a general large-M regression at both tested expert splits. Restoring the paired pipeline is bit-identical and returns performance to the pre-PR level.

I opened FIX_PR_URL against this PR's head branch with the restoration, an explicit compile-time pairing contract, a two-geometry regression test, and a complete-population benchmark.

All measurements are from one NVIDIA GB10, SM121, 48 SMs. I do not have an SM120 measurement, so I am not claiming that result as verified even though the affected kernel path is shared.

| Geometry | M | PR #223 stock us | Paired us | Stock slower |
|:--|--:|--:|--:|--:|
| 148/108 | 33 | 4,225.9 | 3,403.4 | 24.2% |
| 148/108 | 221 | 6,756.8 | 5,639.7 | 19.8% |
| 148/108 | 222 | 6,769.0 | 5,659.1 | 19.6% |
| 148/108 | 3072 | 23,747.3 | 19,414.1 | 22.3% |
| 192/64 | 33 | 4,112.0 | 3,274.0 | 25.6% |
| 192/64 | 221 | 6,583.1 | 5,318.7 | 23.8% |
| 192/64 | 222 | 6,589.2 | 5,342.5 | 23.3% |
| 192/64 | 3072 | 23,705.1 | 19,283.3 | 22.9% |

The paired source stays within 0.91% of stock at M at or below 32. All 16 measured outputs are finite and bit-identical between the stock and paired sources. The paired source is also within 1.4% of the legacy implementation at every large-M cell.

The existing `benchmarks/benchmark_mixed_trellis.py` does not expose this regression because it materializes only 16 experts per tier, pads storage to a nominal 192/64 split, forces six K3 and two K4 routes per token, and uses graph replay. The reproduction in the fix PR materializes all 256 experts and routes across the complete population through the public API.

The source-level mechanism matches the result: without `_run_tile_m8_pair`, the factor-two FC2 schedule invokes the scalar tile path twice and separately loads and decodes the same compressed weight tile. The restoration advances both M8 route blocks through one pipeline. The complete-population regression test passed for both geometries on SM121 in the r18p runtime.

Related production-faithful measurements and the serving impact are tracked in `sjug/rtx6kpro#67`.
