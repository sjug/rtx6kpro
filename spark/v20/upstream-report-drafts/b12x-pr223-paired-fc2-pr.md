# DRAFT: paired-FC2 fix PR

Repository: `local-inference-lab/b12x`

Base: `feat/ii-mixed-trellis-k345-master-20260816`

Head: `sjug:perf/restore-paired-fc2-pr223`

Proposed title: `perf(moe): restore paired FC2 for mixed-Trellis prefill`

## Body

## Summary

Restore the paired-M8 FC2 pipeline in the projection-mixed Trellis path introduced by #223. Commit `b4d6c7593c9fae46f4cb6f7a645e0e10fc9c1faa` removes this pipeline and makes the two scheduled M8 route blocks load and decode the same compressed FC2 weight tile separately.

The restored path advances both route blocks through one MMA pipeline. The scalar tile path remains the fallback whenever the launch does not satisfy the paired-M8 contract.

This PR also makes the pairing contract explicit in compile metadata and adds a complete-population benchmark plus a two-geometry regression test.

## Measured effect

All measurements below use one NVIDIA GB10, SM121, 48 SMs. No SM120 measurement is available yet.

The harness materializes all 256 experts, routes across the complete population through the public mixed-Trellis API, and uses hidden 6144, intermediate 512, top-k 8, MCG K3/K4 weights, and tile `(128, 128, 32, 512)`.

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

The paired source is within 1.4% of the legacy implementation at every large-M cell. M at or below 32 remains within 0.91% of the PR #223 stock source. All 16 paired outputs are finite and bit-identical to the stock source.

The existing `benchmarks/benchmark_mixed_trellis.py` is not a sufficient regression instrument for this change. It materializes only 16 experts per tier, pads storage to a nominal 192/64 split, and forces a fixed six-plus-two route pattern. The new benchmark materializes and routes across all 256 experts.

## Validation

- `ruff check` passes on all changed Python files. `ruff format --check` passes on the changed test, benchmark, and mixed-Trellis API file; the restored historical kernel body retains the branch's existing hand-formatted CuTe DSL style.
- The focused source-contract test passes.
- The complete-population large-M test passes for 148/108 and 192/64 on SM121 in the r18p runtime.
- Both large-M launches report `fc2_schedule_route_block_factor == 2` and `fc2_paired_m8_routes is True`.
- Both large-M outputs match their serial mixed-tier references exactly.

Related production-faithful evidence is tracked in `sjug/rtx6kpro#67`.
