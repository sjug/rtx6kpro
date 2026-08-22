# Paired-M8 removal discriminator and r18-tip source A/B

Date: 2026-08-20

## Question

Determine whether removal of the paired-M8 FC2 path in B12X commit
`b4d6c75` caused the mixed-Trellis large-M regression, and whether restoring
that path on the exact Infernal Invocation r18 B12X tree is a viable prefill
candidate for SM121.

The acceptance conditions established before testing were:

1. The 192/64 control geometry must remain at parity. Movement there voids a
   geometry-specific attribution to 148/108.
2. M <= 32 must remain at parity because the restoration must not affect the
   small-M decode path.
3. Every source arm must be numerically equivalent to its unmodified arm.

## Source construction

The untouched `b4d6c75^` tree is not a valid executable baseline. Its
mixed-Trellis caller and W4A16 kernel body have an ABI mismatch: the caller
omits two LUT arguments, shifting `active_m` into the wrong parameter, and it
lacks the later MCG wiring. The compile failure is recorded in
`parent-148-108.log`; it does not contain performance evidence.

The boundary source arm was therefore built inside `b4d6c75` by reversing
only the paired-FC2 removal. Twenty-two pair-related methods came from the
parent implementation. Three methods whose parent bodies were incompatible
with `dynamic_pair_override` came from the pair-bearing source commit
`7235024`: `_run_tile_m8_pair`, `_run_mma_pipeline_m8_pair`, and
`_load_next_fragment_bundle_m8_pair`. The final diff touched only
`w4a16/kernel.py` and `w4a16/mixed_trellis.py`; all non-selected methods were
kept byte-identical to `b4d6c75`. The compile smoke passed on a GB10.

The same two-file source delta applied cleanly to the exact r18 composed B12X
tree `75787c7a7431b3bea414d2ebf5f2b8671b23eb33`. The r18 source arm has this
identity:

- `kernel.py`: `c863d8136cbf3b950ad2dbcbb1415064fb7d00b1e9375644d72121970baf762a`
- `mixed_trellis.py`: `bdc2237961b3822e1112ee0122e06cac42275a428cb3b0be136cebd5fcb4162f`
- Diff surface: 650 insertions and 15 deletions in `kernel.py`; 40 insertions
  and 4 deletions in `mixed_trellis.py`
- Test image:
  `infernal-invocation-r18-spark-sm121-vllmf560085-b12x75787c7-fi1ac6942-cu133-torch213-20260818`

The r18 source smoke bound the production API on capability 12.1, compiled
decode at 150 registers/thread and prefill at 184 registers/thread, and
produced finite output.

## Method

Each geometry ran M = 1, 4, 16, 32, 33, 221, 222, and 3072 with 10 warmups
and 50 timed repetitions on dusty's GB10. Ordering was unmodified, paired,
unmodified. The tables use the average of the two unmodified medians as B and
the paired median as A. Inputs were deterministic and nonzero. Every paired
output was compared with its corresponding unmodified tensor for finiteness,
allclose, exact-element fraction, and SHA-256 identity.

## Exact r18-tip results

### 148/108 routed-expert split

| M | Plan | Unmodified B (us) | Paired A (us) | Speedup | Time saved |
|---:|:-----|------------------:|--------------:|--------:|-----------:|
| 1 | decode | 277.248 | 278.784 | 0.995x | -0.55% |
| 4 | decode | 570.656 | 566.608 | 1.007x | 0.71% |
| 16 | decode | 2051.376 | 2053.744 | 0.999x | -0.12% |
| 32 | decode | 3266.200 | 3269.664 | 0.999x | -0.11% |
| 33 | prefill | 4225.880 | 3403.440 | 1.242x | 19.46% |
| 221 | prefill | 6756.832 | 5639.696 | 1.198x | 16.53% |
| 222 | prefill | 6768.960 | 5659.136 | 1.196x | 16.40% |
| 3072 | prefill | 23747.288 | 19414.112 | 1.223x | 18.25% |

### 192/64 control split

| M | Plan | Unmodified B (us) | Paired A (us) | Speedup | Time saved |
|---:|:-----|------------------:|--------------:|--------:|-----------:|
| 1 | decode | 280.304 | 282.848 | 0.991x | -0.91% |
| 4 | decode | 552.448 | 551.232 | 1.002x | 0.22% |
| 16 | decode | 1954.168 | 1954.768 | 1.000x | -0.03% |
| 32 | decode | 3105.168 | 3105.440 | 1.000x | -0.01% |
| 33 | prefill | 4111.976 | 3273.968 | 1.256x | 20.38% |
| 221 | prefill | 6583.120 | 5318.656 | 1.238x | 19.21% |
| 222 | prefill | 6589.240 | 5342.464 | 1.233x | 18.92% |
| 3072 | prefill | 23705.120 | 19283.296 | 1.229x | 18.65% |

All 16 paired outputs were finite and bit-identical to the unmodified output:
exact fraction 1.0, maximum absolute error 0, and relative L2 error 0.

## Same-harness r34 control

The corrected-r34 image was subsequently run through this harness at both
expert splits. This supplies the missing legacy reference that the historical
boundary could not provide:

| Geometry | M | r34 legacy (us) | r18 stock B (us) | r18 paired A (us) | Paired vs r34 |
|:---------|--:|----------------:|-----------------:|------------------:|---------------:|
| 148/108 | 33 | 3,441.728 | 4,225.880 | 3,403.440 | -1.11% |
| 148/108 | 221 | 5,615.840 | 6,756.832 | 5,639.696 | +0.42% |
| 148/108 | 222 | 5,652.672 | 6,768.960 | 5,659.136 | +0.11% |
| 148/108 | 3072 | 19,298.368 | 23,747.288 | 19,414.112 | +0.60% |
| 192/64 | 33 | 3,263.680 | 4,111.976 | 3,273.968 | +0.32% |
| 192/64 | 221 | 5,366.944 | 6,583.120 | 5,318.656 | -0.90% |
| 192/64 | 222 | 5,417.888 | 6,589.240 | 5,342.464 | -1.39% |
| 192/64 | 3072 | 19,159.361 | 23,705.120 | 19,283.296 | +0.65% |

The r34 output SHA-256 values match both r18 arms at every measured cell.
Paired r18 is within 1.4% of r34 throughout, while stock r18 is 19.8% to
26.0% slower across these large-M cells. The earlier r34 re-run at 148/108
M=3072 measured 19,480 us, consistent with the 19,298 us control above.

The previous 32.15/32.21 ms 192/64 parity row came from B12X's own
`benchmark_mixed_trellis.py`, not this harness. That benchmark materializes
16 experts per tier, pads them to the compiled 192/64 populations, fixes the
route composition at six K3 and two K4 experts, and times graph replay. It is
not comparable to this harness's complete-expert-population public-API path.

## Verdict

The predeclared 192/64-flat acceptance condition correctly falsified the
geometry-specific premise, but it did not falsify the paired-FC2 mechanism.
The r34 control shows why that geometry moved: legacy paired FC2 was active
and fast at both expert splits. Removing paired FC2 created a general
large-M regression; restoring it on r18 returns both geometries to r34
parity.

Paired FC2 is therefore a general, bit-exact large-M optimization and the
cause of the measured stock-r18 kernel deficit. It leaves M <= 32 flat and
reduces tested large-M kernel time by 16.4% to 20.4% relative to stock r18.
Reverse ordering reproduced the unmodified medians, so the effect is not a
warm-cache ordering artifact. The within-r18 source A/B establishes the
mechanism; the r34 control establishes that the recovered level is the legacy
performance baseline.

This qualifies the source change for an r18p image candidate, not for
production. The textual restoration is large enough to require a dedicated
source review, a permanent numerical/performance gate for both geometries,
and a distinct-image serving A/B. The serving test must determine whether the
kernel recovery closes r18's measured 7.9% to 9.8% prefill deficit while
preserving decode and the exact-four-plus-implicit NCCL candidate contract.
No image was built and no production or staging service was changed during
this experiment.
