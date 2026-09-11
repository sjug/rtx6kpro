# R32 Qwen qualification

Status, 2026-09-10: initial aligned MTP3 qualification and the standard grid
completed, MTP0 boundaries passed, and aligned MTP3 was restored with a correct
first completion at 11:34:49 EDT. The matched second-boot repetition completed
at 12:18:21 EDT. Engine and prefill dips did not persist. C4 output remains
below the historical baseline, so a contemporary R29 C4 control ran next.
That control and restoration finished successfully at 12:45:15 EDT. Aligned
Qwen qualification passes with no demonstrated material overall regression.
R32 is serving on both nodes; a release-level long-context acceptance effect
remains unestablished.

## Identity and serving contract

- Both nodes: `74e53e710bef141f6f68e722582569f9c6aa388bce405ad6f1423566a2300c9c`.
- Checkpoint: `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-4p89`,
  `c374e7e24b54f6cb0017d0c2e6d26823d2f2fb5d`.
- Dusty/kirby TP2, aligned checkpoints, MTP3, context 262144, max sequences 4,
  batched tokens 4096, utilization 0.85, InstantTensor BUFFERED, LL,Simple.
- Build receipts: `../build-receipts/20260910T144935Z-1388331/`.
- Transfer verified identical image IDs and archive SHA-256
  `3bffabe501a07bd0640e7eebeb85d872963bfa6da914ee47c3cd735b8e2dbde0`
  using the switched 200G link. No conversion or compression.

## Initial correctness and behavior

- Semantic: 18/18 over three repetitions, including reasoning, non-thinking,
  vision and a tool round trip. These are bounded admission tests, not a broad
  model-quality assessment.
- MTP3 retrieval: exact at 2848, 2849, 131072 and 262000 tokens, all stop normally.
- MTP0 retrieval: exact at 2784, 2785 and 131072 tokens, all stop normally.
- Fixed-token corpus acceptance: 2.2747 versus R29's 2.3768 on the identical
  corpus hash. The second-boot value is 2.2588. The fresh R29 control will repeat
  the same corpus; these values alone do not identify a cause. Output hashes
  were not identical across waves.
- Padded-transition probe: c1/c3/c1/c2/c4/c1, acceptance 2.465 to 2.612,
  no collapsed phase. Coverage remains predicted, not observed engine dispatch.
- Identical burst: 118.2 aggregate tok/s, compared with surrounding distinct
  bursts at 127.5 and 112.2. No c1-rate serialization observed.
- Head-of-line probe: fresh maximum TTFT 0.321 s, repeat TTFT 0.296 s.
- Prefix probes completed for both MTP profiles. The raw receipts retain their
  hit accounting; long-needle times are not treated as fresh-prefill benchmarks.

## First standard sweep

Historical R29 baseline against R32's first boot, same harness and protocol.
All 15 cells passed structural checks, with no errors, timeouts, capacity
limits or underfilled cells. Geometric means across the five contexts:

| Concurrency | R29 steps/s | R32 steps/s | Change | R29 aggregate tok/s | R32 aggregate tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 21.690 | 21.347 | -1.58% | 45.012 | 46.865 |
| 2 | 36.081 | 35.954 | -0.35% | 75.057 | 73.385 |
| 4 | 55.314 | 54.636 | -1.22% | 119.286 | 115.272 |

| Prefill context | R29 tok/s | R32 tok/s | Change |
| --- | ---: | ---: | ---: |
| 8K | 3089 | 2993 | -3.11% |
| 16K | 2188 | 2141 | -2.15% |
| 32K | 2896 | 2757 | -4.80% |
| 64K | 2694 | 2685 | -0.33% |
| 128K | 2444 | 2463 | +0.78% |

Engine steps, generated output and acceptance are separate measures. The first
grid does not show a large engine slowdown; the output and prefill differences
required the matched repeat reported below. A `_topk_topp_kernel` JIT warning
appeared on both ranks at 11:19:40 during the overall benchmark process.
The present receipts do not resolve whether it fell in warmup or a timed cell;
do not attribute a throughput delta to that warning alone.

Inputs and hashes: `r29-vs-r32.json`. R32 raw grid is the standard campaign
`2026-09-jj-r32-vs-r29`, run
`20260910T111327-0400__jj-r32-aligned-sm121-tp2-mtp3__r01.json`.
The immutable historical R29 grid is
`20260909T092714-0400__jj-r29-aligned-sm121-tp2-mtp3__r01.json`.

## Second-boot repeat

The same full correctness battery passed on the restored R32 MTP3 boot:
semantic 18/18, exact retrieval through 262000 tokens, no padded-transition
collapse, and no identical-burst serialization or head-of-line stall. Maximum
fresh TTFT in the head-of-line probe was 0.306 s. The standard 15-cell sweep
completed at 12:18:20 EDT with no cell errors, underfill or timeouts.

| Concurrency | R29 steps/s | R32 repeat steps/s | Change | R29 aggregate tok/s | R32 repeat aggregate tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 21.690 | 21.843 | +0.71% | 45.012 | 46.398 |
| 2 | 36.081 | 36.517 | +1.21% | 75.057 | 78.000 |
| 4 | 55.314 | 55.787 | +0.86% | 119.286 | 115.085 |

| Prefill context | R29 tok/s | R32 repeat tok/s |
| --- | ---: | ---: |
| 8K | 3089 | 3115 |
| 16K | 2188 | 2996 |
| 32K | 2896 | 2915 |
| 64K | 2694 | 2748 |
| 128K | 2444 | 2483 |

The weak 16K baseline cell is not evidence of a repeatable 37% R32 gain. The
remaining prefill cells are within about 2% of R29. The initial engine and
prefill dips are not reproducible across the two R32 boots.

C4 aggregate output is again lower, by 3.52%, while engine steps are 0.86%
higher. Effective acceptance is 2.0629 versus 2.1565, down 4.34%. This is an
output/acceptance residual, not an engine slowdown. Both comparisons at this point used
one historical R29 boot and randomized benchmark prompts. A fresh R29 control
uses the same fixed-token acceptance check and C4 benchmark cells before
attributing the residual to the release.

One `_topk_topp_kernel` warning appeared on both ranks at 12:11:17 during the
overall repeat process. Its location relative to the timed measurement, rather
than warmup, remains unresolved. The same caveat applies to the first sweep.

Inputs and hashes: `r29-vs-r32-repeat.json`, raw run
`20260910T120516-0400__jj-r32-aligned-sm121-tp2-mtp3__r02.json`.

## Contemporary R29 C4 control

The preserved Qwen R29 containers were restarted on the same pair using image
`ee996ef8e531eb6e9c3208ebe30b141dc41702da1466e6ab2c47daa666d5694f`,
not the later corrected GLM R29 base tag. Live arguments match R32 after release
path normalization. Environment differences are confined to release cache
fingerprints/paths and GLM launcher provenance. Container inspections are in
`qwen-20260910/r29-c4-control/`.

Semantic 18/18, four exact MTP3 retrieval lengths through 262000, and all six
padded-concurrency phases passed. Fixed-token acceptance is 2.2917, compared
with R32's 2.2747 and 2.2588 and the historical R29 value of 2.3768. This already
shows that the historical acceptance number is not a stable release constant.

The C4-only control uses all five standard contexts, the unchanged wrapper,
30-second cells and its standard warmup/prefill scouts. It is a narrow follow-up,
not a replay of the full grid's C1/C2-before-C4 ordering or of the additional
identical/HOL probes preceding those grids. The fixed-token acceptance battery
does follow the same sequence as the R32 checks. All five timed cells passed
with no errors, timeouts, underfill or capacity limits.

| C4 metric, five-context geometric mean | Fresh R29 | R32 repeat | R32 change |
| --- | ---: | ---: | ---: |
| Engine steps/s | 56.090 | 55.787 | -0.54% |
| Aggregate output tok/s | 116.462 | 115.085 | -1.18% |
| Effective acceptance length | 2.0763 | 2.0629 | -0.65% |

| Prefill context | Fresh R29 tok/s | R32 repeat tok/s |
| --- | ---: | ---: |
| 8K | 3093 | 3115 |
| 16K | 2996 | 2996 |
| 32K | 2912 | 2915 |
| 64K | 2749 | 2748 |
| 128K | 2482 | 2483 |

The larger aggregate deficit against historical R29 did not survive this
control at the same magnitude. Prefill differs by at most 0.72%. These are
practical parity observations, not a statistical equivalence proof or a
demonstrated R32 speedup.

The 128K C4 cell remains variable: R29 historical/fresh output is
124.08/120.32 tok/s; R32 first/repeat is 118.41/109.29. Effective acceptance is
2.306/2.216 versus 2.270/2.006. The fresh R29 acceptance falls within R32's
observed range, and the repeat's engine steps are 0.32% above the fresh R29
control despite its 9.17% lower output. Do not relabel that individual cell as
an engine regression, or claim its acceptance difference has been explained.
A repeatable release-specific long-context acceptance effect is not established.

Fresh R29 also emitted `_topk_topp_kernel` at 12:36:48. Thus the warning is not
R32-specific. The precise timed/warmup placement remains a limitation.
The benchmark's simple blocks-times-block-size KV estimate is not used here;
capacity comes from the engine's group-aware boot log.

Inputs and hashes are in `qwen-20260910/r29-c4-control/comparison.json`.
The immutable control run is
`20260910T123535-0400__jj-r29-aligned-sm121-tp2-mtp3-c4-control__r01.json`.

Aligned Qwen correctness qualification passes. Two complete R32 grids and a
contemporary R29 C4 control show no demonstrated material overall performance
regression. Auto-policy testing remains deferred; GLM qualification is separate.

## Boot capacity

These are engine-reported capacities, not a sum across TP ranks or a promise
that every listed token is usable under every request mixture.

| Engine log time (UTC) | Profile | Reported KV tokens |
| --- | --- | ---: |
| 15:05:03 | initial MTP3 | 5457258 |
| 15:29:50 | MTP0 control | 6428358 |
| 15:34:13 | restored MTP3 | 5368874 |
| 16:29:19 | contemporary R29 MTP3 control | 5288576 |
| 16:44:35 | final restored R32 MTP3 | 5392381 |

The difference between the two MTP3 boots is retained rather than collapsed
into a single capacity claim. Boot logs and container configurations are under
`qwen-20260910/<profile>/`.

At 15:34:13 UTC on the restored MTP3 boot, rank 0 (dusty) reported 43.62 GiB
available for KV and rank 1 (kirby) 44.18 GiB. The coordinator reports one
effective token capacity for the TP group; no independent per-rank token
capacity is invented from the byte budgets.

The final R32 boot at 16:44:35 UTC reported 43.81 GiB on dusty and 44.58 GiB on
kirby. Its boot logs are saved in the contemporary-control receipt directory.

## Follow-through

The initial driver exited successfully at 11:34:49 EDT. Result review did not
continue automatically afterward, leaving an avoidable idle gap. The follow-up
service, `jj-r32-qwen-repeat.service`, validated the restored boot, ran
the unchanged grid as repetition 2, and generated the R29 comparison before
exiting successfully at 12:18:21 EDT. Its output is `repeat.log`, with a
separate exit-status receipt. Result review continued immediately into the
contemporary R29 control, `jj-r32-qwen-c4-control.service`. That control
preserved both image/container pairs and restored R32 after the measured arm.
It exited 0 at 12:45:15 EDT after verifying both restored image IDs and the
correct `333` completion with normal stop. No benchmark is left running.

The campaign is closed and its results catalog refreshed and checked. Raw
measurements were not modified. The completed build's recipe manifest still
matches every input byte, and the source-lock digest is unchanged; only
qualification tooling, receipts and documentation were added after the build.

GLM and DS4 remain untouched. The optional auto arm is deferred; GLM cutover
requires its own approval. R29 records remain staged separately and unchanged.
