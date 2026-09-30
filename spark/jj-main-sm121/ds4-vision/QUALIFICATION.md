# DS4 Vision JJ-main qualification, 2026-09-22

Status: correctness and all 15 grid cells passed; performance is mixed and KV
capacity is lower. Rejected for deployment and rolled back to R38 at the user's
direction on September 22. Candidate containers remain stopped and retained.

Candidate `6633e678fee74f5e1060290a01df812d34d10edcf30e34dd1c9bf7db88260ef3`
was transferred unchanged as a Docker archive over dusty .7 to rusty .5 and
toby .6 on the 10.11.11.0/24 switched 200G fabric. Both receiver IDs match.
Retained R38 containers are stopped, not replaced or deleted. Qwen, GLM and
nous were not changed.

Claude's local pre-launch review found no blocker and confirmed that every
runner environment setting matches the retained R38 qualification contract.
The candidate image's real parser passed both roles on rusty; receipts are in
`receipts/head-render.txt` and `receipts/worker-render.txt`. Both ranks passed
the native rms_norm GPU preflight. First real completion returned 333.

The existing text, image, image-follow-up, tool round-trip and concurrent
mixed-request checks all passed. Exact retrieval passed at 16,384 tokens in
7.122 seconds and 524,000 tokens in 336.189 seconds. The latter reused 16,128
tokens, so its elapsed time is not a cold full-context throughput measurement.
The historical R38 needle took 336.510 seconds; prefix reuse must be compared
before attributing that near equality to prefill speed.

All 32 constrained-JSON requests passed in eight waves of four with K3 enabled.
Metrics increased by 108 draft events, 285 proposed tokens and 259 accepted
tokens, confirming speculation actually ran during this probe.
No scheduler assertion occurred. This tests speculative grammar handling, but
does not reproduce the unknown original request behind the R38 failures.

## Capacity and health caveats

Rusty reported 6.85 GiB KV, Toby 7.76 GiB; the shared pool admits 711,894 tokens,
or 1.36 full 524K requests. Weights remain 81.11 GiB and peak activation 1.08 GiB.
Rusty's consumed weights plus non-Torch memory were 93.29 GiB. Estimated graph
memory was 2.22 GiB versus an actual 0.08 GiB. Relative to the original R38
qualification boot, about 5.29 GiB more consumed memory and 0.59 GiB more graph
reservation explain the reduced KV allocation. The owner of the extra consumed
memory is not established. Compilation/cache state and unreclaimed preparation
allocations are hypotheses, not diagnoses.

One Toby NV_ERR_NO_MEMORY warning occurred at 13:28:53 EDT during preparation.
Interim journals show no Xid or OOM kill. First-use wo_projection preparation
warnings and a BuildPrefillChunkMetadataKernel JIT warning occurred during
correctness testing. Full logs and telemetry are retained in
`../qualification/ds4-vision/`.

## Execution note

The first benchmark invocation stopped before timing because campaign.yaml was
missing. Its log and status are retained with the `missing-campaign` suffix.
After adding campaign metadata in this repository, the benchmark resumed on the
same correctness-qualified container IDs. Neither benchmark source nor its
separate results repository was edited. The original execute status remains
exit 2 to record that interruption; the resumed benchmark has its own status.

## Completed grid

The resumed benchmark and post-grid semantic battery completed at 13:51:12 EDT
with exit 0. All 15 cells passed validate-grid.py. Result SHA-256:
`c45ba7a6db6faa5113a8f0311a13dd1310ca147ea70091f01da3e375c12b361b`.
Matched protocol checks pass against the historical September 15 R38 grid
(`c4df4495701dff6e024984e57e06769b593a461b789de766dccbee5819df1d98`).
This is one candidate boot versus a historical baseline, not a repeatability study.

Geometric means across the five contexts:

| Concurrency | R38 output tok/s | JJ-main output tok/s | Output change | R38 steps/s | JJ-main steps/s | Steps change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 36.20 | 35.85 | -0.98% | 17.35 | 16.78 | -3.29% |
| 2 | 56.01 | 57.37 | +2.42% | 26.33 | 26.95 | +2.34% |
| 4 | 79.03 | 82.51 | +4.40% | 37.25 | 38.89 | +4.38% |

| Prefill | R38 tok/s | JJ-main tok/s | Change |
| --- | ---: | ---: | ---: |
| 8K | 2104 | 2108 | +0.19% |
| 16K | 2253 | 2121 | -5.86% |
| 32K | 2219 | 2115 | -4.69% |
| 64K | 2142 | 2060 | -3.83% |
| 128K | 2003 | 1937 | -3.30% |

First-use projection preparation warnings remain part of these scout timings.
No scheduler assertion, Xid, or OOM kill occurred. Rusty had eight sampled 0x20
clock events during the benchmark recording; Toby had none. Mean recorded SM
clocks were 2415.6/2493.0 MHz. Grid MemAvailable minima were 8.97/10.32 GiB,
with no changes in sampled swap, direct-reclaim, allocation-stall, or compaction
stall counters. Full-window kernel logs retain Toby's one preparation-time
allocation warning. No foreign inference POST appears in the retained head log;
one foreign model-discovery GET was observed before the grid.

Verdict: the candidate passed this correctness/stability window and improves
concurrent decode in this comparison. It is not an unqualified replacement:
C1 steps and prefill are lower, KV capacity is substantially lower, and absence
of the historical crash in a finite test does not establish its cause or cure.

## Independent review

Claude independently recomputed every aggregate and prefill comparison from the
raw grids and confirmed protocol parity, 15 full-concurrency cells with no queue
or validity flags, 20 pre-grid checks and 18 post-grid checks, plus all 32
structured cases. C1 steps are lower in every context, rather than one outlier;
C2 and C4 steps are higher in every context. This remains a historical single-boot
comparison.

The review proposed on-demand projection preparation as a possible explanation
for the C1 deficit. Do not treat it as established: warnings during a cell or
the overall grid are not proof of overlap with its measured decode window.
For example, C1/16K records 14.298 seconds of warmup separately from its 29.952
seconds of measurement. Exact event alignment and preparation durations are not
established here. Prefill scouts do include first-use latency in their TTFT.
Likewise eight thermal-flag samples in the C1 phase do not by themselves establish
the size or cause of the performance delta.

## Requested rollback

Stopped JJ-main on toby then rusty with a 60-second graceful timeout. Started
the exact retained R38 containers, worker then head, at 14:20 EDT. IDs remain
`9982b72184bf` on toby and `c906adac4f03` on rusty, image `ea031e1d3d05`.
The real completion returned 333 with finish_reason stop. Restored R38 admits
1,489,106 KV tokens (14.4 GiB), or 2.84 full 524K requests. Rollback logs,
kernel journals, inspect records and completion are alongside the candidate
receipts. Qwen and GLM were not touched. The recurring R38 scheduler assertion
is still an open repair task, not resolved by this rollback.
