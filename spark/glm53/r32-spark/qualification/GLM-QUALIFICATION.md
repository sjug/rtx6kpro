# R32 GLM qualification

Qualified for the frozen aligned/MTP3 contract, 2026-09-10 16:14:16 EDT.
The quiet repeat passed all 15 cells and both R29 comparisons, closing the
earlier traffic-contaminated grid. R32 remains serving on all four GLM nodes.
User authorized the four-node GLM transfer, cutover,
qualification and standard benchmark after reviewing the completed Qwen arm.
The workstation continuation started at 14:33:05 EDT under
`jj-r32-glm-qualification.service`, invocation
`5230899a682a41cdae3836e10e794e79`.

## Frozen contract

- Candidate image: `74e53e710bef141f6f68e722582569f9c6aa388bce405ad6f1423566a2300c9c`.
- Corrected R29 baseline: `0b15723cb87646bb4628c5cd67aa2a7c879ac5a27105109ea5db06c30113dc0b`.
- Model: local-inference-lab/GLM-5.3-Flash-NVFP4,
  revision `46aaae8a82032f77100f2f03e9cc11b391df3b4d`.
- Nodes: sparky, buddy, rocky, lucky. TP4/DCP1/MTP3, aligned, BF16 draft head,
  FP8 KV, FlashKDA, RoCEnante plus PyNCCL, LL,Simple, InstantTensor BUFFERED.
- Context 1048576, max sequences 8, batched tokens 4096, utilization 0.85.
  No fixed KV allocation, LMCache, auto policy or clear_thinking override.
- Standard harness SHA-256:
  `2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3`.

## Execution

The existing Docker archive is copied in parallel from dusty over the verified
switched 200G paths to 10.11.11.1, .2, .4 and .3. No conversion or compression.
Each archive digest and loaded image ID must match before the driver stops
R29, workers first, head last. R29 containers and images are preserved.

The driver checks short-prompt pool tails, semantic output, concurrency, frozen
prefix pairs/triples, native context through 1M, then runs the standard grid.
Short-prompt correctness is assessed on the task answer, not equality with R29
output affected by the corrected attention-tail handling. Node logs and 1 Hz
GPU clocks, clock-event masks, temperature and power are retained.

The historical full R29 grid is the primary complete baseline. Its later
unchanged C1 repeat is also retained for interpretation: 51.4556 output tok/s
and 19.9010 engine steps/s, versus 49.8226 and 19.9209 in the initial grid.
Do not claim a C1 improvement solely against the weaker initial output number.

Receipts: `glm-20260910/`. Qwen and DS4 are not restarted or reconfigured.
The driver finished at 15:08:57 EDT. Its completion/health checks passed, but
the comparison process exited 1 on a capacity-limited cell. R32 is left
serving on all four nodes with corrected R29 stopped and preserved. The
`QUALIFICATION-PASS` driver banner covers its executed checks, not acceptance
of this rejected benchmark grid. The later quiet repeat below supplies the
clean performance evidence. No repeatable speedup or statistical equivalence
is claimed from this same-boot repeat against historical R29 measurements.

## Progress

- All four switched-200G transfers and loaded image checks passed by 14:36:16
  EDT. The existing archive format and image ID were preserved.
- Cutover began at 14:36:25; the four R32 containers were started worker-first
  by 14:38:01. Corrected R29 containers and images remain stopped and preserved.
- Startup selected exactly `['B12X_ROCENANTE', 'PYNCCL']` for TP, with an
  observed first routed all-reduce. FlashKDA remains selected.
- First-completion and all-rank identity/profile checks passed. All seven
  short-pool semantic cases and all 15 semantic checks passed before timing.
- Repeated-prompt bursts remained batched: initial identical C4 114.4 tok/s,
  later repeat 126.6 tok/s, fresh bursts 108.4 to 126.9 tok/s. Fresh requests
  behind the repeated queue head reached first token in 0.6 and 0.7 seconds.
- Frozen long unaligned/aligned extensions reused 126976/129024 tokens,
  exactly as R29; the short unaligned extension reused zero, also as R29.
- The long triple reused 258048 tokens on both the repeat and 1M extension.
  Extension time 391.139 s versus R29 389.450 s, with identical cache hits.
  These are partial-hit request times, not cold-prefill measurements.
- Exact native-context retrieval passed at 2048, 2049, 262000 and 1048000
  tokens, all with `finish_reason=stop`. The two long checks reused already
  exercised prefixes and are correctness receipts, not fresh-prefill timing.
- The unchanged standard 15-cell grid started at 14:56:15 EDT. The qualification
  driver sends no additional requests during its timed window, but a separate
  Pi client began overlapping requests at approximately 15:03:51. Server logs
  show five running requests during C4 and additional multimodal traffic.
  A read-only workstation socket check identified `pi` PID 927081 connected to
  sparky:8000 alongside benchmark `python3` PID 957198. Neither was interrupted.
  These overlapping cells are not a clean R29 comparison. The user was asked
  to pause competing traffic, then confirmed it was stopped and authorized
  the successful quiet repeat below.
- At 15:04:02, all four ranks logged a new `fused_recurrent_kda_fwd_kernel`
  compilation during this overlapping period. This is recorded separately from
  traffic interference; neither is grounds to attribute a release regression.

Boot at 14:41:14 EDT reports 6222210 effective KV tokens. Per-rank KV budgets:
sparky 41.08 GiB, buddy 40.39 GiB, rocky 40.85 GiB, lucky 40.76 GiB. This is a
boot-specific observation, not a guaranteed gain over R29's 6196490 tokens.

## Measurements that preceded competing traffic

C1 cells and all prefill scouts completed before the extra client appeared.
Each C1 cell has zero errors and no capacity-limit, underfill or warmup-timeout
flag. Geometric means across 0/16K/32K/64K/128K:

| Run | Output tok/s | Engine steps/s | Acceptance length |
| --- | ---: | ---: | ---: |
| R29 original full grid | 49.8226 | 19.9209 | 2.5010 |
| R29 later unchanged C1 repeat | 51.4556 | 19.9010 | 2.5856 |
| R32 C1 before interference | 52.6031 | 19.9608 | 2.6353 |

Against the stronger R29 C1 repeat, output is +2.23%, engine steps +0.30%,
acceptance +1.92%. This is engine parity with variable acceptance, not proof
of a repeatable output gain. The per-cell R32 C1 output/steps are
49.85/20.09, 54.90/20.13, 58.27/20.01, 51.71/19.84, 48.84/19.74 in context order.

| Prefill context | R29 tok/s | R32 tok/s | Change |
| --- | ---: | ---: | ---: |
| 8K | 2745 | 2739 | -0.22% |
| 16K | 2938 | 2944 | +0.20% |
| 32K | 2934 | 2944 | +0.34% |
| 64K | 2927 | 2930 | +0.10% |
| 128K | 2853 | 2859 | +0.21% |

## Rejected concurrency comparison

The full 15-cell sweep completed with zero request errors, but it is not an
isolated performance A/B. At C4/16K, the harness observed an average 4.9
running requests, maximum 5, for four benchmark streams. C2/64K averaged 2.5,
maximum 3. Both cells are explicitly `capacity_limited=true`; seven later
cells from C4/16K onward overlap the separate Pi traffic and are excluded
from clean release comparisons, including those without a harness flag.

The comparison helper refused the first flagged cell. Its empty
`glm-20260910/r29-vs-r32.json` is failed-attempt output, not a comparison
receipt. `workflow-status.txt` and `status.txt` truthfully retain exit 1.
No result JSON was edited, and no benchmark or client process was killed.
At that point a quiet repeat was required, with no model restart or recipe
change. The client connection was still present in the read-only check after
the first benchmark ended. The later authorized repeat resolves this item;
the first run remains excluded as a clean full-grid comparison.

Raw run, in the benchmark results repository:
`runs/glm-5.3-flash/nvfp4/2026-09-jj-r32-vs-r29/throughput/20260910T145615-0400__jj-r32-aligned-sm121-tp4-dcp1-mtp3-native1m__r01.json`.
SHA-256: `8cd4f9ba47e4d688431e53427084f0c11cd3ab69cb1133900761d25351b89b1d`.
The harness digest remained unchanged after the run.

## Health and telemetry

The final completion was exactly `333`, with normal stop. Every rank was
running image `74e53e710bef`, `OOMKilled=false`, at the final check. No serving
error, Xid or OOM kill was found in the archived qualification window.
Startup had 40/16/6/1 `NV_ERR_NO_MEMORY` messages on sparky/buddy/rocky/lucky,
all before readiness; these are retained as the known startup allocation
pressure observation, not conflated with a serving OOM.

Four-node telemetry, benchmark window 14:56:15 through 15:08:54 EDT, 758
samples per node:

| Node | Mean SM MHz | Peak temperature C | Nonzero clock-event samples |
| --- | ---: | ---: | ---: |
| sparky | 2457.2 | 82 | 0 |
| buddy | 2479.5 | 86 | 0 |
| rocky | 2519.8 | 85 | 12 |
| lucky | 2433.3 | 81 | 0 |

These are the node logs, not the benchmark client's workstation telemetry.
One new Triton specialization per rank occurred at 15:04:02 during the
overlapping workload; no JIT warning fell in the preceding C1/scout period.
Qwen on dusty/kirby and DS4/nous were not changed. No commit or push was made.

## Quiet repeat and final verdict

The user confirmed Pi traffic was stopped and authorized continuation.
`repeat-glm.sh` ran as `jj-r32-glm-quiet-repeat.service`, invocation
`75eaf463297d47fcbbfdcc5a96415d54`, starting at 16:01:39 EDT. The queue was
empty before measurement. All four image/profile gates passed and the same
container IDs and start times were retained before and after: no restart,
configuration change, cache clearing or new build occurred.

The standard `run_bench.sh` grid ran from 16:01:42 through 16:14:13 EDT.
Every cell had zero errors and no capacity-limit, underfill or warmup-timeout
flag. Average and maximum running requests equalled the requested concurrency
in every cell. No new JIT warning, server error or kernel journal message
occurred during the repeat. Final completion was exactly `333` with normal
stop, and every rank was running the expected image with `OOMKilled=false`.
Both comparisons passed at 16:14:16; `quiet-repeat/status.txt` records exit 0.

Geometric means over 0/16K/32K/64K/128K, against the complete R29 baseline:

| Concurrency | R29 output tok/s | R32 output tok/s | Output change | R29 steps/s | R32 steps/s | Steps change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 49.8226 | 52.7959 | +5.97% | 19.9209 | 20.1018 | +0.91% |
| 2 | 81.4855 | 82.3413 | +1.05% | 31.0777 | 31.2435 | +0.53% |
| 4 | 121.3621 | 120.5124 | -0.70% | 45.6303 | 45.3211 | -0.68% |

Against R29's stronger later C1 repeat, R32 is +2.60% output, +1.01% steps
and +1.58% acceptance. Acceptance lengths in the full R29/R32 grids are
2.5010/2.6264 at C1, 2.6220/2.6355 at C2, 2.6597/2.6591 at C4.
The C1 output difference must not be described as a six-percent engine gain.

Full clean R32 grid; each entry is aggregate output tok/s / engine steps/s:

| Context | C1 | C2 | C4 |
| --- | ---: | ---: | ---: |
| 0 | 52.60 / 20.20 | 84.83 / 32.01 | 122.36 / 45.97 |
| 16K | 55.63 / 20.18 | 76.16 / 31.06 | 123.19 / 45.72 |
| 32K | 54.29 / 20.14 | 83.13 / 31.38 | 122.55 / 46.12 |
| 64K | 52.09 / 20.02 | 82.23 / 30.84 | 117.58 / 44.79 |
| 128K | 49.57 / 19.97 | 85.72 / 30.94 | 117.04 / 44.04 |

Individual output cells vary with acceptance. C2/16K is -7.82% versus R29
while steps are -0.75% and acceptance -7.12%; the earlier uncontaminated R32
C2/16K cell was 83.52 tok/s versus this repeat's 76.16. C4/64K is -3.46%
output, -1.50% steps and -1.98% acceptance. All per-cell engine changes are
between -1.50% and +2.04%. These are retained variations, not evidence of a
material overall engine regression or universal per-cell output parity.

| Prefill context | R29 tok/s | R32 quiet repeat tok/s | Change |
| --- | ---: | ---: | ---: |
| 8K | 2745 | 2780 | +1.28% |
| 16K | 2938 | 2946 | +0.27% |
| 32K | 2934 | 2944 | +0.34% |
| 64K | 2927 | 2938 | +0.38% |
| 128K | 2853 | 2873 | +0.70% |

Quiet-window node telemetry, 16:01:42 through 16:14:12, 750 samples per node:

| Node | Mean SM MHz | Peak temperature C | Nonzero clock-event samples |
| --- | ---: | ---: | ---: |
| sparky | 2461.6 | 79 | 0 |
| buddy | 2483.0 | 82 | 0 |
| rocky | 2522.0 | 81 | 1 |
| lucky | 2435.5 | 82 | 0 |

Verdict: the seven short-pool checks, 15 semantic checks, concurrency/cache
probes, exact retrieval through 1048000 tokens and clean 15-cell grid pass
for R32 aligned/MTP3. No material overall performance regression is
demonstrated. Keep R32 serving, with R29 stopped and preserved as rollback.
The automatic checkpoint policy and other optimization arms remain separate,
unqualified experiments. Qwen and DS4 were untouched; no commit was made.

Raw quiet run:
`runs/glm-5.3-flash/nvfp4/2026-09-jj-r32-vs-r29/throughput/20260910T160142-0400__jj-r32-aligned-sm121-tp4-dcp1-mtp3-native1m__r02.json`.
SHA-256: `785cdd3816beb35bb2de438c28bd73ac91754536a8ecc1e5bc155a1aaf453fcb`.
Comparisons: `glm-20260910/quiet-repeat/r29-vs-r32.json` and
`glm-20260910/quiet-repeat/r29-c1-repeat-vs-r32.json`. The harness SHA remained
`2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3`.
