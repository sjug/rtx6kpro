# R38 DSv4 Vision qualification, 2026-09-15

Qualified for the existing bounded rusty/toby serving profile. R38 remains
serving; R32 remains available as rollback. This is not a multi-boot speed
claim or a broad vision/reasoning accuracy evaluation.

Image: ea031e1d3d051033f077fc986bf6f8fce04cf9ab52483d5a719ba13114567fc5.
Model: deepseek-ai/DeepSeek-V4-Flash-Vision-Exp.
Revision: 6821d6ad3681a4b137b066b76094fa82ebd0a380.
Profile: TP2/DCP1, DSpark probabilistic K3, 524288 context, four sequences,
4096 batched tokens, utilization 0.85, maximum reasoning default, FP8 KV,
NCCL LL/Simple over the f1 pair rails. No channel pin or RoCEnante.
The R38 source launcher is byte-identical to the qualified R32 one.
See README.md and the frozen runtime-file digest manifest for identities.

## Correctness before timing

First meaningful completion passed at 15:15:24 EDT. The complete admission
battery passed at 15:21:39:

- Default maximum-reasoning arithmetic: 3/3 exact 333.
- Non-thinking arithmetic: 3/3 exact 133.
- Image colors and image follow-up: 3/3 each, exact red/blue and blue.
- Tool call: exact multiply arguments; tool return: exact 391.
- Four simultaneous mixed text/image requests: 4/4 exact.
- Exact 16384 and 524000 token retrieval: exact 739184, normal stops.

The 524K request took 336.510 seconds and reused 16128 tokens from the 16K
probe, as did the historical R32 receipt (413.06 seconds). Neither number
is a completely cold 524K prefill measurement. These synthetic needles prove
the tested cases, not general long-context quality or maximum image load.
The same semantic, vision, tool and mixed-load battery passed after timing.

## Standard benchmark

Canonical run_bench.sh, unchanged source and standard 15 cells, C1/C2/C4
at Short/16K/32K/64K/128K with 30-second measured windows. Completed at
15:40:14 EDT; post-grid checks and final container assertions completed
at 15:40:35 with exit 0. No request errors, underfill, warmup timeouts or
capacity flags; actual running-request counts never exceeded each cell's
requested concurrency. Raw grid SHA256:
c4df4495701dff6e024984e57e06769b593a461b789de766dccbee5819df1d98.

| Context | C1 output tok/s | C2 output tok/s | C4 output tok/s | C1 steps/s | C2 steps/s | C4 steps/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Short | 36.80 | 55.08 | 78.87 | 17.62 | 27.46 | 38.55 |
| 16K | 36.55 | 56.86 | 81.46 | 17.54 | 26.43 | 38.37 |
| 32K | 36.21 | 57.66 | 80.91 | 17.32 | 26.81 | 37.64 |
| 64K | 36.50 | 55.60 | 77.54 | 17.18 | 25.93 | 36.31 |
| 128K | 34.98 | 54.89 | 76.51 | 17.08 | 25.09 | 35.49 |

| Geometric mean | R32 | R38 | Change |
| --- | ---: | ---: | ---: |
| C1 output tok/s | 34.483 | 36.203 | +4.99% |
| C2 output tok/s | 57.799 | 56.008 | -3.10% |
| C4 output tok/s | 79.749 | 79.034 | -0.90% |
| C1 engine steps/s | 16.037 | 17.346 | +8.16% |
| C2 engine steps/s | 26.261 | 26.332 | +0.27% |
| C4 engine steps/s | 37.814 | 37.253 | -1.48% |

Effective acceptance lengths were 2.087/2.127/2.122, versus R32's
2.150/2.201/2.109. C2 output loss accompanies lower acceptance despite
flat engine steps; neither a systematic acceptance shift nor a repeatable
C4 regression is established by this single randomized-prompt run.
Steps are the harness's acceptance-normalized aggregate metric, not counts
of distinct batched GPU launches. Output includes reasoning tokens.

| Prefill scout | R32 tok/s | R38 tok/s | Change |
| --- | ---: | ---: | ---: |
| 8K | 1578 | 2104 | +33.33% |
| 16K | 2280 | 2253 | -1.18% |
| 32K | 2242 | 2219 | -1.03% |
| 64K | 1893 | 2142 | +13.15% |
| 128K | 1571 | 2003 | +27.50% |

Scouts are one sample each. The historical 8K cell is weak; do not promote
that difference into a general gain. The long-context improvement is
promising but needs a fresh matched control/repetition to prove a release
speedup. Existing R32 raw receipts remain unchanged.

## Timing audit and capacity

No foreign-address inference POSTs or server errors appeared during the
benchmark. Per-cell request counts also remained matched, unlike the
separate interrupted GLM run. Six extra-token decode specializations per
rank compiled during cell warmups; all finished before the corresponding
ready/measurement event. The mHC compile at 15:27:24 was in calibration,
before the benchmark's timed scouts. No observed compile overlaps a measured
decode window. See receipts/benchmark-audit.json and raw logs for timestamps.

Mean measured-decode SM clocks were 2418 MHz on rusty and 2492.8 MHz on toby,
with zero clock-event samples during measured decode. Rusty had two clock
events elsewhere in the benchmark; peak temperatures were 88 C/84 C.
Node nvidia-smi receipts, not the benchmark's workstation hardware summary,
are the authority for these Spark measurements.

KV: rusty 12.73 GiB, toby 12.81 GiB; engine effective capacity 1,323,142 tokens
at 524288 context, versus historical R32's 1,165,720. This is per-boot capacity,
not a throughput gain or a claim that four maximum-length requests fit.

## Controller correction and provenance

The original controller stopped after successful admission because it
required the rank-zero backend log on the worker too. The model did not
fail. Both actual startup commands disable custom all-reduce; only the head
logs ['PYNCCL']. benchmark.sh checks that real contract and resumed without
restarting or repeating the successful long needle. Original failed status
and successful admission receipts are preserved alongside the separate
benchmark-status.txt exit 0 receipt.

The comparison is recorded in receipts/r32-vs-r38.json with both raw paths
and digests. llm_decode_bench.py SHA256 is
2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3;
run_bench.sh SHA256 is
5c79b9760a2381b4b5233f5bbc8f1f279841b46f596dc718a127eaa8eea3e4f2.
Neither benchmark source nor the historical results were modified.
