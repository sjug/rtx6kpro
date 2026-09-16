# R38 GLM-5.3-Flash qualification, 2026-09-15

Qualified for the existing sparky/buddy/rocky/lucky profile. R38 remains
serving; R32 remains available as stopped rollback. No meaningful throughput
regression was observed in the accepted sweep. Small measured decode gains
are not yet established as repeatable across boots.

## Identity and serving contract

- Image: `localhost/voipmonitor/vllm:jj-r38-spark-sm121`.
- Image ID: `ea031e1d3d051033f077fc986bf6f8fce04cf9ab52483d5a719ba13114567fc5`.
- Checkpoint: `local-inference-lab/GLM-5.3-Flash-NVFP4`.
- Revision: `46aaae8a82032f77100f2f03e9cc11b391df3b4d`.
- TP4/DCP1/MTP3, BF16 draft head, aligned recurrent checkpoints.
- Native context 1,048,576, eight sequences, 4,096 batched tokens, utilization 0.85.
- FlashKDA, RoCEnante plus PyNCCL, NCCL LL/Simple, InstantTensor BUFFERED.
- LMCache disabled, no explicit KV-byte cap, no RTX-only tuning block,
  no clear_thinking override.
- Runner SHA256: `0640ed3178aa93dc0df98d33ad9ca01e6b0b941d5149ffe5dac42bf2d7c356f7`.
- Launcher SHA256: `1014b0945ae23147e2dbd831d694df4efe40cf5db2f0525a3fd68fa114aa44e5`.

The serving boot began at approximately 15:08 EDT. No settings or image
changed between the correctness battery and the final isolated timing run.

## Correctness and cache behavior

Receipts in `glm-20260915-retry/aligned-mtp3/` precede the interfering Pi
requests that invalidated the early timing attempts:

- Seven short-pool cases passed; the semantic battery passed 15/15.
- Identical C4 burst: 121.8 output tok/s versus distinct 115.3/112.8.
  No burst serialization was observed.
- Head-of-line probe: first tokens in 0.4 to 0.7 seconds while long decodes
  remained active.
- Frozen long-prefix extensions reused 126,976 tokens (unaligned) and
  129,024 tokens (aligned). The short 4K-to-8K pair and short triples missed,
  preserving the recorded aligned-policy behavior.
- The 262K/262K/1M triple reused 258,048 tokens on repeat and extension.
  The extension returned the exact needle with a normal stop in 387.222 s.
  This is a partial-cache-hit measurement, not cold 1M prefill throughput.
- Native retrieval at 2,048, 2,049, 262,000 and 1,048,000 tokens returned
  exact 739526 with normal stops. The final quiet run repeated these checks
  successfully. Its long-request durations include warmed prefix reuse.

After the accepted grid, the completion probe returned exactly 333 with
a normal stop. All four final container inspections show the expected
image, running state and no OOM flag. These are bounded synthetic correctness
checks, not a general model-quality evaluation.

## Accepted standard benchmark

The canonical, unchanged run_bench.sh completed the 15-cell grid at
16:05:39 EDT. The driver emitted QUALIFICATION-PASS at 16:05:41.
C1/C2/C4 each cover Short/16K/32K/64K/128K, with 30-second measured windows.
Both average and maximum running-request counts match requested concurrency
in all 15 cells. No errors, capacity flags, underfill or warmup timeouts.

| Context | C1 output tok/s | C2 output tok/s | C4 output tok/s | C1 steps/s | C2 steps/s | C4 steps/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Short | 56.42 | 83.70 | 124.22 | 20.74 | 32.36 | 47.38 |
| 16K | 55.86 | 90.42 | 123.61 | 20.61 | 32.15 | 46.04 |
| 32K | 52.50 | 79.98 | 126.85 | 20.41 | 31.59 | 46.67 |
| 64K | 52.71 | 83.75 | 121.59 | 20.32 | 31.59 | 45.61 |
| 128K | 50.89 | 84.33 | 119.11 | 20.11 | 31.89 | 44.37 |

| Geometric mean | R32 | R38 | Change |
| --- | ---: | ---: | ---: |
| C1 engine steps/s | 20.102 | 20.438 | +1.67% |
| C2 engine steps/s | 31.243 | 31.916 | +2.15% |
| C4 engine steps/s | 45.321 | 46.003 | +1.50% |
| C1 output tok/s | 52.796 | 53.634 | +1.59% |
| C2 output tok/s | 82.341 | 84.371 | +2.46% |
| C4 output tok/s | 120.512 | 123.048 | +2.10% |

Effective acceptance lengths are 2.624/2.644/2.675 versus R32's
2.626/2.635/2.659. Steps/s is the acceptance-normalized aggregate metric,
not a count of distinct batched GPU launches. Output includes reasoning.
These compare one clean R38 sweep against the saved quiet R32 boot, not a
multi-boot distribution or proof of a systematic acceptance change.

| Prefill scout | R32 tok/s | R38 tok/s | Change |
| --- | ---: | ---: | ---: |
| 8K | 2780 | 2775 | -0.18% |
| 16K | 2946 | 2955 | +0.31% |
| 32K | 2944 | 2916 | -0.95% |
| 64K | 2938 | 2947 | +0.31% |
| 128K | 2873 | 2872 | -0.03% |

## Isolation, telemetry and capacity

Three incomplete timing attempts are excluded because independent Pi
requests overlapped them. Their raw and resume receipts remain preserved.
An idle interval was insufficient to establish that the agent turn ended.
The accepted r04 began only after the user's explicit completion confirmation
and a zero-running/zero-waiting preflight at 15:52:50 EDT.

From 15:53:10 through 16:05:39, no JIT warnings, server errors or
foreign-address inference POSTs appeared in the four node logs. Exact
per-cell request counts and a live socket-owner check additionally screened
same-address interference. No clock-event samples occurred during the grid.
Mean decode SM clocks were 2469/2486/2528/2440 MHz on
sparky/buddy/rocky/lucky, with peak temperatures 81/84/81/82 C.
Node telemetry, not workstation hardware metadata, is authoritative.

At the 15:11:42 EDT boot report, per-rank KV allocations were
40.53/40.34/39.79/41.03 GiB. Engine effective capacity was 6,129,223 tokens,
versus historical R32's 6,222,210. Capacity is per boot and envelope;
neither figure implies eight full-native-context requests fit simultaneously.

## Receipt identities

- R38 raw run: `20260915T155309-0400__jj-r38-aligned-sm121-tp4-dcp1-mtp3-native1m__r04.json`.
- R38 SHA256: `87545df132270c92a899675bb4c7159d40e75a35aa40c72a23381d3f95b072d5`.
- R32 quiet control: `20260910T160142-0400__jj-r32-aligned-sm121-tp4-dcp1-mtp3-native1m__r02.json`.
- R32 SHA256: `785cdd3816beb35bb2de438c28bd73ac91754536a8ecc1e5bc155a1aaf453fcb`.
- Full paths and arithmetic: `glm-r32-vs-r38-r04.json`.
- Final validation and timing audit: `glm-20260915-quiet-r04/grid-validation.json`
  and `glm-20260915-quiet-r04/benchmark-audit.json`.
- Driver, native probes and final rank state: `glm-20260915-quiet-r04/aligned-mtp3/`.
- run_bench.sh SHA256: `5c79b9760a2381b4b5233f5bbc8f1f279841b46f596dc718a127eaa8eea3e4f2`.
- llm_decode_bench.py SHA256: `2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3`.

Benchmark source and historical raw results were not modified. Campaign
metadata is closed separately. Qwen remains on R32 on dusty/kirby, and
DSv4 Vision remains on its independently qualified R38 profile on rusty/toby.
