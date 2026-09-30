# Qwen3.8-Flash-Next TP2 on the Karmic beta image, September 29

Image `500ae05b98da0658c1a5e1820387f96f2121c5659bd7954ad1c5861f20934f05`
(`karmic-beta-20260929-spark-sm121`) on dusty (head) and kirby (worker), user-authorized
window. Profile unchanged from the serving `a25bedd4` runner apart from the image and name:
TP2, MTP3, aligned recurrent checkpoints, HC-off, capture 32, utilization 0.85, 262,144
context, 4 sequences, 4,096 batched tokens, checkpoint revision `c374e7e2`.

Receipts: `qualification/` in this kit (git-ignored; private replay responses stay there).
Benchmark results: `runs/qwen3.8-flash-next/nvfp4-4p89/2026-09-karmic-beta-sm121-qualification/`.

## Sequence

1. Image transferred dusty to kirby over 10.11.11.7 to .8 with archive digest check
   (`TRANSFER-OK`); runner byte-identical on the workstation copy, dusty and kirby.
2. `execute.sh`: worker then head, container checks (image, HC_TP=0, MTP3, capture 32).
   Cold first boot (new cache namespace) about 9 minutes, including B12X autotune of
   blockscaled GEMM and `gdn_prefill` (the stage the B12X #417 deadlock fix covers; no stall).
3. Correctness battery `QWEN-CORRECTNESS-BATTERY-PASS`: first completion, semantic admission,
   native-context needle at 2,848 / 2,849 / 131,072 / 262,000, fixed-token acceptance,
   padded concurrency transitions (no collapsed phase), prefix reuse.
4. Counting 21/21 over three rounds plus serial boundary cases 56,976 to 56,993.
   Head-of-line fresh maximum TTFT 0.236 s (a25bedd4: 0.262 s). Identical-vs-distinct c4
   125.7 / 134.8 / 129.5 tok/s aggregate.
5. Private replay (same request `3ec1524a...` as the a25bedd4 review): two completions,
   `tool_calls`, valid JSON arguments (completion and structure only; no tool execution or
   task-correctness check). Health review recorded in `REPLAY-HEALTH-REVIEW-OK`.
6. Two standard grids r01, r02 (unchanged `run_bench.sh`, pinned digests), same boot.
   A first attempt was refused before measuring because the campaign did not exist; it is
   kept as `qualification/mtp3/benchmark-attempt0-campaign-missing/`.

## Results against a25bedd4 (its r01/r02, September 23)

Geometric means over the five contexts; comparison is mean of two grids against mean of two.

| | c1 tok/s | c1 steps/s | c2 tok/s | c2 steps/s | c4 tok/s | c4 steps/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| a25bedd4 r01 | 49.01 | 23.46 | 79.20 | 38.13 | 121.70 | 57.24 |
| a25bedd4 r02 | 47.45 | 23.23 | 78.33 | 37.83 | 116.46 | 56.39 |
| beta r01 | 50.00 | 23.40 | 78.60 | 38.50 | 116.67 | 57.58 |
| beta r02 | 48.56 | 23.31 | 78.50 | 37.61 | 122.07 | 57.34 |
| beta vs a25bedd4 | +2.2% | +0.1% | -0.3% | +0.2% | +0.2% | +1.1% |

Prefill (tok/s, mean of two): 8K 3,023 vs 3,055 (-1.0%), 16K 2,958 vs 2,980 (-0.8%),
32K 2,860 vs 2,864 (-0.2%), 64K 2,714 vs 2,734 (-0.7%), 128K 2,463 vs 2,479 (-0.6%).

c4/64K (the open decode-gap cell): steps/s 56.33 / 58.69 vs 55.54 / 55.48; tok/s 115.0 /
122.0 vs 122.5 / 110.2, with acceptance 2.04 / 2.08 vs 2.21 / 1.99. The output swings in
this cell track acceptance, not engine speed.

KV capacity: 5,174,429 tokens vs 5,237,990 (-1.2%).

Reading: observed parity. Engine speed and prefill fall within the run-to-run spread of the
baseline (its own r01 to r02 differs by up to 1.5% in steps/s and 4.3% in c4 tok/s). Both beta
grids share one boot and are compared with historical a25bedd4 controls, so this is not
demonstrated equivalence across boots, and no performance claim beyond parity is made.

## Health

- Startup `NV_ERR_NO_MEMORY` bursts on both nodes (dusty 50 at weight load, kirby 85 during
  autotune), none after the API came up. Same startup-only pattern as a25bedd4; retained
  as an unresolved observation, not claimed fixed.
- Zero kernel NV_ERR / Xid / OOM events during either benchmark grid, and no allocation
  warnings in the container logs. Containers running, OOMKilled false, restarts 0, swap 0.
- One-time Triton JIT of `_apply_grammar_bitmask_kernel` on the first structured-output
  request (first-use latency only).
- The MTP "no KV cache group could be identified as the draft model's" warning also appears
  on a25bedd4; not new.

## Status

Independent review: Codex recomputed all four grids' geometric means (deltas match), found
all 30 beta cells clean and the health logs event-free, with no material discrepancy; its two
wording qualifications are applied above.


Correctness, counting, replay and both grids passed; performance at parity. **Promoted to
production by the user on 2026-09-29**: the beta image is the production Qwen image on dusty/kirby. The a25bedd4 containers are stopped and
retained for rollback (`podman start`, kirby worker first, then dusty head). DS4 Vision (2026-09-29) and GLM (2026-09-30) were later qualified and promoted on this image; see ds4-vision/QUALIFICATION.md and glm/EXECUTION.md. DS4.1 has not been qualified.
