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

## QAD checkpoint preparation, September 30

At the user's request, Qwen was stopped worker first, then head, for the model
and harness update. Both old containers are retained as
`qwen38-flash-next-nvfp4-karmic-beta-20260929-tp2-c374e7e2-20260930`.
They remain stopped; the results above describe the previous checkpoint.

The user downloaded `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` revision
`7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd` on dusty. Its metadata and referenced
shared Hugging Face cache blobs were transferred to kirby over the direct DAC,
`10.11.1.1` to `10.11.1.2`, without deleting existing files. The blob transfer
completed successfully (105,885,596,990 bytes transferred). Both nodes have 36
weight shards totaling 105,839,492,200 file bytes; index tensor bytes are
105,798,973,864. All shard lengths agree with their safetensors headers, all
snapshot links resolve, and both nodes' index, config and header hashes match.
This was metadata/header/size verification plus rsync's transfer checks, not an
independent full-weight hash pass or runtime qualification.

The runner now pins the renamed repository and new revision, checks the new
`qwen4_exp` / `Qwen4ExpForConditionalGeneration` config, and retains the
client-facing alias `Qwen3.8-Flash-Next`. The runner staged on both nodes matches
the workstation copy (SHA256
`52eba183ac522d53bdbe36fd9095ad312eb2bf70defe279c11f6803364e54224`).

The execution and benchmark wrappers use fresh `qualification/qad-7c4f1bc1/`
receipts and the new `nvfp4/2026-09-qad-7c4f1bc1-qualification` campaign; the
benchmark checks the live checkpoint revision before measuring. Historical
`nvfp4-4p89` results remain unchanged. Shell syntax, shellcheck, head/worker dry
runs and the runner's model checks passed. No new containers were started and
no benchmarks were run during this preparation.

### QAD startup, September 30 20:10 UTC

The user directed startup after preparation. Kirby worker started first, then
dusty head, using the staged runner and image `500ae05b`. Both containers carry
revision `7c4f1bc1`, HC-off, MTP3 and `NCCL_PROTO=LL,Simple`. Weight loading and
B12X preparation completed; the API became ready at 20:10 UTC (16:10 EDT).
`/health` returned 200; `/v1/models` reported `Qwen3.8-Flash-Next` and 262,144
context. A temperature-zero, thinking-disabled chat completion returned exactly
`391` for 17 multiplied by 23. Both containers are running with no OOM kill or
restarts. This is a startup smoke result, not the full correctness battery or a
new benchmark grid. The preparation section's stopped state is superseded.

Kirby logged `NV_ERR_NO_MEMORY` allocation warnings during startup at 20:10:00
to 20:10:03 UTC, before API readiness, matching the previously documented class
of startup warnings. No container OOM kill or restart occurred.


### QAD qualification and October 1 baseline, 2026-10-01

The user authorized a fresh qualification boot from the workstation checkout.
Smoke containers were logged, stopped worker first, and retained as
`qwen38-flash-next-nvfp4-karmic-beta-20260929-tp2-7c4f1bc1-smoke-20260930`.
The `-c374e7e2-20260930` containers remain retained for rollback. The new boot
uses image `500ae05b`, revision `7c4f1bc1`, aligned checkpoints, HC off and MTP3.
API readiness was observed at 16:10:22 UTC, after the 16:07:28 UTC window start.

`execute.sh` exited 0 with `KARMIC-BETA-QWEN-SERVING-GATES-COMPLETE`.
Semantic admission, native-context needles (2,848 / 2,849 / 131,072 / 262,000),
fixed-token acceptance, padded transitions and prefix reuse passed. Counting
passed 21/21 plus serial boundaries 56,976 / 56,977 / 56,978 / 56,984 / 56,992 /
56,993. The serial counting probe retained cached-token usage but did not observe
graph dispatch. Head-of-line fresh maximum TTFT was 0.231 s. C4 distinct,
identical and distinct-again probes measured 125.8 / 129.2 / 124.6 tok/s.

**Private replay was not run:** the `05daceed` source trace is no longer available,
and the user explicitly waived replay. `REPLAY-HEALTH-REVIEW-OK` records the waiver
and the passing health review; it does not claim private-replay or multi-turn
behavior on these weights. Window and post-grid kernel logs showed no Xid, OOM or
NV_ERR_NO_MEMORY, including startup. Containers stayed running with no OOM kills
or restarts; `/health` returned HTTP 200 before and after the grid.

The pinned standard harness completed one full 15-cell C1/C2/C4 grid, exit 0:

| Concurrency | Output tok/s | Engine steps/s | Effective acceptance |
| --- | ---: | ---: | ---: |
| 1 | 51.16 | 24.22 | 2.113 |
| 2 | 80.06 | 38.86 | 2.060 |
| 4 | 120.98 | 57.81 | 2.093 |

These are geometric means over the five decode contexts. Prefill scouts at
8K / 16K / 32K / 64K / 128K measured 3,057 / 2,963 / 2,871 / 2,692 / 2,449 tok/s.
This is one grid from one boot, not a repeatability or checkpoint-quality claim.
The completed grid returned `PRODUCTION-BASELINE-VALID` and is the October 1
Qwen comparison baseline:

`runs/qwen3.8-flash-next/nvfp4/2026-09-qad-7c4f1bc1-qualification/throughput/20261001T122604-0400__karmic-beta-20260929-qad-7c4f1bc1-hc-off-mtp3__r01.json`

Receipts remain under `qualification/qad-7c4f1bc1/`. The freshly qualified pair
continues serving. The October 1 image build has not begun.

## Checkpoint 6909a5be qualification, 2026-10-04/05

Same production image `500ae05b` and unchanged profile; the checkpoint is the declared change, but not the only one (see below). The user
downloaded `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` revision
`6909a5bed089a48fa07e956d3915af2537de9368` on both nodes (40 index-referenced shards,
106,334,488,084 bytes; the index `total_size` 110,091,566,076 is stale; three unreferenced
shard files are left over and not loaded). `HYBRID.json` records a hybrid: trunk from QAD
revision `60215d26` (step-5500), MXFP8 attention and MTP experts from `7c4f1bc1`. The
config lists only `Qwen4ExpForConditionalGeneration` and uses per-layer
`quantized_layers` (384 MXFP8, 49 NVFP4, 2 W4A16_NVFP4).

Kit changes: `run-qwen.sh` pins index size, referenced shard bytes, shard count and
architectures per revision (defaults stay on `7c4f1bc1`); `execute.sh` and `benchmark.sh`
take `MODEL_REVISION`, derive the receipt root and campaign from it, and `benchmark.sh`
accepts `NAME` (a retained production pair) and `KV_BUDGET`. The comparators accept a
declared checkpoint change (`--candidate-revision`, `--checkpoint-change`). Staged runner
SHA256 `64c691c77a71085d473dee3b496c08dc9de479fc6953f7f442e6c193da4f86dc`; check-only
runs passed on both nodes for both revisions.

Sequence: a fresh baseline grid on the serving `7c4f1bc1` pair with the current harness
(`cc9bb06a`, `--coding-peak`, KV budget 78,619,040), `PRODUCTION-BASELINE-VALID`, kernel
logs clean. Production stopped worker first, then head. `execute.sh` booted `6909a5be`
(API ready 03:38:08 UTC) and exited 0: correctness battery, counting 21/21 plus serial
boundaries 56,976 to 56,993, head-of-line fresh maximum TTFT 0.252 s, c4 distinct /
identical / distinct-again 128.8 / 132.3 / 133.2 tok/s. Private replay not run (trace
unavailable), waived on the same terms as `7c4f1bc1`. Two grids on that boot, both exit 0.

| Grid | c1 tok/s | c1 steps/s | c2 tok/s | c2 steps/s | c4 tok/s | c4 steps/s | coding peak median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7c4f1bc1 baseline | 49.63 | 24.00 | 79.83 | 39.45 | 120.47 | 58.76 | 71.6 |
| 6909a5be r01 | 47.15 | 23.86 | 80.49 | 38.56 | 123.63 | 58.76 | 73.6 |
| 6909a5be r02 | 51.12 | 23.70 | 81.78 | 38.20 | 124.64 | 58.02 | 73.7 |
| mean vs baseline | -1.0% | -0.9% | +1.6% | -2.7% | +3.0% | -0.6% | |

Geometric means over the five contexts. Effective acceptance moved -0.1% / +4.5% / +3.7%
(C1 swung from 1.98 in r01 to 2.16 in r02). C2 steps/s was lower in both candidate grids;
with a single baseline grid that is not separated from boot variance. Prefill 16K-128K is
within 1.3%; the baseline's 8K scout (2,536 tok/s) is a first-use outlier. GPU KV
5,132,117 tokens vs 5,191,165 (-1.1%).

What else differed: B12X autotune selections are keyed by model path, so the candidate boot measured its own winners (`0549113e...json`, 87 records) instead of reusing `7c4f1bc1`'s (`d21ec6d1...json`, 93 records). Of 79 shared queries, 41 picked a different winner; 14 queries exist only in the old file and 8 only in the new. The checkpoints also differ in format outside the text path's attention and expert layers: the vision encoder is BF16 (`model.visual.*` ignored by quantization) and the PLE `ngram_embedding` is NVFP4. The steps/s differences above are therefore not attributed to the checkpoint alone; selection variability (b12x#463), the NVFP4 PLE lookup and trunk expert routing are not separated.

Health: dusty logged 59 `NV_ERR_NO_MEMORY` lines at 03:37:48-49 UTC, before API readiness
(the startup-only class recorded above); kirby clean; no Xid or OOM; no kernel events
during either grid; containers ran with no OOM kills or restarts.

Reading: performance parity with `7c4f1bc1`, all serving gates passed. No task-accuracy
evaluation was run, so no quality claim is made beyond the gates. Production was restored
to the `7c4f1bc1` pair (`podman start`, kirby then dusty). The candidate pair is retained
as `qwen38-flash-next-nvfp4-karmic-beta-20260929-tp2-6909a5be-qualified-20261005` for a
`podman start` cutover; promotion is the user's decision. Results:
`runs/qwen3.8-flash-next/nvfp4/2026-10-qad-6909a5be-qualification/`; receipts under
`qualification/qad-6909a5be/` and `qualification/qad-7c4f1bc1/mtp3/baseline-20261004/`.
