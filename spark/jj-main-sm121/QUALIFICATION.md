# JJ-main Qwen qualification, September 21/22

Status: patched retry passed correctness and the full standard grid; not promoted.
The user authorized Qwen on dusty/kirby first
after the build window; this supersedes the earlier GLM-first preparation note.
GLM and DS4 remain serving and untouched. No MTP0 run or fresh R32 grid is planned.

Current candidate: `6633e678fee74f5e1060290a01df812d34d10edcf30e34dd1c9bf7db88260ef3`.
Failed parent: `e7926f763859ba5800cc24198ef297a12322fdaac93b55ac89b88abe02c1b1fb`.
Checkpoint: `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-4p89`, revision
`c374e7e24b54f6cb0017d0c2e6d26823d2f2fb5d`, alias `Qwen3.8-Flash-Next`.
Contract: TP2, MTP3, aligned checkpoints, 262144 context, four sequences,
4096 batched tokens, utilization 0.85, InstantTensor BUFFERED, LL/Simple.

Claude reviewed the runner, correctness driver and benchmark wrapper after a
completed long wait. No launch blocker. The identified missing container receipts
are now written by the driver; the execution wrapper collects per-node kernel and
container logs plus one-second GPU/memory/reclaim telemetry. FlashKDA's stack
growth is a GLM follow-up, not a Qwen GDN-path observation.

Both R32 containers were gracefully stopped worker-first and retained without
deletion. Their pre-window inspections are in `qualification/window/`. Transfer
uses uncompressed Docker archive over verified `10.11.11.7 -> 10.11.11.8`, with
image ID, manifest type, rootfs and label equality checked on receipt. The first
transfer attempt failed before saving the archive because its local disk-space
check unnecessarily used SSH-to-self; corrected to local df, no host-key bypass.

Correctness: exact first completion, semantic battery three times, retrieval at
2848/2849/131072/262000 tokens, fixed-token acceptance, padded concurrency
transitions, prefix reuse and head-of-line probes. Timing starts only after this
battery succeeds. Standard `run_bench.sh` and `llm_decode_bench.py` are unchanged,
with receipts written in this repository. Compare against the completed R32
September 21 return grid; distinguish output throughput, steps and acceptance.

No model correctness or performance result: the endpoint never became ready.
No benchmark launched. No commit or push.

## Startup blocker

Both ranks loaded the checkpoint, then reported (on the head) `0/792 ready`, zero
measured, cached or compiled, and `waiting for ranks`. The engine logged shared
memory broadcast waits. The failure is a source protocol mismatch:

- B12X `0f3a8cbf` includes `06809d5307f27e5dfb71484cfa322109876ef98d`.
  In `preparation/session.py:1447-1451`, multi-rank tuning yields a
  `TuningCacheRequirement` before planning. Subsequent advances return
  `ready_cache` until an agreement is supplied with `cache=` (lines 671-673).
- vLLM `8e1f1e587` configures the TP group's tuning ranks, but its
  `v1/worker/b12x_startup.py` recognizes only collective and tuning readiness.
  The line-147 guard ignores cache-only progress; `_advance_local` does not pass
  `cache=` to the job. Therefore the stage cannot advance.
- Five-second syscall summary of kirby's worker: 15,712 recvfrom and 15,736
  sendto calls. A bounded trace showed repeated TCPStore checks for `/cancel`,
  `stage-0/stop` and `stage-0/failed`, matching this loop, not compilation.
- Local CPU probe `tests/reproduce_cache_agreement_stall.py` executes the pinned
  coordinator against synthetic cache-only B12X progress: 100 advances, zero
  control exchanges, final round zero, no error or completion. This verifies the
  source-level failure; no live Python locals were inspected.
- Claude independently identified the same mismatch after a completed long wait.
  Matching upstream vLLM fix: `0f60770e859ec95fd161b2a64c4925dfcfbd86ed`, currently
  present on the locally fetched Karmic line, not the pinned JJ-main tree.

At the end of the failed attempt, the proposed next step was to port that coordinator fix and its tests into an
explicitly pinned derivative, retaining autotuning and native artifacts where
the final changed-path check permits. Re-run the gates and TP2 qualification.
Single-GPU build gates did not exercise the required multi-rank cache handshake.

## Failed-window health and restoration

Telemetry minima: dusty 30.86 GiB and kirby 34.33 GiB MemAvailable. Kernel logs
record 85 NV_ERR_NO_MEMORY messages on dusty, zero on kirby, no Xids on either.
This is not evidence that the warnings caused the protocol stall.

The readiness driver was terminated deliberately after diagnosis (exit 143);
its exit handler retained container logs and kernel journals and stopped its
observers. The standard grid was never entered. Candidate containers were stopped
worker-first and retained: kirby exit 0; dusty required Podman's SIGKILL fallback
after the requested `stop -t 60`. R32 was restarted worker-first without changing
its containers or configuration. At 02:53 UTC the real arithmetic completion
returned `333` with finish_reason `stop` and the R32 fingerprint; receipt in
`qualification/window/r32-restored-completion.json`. Qwen is back on R32.

## Authorized cache-agreement retry

The user approved backporting upstream `0f60770e859ec95fd161b2a64c4925dfcfbd86ed`
and retrying Qwen with autotuning preserved. `cache-agreement.lock.json` records
the two-file Python-only adaptation and resulting tree
`976a9ac502a7bcdb25b18cb5597d21caafc97252`. Native artifacts, launch settings and
B12X source are unchanged. Claude reviewed the adaptation before this build.

Image `6633e678fee74f5e1060290a01df812d34d10edcf30e34dd1c9bf7db88260ef3`
passed the cache-agreement 25-case gate and existing native, FlashKDA, regression,
draft-head and launcher gates. Build receipt on dusty:
`build-receipts/cache-agreement-20260922T031658Z-3132975`.
Build gates are not model qualification. The R32 pair was gracefully stopped
worker-first for this authorized window. Distribution and the model retry are
in progress; GLM and DS4 are untouched. New qualification receipts use
`qualification/qwen-mtp3-cache-agreement` and `qualification/window-cache-agreement`.

### Retry correctness and startup evidence

The cache handshake now completes across TP2 with autotuning enabled. The
post-KV preparation stage reached 82/82 ready, 1,472 measured candidates and
3,321 compilations. The first real completion returned `333` with normal stop.
Semantic admission passed 18/18, including image and tool round trips. Retrieval
at 2,848, 2,849, 131,072 and 262,000 tokens returned the exact needle and stopped.
Fixed-token acceptance was 2.314996 over 2,654 verifier steps. Output hashes differ
across waves, as in earlier controls; this is not a bitwise-determinism claim.

The padded-transition sequence c1/c3/c1/c2/c4/c1 passed, with acceptance lengths
2.474 to 2.698 and no collapsed phase. Graph-padding coverage is predicted by
the probe, not observed directly from engine internals. Prefix extensions and
repeats reused 28,480 tokens. Head-of-line fresh-request TTFT was at most 0.318 s;
the repeat was 0.301 s. Identical c4 requests did not serialize (131.8 tok/s,
versus distinct probes at 128.2 and 136.9 tok/s). These short probes are not the
standard-grid performance comparison.

Engine-reported KV capacity is 4,857,562 tokens; rank-local budgets are 39.46 GiB
on dusty and 40.75 GiB on kirby. The older benchmark harness prints a different
capacity inferred from block metrics; use the engine admission line, not that
derived 73.6M figure, for this hybrid model's capacity.

Claude's completed independent review confirmed image identity, all gates,
driver pins, and source scope. It identified a startup memory transient during
post-KV preparation: MemAvailable fell to roughly 2 GiB and swap was consumed,
then memory recovered before correctness testing. Startup journals retained now
show 86 NV_ERR_NO_MEMORY lines on dusty and 11 on kirby, no Xids. The missing
Karmic startup-resource cleanup commit `298f265c67` is a source-level lead, not
an established cause or an authorized additional backport. The handshake fix
does not resolve this warning exposure. Full grid and health review remain
separate from these correctness passes.

### Completed standard grid

The unchanged `run_bench.sh` completed all 15 cells, with no request errors,
underfilled cells or capacity limits. Driver exit status was zero. Raw result:
`runs/qwen3.8-flash-next/nvfp4-4p89/2026-09-jj-main-sm121-qualification/throughput/20260921T234306-0400__jj-main-cache-agreement-aligned-mtp3__r01.json`
(SHA256 `2d6527f87bbfa0d47e659df41aafe78702f6904544f179a18e87c2095d2f955a`).
`compare-r32.json` verifies protocol and identity comparability against the
existing September 21 R32 return grid. No new R32 run was made.

Geometric means across five contexts:

| Concurrency | R32 tok/s | JJ-main tok/s | Output change | R32 steps/s | JJ-main steps/s | Step change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 44.19 | 49.79 | +12.68% | 22.07 | 22.78 | +3.21% |
| 2 | 74.92 | 79.67 | +6.35% | 36.58 | 37.36 | +2.14% |
| 4 | 115.79 | 121.51 | +4.94% | 55.79 | 58.05 | +4.05% |

The output gains include acceptance changes (+9.18%, +4.12%, +0.85% respectively),
not just faster verifier steps. These are one-boot observations, not demonstrated
repeatable gains.

| Prefill context | R32 tok/s | JJ-main tok/s | Change |
| --- | ---: | ---: | ---: |
| 8K | 3124 | 3005 | -3.81% |
| 16K | 3004 | 2930 | -2.46% |
| 32K | 2918 | 2838 | -2.74% |
| 64K | 2764 | 2701 | -2.28% |
| 128K | 2501 | 2461 | -1.60% |

Prefill uses one scout per length. The 16K-128K server-side measurements also
show the deficit; neither its cause nor repeatability is established here.

Timed-grid health (03:43:06-03:56:17 UTC): no Xids, driver allocation warnings,
JIT warnings or nonzero GPU throttle samples on either node. Direct reclaim,
allocation-stall and compaction counters did not advance. Minimum MemAvailable
was 7.71 GiB on dusty and 10.20 GiB on kirby. All logged inference clients were
the workstation, 192.168.2.2; this does not distinguish another client on the
same workstation. Use the retained per-node observer data, not the benchmark's
workstation-local hardware table, for remote-node hardware conclusions.

The whole startup window reached minima of 1.57/1.61 GiB MemAvailable and consumed
up to 8.37/7.62 GiB of swap relative to observer start. This exposure is not fixed
by the cache-handshake backport. `health-summary.json` and final journals retain
the evidence. The first-use layer-norm warnings (both ranks) and image-position
embedding warning (head) occurred before the timed grid.

Current state after the run: both nodes remain on the candidate, running and not
OOM-killed, as evaluation serving. R32 containers remain retained as rollback.
No production promotion, GLM/DS4 change, commit, push or benchmark-repository edit.

Claude's completed final review independently reproduced every comparison number
and confirmed all 15 cells met their concurrency targets, with no queueing or
validity flags. Its additional capacity finding is verified against the real
R32 September 21 13:04:55 UTC boot log: 5,272,780 tokens / 42.84 GiB versus this
candidate's September 22 03:33:30 UTC 4,857,562 tokens / 39.46 GiB. The candidate
admits 7.87% fewer KV tokens in these boots. The cause is not established.
This is still well above the configured four 262K requests, but belongs in the
comparison rather than being hidden by the throughput result.

The review's statement that nothing paged during measurement is stronger than
the retained counters support: flat SwapFree and reclaim counters do not measure
swap-in/out directly. Likewise matching client addresses/concurrency supports
an uncontaminated run but cannot prove absence of a second workstation client.
The review's stale-document notes preceded the edits above and are now closed.
