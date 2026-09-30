# DS4 Vision JJ-main KV capacity: recommendation and deferred plan

Written 2026-09-22 from local read-only analysis of retained receipts and pinned
sources. Deferred until upstream developments are synced to `master`; re-check
every pin and the JJ-main image identity before acting on anything here.
Companion evidence: `docs/ds4-vision-jj-main-kv-investigation-20260922.md`.

Pins used: vLLM R38 composed tree `077347fd` (base `66c29357`) versus tested
JJ-main `976a9ac502a7` (source `8e1f1e587f` plus Spark overlay and `0f60770e85`);
B12X `6abad734` versus `0f3a8cbf`. Receipts:
`spark/jj-main-sm121/qualification/ds4-vision/` and
`spark/ds4-vision/r38/grammar-repair/receipts/qualification/` (R38p, same day).

## Reframing

The question is not "why does the profiler report less KV" but "how much KV can
the node hold with a safe host floor, and how do we know". On unified memory the
profiler's free-memory proxy is host `MemAvailable`
(`vllm/utils/mem_utils.py:147-155`, unchanged between pins), so the admitted KV
is a snapshot of host state at one instant of one boot. Reclamation attacks one
term of that snapshot; the evidence says the binding constraint is elsewhere.

## Proven facts

1. Pre-admission retention. The JJ weights-stage preparation runs inside the
   profiled window (`vllm/v1/executor/abstract.py:357-359`; `init_snapshot` at
   `vllm/v1/worker/gpu_worker.py:439`) and leaves 4.43 GiB (rusty) / 4.45 GiB
   (toby) less `MemAvailable` before profiling starts (memory logs 13:23:52 to
   13:25:02). R38 has no such stage; its B12X tree has no `preparation/` package.
   Introduced by `9c27ec090d`, restructured by `26c055577b`.
2. The qualified JJ boot was cold: every progress line reports `0 cached`, 621
   compilations, 7,040 timing trials, 2,136 prepared units. The selection cache
   persists under the cute compile cache directory
   (`b12x/preparation/session.py:300-307`), which is
   `B12X_CUTE_COMPILE_CACHE_DIR=/cache/jit/jj-main-976a9ac502a7/b12x/cute` on the
   RW host bind mount `/home/jugs/.cache/vllm-jj-ds4-vision -> /cache`. A restart
   of the same image is a warm boot. The warm boot has never been measured.
3. R38 shows the same cold/warm effect: the 2026-09-15 boot compiled 102
   programs and admitted 12.73 GiB; the 2026-09-22 R38p boot compiled 5 and
   admitted 16.85 GiB; the rollback boot compiled 4 and admitted 14.4 GiB. The
   12.73 GiB comparison point is R38's cold boot.
4. Operating points after admission (host `MemAvailable`): JJ steady state
   9.3 GiB (rusty) / 10.6 GiB (toby); R38p ran at 3.7 GiB. JJ's post-admission
   minimum was 3.52 GiB (rusty, 13:29:22) / 4.20 GiB (toby) during state-stage
   tuning plus graph capture; R38p's was 2.68 GiB. JJ leaves about 5.6 GiB more
   idle at steady state than R38p, but its post-admission startup transient
   swings 5.8 GiB (R38p: 1.0 GiB). The transient, not steady state, is what
   would OOM if KV were simply raised.
5. Tuning transient budget: `race_budget` defaults to `cudaMemGetInfo free / 2`
   (`b12x/preparation/session.py:328-335`), the device-side number vLLM rejects
   on UMA; vLLM passes no budget
   (`vllm/model_executor/warmup/b12x_prepare.py:475-483`).
6. Existing cleanup is complete as far as source shows: losers evicted and the
   CUDA stack limit reset to the retained kernels' floor
   (`session.py:1570-1595`, `b12x/_lib/program_cache.py:56-77, 118-142`);
   compiler pool closed at job end (`session.py:770-789`); vLLM gc plus
   `empty_cache` twice before admission (`gpu_worker.py:545-549, 629-633`).
   No preparation-only leftover has been identified.
7. Over-reservation exists but is bounded. `reserve_all` equalizes both DSpark
   workspace lanes (`vllm/v1/worker/workspace.py:211-231`,
   `gpu_worker.py:101-107`, added by `a6994a6b6a`). `a663db14f8` raises the
   compressed-MLA `max_q_chunks` for C128 layers from 20,480 to 47,872 (CPU
   recomputation of `b12x/attention/_shared/mla/compressed_config.py:20-69`
   with 32 padded heads, head_dim 512, bf16, rows 4096, dspark width 192):
   0.63 to 1.47 GiB per lane at `vision_max_n_token=0`; delta per lane is
   +0.84 (0), +0.38 (1024), 0 (2048), +0.34 (4096), 0 (>=8192). Together at most
   about 1.7 GiB plus one lane's true requirement. `vision_max_n_token` for
   Vision-Exp is not in any local config and must be read from the model.
8. Not factors: `10c53de5af` (MoE tuning under graph replay) was reverted by
   `5dafb63d7c` before the pin; `0493cdc071` is present and reduces the gap;
   the 16 compiler processes (`b12x_prepare.py:483`) are transient (8 GiB
   swing that returns at 13:25:02) and never reach the snapshot.

## Hypotheses

- H1: most of the 4.4 GiB is winners (installed payloads and their local-memory
  floor), not losers; a warm boot shrinks it only by the loser share.
- H2: retention scales with prepared-variant count (per layer, mode, row count,
  width), so the structural lever is variant count, not cleanup.
- H3: a host-aware race budget cuts the post-admission transient enough to
  admit several GiB more KV safely.

## Recommendation, in order

1. Measure the warm boot before any reclamation code. Same image, same runner,
   second start with the persisted cache, in a new rusty/toby window. Record
   admitted KV, `MemAvailable` at the four vLLM snapshot points, and the
   post-admission minimum. Restarts are the production operating point, and
   R38's own 4 GiB cold/warm swing suggests this may remove much of the gap.
   Qualification records must state cold versus warm from now on.
2. Change the budget signal, not the accounting: give the B12X race budget a
   host-aware value (available minus a floor) and let vLLM pass it
   (`session.py:226-244` already accepts `race_budget`). This targets the
   binding constraint (the post-admission transient) without touching winners,
   snapshots or scratch envelopes.
3. If the warm boot still admits far less than the measured steady state
   allows, size KV explicitly with `--kv-cache-memory`
   (`gpu_worker.py:565-586`) from warm-boot measurements with a documented
   floor. vLLM already prints the value it would recommend. Runner default
   change: user decision.
4. Only then lane-specific workspace sizing and the `a663db14f8` envelope, with
   a lane-1 plan trace and `VLLM_DEBUG_WORKSPACE=1` slot sizes. The guards fail
   closed on an undersized lane (`workspace.py:237-240` when locked,
   `:310-316` during capture). Expected gain at most about 1.8 GiB.
5. Longer term: why DS4 needs 2,136 prepared units where R38 ran 102 programs.
   If instrumentation confirms H2, the fix is upstream variant coalescing.

## Do not

- Move `init_snapshot` after the weights stage or exclude preparation retention;
  that hides real serving state.
- Write "release preparation-only objects" code before a warm boot and a
  per-stage split identify an object.
- Reduce compiler parallelism for KV reasons.
- Raise `gpu-memory-utilization`, or claim the 4.4 GiB is recoverable.

## Minimum instrumentation (env-gated log lines only)

At each vLLM snapshot point and at the done branch of
`advance_b12x_preparation` (`gpu_worker.py:532-549`): `psutil` available,
`torch.cuda.memory_allocated/reserved`, `cudaDeviceGetLimit(StackSize)`,
retained program count (`len(keep)`, `session.py:1570`), and
`VLLM_DEBUG_WORKSPACE=1` slot sizes. One warm and one cold boot then test H1-H3.

## Recheck after upstream sync (2026-09-22 evening)

Fetch details in `docs/upstream-check-20260922.md`. Consequences for this plan:

- The JJ-main line is retired upstream: recipe `ef8008c` removed the Jovian
  channels from `community-channel.json`; `dev/jovian-judgement` is still
  `8e1f1e587f`. Every pin above stays valid for the JJ image, but a fix there
  would be a local patch on an unpublished line. The next candidate line for
  DS4 Vision is Karmic (`dev/karmic-kraken` 6afb999825, B12X `master` 4f3028b1).
- Item 4 (lane-specific workspace sizing) is already implemented on Karmic by
  `50554de256`: `workspace.py` `reserve_by_lane()`, coordinator lane assignment
  in `b12x_startup.py:306-330`, unit `workspace_lanes` in `b12x_prepare.py`.
  DS4's `_reserve_profile_workspace` was removed; the C128 width envelope
  (`_c128a_profile_widths`) moved into the MLA preparation units. Its effect
  on Spark DS4 Vision admission is unmeasured.
- Items 1 to 3 still apply on Karmic: weights stage inside the profiled window
  (`abstract.py:401`, `gpu_worker.py:617-655`), `race_budget` still
  `cudaMemGetInfo free / 2` and not passed by vLLM, `mem_utils.py` unchanged
  upstream and in the fork. The warm-boot measurement and the host-aware race
  budget remain the first two steps on whichever line is next.
- Selection cache identity changed (B12X `f818b3ab`: compute capability and SM
  count instead of device name); a Karmic image starts with a cold selection
  cache. Qualification must record cold versus warm boot on it too.
- Upstream `ds4-flash.yaml` (recipe `89ce009`) now loads the text checkpoint
  with `INSTANTTENSOR_BACKEND=URING,AIO` (direct I/O). Our launchers use
  `BUFFERED`, which leaves the 81 GiB checkpoint in page cache (R38p: Cached
  106 GiB at boot). Hypothesis, untested: direct I/O changes what
  `psutil.virtual_memory().available` sees at the init and profile snapshots on
  unified memory, so it belongs in the instrumented boot comparison, not in a
  launcher change.
- B12X `master` changes to MoE/MHC tuning predicates and prepared capacity
  (`1dc77276`, `6debca95`, `f20ab3ba`) may change the prepared-unit count
  (H2); recount on the next candidate rather than reusing 2,136.

## Evidence required before reclaiming any buffer

- A per-stage split with a bucket unexplained by torch allocated, the retained
  kernels' stack floor and workspace slots, or torch reserved above allocated
  after `empty_cache`.
- The warm boot agreeing with that attribution.
- For lane sizing: the lane-1 plan trace with scratch bytes and a boot showing
  the workspace guards do not trip.
- After any change: admitted KV, post-admission `MemAvailable` minimum, and the
  full battery (20 semantic including 524K, 32 structured, 15-cell grid) on
  both ranks. Serving evaluation only; no promotion claim.
