# GLM on the Karmic beta image, September 29 to 30 (promoted to production)

Kit: this directory. Image `500ae05b...`, profile per `../../../../runs/glm-5.3-flash/nvfp4/2026-09-karmic-beta-sm121-qualification/campaign.yaml`
(nvfp4 draft head, 4 sequences, native 1M, upstream verbatim chat template, aligned checkpoints,
RoCEnante plus PyNCCL, `NCCL_PROTO=LL`).

- 19:41 EDT production R38 stopped worker first (logs `~/logs/glm53-flash-nvfp4-jj-r38-spark-tp4-*`).
- Transfer dusty to sparky (one niced save and send), sparky fan-out; image verified on all four.
- 19:57 ready (cold boot about 12 minutes). Container contract checks and RoCEnante/PyNCCL backend passed.
- Passed: short pool 7/7; semantic battery x3 including every tool round trip (the Sept 21 blocker);
  concurrency; frozen prefix matrix; short prefix triples.
- **Long prefix triple, 262,000 tokens:** cold first stage 316.6 s (R27 to R38: 91.3 to 92.5 s on the same
  test). The head logged no engine stats from 20:11:19 to 20:16:39, then one window credited about
  26,200 tok/s. The repeat stage hit the prefix cache (5.8 s). The extension stage (from 20:16:41)
  stalled the same way. During the stall all four GPUs showed 96 percent utilization at 39 to 44 W
  (spin-waiting, not compute), no memory pressure (PSI 0, no swap, 3 to 7 GiB available), with
  sparky's EngineCore at about 92 percent CPU and the worker at about 183 percent. No cache files were
  being written, so not an obvious JIT/tuning write. py-spy is not in the image; no debugger was attached.
- 20:2x EDT stopped by the user. Driver stopped first; beta containers stopped worker first, exit 0, logs
  archived as `~/logs/glm53-flash-nvfp4-karmic-beta-20260929-tp4-stall-*`, containers retained. R38 restored.

Not run: native-context retrieval, standard grid.

## Root cause (2026-09-29 follow-up)

Not a stall and not a beta source regression. It was a launcher transport error in this kit. `run-glm-tp4-node.sh`
set `NCCL_PROTO=LL` and one NCCL rail (`NCCL_IB_HCA=rocep1s0f0`). Every other run uses `NCCL_PROTO=LL,Simple`
over both rails (`rocep1s0f0,roceP2p1s0f0`): live production R38, the September 21 Karmic runner, the September 22
base-image run (image `1a7a8acf`, which this beta refreshes) and our Qwen and DS4 Vision beta runners. The kit's
note that LL single-rail was "production's proven" setting was wrong.

Mechanism: a 4,096-token prefill chunk produces all-reduces of about 32 MiB (4096 x 4096 x bf16). RoCEnante
carries only up to 2 MiB ("NCCL remains the fallback above 2MB" in the boot log), so prefill all-reduces go to
NCCL. There, LL-only forces the flag-polling low-latency protocol onto bulk transfers on half the fabric. Decode
messages stay under 2 MiB on RoCEnante, which is why short requests, semantics and tools passed.

Evidence from the receipts:
- Prefill was slow at every length, not only long ones. 131K fresh prefix-matrix base took 156 to 157 s on beta
  against 44.6 to 48.0 s on R27 through R38 (3.4x). 4K took 5.8 s and 8K took 10.5 s. 262K took 316.6 s. That is
  a flat ~830 tok/s, so there is no context-length cliff.
- Median GPU power over the 131K fresh prefill was 34 to 37 W on beta against 61 to 66 W on R38, at the same SM
  clock and at 96 percent "utilization". That is communication-bound spinning.
- The base image `1a7a8acf` on LL,Simple dual rail prefilled at 2,759 to 2,917 tok/s (R38 parity).
- The engine "silence" is normal: prompt tokens are credited only when a request's prefill finishes, and
  all-zero stat windows log at debug level.

Upstream check: beta advanced to `2981ddf5` (restore-admission scheduler stall #946, Qwen GDN trial pool
retention, MiMo loader, #926 ring mapping). None of these touches this path. #946 is a full-KV-pool decode
starvation with idle GPUs, which is a different symptom.

Fix: the runner now sets `NCCL_PROTO=LL,Simple`, `NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0` and
`NCCL_SOCKET_IFNAME=enp1s0f0np0,enP2p1s0f0np0`. It still needs a boot with the corrected runner: at minimum the 131K
prefix-matrix pair and the long triple, then the rest of the qualification. Receipts: `../qualification/glm/`
(git-ignored).

## Confirmation and full qualification with the corrected transport (2026-09-29, user-approved)

Same image `500ae05b`, model revision, launcher and profile as the stopped attempt. The only change is NCCL:
`LL,Simple` over both rails. NCCL logging was at INFO for this boot. R38 was stopped worker first at 21:09
(logs archived, containers retained). The stopped LL single-rail containers are kept as
`glm53-flash-nvfp4-karmic-beta-20260929-tp4-ll1rail`. The first confirmation attempt stopped at its own
render check (`printf %q` escapes commas), before any node changed state (`../qualification/glm-transport-attempt0-render-grep/`).
Boot to first correct completion took 3 minutes on warm caches. NCCL reported `NET/IB : Using [0]rocep1s0f0 [1]roceP2p1s0f0` on all four ranks.

Transport confirmation (`confirm-transport.sh`, `probe-fresh-prefill.py`, nonce-prefixed, zero cache hits):
4K 1.84 s; 8K 4.09 s and 2.72 s; 131K 42.33 s and 42.50 s (about 3,090 tok/s). The broken run took 5.8 s at 4K,
10.5 s at 8K and 156 to 157 s at 131K. **Root cause confirmed: the runner's NCCL transport, not the image.**

Full battery (`qualify-glm.sh`, receipts `../qualification/glm-dualrail/`), all passed:
- Short pool 7/7; semantic x3 including tool round trips; concurrency.
- Frozen prefix matrix: long-unaligned 43.2 s / 49.3 s, long-aligned 42.9 s / 47.9 s (R38: 47.1/50.3, 44.7/49.7);
  short 1.9 s / 3.8 s.
- Long triple: 262K cold first 89.8 s (R27-R38 91.3 to 92.5 s, broken run 316.6 s); repeat 2.1 s (258,048 cached);
  1,048,000-token extension 380.5 s (R38 387.2 s).
- Native-context needles 2048, 2049, 262000 and 1048000 exact. The 262K and 1M timings are cache reuse, not prefill.
- Grid: 15 valid cells (`grid-validation.json`, sha256 `bcf88f33...`). Post-grid completion correct.

Against R38 (Sept 15 r04, `r38-vs-karmic-beta.json`):

| | c1 | c2 | c4 |
| --- | ---: | ---: | ---: |
| Output tok/s | -5.4% | -6.7% | -5.4% |
| Engine steps/s | +0.3% | -2.3% | -0.4% |
| Effective acceptance length | -5.7% | -4.5% | -5.1% |

Prefill scouts 8K to 128K: -1.4, -1.2, +1.6, +0.5, +1.0 percent. Engine speed and prefill are at parity. The decode
output loss comes from speculative acceptance. This profile uses the NVFP4 MTP draft head, while R38 uses BF16. The
September 22 base-image run with the NVFP4 head also showed acceptance 2.9 to 4.9 percent lower than R38. This
suggests the draft head as the cause, but no same-boot arm tested it. The expected steps gain from the NVFP4 head in
the September 24 bisect (4.2 to 4.9 percent) did not appear here. This is one boot against a historical baseline.

Health: zero Xid/OOM; one `NV_ERR_NO_MEMORY` on lucky at 21:12:09, during startup (the September 22 base run logged
92 at startup). Containers running, no OOM kill, 0 restarts; MemAvailable 5.3 / 5.7 / 6.2 / 8.7 GB.

Status: correctness qualified. Decode output is about 5 to 7 percent below R38, so beta does not yet replace R38 on
performance. The beta containers are currently serving; R38 is stopped and retained. Promotion or restore is the
user's decision. The next candidate arm is `VLLM_GLM53_MTP_DRAFT_HEAD=bf16` on this image.

## Decode gap attribution (2026-09-29 late, same beta boot, request-level A/B)

The benchmark sends chat requests without temperature, top_p or template kwargs, so server defaults apply. R38
launched with no template or sampling defaults. The checkpoint `generation_config.json` has `top_p: null`, so R38
used top_p 1.0. The template renders "Reasoning Effort: Max" when the effort is unset (template line 8). This
kit's launcher (from upstream recipe `f28aea04`) sets `reasoning_effort: high` and `top_p: 0.95`. The two grids
therefore measured different generated text. In the V2 model runner, draft sampling ignores top_p
(`vllm/v1/worker/gpu/spec_decode/speculator.py` `_copy_request_inputs`, unchanged since R38). The target applies it
before the ratio test (`rejection_sampler.py` `_verify` via `apply_sampling_params`), and it adds a logits
processing pass per verify step.

`probe-acceptance-ab.py`: sequential c1, alternating arms, idle endpoint, acceptance from the server's
spec-decode counters. Receipts are in `../qualification/glm-top-p/`.

| Prompt | Arm vs beta server defaults | Acceptance length | tok/s |
| --- | --- | --- | --- |
| 3 generic prompts, 12 x 1024 | top_p 1.0 | 2.5929 vs 2.5617 (+1.2%) | +1.2% |
| 3 generic prompts, 12 x 1024 | effort max | 2.6275 vs 2.6141 (+0.5%, per-prompt signs mixed) | -0.8% |
| grid prompt (`GENERATION_PROMPT`), 8 x 2048 | effort max + top_p 1.0 (R38 defaults) | 2.6144 vs 2.4961 (+4.7%) | +5.3% |
| grid prompt, 8 x 2048 | effort max only | 2.6329 vs 2.5639 (+2.7%) | +2.6% |

The default arm varied 2.7 percent between the two grid-prompt runs, so single-factor gains cannot be subtracted.
The combined change improved 6 of 8 matched round pairs, and effort alone improved 5 of 8. Conclusion (reviewed by
Codex): same-boot c1 probes show a workload-dependent acceptance gain from restoring R38-like request defaults,
including +4.7 percent on the zero-context grid prompt. That makes the launch defaults a substantial contributor to the
decode gap. It is not a full decomposition. Only c1 at context zero was probed with the grid prompt. The probe's
metrics (whole-request acceptance and tok/s, which include prefill) are not the grid's steady-window metrics. The
matching historical R38 cell (c1, context 0) had acceptance 2.7203, about 3.9 percent above the probe's 2.6144. No
residual engine or head effect is bounded: the c2 steps -2.3 percent and the NVFP4 head remain open.

No sampler code difference: the Gumbel draft-noise salt (`fe755c8899`, upstream #54282) is absent from R38's bare
integration commit, but it is present in R38's built package tree `2d387573` (`gumbel.py` blob `5d1660a7`, byte-identical
to beta). An earlier line in this section claiming otherwise was wrong and has been withdrawn.

A like-for-like grid needs matched server defaults on both sides. Which defaults production should serve (upstream
recipe High/0.95, or R38's implicit Max/1.0) is a user decision.

## Matched-defaults comparison, R38 versus beta (2026-09-30, user-approved)

User decision: production request defaults are `reasoning_effort` max, `clear_thinking` false, temperature 1.0,
top_p 0.95. Beta's runner now sets Max. `run-glm-r38-matched-defaults-node.sh` is production R38 unchanged except
for the same request defaults, 4 slots (capture to 16, user rule) and a separate container name. It uses the
production launcher `1014b094` on the nodes. `compare-matched-defaults.sh` ran both arms on one cluster. Receipts are
in `../qualification/glm-matched-defaults/`. Every rank's contract was checked: image, `LL,Simple`, 4 slots, both
defaults, aligned checkpoints. Draft heads remain each image's own (R38 BF16, beta NVFP4).

Sequence: the 2026-09-29 beta (effort high) was stopped and kept as `...-tp4-effort-high`. The R38 comparison was
ready at 09:51. Its semantic battery missed the exact tool format (`17 × 23 = 391\n\nFINAL: 391`), so the rate-based
diagnostic ran: 84/100 against the 85/100 baseline. All 16 misses had the correct 391 plus extra text and made no tool
call, so it passed under the user's tool gate. Grid 09:54 to 10:07. Two run bookkeeping stops (a missing `RESULTS_REPO`,
then the comparator's metadata key) changed no state. Beta with Max was ready at 10:11, passed the semantic battery
including the exact tool format, and ran its grid 10:11 to 10:24. Both grids have 15 valid cells (R38 `86e39993...`,
beta `acd2ad53...`). The grids omitted `recurrent_checkpoint_policy` metadata, so that identity key was checked from all
eight container receipts (all aligned), and the unchanged `compare-grids.py` logic ran with that one key removed
(`r38m-vs-beta.json`).

| Beta vs R38, matched defaults | c1 | c2 | c4 |
| --- | ---: | ---: | ---: |
| Output tok/s | +8.6% (49.96 to 54.23) | +2.3% (80.64 to 82.51) | +6.7% (119.31 to 127.36) |
| Engine steps/s | +4.7% | +3.1% | +6.7% |
| Effective acceptance | +3.7% | -0.8% | 0.0% |

Prefill scouts 8K to 128K: -4.1, -3.0, -1.9, -0.9, +0.5 percent (shorter contexts slightly lower).

Reading: in this matched run, beta decoded faster than R38 at every concurrency; prefill was slightly lower at
8K to 64K and slightly higher at 128K. The steps gain is consistent with the September 24 bisect's NVFP4-head
advantage (4.2 to 4.9 percent), but this comparison does not isolate that cause. Request defaults and slots were
matched; the images, draft-head precision and other runtime components still differed. This is one boot per arm,
not evidence of a repeatable gain or proof that request defaults alone caused the September 29 regression.
Repeated matched runs and a focused draft-head control would be needed for those claims.

Health: no Xid or OOM. `NV_ERR_NO_MEMORY` warnings fell almost entirely inside the two boots (09:49 to 09:50 and 10:10 to
10:11). One on sparky at 09:54 fell between the tool diagnostic and the R38 grid. Containers: beta running, not OOM
killed. The R38 comparison exited 0 and is kept. MemAvailable 5.3 / 6.0 / 6.3 / 8.3 GB.

State: beta (Max) is serving GLM. R38 production (`glm53-flash-nvfp4-jj-r38-spark-tp4`), the R38 comparison and the
effort-high beta are stopped and retained. Promotion is the user's decision.

## Promotion (2026-09-30)

**Promoted to production by the user on 2026-09-30.** GLM-5.3-Flash TP4 serves from image `500ae05b` as
`glm53-flash-nvfp4-karmic-beta-20260929-tp4`, using `run-glm-tp4-node.sh`: NCCL `LL,Simple` over both rails; request
defaults `reasoning_effort` max, `clear_thinking` false, temperature 1.0, top_p 0.95; 4 slots; NVFP4 draft head; aligned
checkpoints; native 1M context. That boot started 2026-09-30 10:11 EDT with NCCL logging at WARN (the runner default).
Rollback: production R38 `glm53-flash-nvfp4-jj-r38-spark-tp4` is stopped and retained. Stop beta worker first, then
`podman start` R38 workers, then sparky. Other retained containers: `...-jj-r38-spark-tp4-matched-defaults`,
`...-karmic-beta-20260929-tp4-effort-high` and `...-tp4-ll1rail`. The restart policy is `no`: after a node reboot, start
the containers manually, workers first.

## Prefill attribution (2026-09-30, user-approved two-boot profile)

Question: the matched grid showed beta prefill 4.1, 3.0, 1.9 and 0.9 percent below R38 at 8K to 64K (+0.5 at 128K).
Server-side prefill times agreed with client time (the gap is not in the API layer). Method (`profile-prefill.sh`,
receipts `../qualification/glm-prefill-profile/`):
- Both runners got an opt-in `PROFILER_DIR`; renders are byte-identical when it is unset.
- R38 (matched-defaults runner) was booted, then beta; beta stays serving. The profiler is idle unless
  `/start_profile` is called.
- On each boot: one 4K warmup, three timed fresh 16K requests, then one profiled fresh 16K (torch profiler, no stacks).
- Kernel families were compared with `summarize-prefill-profile.py` (`kernel-compare.json`).
- Retained renames: the production beta boot became `...-tp4-noprof-20260930`, and the grid R38 comparison became
  `...-matched-defaults-grid-20260930`.

Results:
- **Steady-state kernels are at parity.** Summed GPU kernel time for the profiled 16K prefill was 5,187 to 5,189 ms per
  rank on R38 and 5,180 to 5,191 ms on beta, with a wall span of 5,246 ms on both. The family shifts cancel: BF16
  CUTLASS GEMM +38.6 ms and NCCL all-reduce +12.9 ms; mHC -21.3, MLA prefill -26.1 and torch elementwise -34.1 ms.
- **Steady-state wall time is at parity or better.** Fresh 16K: R38 5.37 / 5.25 / 5.26 s (profiled 5.27); beta
  6.32 / 5.22 / 5.25 s (profiled 5.29). Later fresh requests on the serving beta: 8K 2.86 / 2.71 s, 32K 10.85 / 10.86 s,
  64K 21.50 / 21.02 s. These match or beat the R38 grid scouts (2.94, 10.89 and 21.87 s).
- **A one-time first-use cost remains.** Beta's first long prefill after boot cost about 1.1 s extra (first 16K 6.32 s
  against 5.22 s after), while R38's was about 0.1 s. Smaller first-at-size costs followed (8K +0.15 s, 64K +0.48 s,
  32K none). Nothing was logged during it: both images' Triton JIT-monitor warnings finish earlier, during the first
  completion and the 4K warmup. So this is untracked first-use preparation, not an inference-time Triton compile.

Conclusion: the grid's 3 to 4 percent short-context prefill deficit is not a steady-state kernel regression. The grid
takes one prefill scout per context on a fresh boot, so it absorbs first-use preparation and boot-to-boot variance
(beta itself moved 1 to 2 percent between two boots). What remains attributable to beta is the larger one-time
preparation on the first long prefill after boot (about 1 s). Its source is not identified. Pinning it would need a
profile of the first long prefill on a fresh boot (the profiler flag is already set in the serving boot, but that
boot's first-use has already passed).

## Production boot without profiler (2026-09-30 12:04 EDT)

At the user's request, the diagnostic `--profiler-config` was removed from production. The profiled boot was stopped
worker first (logs archived as `~/logs/glm53-flash-nvfp4-karmic-beta-20260929-tp4-profiler-*`) and kept as
`...-tp4-profiler-20260930`. `run-glm-tp4-node.sh` was restarted without `PROFILER_DIR`: GLM was down from 12:00 to
12:04. On all four ranks: image `500ae05b`, no `--profiler-config`, Max/top_p 0.95 defaults, `LL,Simple`. The first
completion was correct. Lesson: diagnostic flags come off production at the end of the same window, never at "the next
planned restart".

## QAD checkpoint 175ae8ce (2026-09-30, user-approved)

The publisher replaced HF `main` with a quantization-aware distilled checkpoint on 2026-09-16 (`175ae8ce`; 40 of 47
weight shards differ; `generation_config.json` now carries temperature 1.0 and top_p 0.95). The user downloaded it on
sparky. It was pushed to buddy, rocky and lucky over 10.11.11.x (190 GB, 13:17 to 13:23, one rsync per peer, existing
files kept). On every node the snapshot matches sparky name for name and size for size, and it passes the runner's
model checks (index total 198042331512, 44 shards, `Glm5NextForConditionalGeneration`, 1,048,576). The
`run-glm-tp4-node.sh` default `MODEL_REVISION` is now `175ae8ce`. `execute-glm-qad.sh` stopped the old-weights boot
worker first and kept it as `...-tp4-46aaae8a-20260930`. It then booted the new weights with the plain production
runner (no diagnostic flags; every rank checked for `MODEL_REVISION` and no profiler) and ran `qualify-glm.sh`.
Campaign: `runs/glm-5.3-flash/nvfp4/2026-09-qad-175ae8ce-qualification/`. Receipts: `../qualification/glm-qad-175ae8ce/`.

Correctness, all passed on the new weights:
- Ready at 13:32:57. Short pool 7/7. Semantic x3 including the exact tool format, so no diagnostic was needed.
  Concurrency passed.
- Prefix matrix: long 43.9 / 49.6 s and 43.8 / 48.6 s (old weights 43.2 / 49.3 and 42.9 / 47.9).
- Long triple: 262K cold 90.8 s; 1M extension 384.8 s (old 89.8 and 380.5). The 262K first stage now fills its
  16-token budget (`length`) where the old weights stopped after 6 tokens, so the budget arm now exercises truncation.
- Native-context needles 2048, 2049, 262000 and 1048000 exact.

The first driver was stopped by the tool's background time limit 12 minutes into the grid, before results were saved.
Its partial log is `benchmark-killed-at-time-limit.log`. The grid was rerun detached with `qualify-glm.sh --from-native`,
after moving the passing first native receipt to `native-context-glm-first-run.jsonl`. Native passed again. The grid ran
14:00 to 14:13 with 15 valid cells (`75f9475f...`), and the post-grid completion and final container checks passed.

Against the old weights on the same image, defaults and slots (the 2026-09-30 10:11 beta matched grid, `acd2ad53...`),
compared in `old-vs-qad.json` with the unchanged compare logic minus its identity keys (the checkpoint is the variable;
checkpoint policy verified aligned on all ranks from container receipts):

| New vs old weights | c1 | c2 | c4 |
| --- | ---: | ---: | ---: |
| Output tok/s | -0.6% (54.23 to 53.89) | +0.4% (82.51 to 82.81) | +0.5% (127.36 to 128.03) |
| Engine steps/s | +0.4% | -0.2% | +0.1% |
| Effective acceptance | -1.1% | +0.5% | +0.5% |

Prefill scouts 8K to 128K: +2.2, +1.3, +2.2, +1.2 and +1.4 percent, within the observed boot-to-boot range.

Reading: the QAD checkpoint serves at performance parity with the previous weights. Health: no Xid or OOM;
`NV_ERR_NO_MEMORY` warnings only during the 13:32 boot. MemAvailable 5.8 / 6.4 / 6.4 / 8.9 GB. GLM is serving `175ae8ce`.
Rollback to the old weights: `podman start` the retained `-46aaae8a-20260930` containers, workers then sparky, after
stopping the current boot. This window measured correctness and speed only; it measured no quality improvement from
the distillation.
