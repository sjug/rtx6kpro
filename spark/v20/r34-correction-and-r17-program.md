# Program: corrected r34 + II r17 unified qualification (2026-08-18)

Verified inputs:
- Upstream vLLM PR #45224 = commit 10c75477b07c (merged 2026-07-23):
  "shm_broadcast: bound idle reader waits and release read slots".
  Files: vllm/distributed/device_communicators/shm_broadcast.py,
  tests/distributed/test_shm_broadcast.py.
- GG r34 vLLM base e2666d9a65f4 LACKS #45224 (0 recheck markers, no
  finally-release). r34p (tree 9eb28db0) therefore ran busy_loop_s=0.002
  on an unprotected queue.
- II r15 base ad848fc4 CONTAINS #45224 (4 markers, finally in read
  path); r17 base d6e0bb7c is later on the same dev branch (verify by
  markers + presence of the 3 tests at compose, never by assumption).
- r17 locks: vLLM d6e0bb7c -> tree c53cc73, patch sha 03824044...;
  b12x tree c0a44a1 (from image tag; read lock at compose); FlashInfer
  1ac6942 unchanged.

P0 - interim production posture (user decision, program-independent):
rusty/toby serve r34p = 0.002 on the unprotected queue. Options:
(a) hold with an INFERENCE watchdog: a wedged EngineCore can leave
    the API process, /health, and :9101 all alive, so port/process
    checks protect nothing - the watchdog must send a small completion
    request (few tokens) with a hard deadline every ~60 s and alert
    (or restart) the moment one fails or times out
    (evidence for holding: ~7 clean TP2 boots + serving incl.
    idle->burst),
(b) roll back to r33 now (two bare relaunches).
Either way corrected r34 supersedes within the day. r34p (9eb28db0) is
RETIRED as a fallback everywhere once corrected r34 exists; keep the
image only for forensics. Fallback ladder: corrected-r34 -> r33.

## 1. Correct the r34 composition - P1

On dusty ~/r34-spark/vllm at tree 9eb28db0 (base HEAD e2666d9a65f4):
- git remote add vllm-upstream https://github.com/vllm-project/vllm.git
  && git fetch vllm-upstream 10c75477b07c.
- Apply ONLY the shm_broadcast.py side with a PLAIN apply - NO 3-way:
  git show 10c75477b07c -- vllm/.../shm_broadcast.py | git apply --check
  first, then git apply. The exact #45224 patch applies cleanly to tree
  9eb28db0; if --check ever fails, STOP (fail closed) - a 3-way merge
  could silently produce a noncanonical resolution. The fix content:
  recheck interval constant, bounded indefinite reader waits that reread
  the authoritative shm flag, finally-release of read slots on consumer
  raise. Retain busy_loop_s: float = 0.002 afterward (our overlay line).
- Vendor tests/distributed/test_shm_broadcast.py from 10c75477b07c into
  tests/upstream/test_shm_broadcast_45224.py with a sha256 digest pin in
  the build script (r34-flow vendored-test convention). Adapt imports
  minimally; keep the three tests verbatim:
  test_reader_timeout_caps_indefinite_waits,
  test_reader_rechecks_shm_after_idle_wait_timeout_without_notify,
  test_acquire_read_releases_slot_when_reader_raises.
- Regenerate: git add -A; write-tree -> T3; git diff HEAD ->
  integration.patch; sha256; lock update (result.tree=T3, patch sha).
  Overlay inventory stays FOUR runtime source files: shm_broadcast.py
  simply gains the complete #45224 fix, with 10c75477b07c recorded as
  provenance on that file's entry (it would become five files only if
  the upstream test file were incorporated into the composed vLLM tree,
  which it is not - the tests live as a separately vendored,
  digest-pinned gate). DATE_TAG=20260818; image tag auto-derives
  vllm<T3:0:7> (distinct from r34p); cache fingerprint derives from the
  tree (verify it changed). Round-trip verify patch->T3 on a clean
  checkout. bash -n + shellcheck + ruff on any touched python.
Completion: source contains BOTH the queue safety mechanism and the
2 ms default; r34p tree 9eb28db0 not reused anywhere.

## 2. Gate the corrected r34 queue - P1

In the BUILT image (composed source at /opt/... via the image's
installed vllm), not source-marker greps:
- pytest the three vendored #45224 tests (CPU-only, run before the GPU
  ladder).
- Constructor-default assertion via inspect:
  signature(SpinCondition.__init__).parameters["busy_loop_s"].default
  == 0.002, plus an instantiation check of the attribute.
- Keep the existing 5-gate GPU ladder unchanged.
Completion: all three behavioral tests pass in-image + 0.002 asserted.

## 3. Review and compose II r17 - P1

Review BEFORE building (workstation, read-only):
- Diff r15->r17 locks: vLLM PR-head set (r16: DSpark graph contract
  fix, inactive-route + DSML fixes, mapped W4A16 graph contracts, FC2 +
  direct-micro compile contracts; r17: B12X route-pack contract + GLM
  EXL3 qualification; SQG REMOVED at r16), b12x base/PRs, LMCache,
  FlashInfer/InstantTensor/CUTLASS-DSL/CUDA/Torch/NCCL pins (upstream
  base-script diff r15->r17), launcher + compose deltas (GLM r17
  profile: TR3v4-3.5bpw-MTP78 @ DCP1, online K6 trellis-mcg-b6,
  nvfp4_ds_mla KV - x86 reference only, NOT our profile).
- #45224 presence in r17: verified by SOURCE BEHAVIOR and TEST
  EXECUTION (run the three tests against the composed r17 tree), plus
  patch identity or ancestry where determinable - never markers alone.
  Expected present (r15 already has it) -> overlay carries ONLY the
  busy_loop_s 1->0.002 line for the queue.
- RESOLVED 2026-08-18 (read-only tree reconstruction on the
  workstation; b12x tree c0a44a16 + vllm tree c53cc73 both byte-verified):
  (1) #305 trace: SUPERSEDED by a stronger base guarantee -
  _own_deferred_accelerator_tensors (reload/layerwise.py:202, called at
  :273) clones EVERY non-param accelerator tensor entering the deferred
  queue, explicitly because attribute-marked detection (#305's method)
  misses views that lose the borrowed marker. exl3.py additionally clones
  borrowed tensors at both loader sites (:1722, :2423). NO ownership
  overlay needed.
  (2) Scratch contract: the #221 CuTe K6/MCG kernel is SM121-admissible
  (is_b12x accepts (12,0) AND (12,1); no capability gate anywhere in the
  new namespace; old _use_k6_small deleted). Launch binding is per-WEIGHT
  (plan_k6_mcg_small_m: unpaired K6/MCG, fp16, %128 dims, env kill-switch
  honored); dispatch admission accepts_input caps m<=16 = _MAX_ROWS and
  matches the planner's rows<=16 query branch; the query returns the
  kernel's REAL split-K scratch; capture with missing storage RAISES.
  vLLM's dense-path gate _b12x_trellis_k6_supported admits only K6/MCG
  checkpoint weights, and online-K6 weights are always eligible, so every
  weight reaching _b12x_trellis_linear on SM121 binds the launch. The
  user's dangerous case (query scratch + generic dispatch) survives only
  in two fail-LOUD corners: B12X_DISABLE_STANDALONE_K6=1 (document: never
  set on capture configs) and non-%128 dims (query raises ValueError;
  GLM TP4 shapes are all %128). NO planner overlay needed on r17.
  r17-spark overlay FREEZES at: CMake 12.1 (verify at compose) +
  busy_loop_s 0.002 + EXL3 combined patch (pin unmoved at 704aefd).
  P2 HARDENING (user, 2026-08-18): B12X_DISABLE_STANDALONE_K6 is an
  INCOMPATIBLE runtime setting (intentionally breaks planner/dispatch
  agreement; fails during capture). Build gate asserts it is absent
  from image ENV; launcher DRY_RUN gates assert it is absent from the
  rendered serving environment; runner docs mark it forbidden.
- Gate re-derivation (replaces the old planner matrix + graph gate):
  planner matrix: caps {(12,0),(12,1)} x rows {1,4,16,17,128,129} x
  {K6/MCG-with-query, query-absent non-small case}: rows<=16+query ->
  query value; rows 17..128 -> padded on BOTH capabilities (the <=128->1
  branch is dead when the query exists); query raises on non-%128.
  Graph gate: retargeted to the b12x.gemm.trellis_linear namespace, both
  GLM TP4 shapes, M in {4, 16, 17, 32} - asserting the small-M launch is
  BOUND and ENGAGED at M<=16 on SM121 (introspect
  prepared.k6_mcg_small_m_launch + accepts_input) and generic at 17+,
  with capture + replay at each M.
- Historical overlay re-derivation notes (superseded by the above):
  a) CMake 12.1 arch: expect still required (check the CUDA>=13 branch).
  b) B12X scratch planner: check whether r16's mapped-W4A16 graph
     contracts made _b12x_trellis_c_tmp_elements capability-aware
     upstream; DROP our fix if superseded, port if not. Check
     _use_k6_small admission set (SQG removal may have changed it) and
     re-align the pipeline gate's dispatch assert.
  c) EXL3: confirm EXLLAMAV3_COMMIT (704aefd or moved for TR3v4); if
     moved, re-verify the combined guards+contract patch applies and
     the quantize.py key-presence defect still exists at the new pin
     (re-derive patch + sha if needed).
  d) NCCL/RoCE: runner-owned as always; strip any same-host policy the
     r17 image env carries (r10-base convention already does this);
     keep c4 channel pin + LL,Simple + plugin neutralization in the
     runner, not the image.
- New worktree: git worktree add ~/git/bld-ii-r17 -b
  spark/sm121-arm64-ii-r17 9350b50; carry-over checklist = the ii-r15
  file set (this time including the spin+#45224 posture), rename
  r15->r17 throughout, release dir
  patches/releases/infernal-invocation-r17-spark/.
- Compose trees on dusty ~/ii-r17-spark (cp -al from ii-r15), verify
  all three trees + patch shas against the r17 locks byte-exact;
  overlay applied on top -> overlay tree hash recorded.
Completion: reproducible r17 composition; every retained overlay file
justified against the r17 tree; queue has safety + 2 ms.

## 4. Build and gate II r17 - P1

Native aarch64 build on dusty (staging pair idle; one job per node).
Base image: reuse ii-spark base r1 IF upstream base pins unmoved
r15->r17 (check the Kimi lmcache merge did not touch the base recipe);
otherwise rebuild base first.
Gate ladder additions on top of the r15 set: the three #45224 tests +
0.002 assertion; tree/patch-hash verification (existing compose_source
mechanism); runtime package/arch checks; oracle/parity/pipeline/
scratch/graph gates re-derived per step 3(b,c); DS4 + GLM launcher
DRY_RUNs including the r17 GLM profile flags. Markers + timestamped
monitor per standing watcher discipline.
Completion: full static + GPU ladder green before any serving qual.

## 5. Qualify corrected r34 GLM - P1 (first half of ONE GLM window)

Pre-window (no downtime): install py-spy on all 4 GLM nodes and verify
py-spy --version; capture the missing r33 baseline - 60 s :9101 sampler
on all 4 nodes under a decode load (closes the lost-baseline gap; this
pre-teardown snapshot is now a STANDING cutover step); stage nothing -
image already on all 4 nodes? NO - corrected r34 is a NEW image: mesh
fan per the bulk-transfer rule (dusty 2-wide to .1/.2 then .3/.4).
Boot the EXACT failed config: GLM-5.2 EXL3 TR3 3.42bpw (rev
ae68c659...), TP4/DCP2, MTP3, B12X backend, FP8 KV, InstantTensor,
FULL_AND_PIECEWISE, graph=32 ladder, len 262144, gpu_mem 0.86 (runner
defaults + IMAGE= override).
Watcher: ready OR post-init stall >5 min. ON STALL: py-spy dump of
EngineCore + every worker on every node FIRST, full logs second,
teardown LAST.
Pass: engine READY; no repeated broadcast-block warnings (isolated
singles during compile tolerated); serving smoke (correct arithmetic,
served name GLM-5.2-EXL3-TR3-3.42bpw); ONE warm restart also clean.
Then the serving qual numbers vs the r33 record: decode cc1 vs 23.7,
prefill 8k/64k/128k vs 798/793/767, MTP3 acceptance vs 3.20, pool vs
309k band, fp8 banner, plus the 4-node :9101 spin check (expect <=1
hot core/node - the GLM after-picture).
Conclusion discipline (tightened): a CLEAN boot means the corrected
queue eliminates the observed failure in this reproduction and r34
becomes operationally qualified - it does NOT prove the original hang
was definitely the queue race. A REPEAT hang means #45224 did not
resolve the failure; the captured stacks then discriminate between
another queue defect and a stalled GLM worker (#280 EXL3 runtime first,
then #281 InstantTensor, then the scratch overlay) - the persisting
writer warning alone proves nothing, since it can be either cause or
symptom. On repeat hang: GLM returns to r33, investigation moves
off-window, and the queue is not modified again without stack evidence.

## 6. Narrowly confirm corrected r34 DS4 - P2 (staging pair)

Explicitly NOT the full battery. On dusty/kirby:
- one cold TP2 boot, one warm TP2 boot (times recorded);
- serving smoke (both names);
- idle->burst: >=10 min zero traffic, then an 8-request concurrent
  burst - the sleep/wakeup path the race lives in; response times and
  zero stalls gate it;
- one acceptance-normalized bench cell set (0-ctx cc1 + cc4) against
  the r34p band (steps-geo 21.3-21.7);
- 60 s :9101 CPU/thermal window under load (expect the fix's 1-hot-core
  signature preserved).
Then production swap rusty/toby to corrected r34 (graceful, worker
first) ON USER GO; r33 remains loaded + runner default until the swap
is verified, then corrected-r34 becomes the default and r33 the
fallback.

## 7. Qualify r17 on both production workloads - P1

DS4 (staging pair): production TP2 profile boot; correctness probes
(smoke, decode cells, 1x200 race probe, quick strict-tools pass);
idle->burst; the same bench cells as step 6; spin check. Change-driven
additions from the r15->r17 diff: rerun the #214-sensitive decode
comparison (r16 inactive-route follow-ups) and a capture-ladder
boundary probe at cc4 verify multiples (r16 DSpark graph contract fix).
GLM (second half of the SAME window as step 5): boot OUR production
profile (TR3 3.42bpw, TP4/DCP2, MTP3, FP8 KV) on r17 - this exact
combination is qualified nowhere upstream (their receipt is TR3v4 @
DCP1 + NVFP4 KV on x86), so this IS the qualification: readiness, EXL3
load path (projection-mixed loaders vs TR3), MTP3 acceptance, graph
config, no queue warnings/stalls; decode/prefill/pool vs the r33
record; spin check. Follow-up OUT of this program: TR3v4-3.5bpw-MTP78
checkpoint as a candidate upgrade (mesh transfer, own A/B).
Test selection principle: derive extra tests from the r15->r17 source
diff, not by replaying every historical battery.

## 8. Compare r17 vs corrected r34 - P1

Same hardware, runner env, checkpoint revisions, sampling, concurrency,
prompt set, and measurement method (acceptance-normalized steps/s
primary; A/B/A on any delta >2% before believing it - lucky-boot
lesson).
DS4: raw + steps-geo, cc1/2/4 shape, KV pool, cold/warm boot, CPU/GPU
temps. GLM: boot + stability FIRST; then decode + pool under TP4/DCP2;
prefill 8k/64k/128k.
Decision: correctness is a hard gate. r17 correct but materially slower
(>3% steps-geo beyond the boot band) -> corrected r34 stays production,
r17 held. r17 correct and within band -> r17 becomes the preferred
UNIFIED image (one line, both workloads, #45224 native, upstream GLM
receipt) and corrected r34 the fallback.
Afterward: ceremony (r34 corrections, ii-r15, ii-r17, bench merge,
records - user's key), upstream report items (spin finding + GB10
default, TMA autotune guard, retraction), and the GLM hang postmortem
closes with whichever root cause the discriminator proved.

SCHEDULING CORRECTION (user, 2026-08-18): dusty cannot run the
corrected-r34 DS4 confirmation and the r17 build simultaneously (one job
per node; concurrent build contaminates perf/thermal measurements).
Serialized order: (1) corrected-r34 BUILD-OK; (2) narrow DS4
confirmation on dusty/kirby; (3) stop staging containers; (4) reverify
r17 trees on dusty; (5) start the r17 build. The r17 source transfer
and lightweight tree verification MAY overlap with DS4 serving; the
image build may not.

Estimated wall-clock: steps 1-2 ~1.5 h (warm build cache); step 3 ~1 h
review + compose; step 4 ~1.5-2 h; step 6 ~1 h staging (+prod swap);
steps 5+7-GLM one ~2.5-3 h cluster window; step 7-DS4 ~1.5 h; step 8
analysis folded in. Builds serialize on dusty; the GLM window is the
only production downtime beyond the rusty/toby swap.

DECISIONS (user, 2026-08-18):
1. Production swap = CONDITIONAL GO: after corrected-r34 passes every
   build gate AND the narrow DS4 confirmation, distribute (mesh) and
   replace rusty/toby worker-first. Watchdog stays active through the
   restart and initial completion smoke; stand down only after the
   corrected image serves successfully.
2. GLM window: ONE window AFTER the r17 build passes. Order inside the
   window: capture r33 baseline (:9101 under load, py-spy pre-staged) ->
   corrected-r34 GLM (discriminator) -> r17 GLM (our TR3/DCP2 profile) ->
   restore r33 if either candidate fails.
3. Ceremony: deferred until qualification finishes (avoids a second
   signing round if changes emerge).

EXECUTION RECORD 2026-08-18 (steps 1-2, 6, swap):
- Corrected r34 built + ALL gates green: image gilded-gnosis-v20-r34-
  spark-sm121-vllm17b78ef-b12xcd3ce19-fi1ac6942-cu132-20260818 (tree
  17b78ef4, #45224 behavioral tests 4-passed in-image, 0.002 asserted).
  Two build incidents en route, both environmental + fixed structurally:
  (1) OOM from my failure to idle-check before relaunch (build wrapper
  now refuses to start beside serving containers); (2) vendored test's
  module-level `multiprocess` import (guarded; digest repinned f2c2edda).
- Step 6 narrow DS4 confirmation PASS on dusty/kirby: cold 355s/KV 955k,
  warm 217s/KV 1,019k, smokes correct, idle(11min)->burst 8/8 in 12.4s
  (the repaired wakeup path), steps-geo 21.57 in the 21.3-21.7 band.
  Incident: first cold boot SIGTERMed 11s in by a systemd user-manager
  restart - ROOT CAUSE Linger=no; linger now ENABLED on ALL 8 nodes
  (only rusty/toby had it; GLM cluster had been surviving on session
  overlap). dusty nv-monitor died in that restart - needs user attention.
- PRODUCTION SWAP DONE (conditional GO satisfied): rusty/toby now serve
  corrected r34. Boot 347s, KV 938k, smokes correct, decode
  41.9/59.7/90.0 (r34p band: 41.7/59.4/91.3). Watchdog rode through
  (alerts during planned downtime, WATCHDOG-RECOVERED 12:01:28) and is
  STOOD DOWN. r34p (9eb28db0) RETIRED from production - forensics only.
- II r17 BUILT + all gates green 2026-08-18 (image vllme879345-...-20260818;
  lmcache r17 patch legitimately empty - composition == new base).

WARMUP PROTOCOL (finding 4, 2026-08-18): r17 DS4 cold cache recorded 33
post-engine-start CuTe compilations under live probing (16 extra-token
MLA decode variants, 9 dense GEMMs, 3 MLA decode, 3 multi-grid prefill,
1 sink merge, 1 sparse indexer; inventory captured in
spark/ii-r15/r17-ds4-jit-miss-inventory.log). Measurement rule for ALL
A/B probes from now on: after boot, run a SEEDING SWEEP (decode-cells
cc1/2/4 + one strict-tools pass + one 16k prefill) to populate the JIT
buckets, snapshot `podman logs | grep -c "reason=post-engine-start"`,
run the measured probe, snapshot again - REQUIRE zero new misses during
the measured window or discard the measurement. Longer-term: extend the
image warmup to cover the extra-token MLA variants (respin item).

WARMUP CLOSURE (user decision, 2026-08-18): over-scoped; descoped to an
INFORMATIONAL P3 (user final scoping 2026-08-18: P1 launcher corrections
required; benchmarks valid, no rerun; two first-use JIT spikes deferred;
no warmup overlay, no cache investigation, no extra respin). Facts established: b12x disk cache WORKS (124
artifacts persist per fingerprint; most "misses" were first-encounter
compiles or misread cache hits); exactly TWO kernels are non-disk-
cacheable and recompile once per boot on first use
(SparseNSAFusedIndexerKernel deep-context bucket;
UnifiedDecodeKernel.call_extra_pertok spec-verify variant). P3 (informational, deferred): after a
fresh boot, the first deep-context and first spec-extra-token requests
may see a one-time compile spike. NO warmup code; NO benchmark repeats
(the harness warms each cell and the qual battery pre-seeded the r17
grid). Seed sweep stays MANDATORY before production qualification only.
r17 respin = launcher-only (four GG EXL3 exports removed from
serve-gilded-gnosis.sh + fail-closed runner/launcher gates), same
kernel-cache fingerprint, tag suffix -20260818b.

GLM WINDOW EXECUTED 2026-08-18/19 (record):
- Stage A r33 baseline: grid 15/15, steps-geo 11.42, prefill 774-838,
  acc 2.71; sampler: 26-28% CPU, 5 hot cores, 88-95C (NCCL-dominated).
- DISCRIMINATOR (corrected-r34, exact r34p failed config): PASSED.
  Boot 440s, post-init 21s (r34p: fatal 11+min hang), smoke correct,
  watchdog corroborated. The #45224 queue fix eliminates the observed
  failure in this reproduction; corrected-r34 operationally qualified.
- corrected-r34 battery: 15/15 cells, steps-geo 11.48 (+0.5% vs r33 =
  parity, no A/B/A needed), prefill 779-829 (band), acc 2.73, KV pool
  310,528 (band), bracket CLEAN, warm restart 265s + smoke. Thermal:
  ~24% CPU / 5 hot cores - GLM CPU burn is NCCL-polling-dominated; the
  spin fix's DS4 win does not transfer (prediction corrected).
- r17 leg (II runner, first GLM-on-II serving since r10): FAILED.
  (a) First boot: II request_memory free-at-start check (known r10
  issue) -> relaunched with the r10-qualified GPU_MEM=0.84 +
  KV_CACHE_MEMORY_BYTES pin (runner gained --kv-cache-memory-bytes
  forwarding). (b) Booted 1109s cold, smoke OK, KV pool 353,152 (best),
  BUT: prefill 128-553 tok/s = THE R10 PREFILL PATHOLOGY REPRODUCED
  (unfixed by r15-r17 or GG-tuning removal), decode degraded (5.2
  steps 0ctx) then ENGINE FATAL mid-grid: worker response exceeded the
  executor timeout during 64k prefill; #45224's bounded read turned it
  into a clean fatal with dump (the fix working as designed - the
  disease is II GLM prefill, the queue was the messenger). 3/15 cells.
  Logs: spark/v20/glm-r17-window-logs/.
- r33 RESTORED per window rule (verification pending at record time).

VERDICT: unified-image recommendation = CORRECTED-R34 for both
production workloads (DS4: parity + protected queue + spin fix; GLM:
full r33 parity + boot-path fix). r17: DS4-qualified but GLM-failed -
not a unified candidate. r18 (admitted, gates green) becomes the II
line's next GLM attempt AND IS GENUINELY PROMISING: its #432 "Preserve
GLM sparse attention semantics during prefill" + #434 metadata-kernel
work target exactly the observed pathology. r18 DS4+GLM staging qual =
next program step; GLM cutover to corrected-r34 = user decision.

R18 STAGING QUALIFICATION EXECUTED 2026-08-19 (record):

DS4 leg (dusty/kirby, full six-stage chain): PASS.
- Cold boot 293s; KV pool 1,054,461 tokens (~8.5% below r17's
  1,152k on the identical profile - upstream memory-accounting drift,
  recorded as informational).
- Smoke correct; race probe 0/200; strict-tools 12/12 + 32/32
  conformant, long-ctx needle EXACT at 115,541 prompt tokens.
- Decode cells 40.0/58.9/89.7 (cc1/2/4); idle 660s -> burst 8/8 in
  6.3s.
- Grid steps-geo 21.38 - INSIDE the r34c band (21.3-21.7), above
  r17's 21.14. Spin signature preserved (cpu-avg 7.7%, max 2 hot
  cores). Bench JSON: benchmark_results-ds4-0731-iir18-b12x-a8_
  20260819_090642.json. Staging containers stopped post-qual.

GLM leg (window #2, sparky cluster, r10-qualified profile
GPU_MEM=0.84 + KV pin 9663676416): HARD FAIL.
- Boot 552s cold (vs r17 1109s), image asserted, smoke correct, KV
  pool 353,152 (same as r17, best of the three lines).
- SCOUT FALSE-PASS (measurement defect, retracted): the early
  16k prefill verdict probe read 957 tok/s on its second pass and I
  declared the pathology fixed. Both scout passes used the IDENTICAL
  'x '*16000 prompt; the second pass hit the prefix cache (the fatal
  dump shows the signature: queries=64478, hits=64256) and measured
  cache reuse, not prefill. The first pass's 185 tok/s - which I
  dismissed as first-touch JIT - was the true rate. RULE: verdict
  probes MUST use fresh, non-cacheable prompts (vary the token
  stream per pass); a reading better than the healthy band on a line
  that has never been healthy is a measurement bug until proven
  otherwise (same lesson as the lucky-boot A/B/A rule).
- TRUTH (harness-integrated scouts, server-validated cached_tokens=0):
  prefill 288/177/195 tok/s at 8k/16k/32k = THE R10/R17 PATHOLOGY,
  UNFIXED. #432/#434 did not cure it on SM121.
- Decode catastrophic, worse than r17: cc1 aggregate 1.61/0.34/1.20
  tok/s at 0/16k/32k, ITL 437-693ms, TTFT 30.6s at 0-ctx (r17: ~5.2
  at 0-ctx). Hypothesis only: the pathology lives in small-M/chunked
  sparse-attention processing, so it poisons MTP verify steps as well
  as prefill chunks - unproven, needs upstream profiling.
- ENGINE FATAL at the 64k row, same terminal signature as r17:
  worker sample_tokens RPC exceeded the bounded executor dequeue
  (#45224 clean-fatal working as designed). Full logs from all 4
  nodes: spark/v20/glm-r18-window-logs/fatal-*.log. 9/15 cells ran
  (all sick), 64k/128k rows no-op'd post-fatal. No shm broadcast-block
  warnings pre-fatal (an earlier count of 7 was a grep matching the
  death traceback's shm_broadcast.py frames - noise).
- r33 restored per window rule; watchdog corroborated the whole
  window (down at teardown 09:31, recovered on r18 ready 09:42,
  saturation-then-death alerts from 09:50, restore in progress at
  record time).

THREE-WAY VERDICT (final for this program): corrected-r34 is the
unified production recommendation for BOTH workloads - unchanged.
r17 and r18 are DS4-qualified only. Within the II line r18 is the
preferred DS4 candidate (#45224 native, clean image-series break,
faster GLM boot, all gates green). The II GLM prefill pathology now
has a three-release track record (r10, r15/r17, r18) with receipts
on identical hardware where GG runs 774-838 tok/s: this is
upstream-report material, not a config problem on our side. GLM
production cutover to corrected-r34 remains a user decision.

## Release-source identities (2026-08-21)

- Corrected GG r34: `a565fa90df2bb0ac4480233cf7cd909001f8e32d`
- Infernal Invocation r17: `c0d8b47fad1c3b4c50ba4bc74654eb3dd64f6646`
- Infernal Invocation r18: `5ab7ab4126b38be9c50522b4485e068d8f14f45d`
- Infernal Invocation r18p: `520899278a984e37a2f1d873b1b2042cf5b5b7c7`
- Acceptance-normalized benchmark tool:
  `88acca8b8e5a5f13b24c494d56f35ea2f0b4375f`
