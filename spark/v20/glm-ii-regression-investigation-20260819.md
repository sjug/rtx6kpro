# II-line GLM regression investigation (opened 2026-08-19)

User directive: three II releases (r10, r15/r17, r18) carry the same
GLM-5.2 regression on receipts; upstream is not fixing it (#432/#434
did not); root cause is on us.

## The two observed symptom groups

A. PREFILL (constant since r10): 130-310 tok/s vs GG's 774-838 on the
   same nodes, same checkpoint (EXL3-TR3-3.42bpw), same profile.
   - r10: 169-299 (8k ladder)   - r17: 128-553   - r18: 177-307
B. DECODE (progressive within the II line):
   - r10: 23.1 tok/s cc1 = r33 PARITY (23.7)
   - r17: ~5.2 steps at 0-ctx (~30% down)
   - r18: 0.3-1.6 tok/s cc1, ITL 0.4-0.7s, TTFT 30.6s at 0-ctx,
     windows oscillating healthy(17-19)/dead(0-2) = stall-shaped.
   Both lines end the 64k row with the same clean fatal
   (sample_tokens RPC > bounded dequeue; #45224 as designed).

## Evidence chain (2026-08-19, all zero-downtime)

1. Dense-MHA prefill warning is a RED HERRING for the perf delta.
   r18 warns "Sparse MLA layer has no dense-MHA prefill path; using
   the top-k MQA path only" (mla_attention.py:843, gate =
   impl.supports_dense_mha_prefill AND layer flag). But BOTH lines
   hard-code the same choice (GG b12x_mla_sparse.py:
   "supports_mha_prefill = False", same 'dead dense-MHA routing
   check' comment). GG runs the SAME top-k MQA prefill route at
   ~800 tok/s. The II slowness is inside execution, not routing.
2. index_topk_pattern / use_index_cache honored on BOTH lines
   (identical "skip sparse MLA indexer computation on layer N" boot
   lines, r18 deepseek_v2.py:1335 vs GG :1311). Eliminated.
3. Chunk budget identical (max_num_batched_tokens 3072 both). Elim.
4. B12X env parity total: runners + in-image launchers export the
   same CKV-gather/query-split/workspace set; CKV gather code and
   defaults byte-similar in both trees. Eliminated.
5. RAW EXL3 dense kernels IDENTICAL across images (M-ladder
   microbench on dusty, bench_exl3_gemm_mladder.py, GLM shapes
   6144x1024 + 512x6144, K3+K4, M=1..3072):
   - fused-trellis M=3072: II 3.394ms vs GG 3.426ms (same).
   - dequant+hgemm M=3072: II 0.815 vs GG 0.898 (same).
   => cu133/torch2.13 toolchain exonerated AT THE KERNEL LEVEL for
   dense GEMMs. Note for later: fused-trellis is 3-4x slower than
   dequant+hgemm at M>=256 ON BOTH - the strategy crossover matters.
6. The four GG EXL3 tuning vars are a MIRAGE at r17/r18: code
   defaults == GG production values exactly (MAX_M 32, BLOCK_M 8,
   PREFILL_TRELLIS 1, CHUNK 128; identical consumption sites both
   trees). GG setting them is a no-op; II omitting them lands on the
   same policy. (r10's "envs 169 vs defaults 253" ladder was r10-era
   code; not the r17/r18 story.)
7. vllm-side exl3.py effectively identical (486-line diff = BTX
   weight adoption + pre-bound launch states, both LOAD-TIME; the
   qualified GLM-5.2 prefill route resolver, allowlist
   (128,128,32,512) -> block 32, hidden 6144 / intermediate 512 /
   topk 8 / K3-K4 tier signature, is byte-identical).
8. PREFILL RATE IS CONTEXT-FLAT: 85/90 active samples in the
   250-320 band, median 307.2 (= one 3072 chunk / ~10s; GG: ~3.8s),
   from 16k through the 128k prefill. Context-independent cost
   => NOT attention/indexer-over-KV; the per-token WEIGHT PATH.
   (The alternating 307.2/0.0 logger pattern = chunks taking >10s.)

## Round-1 suspect (later exonerated)

The b12x mixed-Trellis fused-MoE PREFILL engine - the routed-experts
kernels that carry most of GLM's prefill FLOPs:
  GG: b12x cd3ce19, run_mixed_trellis / trellis3_t256_proj
  II: b12x 75787c7 (BTX generation), bind_mixed_trellis +
      run_bound_mixed_trellis / trellis_t256_proj, source_format=btx
Same vllm orchestration above; the engine itself diverged. A ~2.6-3x
gap in that engine alone reproduces the observed 3.8s-vs-10s+ chunk.
Sparse-MQA prefill kernels also differ by b12x rev but the
context-flatness argues against attention dominance.

## Round 2 results (2026-08-19 afternoon, all no-window)

9. MIXED-TRELLIS MoE ENGINE EXONERATED AT KERNEL LEVEL AT THE TIME,
   LATER INVALIDATED BY ROUND 8: ran each
   line's own upstream benchmark (benchmark_mixed_trellis.py; II
   copy patched to the bound API - upstream's own benchmark is stale
   and still imports the removed run_mixed_trellis) at GLM
   production geometry (H6144/I512/topk8, 192xK3+64xK4, tile
   (128,128,32,512), graph-captured):
     m=1: GG 161.9us / II 152.9us ... m=3072: GG 32.15ms / II
     32.21ms. Identical at every rung, decode and prefill M.
   Note: benchmark methodology was byte-identical between revs except
   the trellis3_t256->trellis_t256 layout rename, but the instrument was
   not production-faithful. It materializes only 16 experts per tier,
   pads to the 192/64 compiled populations, fixes each token at six K3
   plus two K4 routes, and times graph replay. Round 8 shows that its
   parity result was a false negative and cannot exonerate the engine.
10. The first NCCL check established only that no GLM runner, including
    r10's, carried the exact-four channel pin. Treating that absence plus
    r10's healthy decode as rejection was invalid: r10 prefill was already
    slow, the DCP collective mix changed in later releases, and the planned
    TP4 GLM A/B had never run. The source-faithful test in item 18 resolves
    this theory causally.
11. vllm sparse orchestration exonerated by inspection:
   b12x_mla_sparse.py GG-vs-II diff is 49 cosmetic lines (CKV pool
   ownership refactor, flag renames, asserts). mla env/config
   parity re-confirmed.
12. CUDA-graph coverage parity at boot: both lines capture
   PIECEWISE 7 + FULL 8 for target AND speculator (prefill +
   decode FULL). The DS4-style "fell off the capture ladder"
   signature is not visible at boot; runtime dispatch remains
   unverified (cudagraph_stats disabled in production).
13. Upstream receipt blind spot (P4 answered): upstream's GLM-on-II
   quals (r11 6f92a9e, r17 9350b50) gate DECODE ONLY (r17: 20s C1
   gate, 94.06 tok/s, 33.42 steps/s, x86) + boot/coherence. No
   upstream prefill number exists on any hardware. Their compose
   ships VLLM_EXL3_PREFILL_CAPACITY=2048. The prefill disease may
   be II-line-wide and simply unmeasured upstream => strong
   upstream-report item regardless of root cause.
14. r18 prefill runs the transient full-CKV gather as intended
   ("capacity=524288 logical tokens" boot line).

15. Sparse-attention kernels EXONERATED (round 3): paged-MSA bench
   (byte-identical file both revs, run from each package root, fp8
   KV, contexts 32k/131k, batches 1/16): II == GG at every cell
   (b1 ~25.6us, b16 ~168-217us, dense reference equal). The
   nsa-indexer bench needs a GLM-5.1 config mount neither image
   has - skipped on both, equally.

## Round 4: production geometry and DCP transport (2026-08-19)

16. The checkpoint headers establish the dominant routed-expert tier
    geometry. Layer 3 is K3/K4=206/50 for gate, up, and down; the
    remaining routed layers are 148/108 for all three projections.
    No projection-mixed expert appears in this checkpoint. An exact
    148/108, H=6144, I=512, top-k=8 benchmark found II slower only
    on the large-M mixed-Trellis launch (M=3072: 23.75ms vs 19.48ms;
    M=222: 7.00ms vs 5.89ms). Decode M<=32 is parity or slightly
    faster and outputs are numerically identical. Round 8 establishes
    that this is a general paired-FC2 regression rather than a
    148/108-specific effect. Across 75 routed layers the M=3072 delta
    is ~0.32s. Before the NCCL mitigation that could not explain the
    ~6s missing from each production prefill chunk; after exact-four
    recovered transport progress, it matches the residual 7.9-9.8%
    serving-prefill gap.
17. The production-shaped sparse-attention/indexer path is also equal.
    At q=3072, context=16384, top-k=2048, FP8 paged KV, GG totals
    12.04ms (3.74ms indexer + 8.34ms MLA) and II totals 11.88ms
    (3.67ms + 8.16ms). The benchmark uses the production paged-cache
    view; the upstream benchmark's token-flat view was corrected only
    in the harness.
18. The exact DCP2 overlap reproduces the failure and identifies its
    cause. Two earlier harnesses were invalid and are not evidence: the
    first used ProcessGroupNCCL instead of production PYNCCL; the second
    modeled an LSE all-gather/output reduce-scatter pair that this path
    does not execute and omitted `/dev/infiniband`. Source tracing closed
    the actual contract:

    - `_dcp_gather_ckv` launches the full-CKV all-gather on the dedicated
      prefetch communicator and side stream.
    - `_merge_b12x_dcp_topk` concurrently all-gathers the sparse-indexer
      candidate tensor (`q x 2 x 2048`, int32) on the main communicator.

    The corrected weight-free harness uses two real
    `PyNcclCommunicator` objects, the exact image-owned library from
    `VLLM_NCCL_SO_PATH`, `/dev/infiniband`, and the two RoCE f1 pair
    interfaces on idle dusty/kirby. At context 32768, q=3072, and 75
    layers with II's NCCL 2.31.2:

    - CKV all-gather alone at the default channel count: 6.589ms/layer.
    - Indexer candidate all-gather alone: 28.461ms/layer.
    - Both communicators concurrently at the default: no completion in
      60 seconds; both ranks time out.
    - Both concurrently with `NCCL_MIN_NCHANNELS=4` and
      `NCCL_MAX_NCHANNELS=4`: 3.136ms/layer, 235ms total.
    - `NCCL_MAX_NCHANNELS=4` alone also passes at 3.143ms/layer, proving
      that the cap is the mechanically necessary term.

    A bounded maximum-channel sweep on that same case found 1=6.242,
    2=3.505, 4=3.143, 8=4.109, 16=3.523, and 32=3.928ms/layer;
    automatic 64 channels hangs. Four is the measured optimum, not an
    arbitrary escape value. The full production-shape matrix then found:

    - corrected-r34/NCCL 2.30.4/default at (8K,3072), (32K,3072), and
      (64256,222): 3.304, 3.562, and 1.604ms/layer.
    - II-r18/NCCL 2.31.2/exact-four: 2.770, 3.223, and 1.342ms/layer.

    NCCL debug from the failed four-node serving run independently shows
    NCCL 2.31.2 creating 64 collective channels per communicator and
    issuing the large all-gathers across channels 0..63. With two
    simultaneous communicators, GLM exposes 128 channels to the same
    links. The PYNCCL wrapper uses `ncclCommInitRank`, not
    `ncclCommInitRankConfig`, so it has no communicator-local channel
    control. A drop-in attempt to load corrected-r34's NCCL 2.30.4 in the
    II image failed communicator initialization with `invalid argument`;
    library rollback therefore requires a rebuilt, internally aligned
    image and is not the narrow fix.

    This is also consistent with the r10 record: its NCCL 2.31.2 review
    already traced Turin's "prefer more channels" change and selected
    exact-four as the best two-rank RoCE configuration. The II DS4 runner
    carried that cap; the GLM TP4 runner did not, because its planned A/B
    was deferred. The r10 prefill regression and r17/r18 fatal behavior
    can therefore be one workload-amplified NCCL defect rather than two
    independent diseases. Four-node serving with the cap remains the
    final end-to-end confirmation of that inference.

## Executed from the original plan

P1 RUN (mixed-Trellis MoE bench): approximately equal under the
upstream geometry; exact checkpoint tier geometry found a 19-22%
large-M delta that is too small to explain serving - see items 9 and 16.
P2 OBSOLETED for kernels (r18 kernels == GG kernels, so earlier II
   generations need no kernel bisect); the final transport reproduction
   also obsoletes the proposed vLLM engine-layer bisect.
P3 RULED OUT as TP2: the TR3 checkpoint is TP4-RANK-SLICED
   (checkpoint TP must equal runtime TP; the loader raises). No
   off-cluster GLM serving repro exists on this hardware.
P4 ANSWERED: upstream gates decode only - see round 2, item 13.

## Window #3 intermediate attribution, superseded

The 2026-08-19 instrumented run produced valid observations but the
initial shared-memory-queue causal attribution was wrong. It is kept
here to make the correction explicit.

Boot: r18 II runner, 0.84 + KV pin; sparky head instrumented
(NCCL_DEBUG=INFO,SUBSYS=INIT,TUNING via one-off runner variant,
removed after the window). Ready 354s, smoke OK. NCCL init shows
64 channels for the 4-rank communicator (Turin over-channeling
visible in production config; item 18 cleared it in isolation).

The apparent finding was rank-0 worker starvation on the shm input
queue. Three instruments observed:
- py-spy dumps (every snapshot, decode AND prefill): sparky rank-0
  Worker MainThread asleep in worker_busy_loop -> dequeue ->
  acquire_read -> SpinCondition.wait -> zmq poll
  (shm_broadcast.py:206/794/889), while EngineCore MainThread is
  simultaneously asleep in step_with_batch_queue ->
  _wait_for_response -> get_response -> dequeue -> the same
  acquire_read path. Meanwhile buddy (remote rank, zmq transport)
  was mid-model-forward at the same instant.
- GPU utilization: sparky 0% for 38 of 43 sampled seconds DURING
  PREFILL (decode window ~96%, but GB10 counts resident NCCL-wait
  kernels as busy).
- py-spy record (40s, prefill): only 5.2s of model-forward Python
  executed = ~13% duty cycle on the rank feeding the local GPU.
  The initial interpretation assigned the missing time to the queue;
  the later causal test shows this was wait time induced by another
  rank's incomplete collective.

Those stacks show where rank 0 and EngineCore wait, not why the remote
rank has not responded. Once the source-faithful concurrent-collective
test reproduced the hang, the stacks became consistent with the normal
protocol: the local worker has completed and waits for its next command;
EngineCore waits for every worker response; a remote worker is still
inside model forward because its NCCL collective has not completed.
The bounded shared-memory dequeue and #45224 then report the missing
response as designed. They are the messenger, not the cause.

Engine fatal #3 occurred DURING the probe (12:03:29, same
sample_tokens bounded-dequeue signature, triggered by a few
sequential fresh 16k prefills) - logs from all 4 nodes preserved in
glm-w3-probe/fatal-*.log (sparky log includes NCCL_DEBUG init).

CUTOVER EXECUTED (user GO): after the probe, GLM production moved
to CORRECTED-R34 (per-node graceful teardown of dead r18, workers
first; boot corrected-r34 workers-then-head). Ready 193s (warm),
image asserted, smoke correct, KV pool 339,840 (above the 310,528
window #1 reading - favorable boot variance), watchdog recovered
12:15:29. GG runner default IMAGE flipped to corrected-r34 and
deployed to all 4 nodes; r33 image stays loaded = instant rollback.

## Fix and remaining validation

The narrow fix is a runner-level exact-four channel contract:
`NCCL_MIN_NCHANNELS=4` and `NCCL_MAX_NCHANNELS=4`. The image continues
to own its NCCL library path and cache fingerprint; channel count is a
topology/workload property and belongs in the four-node GLM runner.
The build recipe must fail closed unless the runner renders both values,
while continuing to reject `LD_PRELOAD` and `VLLM_NCCL_SO_PATH`
overrides. No image rebuild is required to test or deploy this runner
change.

The only required live validation, on user GO, is one bounded r18
four-node window with the exact-four runner:

1. Run fresh 16K/32K prefill and the former 64K cached-tail fatal
   reproducer.
2. Run cc1/cc4 decode cells; if healthy, run the normal benchmark grid.
3. Restore corrected-r34 unless the user separately approves an r18
   cutover.

Profiler work is now fallback-only. If exact-four does not close both
serving symptoms, retain the cap because it independently fixes the
reproduced transport defect, then capture the residual rather than
resuming a broad engine bisect.

## Assets

- bench_exl3_gemm_mladder.py + exl3-gg.py / exl3-ii18.py /
  exl3-gg-vs-ii18.diff in the session scratchpad; mladder script
  staged at dusty:~/exl3-mladder.py.
- Window logs: glm-r18-window-logs/ (fatal-*.log), glm-r17-window-logs/.
- r10 baseline findings: spark/ii-r10/ii-r10-spark-qualification-20260814.md
  ("GLM-on-II attempt" section: the original ladder + hardware control).

## Window #4 EXECUTED 2026-08-19 ~14:00-14:20 (bounded confirmation, PASS)

Pinned runner (exact-four channels) on the four-node cluster, user-
specified scope, all fresh randomized prompts:
- Boot 223s (fastest r18 boot; 552s window-2, 354s window-3).
- Prefill: 739 tok/s @34,943 tokens; 747 @31,799; 731 @63,641 -
  GG-band rates at every size including the formerly fatal one
  (disease band was 177-307 with fatal at 64k).
- 64K cached-tail reproducer (pass-2 on the same prompt, 63,641
  cached): SURVIVED, 42 tokens in 3.4s. This exact shape killed
  windows 2, 3-grid, and 3-probe.
- Decode quick-probes: cc1 18.1 tok/s, cc4 42.4 aggregate (~11x
  recovery from 1.6; formal band parity deferred to the acceptance-
  normalized battery - these probes use temp-0.7 random-word prompts
  that depress MTP acceptance).
- Zero fatal markers; all four ranks up through the battery.
Production restored to corrected-r34 via the flipped runner DEFAULT
(no IMAGE override): ready 254s, image asserted, smoke, watchdog
recovered 14:19:41.
SCOPE (per review): window #4 proves RECOVERY from the fatal
regression and plausible performance - it supports re-entering r18
into qualification, not selecting it as the unified image; that
requires the acceptance-normalized battery and A/B/A.

## P1 six-arm matrix (dusty/kirby f1 rails, user's reproducer)

Harness notes: containers need --ipc host (NCCL /dev/shm segments;
production runner always sets it) and env strings must be
newline-free through ssh. GID index 3 works on the f1 rails.
Fork-lib arms (DECISIVE):
- fork-auto: HANG (300s timeout) - defect reproduced.
- fork-auto + NCCL_LAUNCH_ORDER_IMPLICIT=1: PASS,
  3076/3434/1313 us/layer at (8k,3072)/(32k,3072)/(64256,222).
- fork-c4: PASS, 2769/3607/1415 us/layer (matches user's runs).
=> TWO independent mitigations: implicit launch ordering OR the
channel cap. NARROWED MECHANISM CLAIM (user review 2026-08-19): the
matrix establishes an NCCL multi-communicator PROGRESS FAILURE under
automatic resource selection, avoided either by implicit cross-
communicator ordering or by reducing communicator resource usage.
It does NOT distinguish ordering inconsistency from resource
starvation (or a combination); NVIDIA documents both hazards
(concurrent-communicator ordering docs; issue #592 block
exhaustion). The fork's ungated fb6f4099 comparator remains a
hygiene item only. The vLLM/B12X two-communicator overlap
establishes neither launch ordering nor bounded communicator
resources - upstream-report item regardless of lib.
Official-lib arms: first attempt INVALID - my library build lacked
the recipe's "compiled with CUDA 13.3" version stamp (built in the
wrong stage image without CUDA_HOME) and segfaulted even
single-communicator. Rebuild in the pinned NGC builder with the
exact recipe line is in progress; the official-auto verdict (the
repin decision) awaits it.

## P1 verdict: official vs fork (completed 2026-08-19 ~14:47)

Harness corrections first (three artifacts fixed before any verdict):
(1) --ipc host required (NCCL shm segments); (2) newline-free env
strings over ssh; (3) THE IMAGE LD_PRELOADS THE FORK NCCL - any
library-swap test must set LD_PRELOAD to the lib under test or the
process holds two NCCL builds and segfaults at the first collective
(this also retro-explains item 18's 2.30.4 "invalid argument"
drop-in note: that attempt shared the dual-library hazard). The
"official segfaults" results before this fix were artifacts. First
official build was additionally non-conformant (wrong builder
stage); rebuilt in the pinned NGC digest (7531d90) with the exact
recipe line and the CUDA 13.3 version-stamp gate.

Final matrix (conformant lib, coherent loading, user's reproducer,
75 layers, (8k,3072)/(32k,3072)/(64256,222)):
- fork-auto:                HANG (300s)
- fork + implicit order:    PASS 3076/3434/1313 us/layer
- fork + MIN/MAX_NCHANNELS=4: PASS 2769/3607/1415
- official-auto:            HANG (300s)  <- decision datum
- official + implicit order: PASS 3110/3418/1314
- official + MIN/MAX_CTAS=4: PASS 2785/3644/1444
Official 2.31.2 selects 40+ channels on the 2-rank pair by itself
(NCCL_DEBUG receipts) and deadlocks the un-ordered dual-communicator
overlap exactly like the fork. The fork's ungated fb6f4099
comparator is EXONERATED as the cause of this defect (it remains a
hygiene item - un-gated x86 policy in ARM images - for the next
base respin, not a fix).

DECISION (user's branch 3, narrowed per review): no repin required
or helpful for this defect; 2.30.4 rollback dead. Exact-four is the
PROVEN-SAFE production mitigation (deployed, window #4), and was
optimal in the earlier bounded single-case sweep - but it is NOT
globally optimal: official CTAS=4 vs official implicit-auto was
+10.5% at 8K, -6.6% at 32K, -9.8% at the 64K tail. Implicit
ordering may be the better general contract; that is decided by a
serving A/B (exact-four vs NCCL_LAUNCH_ORDER_IMPLICIT=1, same
acceptance-normalized GLM workload incl. the cached 64K tail),
after which implicit becomes the r18+ correctness contract if
stable and equal-or-faster, with CTAS=4 retained only on a material
serving win.

ISSUE ROUTING (per the user's P2 tree): fixed solely by
ordering/caps on BOTH libraries => primary issue against the
vLLM/II integration that owns these communicator launches (overlap
establishes neither ordering nor resource bounds); the
auto-resource-selection reproducer goes SEPARATELY to NVIDIA NCCL;
the fork comparator is a separate hygiene PR. UPSTREAM-REPORT GATE:
the archived arm logs lack the timeout invocation/exit status, the
selected-channel NCCL lines, the env values, library maps/hashes,
and rank-1 completion output - the three official arms must be
rerun with full receipts (incl. output-correctness validation, since
the harness gathers zero-filled buffers) before filing. Secondary: NVIDIA NCCL perf/tuning report - automatic
channel selection picks 40+ (fork: 64) channels on a 2-rank
RoCE/GB10 pair where measured optimum is 4 (bounded sweep: 1=6.242,
2=3.505, 4=3.143, 8=4.109, 16=3.523, 32=3.928 ms/layer, auto=hang);
weight-free reproducer + channel logs ready. Tertiary:
nccl-canonical PR to gate or revert fb6f4099's un-gated comparator
hunks. The pipeline positive-control (fork source through the same
build path) was built but not run - official reproducing BOTH the
hang and the mitigated passes BEHAVIORALLY CONVERGENT with the
image fork lib (same qualitative outcomes, timings within ~2%; NOT
bitwise-compared, and the harness gathers zero-filled buffers
without output validation) validates the pipeline by convergence.

Assets: spark/v20/bench/nccl-6arm-results/ (all arm logs),
~/nccl-official-7b83616 + ~/nccl-fork-ctl on dusty (builds),
nccl-official-lib.so staged dusty+kirby.

## Receipts-grade official-arm rerun (plan step 2, 2026-08-19 15:00-15:06)

Artifacts: spark/v20/bench/nccl-6arm-results/receipts-20260819/
(per arm: .receipt with exact commands/timeout/wall/exit status,
full NCCL_DEBUG=INFO INIT,GRAPH,ENV,TUNING logs both ranks; the
PASSING arms embed the env receipt + library sha256 + /proc/maps in
their result JSON and print rank-1 completion; the HUNG auto arm
never reaches its final provenance record - its library identity is
established by the recorded command, mount, LD_PRELOAD, and NCCL
debug output, which is sufficient for filing). Harness: receipts VARIANT of the reproducer
(patterned per-rank fills + rank-major output validation; collective
structure unchanged; delta documented in the file header).

- official-auto: HANG, exit 124 at 300s wall. Channel receipt:
  "Channel 63/64" - OFFICIAL v2.31.2-1 auto-selects 64 channels on
  the 2-rank RoCE pair, same as the fork. The fork comparator did
  not even change the channel count on this topology.
- official-implicit: PASS in 11s, validation OK x3 cases, rank-1
  complete - at the SAME CONFIGURED 64-channel width. (Implicit
  ordering may prevent simultaneous residency, so the actual
  CONCURRENT footprint is not proven.) (Datum leans toward ordering
  inconsistency over pure exhaustion, but is not uniquely
  discriminating: implicit ordering may serialize the kernels and
  avoid contention as a side effect. Narrowed mechanism claim
  stands.)
- official-ctas4: PASS in 8s, validation OK x3; channel receipt
  "Channel 03/04" (the CTAS cap bounds channel width on this path).

NVIDIA-report core claim, now receipted: official NCCL v2.31.2-1,
built for SM121 from 7b83616 in the pinned NGC builder, auto-selects
64 channels on a 2-rank dual-rail RoCE GB10 pair and fails to make
progress on a two-communicator all-gather overlap (CKV 656B/token +
indexer qx2x2048 int32) within 300s; NCCL_LAUNCH_ORDER_IMPLICIT=1
completes the identical shape at the same width in 11s with
validated output; MIN/MAX_CTAS=4 also completes.

Remaining plan items: (3) serving A/B exact-four vs implicit on the
acceptance-normalized GLM workload incl. cached 64K tail (one
cluster window, user schedules) -> (4/5) contract decision ->
(6) three separate filings (vLLM-II integration primary, NVIDIA
reproducer, nccl-canonical hygiene PR).

## Window #5 EXECUTED 2026-08-19 21:49-23:00 (serving A/B, plan step 3)

Arm A (exact-four, deployed runner): PASS end-to-end.
- Boot 223s (= window #4, reproducible). Full acceptance-normalized
  grid 15/15 (LABEL glm52-r18-c4-ab): tok/s-geo 31.33 vs r34c 31.31
  and r33 30.92 = FULL DECODE PARITY; heavy rows on the r33 band
  (64k 7.5/11.8/17.7 steps, 128k 7.3/11.6/17.3).
- Prefill scouts 710-760 tok/s across 8k-128k: 7.9-9.8% below
  corrected-r34 per size (8K -7.88, 16K -9.17, 32K -9.81, 64K
  -9.27, 128K -8.86; call it ~9% overall). The remaining gap is
  consistent with, and likely dominated by, the measured
  mixed-Trellis large-M regression (item 16, ~0.32s of
  ~3.8s/chunk) - full causal proof needs dispatch-level profiling
  or an optimized-kernel A/B. Decode: r18 steps/s ~2.7% above
  corrected-r34 with lower acceptance; tok/s parity.
- Cached 64K tail survived (pass1 722 tok/s, pass2 1.7s); zero fatal
  markers; sampler 24.7% CPU / 5 hot cores / GPU 77-84C = the
  healthy NCCL-polling profile.

Arm B (NCCL_LAUNCH_ORDER_IMPLICIT=1 WITH automatic 64-channel
selection - the arm did NOT isolate implicit ordering; both settings
changed together): HARD FAIL in serving, with collateral.
- Boot 423s (64-channel init costs ~200s vs the pin).
- Sequencing per the bench JSON: prefill had already collapsed
  before the formal grid (8K scout 156 tok/s); the first formal
  cell (C1/128K) hit its ten-minute warmup timeout at 0 tok/s,
  C1/context-0 also timed out, and the engine died during the 16K
  path; all 15 cells errored. APIServer answered /v1/models as a
  zombie until ~22:48 then exited (connection refused). Same
  bounded-dequeue fatal signature on sparky (logs:
  glm-w5-armB-incident/).
- COLLATERAL: buddy and rocky hosts became unresponsive - sshd
  unable to complete banner exchange on LAN and mesh for 30+ min.
  Previous-boot journals (archived: glm-w5-armB-incident/
  *-prevboot-journal-full.log.gz + *-prevboot-nvrm.log) show the
  direct evidence: bursts of NVRM NV_ERR_NO_MEMORY (buddy 56, rocky
  59 in the extracts) plus repeated systemd memory-pressure
  notifications, and NO Xids, soft/hard lockups, RCU stalls, or
  hung-task reports. I.e. severe UMA/NVIDIA allocation pressure
  during the auto-64 deployment followed by host unresponsiveness;
  "spin-starvation" remains plausible but unproven, and the bytes
  attributable to NCCL channel resources were not measured. buddy
  self-recovered after ~33 min; rocky did not and both were
  power-cycled by the user.
- LESSON (measurement): /v1/models liveness checks are worthless -
  the zombie APIServer answers them with a dead engine. Only
  completion probes count (the watchdog was right each time).

CONTRACT VERDICT (corrected phrasing per user review): implicit
ordering WITH automatic 64-channel resource selection hard-failed
under full TP4/DCP2 serving; exact-four WITHOUT implicit ordering
passed and is the qualified production contract for THIS GLM
TP4/DCP2 GB10/RoCE deployment (not generalized to DS4 or other
topologies). The arm confounded two variables, so no claim is made
about implicit ordering in isolation; NCCL's implicit ordering
guarantee also requires consistent host-side launch order across
devices, unproven for the asynchronous CKV-prefetch/indexer paths.
The serving counter-example still upgrades the reports: the
two-rank reproducer under-models serving concurrency, and the
resource cap is the setting with serving-grade validation.

## Arm-B incident resolution (2026-08-19 23:43)

buddy self-recovered after ~33 min (uptime preserved; load draining
before cleanup, consistent with torch NCCL watchdog timeouts finally
aborting the spinning comms). rocky regressed to TCP-connect timeout
and did not self-recover; user powered BOTH nodes off. Post-boot
checks on both: GPU enumerates (GB10), stale containers cleared,
linger intact. Production restored to corrected-r34 (runner
default): ready 254s, image asserted, COMPLETION smoke '391',
watchdog independently recovered 23:43:05. Total GLM downtime for
window #5 + incident: ~1h54m (21:49-23:43), of which ~55 min was the
arm-B livelock tail. Host-livelock collateral is now part of the
vLLM-II filing narrative.

## Post-review hardening + posture (2026-08-19/20, user review round 3)

Actions taken:
- Previous-boot journals ARCHIVED from buddy and rocky before
  rotation (glm-w5-armB-incident/: full gzipped journals + NVRM
  extracts; NV_ERR_NO_MEMORY buddy 56 / rocky 59, zero
  Xid/lockup/RCU/hung-task lines in the reviewed interval).
- r18 build gate HARDENED: now rejects NCCL_LAUNCH_ORDER_IMPLICIT
  in the rendered runner in addition to requiring both exact-four
  values (build-ii-r18-spark-sm121.sh). The unqualified arm-B
  runner variant was removed from all four GLM nodes.
- NCHANNELS vars stay AS-IS for this image: deprecated upstream in
  favor of MIN/MAX_CTAS, but these exact variables are what was
  serving-qualified; a silent swap would create a new unqualified
  contract.

Standing posture (user decision):
- corrected-r34 remains GLM and DS4 production.
- Exact-four is mandatory for r18+ on THIS GB10 TP4/DCP2 dual-rail
  deployment only; not generalized.
- r18 is a DEVELOPMENT CANDIDATE, no longer disqualified by the
  NCCL failure, but a consistent ~9% prefill regression with decode
  only at parity REJECTS it as the unified production image unless
  it closes the gap (mixed-Trellis large-M optimization, gated on
  both 148/108 and 192/64, numerical equivalence, M=222/M=3072
  legacy parity, M<=32 decode preserved) or gains a
  compensating capability worth the measured cost.
- Upstream vLLM improvement to file: expose ncclCommInitRankConfig
  so overlapping communicators can carry explicit per-communicator
  resource budgets (maxCTAs) and ordering policy; today's PYNCCL
  uses only ncclCommInitRank, so channel topology can only be set
  process-globally via env. Four stays a deployment-derived value.
- NVIDIA filing scope check: NCCL #1766 is related
  multi-communicator ordering evidence but does not cover automatic
  64-channel selection on GB10 nor the validated CTA cap; #2334 is
  a different RoCE completion-path failure. Our reproducer is not a
  duplicate.

## Round 6: native-stack discrimination and candidate contract (2026-08-20)

The unmodified official NCCL v2.31.2-1 automatic-64 arm was rerun on
dusty/kirby with native stacks captured on both ranks. The #2353
signature is absent: neither application thread is in
ncclLocalOpAppend/uploadProxyOps and neither proxy thread is in
ncclProxyGetPostedOps. The application had completed enqueueing and
was waiting in CUDA synchronization while both communicator proxy
threads remained in progress code. Therefore #2353's proposed
flush-before-block patch was not ported. The evidence bundle is
spark/v20/bench/nccl-stack-signature-20260820/.

The source-correct r18 candidate contract is process-wide
NCCL_LAUNCH_ORDER_IMPLICIT=1 plus the existing exact-four resource
cap. The official 2.31.2 weight-free reproducer passed with validated
outputs using implicit ordering plus MIN/MAX_CTAS=4, and passed again
using the exact deployed MIN/MAX_NCHANNELS=4 form. These are admission
results, not serving qualification; automatic-64 remains unsafe because
implicit ordering does not bound the separate serving UMA/resource
exposure.

Transitional state is deliberate:
- The repo and bld-ii-r18 runner copies carry the candidate contract and
  are byte-identical (md5 6ed963aae114b054ba373aef3347f03a).
- The r18 build gate now requires literal
  NCCL_LAUNCH_ORDER_IMPLICIT=1 together with both exact-four values,
  replacing the earlier reject-implicit assertion.
- All four deployed GLM nodes still carry the old qualified two-setting
  runner (md5 bfbb27ac...) with exact-four and implicit ordering unset.
  Nothing has been deployed or built from the candidate state.
- corrected-r34 remains production. Before the eventual window, deploy
  the candidate runner with a four-node byte assert, monitor UMA/NVRM
  pressure through boot and measurement, use completion probes for
  liveness, seed before the measured grid, and compare against both
  corrected-r34 and window #5 arm A (glm52-r18-c4-ab).

## Round 7: paired-FC2 source discriminator (2026-08-20)

The untouched `b4d6c75^` tree could not serve as the proposed boundary
control: its mixed-Trellis caller and W4A16 kernel body have an ABI mismatch
and its path lacks the later MCG wiring. The compile failure is therefore not
performance evidence. A synthetic source arm was built inside `b4d6c75` by
reversing only the paired-FC2 removal, auditing the selected methods
line-by-line, and checking all non-selected methods for byte identity. The
same two-file source delta applied cleanly to the exact r18 composed B12X tree
`75787c7a7431b3bea414d2ebf5f2b8671b23eb33`.

The boundary and exact-r18 A/Bs used deterministic nonzero inputs, the full
M = {1, 4, 16, 32, 33, 221, 222, 3072} ladder, both 148/108 and 192/64
expert splits, and unmodified/paired/unmodified ordering. All source-arm
outputs were finite and bit-identical to their unmodified references at
every shape.

On the exact r18 tree, M <= 32 stayed within 0.91% in both directions. At
large M, paired FC2 reduced kernel time by 16.4% to 19.5% for 148/108 and by
18.7% to 20.4% for the 192/64 control. The control therefore failed the
predeclared flatness condition and correctly rejected the geometry-specific
premise. At this point the legacy 192/64 baseline was still missing, so the
general historical regression claim remained open for round 8.

The positive finding is narrower and directly useful: paired FC2 is a
general, bit-exact large-M optimization on the r18 source, with the decode
range preserved. It is eligible for a separately reviewed r18p image
candidate. It is not qualified for serving until a distinct image passes the
permanent two-geometry numerical/performance gate and a controlled GLM A/B
shows that the kernel recovery closes the measured 7.9% to 9.8% prefill gap
without losing decode. Full receipts and source identities are in
`spark/v20/bench/b4d6c75-paired-m8-boundary/RESULTS.md`. No image or service
was changed in this round.

## Round 8: same-harness legacy control closes paired-FC2 causality (2026-08-20)

The corrected-r34 legacy image ran the production-faithful harness at both
148/108 and 192/64. Its output hashes match stock and paired r18 at every
measured cell. The large-M medians are:

| Geometry | M | r34 legacy (us) | r18 stock B/A/B (us) | r18 paired (us) | Paired vs r34 |
|:---------|--:|----------------:|----------------------:|----------------:|---------------:|
| 148/108 | 33 | 3,441.7 | 4,225.9 | 3,403.4 | -1.1% |
| 148/108 | 221 | 5,615.8 | 6,756.8 | 5,639.7 | +0.4% |
| 148/108 | 222 | 5,652.7 | 6,769.0 | 5,659.1 | +0.1% |
| 148/108 | 3072 | 19,298.4 | 23,747.3 | 19,414.1 | +0.6% |
| 192/64 | 33 | 3,263.7 | 4,112.0 | 3,274.0 | +0.3% |
| 192/64 | 221 | 5,366.9 | 6,583.1 | 5,318.7 | -0.9% |
| 192/64 | 222 | 5,417.9 | 6,589.2 | 5,342.5 | -1.4% |
| 192/64 | 3072 | 19,159.4 | 23,705.1 | 19,283.3 | +0.6% |

The causal story is now general rather than geometry-specific. Legacy r34
paired FC2 is fast at both expert splits; stock r18 without paired execution
is 19.8% to 26.0% slower across these cells; and restoring paired FC2 returns
r18 to within 1.4% of r34 everywhere while leaving M <= 32 flat and outputs
bit-identical. The failed 192/64-flat criterion in round 7 rejected only the
incorrect split-sensitive premise: that control moved because paired FC2 was
the legacy optimization at 192/64 too.

Round 2 item 9 is therefore an instrument false negative. B12X's upstream
benchmark materializes 16 experts per tier, pads them to 192/64, fixes the
route mix at six K3 and two K4 experts, and graph-replays that synthetic
workload. Its 32.15/32.21 ms parity row cannot be mixed with this harness's
complete-expert-population public-API results. This is the third harness-
fidelity failure in the program, after the two rejected DCP/NCCL models in
item 18.

At M=3072 the same-harness stock-r18 minus r34 delta is approximately 4.45 ms
per routed layer, or 0.33 seconds over 75 layers. Paired r18 removes that
delta. This supplies a well-founded prediction that an r18p image will close
the 7.9-9.8% serving-prefill gap under the candidate NCCL contract; only the
distinct-image serving A/B can qualify that prediction.

## Round 9: r18p image and same-host DS4 qualification (2026-08-21)

The r18p image is
`445f9ac3196dd10a47d0d90441ba43a8417903ba029f59a1a9c7fbef8ecfa4a1`.
Its qualified B12X composition is bound by integration tree
`07cdf4567b50fa983462f0f0e1bc992de3033adc`, integration patch SHA-256
`e474392dea52c98fd16ca180e7ad54327c63c2623ed87ef5652bdb837a2175ab`,
composition-lock SHA-256
`ad98c1b196eba0a4e84f7f907c5b09feb6e4b09be7dae477bf6c9bccd5e7624d`,
and paired-FC2 patch SHA-256
`e8d399a1c12a3a2ad8a65ab49437b5fb75ae78a00aba551fe2f193c029805ca1`.
The image was built with `ALLOW_DIRTY_BUILD=1`, so its recipe-commit label
still names the r18 base commit `1e0c1ab` rather than the later r18p source
commit. The exact qualified lock and patch bytes were recovered from the
surviving dusty build context and independently reproduced from the staged
B12X source tree. They are committed byte-for-byte in signed release-source
commit `520899278a984e37a2f1d873b1b2042cf5b5b7c7`; the baked recipe-commit
label predates that commit, but its lock, integration-tree, integration-patch,
and paired-patch identities match.

Pagure was unavailable during the build, so the image used the explicit
libaio source override `https://github.com/joninco/libaio.git` at commit
`1b18bfafc6a2f7b9fa2c6be77a95afed8b7be448` and tree
`c9442e111b747e9329ea782c6edb9d13a827cc08`. The committed recipe retains
the restored canonical `https://pagure.io/libaio.git` default; reproducing
the qualified image exactly requires the recorded GitHub override.

The first r18p, stock-r18, and corrected-r34 sweeps seeded lazy kernels and
are excluded from performance conclusions:

- `benchmark_results-ds4-0731-iir18p-b12x-a8_20260821_125000.json`
- `benchmark_results-ds4-0731-iir18-stock-cc4-ab_20260821_132500.json`
- `benchmark_results-ds4-0731-r34c-contemporary_20260821_144610.json`

The accepted JIT-free results are the corresponding `-warm` files at
13:06, 13:33, and 14:59. On the same dusty/kirby hosts and TP2 DSpark K5/a8
profile, warmed r18p and warmed stock r18 were equal at concurrency four:
28.249 versus 28.325 steps/s geometric mean, a -0.27% delta. The paired-FC2
restoration therefore does not regress this DS4 workload.

The contemporary full-sweep comparison against corrected-r34 measured
r18p at -0.34% prefill geometric mean and -1.26% acceptance-normalized
decode geometric mean overall. By concurrency, r18p was -1.34% at cc1,
+1.73% at cc2, and -4.09% at cc4. This is DS4 qualification evidence only:
DS4 does not exercise the GLM mixed-Trellis path changed by r18p. It neither
proves nor disproves closure of GLM's earlier 7.9-9.8% prefill deficit. The
production-faithful single-GPU mixed-Trellis A/B predicts closure, but the
four-node GLM TP4/DCP2 serving A/B remains the qualification boundary.
