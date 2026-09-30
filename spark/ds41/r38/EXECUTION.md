# DS4.1 TP4 admission, 2026-09-16

Status: **blocked at native-context retrieval.** The K7 native 1M boot is
serving and healthy. It passed the short battery and dual retrieval at 16K,
131K and 262K, then failed the unchanged 524K dual-needle gate. That failure
is unresolved and blocks promotion. The 1M, conversation and mixed-load gates
and the benchmark grid were not run. See the last section.

## Authorized window

The user authorized stopping dusty, toby, rusty and kirby and proceeding.
Kirby's Qwen worker and toby's DS4 Vision worker were stopped gracefully with
`podman stop -t 60`, then dusty's Qwen head. Rusty was already idle. All old
containers and images are retained. No GLM node was stopped or modified.
The four stopped-state receipts and previous-container commands are under
`receipts/20260916/`. The initial pre-cutover template inspection failed;
the separate `previous-*.json` records supersede that failed field.

All four nodes have R38 image `ea031e1d3d05`, GPUs idle and about 117 GiB
MemAvailable. Toby's 1.79 GB swap occupancy dropped below 1 MB on shutdown.
Both f0 rails pass the runner's address, peer-route, HCA and GID checks.
Persistent model storage is local NVMe/ext4. No cache cleanup was performed.

## Checkpoint distribution

No DS4.1 snapshot was found under the inspected cache/model locations on the
eight Sparks. Only rusty downloads the pinned checkpoint from Hugging Face,
under `ds41-checkpoint-download.service`. The 48 weight files total
510,296,708,312 bytes. The existing HF CLI is managed by uv; nothing was installed.

`stage-model.sh`, under `ds41-checkpoint-fanout.service` on rusty, copies
completed blobs and relative snapshot links to dusty, toby and kirby while the
download continues. It excludes incomplete blobs, never deletes files, and
runs one final pass after download success. Bulk data uses only verified
`10.11.11.5 -> .7/.6/.8` switched f0 routes. Rsync uses whole-file transfers,
no compression and AES128-GCM. Peer host keys were obtained through existing
trusted hostname connections, not accepted blindly over the fabric.

Download completed successfully at 15:42:59 EDT. The final fanout completed
at 15:49:57 EDT, all copies exit 0. See the download/fanout journal receipt.
Full payload SHA256 and metadata checks passed on all four nodes (48 shards
each): rusty 695.74 s, dusty 655.29 s, toby 626.37 s, kirby 624.10 s.
Rusty's check overlapped with final fanout. All four complete host preflights
also pass; those file-size checks were not substituted for payload hashing.
The download explicitly pinned `--revision fb2764a5...`; it never followed
`refs/main`. A later review's assumption of a plain main download was incorrect.

## Real-image gates completed

- Eight actual CLI parser cases pass: four ranks, DSpark K0 and K7.
- Installed InstantTensor `_impl.py` reads BUFFER_SIZE, CONCURRENCY and IO_DEPTH
  at lines 162, 150 and 154 respectively.
- Default-policy disk probe fails reproducibly with io_uring ENOSYS. The host's
  `kernel.io_uring_disabled=0`. An identical invalid-argument syscall returns
  EFAULT on the host and ENOSYS in the default container.
- All four baseline seccomp profiles have SHA256
  `9b755202516aee4b45d9d411ab800c20fe4f7af97166a93b7f07d5b16c1a4ecd`.
  The profile is default-deny with errno 38 and no io_uring entries.
- Claude independently reviewed `seccomp-io-uring.json`: all 40 baseline rules
  unchanged, only io_uring_setup/enter/register added. SHA256
  `79bb4c4789e417dd024c19bb7f16255e5ebf940133f117743c86a6feed9dc9b1`.
  No capability, privilege, sysctl or host security policy change.
- The same native O_DIRECT registered-buffer/file probe then passes on rusty
  against the pinned first model shard: three rows, 792 logical bytes,
  12,288 read bytes, one read/submit, 8 MiB staging, exact byte parity.
- Ten upstream Engram GPU cases pass on SM121, zero skips, 7.76 seconds:
  eight separate-plane/TP-edge/raw-E8M0 cases, consumer-graph/stream reuse, and
  sparse global row addresses above 2^31. The image's RequiredMatrix wrapper
  enforced the exact collected/passed count. Temporary files were on NVMe.
- All four nodes then passed the complete in-container preflight: SM121,
  actual vLLM RMSNorm operation, native linkage, image-owned NCCL preload,
  liburing loader, registered O_DIRECT byte parity and eight parser renders.
  The `native-*-attempt1.json` receipts retain the full outputs and exit codes.

The semantic driver is adapted from `spark/ds4-vision/qualify.py` with the
DS4.1 served name, dusty endpoint, an explicit first non-thinking completion,
and a 131,000-token long probe inside the 131,072-token admission envelope.
It requires normal stops and exact expected answers, with all responses saved.

These are component gates, not model-output qualification or throughput results.
The narrowed policy allows io_uring for all processes inside this container;
InstantTensor remains explicitly BUFFERED and its actual boot selection must
still be recorded. No benchmark repository was modified.

## Initial four-node boot and short output admission

Workers started first on toby, rusty and kirby, followed by dusty at about
16:02 EDT. Every container uses image `ea031e1d3d05...` and the frozen kit
manifest `3be563d40ff8469b1acb2b807f7d3ffff7c87f3e5a31172f1256e75cf52226b5`.
The initial profile is TP4, DCP1, DSpark K7, 131,072 maximum context and
0.85 utilization. The previous services remain stopped and retained.

Model loading and graph capture completed. Head reported 7,643,686 effective
KV tokens at this 131,072-token envelope. This group-aware figure must not be
extrapolated to the native 1M envelope. Per-rank KV bytes and UMA headroom
need to be read alongside each boot, not treated as fixed model properties.
Post-capture MemAvailable was about 2.6 to 5.3 GiB across nodes; this is
limited headroom, not evidence of an OOM. Read-only `ds41-observer.service`
units record memory, GPU clocks/power/temperature and NVMe counters at 1 Hz.
They do not stop serving containers.

The short admission battery passed all 19 requests, including the first
non-thinking completion (333), three repetitions each of maximum-reasoning
arithmetic, non-thinking arithmetic, vision and vision continuation, a
validated tool call and return, and four concurrent text/image requests.
Receipts are in `receipts/20260916/admission-k7/`. All answer checks completed
with `finish_reason=stop`; tool arguments were checked separately. These are
simple admission checks, not a general quality evaluation. DSpark logged
nonzero acceptance during them. Long retrieval and target-only controls are
not included in that verdict, and no benchmark throughput is claimed.

Claude reviewed the graph setting against the frozen source. Explicit
`VLLM_USE_BREAKABLE_CUDAGRAPH=0` matches upstream `serve-ds41-flash.sh:317`
and is a supported path: disk rows are prepared before graph replay, full
decode graphs remain enabled, and mixed/prefill batches run eagerly.
The boot records successful graph capture, not a graph-disable fallback.
Breakable prefill graphs are a separate unmeasured optimization, not a
correctness requirement for disk Engram. No runtime setting was changed.

The second admission pass repeated the 19 short requests and added exact-answer
retrieval at 16,384 and 131,000 input tokens. All 21 passed with the required
answers; receipts are in `receipts/20260916/admission-k7-long/`. The longer
request reused 16,128 prefix tokens from the preceding probe, so its 31.58 s
wall time is not a cold-prefill benchmark. The first retrieval took 3.68 s
with zero cached tokens. No performance grid has run.

Kernel-journal inspection found `NV_ERR_NO_MEMORY` on rusty at 16:07:25 EDT,
during graph capture and before readiness. The engine remained running and
passed the output probes; no such messages appeared on the other three nodes
in the inspected startup interval. This is an allocation-pressure finding,
not an OOM-killed container or a clean-memory qualification. Current memory
and startup journals are in `host-after-admission-*.log`. Compact boot logs
omit verbose compile parameter payloads only; complete originals remain in
the rootless Podman journals. Larger context and sustained load remain
unqualified, along with the target-only control. GLM serving was untouched.

## Target-only control and bounded diagnostic retry

The first K0/131K/0.85 boot reached readiness but the first 21-token arithmetic
request returned HTTP 500. The exact first exception was the JIT monitor
raising on `_quantize_attention_inv_rope_to_tdg_kernel` during inference,
not a CUDA illegal access or engine allocation exception. Full logs and
telemetry are in `receipts/20260916/k0-131k-failure/`. Separately, rusty again
logged NV_ERR_NO_MEMORY after KV allocation and before the second graph capture
at 16:34:50 to 16:34:52 EDT.
The prior K7 and failed K0 containers are stopped and retained by profile name.
All four nodes recovered more than 115 GiB available with little swap used.

The retry keeps the image, model, topology, K0 and 131K context. It uses
0.80 utilization for additional UMA headroom after repeat capture pressure.
It uses explicit JIT warning mode only for untimed target-only correctness,
so missing warmup specializations are recorded rather than killing requests.
The default remains strict error mode; the contract refuses warning mode
when speculation is enabled. The new local contract test failed before the
change and passed after it. This is a diagnostic exception, not a warmup fix
or a numerical/performance qualification. No source kernels were modified.

The K0 retry passed all 21 requests, including cold, unique-prefix retrieval
at exactly 16,384 and 131,000 input tokens with zero cached tokens. Complete
responses are in `receipts/20260916/admission-k0-u80/`; logs, journals and
telemetry are in `receipts/20260916/k0-u80-final/`. No new NV_ERR_NO_MEMORY or
Xid appeared during this boot and battery. Post-battery MemAvailable was
9.64 to 12.28 GiB across the four nodes. Head reported 7,599,494 effective
KV tokens at the 131K envelope, with rank-local available KV 17.12 to
18.04 GiB. Six distinct JIT kernel names were logged during correctness
requests. Warning mode allowed them to compile, not to evade answer checks;
this is not a strict cold-start or timing qualification for K0.

Full-response comparison against K7 found the same validated final answers
and tool name/arguments. Some vision answers differ in capitalization;
thinking text and output lengths differ, including within repetitions of
each arm. No byte-identical or general quality parity claim is made. The
target-only control validates these simple tasks, not DSpark's distribution
over all prompts. The original K7 and K0 retry also differ in utilization,
so their wall times are not an isolated performance comparison.

The next boot returns to K7 with strict JIT error mode, retains utilization
0.80, and attempts the native 1,048,576-token envelope. Its own per-rank
capacity, allocation errors and headroom are admission gates before cold
unique-prefix probes, conversation checks and the standard grid. The old
131K effective capacity is not used to admit the 1M profile. All completed
control containers are stopped and retained, with GLM untouched.

## Native 1M K7 boot and retrieval admission

Boot at 16:47 to 16:50 EDT: all four containers use image `ea031e1d3d05...`
and kit label `03ebcfe8...`. The profile is K7, strict JIT error mode,
utilization 0.80 and `max_model_len` 1,048,576. Available KV memory by rank was
12.91, 12.03, 12.58 and 12.16 GiB (dusty, toby, rusty, kirby). The head
reported 11,682,954 effective KV tokens for this boot; this is not an
extrapolation from 131K. Boot evidence is in `receipts/20260916/k7-1m-boot/`.
No NV_ERR_NO_MEMORY, Xid or OOM was logged on any node from this boot through
the diagnostics below. Minimum observer MemAvailable in that interval was
9.72, 8.71, 8.13 and 11.26 GiB. The JIT monitor raised nothing.

`qualify.py --long --dual-needle` (sha256 `108adc19...`) passed the 19 short
checks and cold two-code retrieval at 16,384, 131,000 and 262,144 tokens.
Those took 4.1, 32.6 and 72.8 s, all with 0 cached tokens. At 524,288 tokens it
**failed**: the cold request stopped normally and returned the exact archive
identity instead of the initial code (`510c94b1..., 482617`, expected
`739184, 482617`). The receipts are unchanged in `admission-k7-1m-u80/`, and
that log has no PASS marker.

The diagnosis is in `receipts/20260916/needle-524k-diagnostics/VERDICT.md`;
it was produced by the new, provenance-labelled
`probe-needle-diagnostics.py`. In summary:

- On the identical failing input, the original instruction was wrong in 8 of
  9 runs. That includes 2 of 3 cold runs, two of which used a unique
  `cache_salt` so no prefix was reused. Changing only the final instruction to
  the single-needle phrase "retrieval code stated before the archive" gave 4
  of 4 correct (1 cold salted, 3 warm with 524,032 cached tokens), with a
  first-token margin of 5.75 to 6.88 logprob.
- Identical cold requests on the idle server do not give identical logits.
  At 16K and 131K, answers were unchanged, but shared top-5 logprobs moved by
  up to 2.38 and 1.81. At 524K, the margin between the identity and the code
  swung by about 9 between two cold repeats and flipped the answer. The source
  of this variation is not identified.
- Fresh-identity passes (25 of 25 cold across 16K to 524K) change token
  positions as well as wording. They do not isolate the instruction and are
  not used to compose a pass. No reference implementation was run.

Verdict: a runtime crash or visible error is excluded. A near tie between the
code and a digit-leading identity, combined with unexplained run-to-run
variation, explains the observed flip. But neither model behavior nor this
runtime's numerics is proven responsible. The original failure is **not
resolved and remains a promotion blocker**. Continuing needs a user decision.
One option is a revised retrieval gate, run with repeated cold salted trials
and keeping this failure on record. The other is an approved investigation of
the nondeterminism, which changes the runtime. The candidate is left serving,
healthy and idle; nothing was restarted, rolled back or promoted.

`probe-conversations.py` now requires an idle `/metrics` baseline (running and
waiting both 0) before the mixed long request. It also requires the long
request to be observed running. It records sampled running/waiting gauges and
client-side turn timings, and passes only if at least 2 requests were observed
running while the long request was active and at least one turn finished
before it. Submission alone never counts as overlap. Exact-answer checks are
unchanged. Seven offline tests in `test_probe_conversations.py` pass, using a
mocked server, no sockets, and temporary files under this kit. The probe itself
has not been run against the server.

## Approved diagnostic restart cycle: K7 boot 1, K0, K7 boot 2

The user approved diagnostic restarts of the four DS4.1 nodes. Each cycle
stopped workers first, then the head, with `podman stop -t 60`; all exits were
0 with no OOM kill; stopped containers were renamed and retained
(`...-k7-1m-u80-boot1-20260916`, `...-k0-1m-u80-warn-20260916`). Launches ran
workers first through the unchanged node runner (kit label `03ebcfe8...`),
with the same image, model, seccomp profile, 1M envelope and utilization
0.80. Only speculation varied. Logs and dry renders are in
`receipts/20260916/restarts/`; node captures per boot are listed in the verdict.

- Before teardown, boot 1 was sampled further: original instruction 1 of 6
  cold and 0 of 14 warm correct; instruction replaced 3 of 3 cold, 13 of 13 warm.
- K0 (18:36): JIT monitor in warn mode, because the strict K0 boot earlier in
  the day died on its first request; four kernels compiled during the first
  requests. Head KV 15.86 GiB, 13,882,775 effective tokens at 1M. Original
  instruction 0 of 5 cold, 0 of 10 warm; replaced 3 of 3 cold, 11 of 11 warm.
- K7 boot 2 (19:07): strict JIT, no compilation raised. Head KV 12.0 GiB,
  11,605,244 effective tokens at 1M (boot 1: 12.91 GiB, 11,682,954).
  Original 1 of 5 cold, 0 of 10 warm; replaced 3 of 3 cold, 11 of 11 warm.
- 16K and 131K passed in every trial of every arm (22 cold, 60 warm).
- No NV_ERR_NO_MEMORY, Xid, OOM, JIT raise or engine exception in any arm.
  The 1 Hz observer units had exited with boot 1's container; K0 and boot 2
  have kernel journals and memory snapshots only. The observers were
  recreated after boot 2's series and are running now.

Speculation is excluded as the source of the wrong answer and of the
run-to-run variation; a fresh boot reproduces both. The read-only source
scan (`needle-524k-diagnostics/source-analysis.md`) lists candidate
mechanisms, chiefly bf16 atomic MoE output accumulation with a dynamic work
queue and atomic-order tie-breaking in the DSA indexer top-k; neither is
measured. Testing the MoE candidate needs an admitted environment variable
(`B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1`), which is a contract change and
awaits approval. The candidate is serving under K7 boot 2, idle and healthy.
Qualification, benchmark and promotion remain blocked.

## Approved fourth arm: deterministic MoE output

The user approved one more diagnostic arm. `contract.py` gained a narrow,
provenance-labelled passthrough for `B12X_DYNAMIC_DETERMINISTIC_OUTPUT`
(0 or 1, only when given, no CLI change), with a new test in
`test_runner.py` (21 tests pass, also under `python -O`) and a regenerated
`runtime-files.sha256`. Kit label is now `f6166ebb...` (was `03ebcfe8...`);
the pre-change files are kept in `receipts/20260916/restarts/kit-before-deterministic-arm/`.
The three changed files were copied to `/home/jugs/git/ds41-r38` on all four
nodes, where the manifest verifies and the tests pass. `benchmark.sh` still
pins the previous label and will refuse the current one; benchmarking is
blocked regardless.

Boot 2 was stopped worker-first (all exit 0) and retained as
`...-k7-1m-u80-boot2-20260916`. The fourth arm booted K7, strict JIT, 1M,
u0.80 with the flag. The variable is in the API-server environment and the
container's settings receipt; all four ranks compiled the deterministic
path's `moe.w4a16.topk_sum` reduction kernel. Because the B12X compile cache
fingerprints every `B12X_*` variable, all 180 kernels were recompiled from
the same source into a new cache namespace (about 33 s on the head). Head
KV 12.64 GiB, 10,965,833 effective tokens at 1M. No JIT raise, NVRM, Xid or
OOM. The observer units were recreated before the series and covered it.

Result: no change. At 524K the original instruction was wrong in 6 of 6 cold
and 9 of 10 warm trials; the replaced instruction was right in 3 of 3 cold
and 11 of 11 warm. The first-token spread did not shrink (cold sd 1.6 at
524K, 1.8 at 16K, 1.2 at 131K) and cold repeats at 16K still differ by up to
3.1 logprob. Deterministic MoE accumulation is therefore not the dominant
source of the run-to-run variation; the remaining source-scan candidate
(indexer top-k tie-breaking) has no runtime knob. Full record:
`needle-524k-diagnostics/VERDICT.md`. The candidate is left serving under
this diagnostic arm, idle and healthy; qualification, benchmark and
promotion remain blocked.

## Attribution work, 2026-09-17

After the fourth arm the user asked for an attribution before filing. Steps
and results (details in `needle-524k-diagnostics/ATTRIBUTION.md`):

- Measurement validation over all receipts: 0 request-equivalence issues
  across 32 inputs; 3,020 logprob positions consistent; logprobs never changed
  an answer; cold and warm paths agree within about one sd at 16K and 131K.
- Length sweep on the deterministic-MoE arm (`10-`, `11-`): identical cold
  salted requests are bitwise identical in the top-20 up to 513 tokens of KV
  and diverge from 514 to 515, matching the indexer's `> 512` selection
  threshold; variation grows with length.
- The deterministic arm was stopped worker-first and retained
  (`...-k7-1m-u80-det-20260916`); the baseline K7 strict profile was relaunched
  (boot 3, 11,390,458 effective tokens at 1M, no JIT raise, clean kernel logs,
  observers recreated) and is serving now. On it, the reproducer
  (`repro-nondeterminism.py`, `12-`) shows identical requests already differ
  at 256 tokens (max top-20 diff 2.9) while the top-1 token never changes.
- Kernel oracle (`diag/topk_oracle.py`, run inside the kirby container,
  `13-`): 375 calls of the production indexer top-k, all valid by B12X's own
  criterion; stable without ties; different sets whenever bf16 ties straddle
  rank 512. The extra `diag/` file sits in the node kit directory and is not
  part of the runtime manifest.
- Verdict: reproducibility is attributed to two design-level nondeterminism
  sources (bf16 atomic MoE accumulation, atomic-order indexer tie-break);
  accuracy is consistent with instruction sensitivity under that variability
  but needs a non-B12X reference to be proven; numerical correctness is
  verified where a reference exists and otherwise open. An issue on
  reproducibility is justified (`ISSUE-DRAFT.md`, not posted). Qualification,
  benchmark and promotion remain blocked.

## DS4.1 brought down; previous serving restored, 21:30 EDT

On the user's direction, DS4.1 was stopped worker-first, head-last and
retained (`...-k7-1m-u80-boot3-20260916`), its observer units stopped, and
the retained previous containers restarted from their recorded identities:
Qwen R32 on kirby then dusty, DS4 Vision R38 on toby then rusty. Both heads
answered a real completion and advertise only their previous model names.
Kirby logged NV_ERR_NO_MEMORY during the Qwen worker's graph capture; the pair
still came up. Record: `receipts/20260916/restore/RESTORE.md`.
