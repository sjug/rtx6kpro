# Engram progress-safe handoff candidates

Design investigation, 2026-09-25. Candidate implementation in progress, not qualified.

## Stock-runtime reference feasibility (2026-09-26)

The user suggested official vLLM v0.30.0 or nightly as an independent
non-B12X reference. Both Docker Hub tags have ARM64 manifests (queried live):
v0.30.0 `sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56`;
nightly `sha256:3d538f222257ccd4a4e6d133b07d2dea42ce6fdc849ea946584b54b4a9a67613`.
These are architecture manifests, not proof of successful SM121 execution.

Local upstream vLLM at `4bb804cc5b` and tag v0.30.0 contain DS4.1 support;
the NVIDIA model selector chooses FlashInfer for capability major 12.
However, the inspected Engram implementation and configuration provide
RAM-resident tables, including pinned-host UVA offload, not disk-backed
lookups. On GB10 that host memory consumes the same physical pool as GPU
allocations. The current disk-backed boot already reports approximately
74.59 GiB of weights per rank, before restoring the large Engram tables to
RAM. The user confirmed that this does not fit without disk Engram.

Consequently neither stock tag is a drop-in TP4 reference here. No stock
image was pulled or launched, and the running diagnostic was not interrupted.
An upstream-compute reference would need a separately reviewed disk-storage
adaptation or hardware with enough RAM. Do not describe such an adaptation
as an unmodified upstream reference or silently relax the original retrieval
gate in its absence.

### B2 long repeatability continuation

`receipts/router-b2-long-diagnostics-20260926/` completed with exit 0 on
the same B2 boot: 131072 tokens eight cold repetitions, 524288 tokens four
cold repetitions. Every answer was `739184`, with one identical choice
object (including logprobs) per length. The 524K times were 180.369,
181.076, 179.514 and 174.917 seconds. All eight final node-log collections
succeeded; no Xid, NV_ERR_NO_MEMORY or OOM lines appeared in the captured
kernel windows. Owned observers were cleaned up.

This is the frozen single-code corpus, not the failing historical dual-code
input. It strengthens repeatability evidence without satisfying or replacing
the original gate. The latter still has three identical wrong responses.
The separate remaining-coverage driver runs the unchanged context envelope
and historical mixed-load needle with same-boot identity checks; its results
must not be presented as a pass of the failed full qualification.

The remaining-coverage context stage subsequently passed new dual-code
requests at 262000, 500000 and 599936 tokens (73.062, 175.596 and 215.894
seconds respectively), all returning `739184, 482617`. These have fresh
archive identities and are not the unchanged historical input. The serial
32K prefix tests also passed: repeat, extension and divergent suffix each
reused 32512 tokens, approximately 99.1 percent. At this recording the
concurrent/mixed-load stage was still running, not yet a pass.

That stage subsequently failed its overlap requirement. All 13 responses
were correct, with zero aggregate cache hits and zero preemptions, but
none of the conversation turns finished before the long needle. The long
request took 337.815 seconds and first conversation turns about 334.899
seconds. The new identity made that needle 524285 tokens; it is not an
exact replay of the frozen 524288-token historical input. All eight final
log collections succeeded and all four request-window kernel logs were
free of Xid/NV_ERR_NO_MEMORY/OOM. The failed driver did not run its final
short probe or claim qualification success.

Source explanation for admission delay: pinned vLLM `1794dcf1`,
`config/scheduler.py:213`, defaults `max_parallel_prefills=1`, inherited
by `engine/arg_utils.py:723`. The launch receipt carries no override.
At `scheduler.py:863` the interleaver is disabled for one lane; running
requests are scheduled at line 1249, and waiting requests require a
positive remaining token budget at line 1265. With the default disabled
long-prefill cap, an existing long prefill can consume the whole budget.
This supports a separate scheduling-policy finding, not attribution to
the determinism patches. No scheduling setting was changed.

## New isolated evidence

`probe_cuda_producer_progress.py` now reproduces the progress failure without
DS4.1, disk I/O, NCCL or any B12X operation. Receipt:
`receipts/cuda-producer-progress-20260925T174435Z/` (three fresh processes per
arm, timing-image identity and script digest recorded). All runs use LAZY and
idle dusty. The consumer spins for at most one second; a producer attempts to
publish after 100 ms. The GEMM is BF16 385x512 by 512x128, using `torch.mm`.

| Arm | Consumer timeouts | Producer host launch | Total |
| --- | --- | --- | --- |
| No GEMM | 0/3 | 0.20 to 0.26 ms | 101 to 102 ms |
| First-use GEMM after consumer submission | 3/3 | about 900 ms | 1069 to 1085 ms |
| Same GEMM prewarmed | 0/3 | 0.24 to 0.26 ms | 101 to 102 ms |
| Delayed producer pre-submitted, GEMM still cold | 0/3 | not a host-thread producer | 184 to 185 ms |

Every GEMM result is correct. The last arm uses a bounded GPU delay to model
pre-submission, not CPU disk reads, so it is structural evidence rather than
validation of either repair below. This establishes that first-use `torch.mm`
can cause the same producer-launch blocking pattern in this runtime. It does
not locate the internal CUDA API/lock or prove every serving stall has that
cause. The original serving traces and this isolated red/green discriminator
together support removing the dependency on a future host CUDA launch.

A subsequent native CPU-only CUDA host callback passed three fresh-process
trials with cold GEMM (`receipts/cuda-producer-progress-20260925T174736Z/`).
The extended callback wrote 8,192 bytes into B12X's actual write-combined
`MappedHostAllocation`; the queued GPU lookup saw the exact checksum 1,044,480
in all three trials, with no consumer timeout and correct GEMM output
(`receipts/cuda-producer-progress-20260925T174915Z/`). This tests visibility and
progress for that mapped payload, not real disk-reader ownership or full-model
qualification. The repair must still pass those integration gates.

The existing consumer spin runs before the producer has submitted all CUDA
operations needed to release it. A later producer launch can block behind device
progress, leaving the consumer to time out and use unavailable rows. Eager loading
is a discriminator, not a sufficient general progress contract.

## Required properties

- Submit the IDs transfer, row decoding, publication and their ordering before
  submitting the model forward that consumes them.
- The CPU disk-read producer must not require a later CUDA call to make progress.
- Keep disk reads overlapped with independent model work and keep the two table
  reads concurrent where possible.
- Preserve cache lifetime, previous-use completion, resident scales, graph replay
  and real token counts. Do not reuse staging until prior decoding completes.
- Propagate disk errors and timeouts to all TP ranks before publishing output.
  A stale timeout from a previous epoch must not invalidate a later step, and an
  asynchronous later step must not overwrite an earlier step's failure receipt.
- The current non-GDS, io_uring path is the initial scope. A GDS implementation
  cannot silently inherit a CPU-only producer contract.

## Candidate A: pre-enqueued mapped-word handoff

Claude proposed enqueueing IDs copy and a mapped readiness signal, then a GPU
gate, lookup kernels, and publication before the forward. A CPU-only worker
acquires IDs readiness, performs native reads, and release-publishes completion.
This needs explicit native CPU release/acquire and GPU system-scope ordering,
bounded failure handling, and a transaction API that separates enqueue ownership
from CPU read ownership. GPU scheduling of the gate must itself be tested under
eager and graph load; moving the spin is not by itself a progress proof.

## Candidate B: native stream-ordered host I/O

Investigate whether the existing native io_uring reader can run as a CUDA host
function between the pre-enqueued IDs copy and lookup kernels. Its callback would
perform only native CPU I/O, no Python, CUDA, tensor destruction or CUDA-backed
allocation. All pointers and readers must remain owned until completion. A
callback must record native errors rather than raise across the C ABI.

The CUDA driver contract orders a host function after prior work and before
subsequent work in that stream. It prohibits CUDA calls and synchronization on
CUDA work not ordered earlier, and permits independent callbacks to serialize.
It also omits the callback on a CUDA-context error. See
[CUDA execution-control reference](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__EXEC.html).

Pinned-source feasibility: `a7d7d29b:b12x/loader/_ple_reader.c:334-363`
already encloses the actual read in a GIL-released region. Its core uses only
the reader mutex, clocks, `ple_plan`, `ple_uring_run` and a native `failure_t`.
It makes no CUDA or Python calls in that region. A callback must extract that
core, not call the Python wrapper: buffer acquisition/release and exception
construction remain Python operations outside it. The reader mutex must not be
held by a host thread waiting on the callback's CUDA stream. Async buffer owners
must be released on the normal Python thread, not from the CUDA callback.

Therefore a single native batch callback may need its own CPU workers to retain
the two-table I/O overlap. Its error and lifetime paths need tests even when the
callback never runs. This option could avoid a mapped host polling protocol,
but its compatibility with the current reader, graph capture, and failure
propagation is not yet established.

With a fully submitted side stream, a real stream-event dependency at the Engram
consumer may be preferable to a spinning kernel. External-event replay semantics
and per-step event reuse must be demonstrated in the actual graph path before
selecting that design. No source or runtime change is implied by this proposal.

## Acceptance evidence required

1. A minimized producer/consumer test that reproduces the current failure, with
   delayed CPU reads and first-use CUDA work, in eager and graph modes.
2. Correct row output across repeated epochs, alternating shapes, zero/padded
   rows and cache reuse; explicit old-epoch and CPU-read-failure tests.
3. No silent outputs on injected timeout or a failure on only one TP rank.
4. Clean-image repeated first-cold 385-token tests, the unrelated-content ring
   transition, and the unchanged historical 524K input.
5. Full approved 600K-envelope qualification and standard benchmark, with GLM
   left serving throughout.

## Integration gates completed so far

The proposed vLLM main-thread enqueue integration passes four CPU tests for
submission order, captured epoch identity, enqueue-error propagation and exact
generated-source composition. `test_engram_epoch_gpu.py` executes the actual
generated `_publish_engram_epoch` and `_wait_engram_rows` bodies on GB10, without
substituting a mock kernel. Its eight cases pass for eager and captured consumers:
success, I/O error, unpublished status and success after a previous timeout.
Receipt: `receipts/engram-epoch-gate-20260925T180045Z/`.

These tests do not establish native-reader lifetime safety, cross-rank fault
propagation or full-model determinism. Those remain required before qualification.

## Reviewed candidate built

Image `ee03505df7a3b5b251d657ab201f62ac9d74eb2aee033cc5a094d7996caab4c5`
derives from the uninstrumented ring-fix base `e7b273407022…`. Its complete
tracked source identities are vLLM `ce1b9bcdb91706251314d3c1ea955887c779901c`
and B12X `ab068013c6bedbbd9435b2bb33de3f0a04a18a82`. Existing native artifacts
are unchanged; the source-hashed C99 loader recompiles with the additive batch
callback API. The GB10-built loader links liburing and exposes all three new
entry points. Build receipt: `receipts/engram-repair-build-20260925T181257Z/`.

Claude's native revision 2 lock is
`334eedf4692c8be65832bfef7277aa67a4416dac8d46142f0ea5783684e4779e`.
Independent reruns passed 16 native tests and 12 Python API tests. The initial
sandboxed native run was refused by io_uring policy; the authorized unsandboxed
rerun passed. This was an execution-policy failure, not a skipped test.

The integration now bounds prior-job completion reporting to 60 seconds,
pins the exact native API lock, and uses a volatile mapped-status load.
No status reset occurs between steps. Ownership remains with the persistent
model caches/status; a pending native callback retains its own references.
Hung I/O can still block its CUDA stream; timeout handling refuses outputs,
not disk cancellation or automatic recovery.

Build GPU gates passed: a real vLLM RMSNorm custom op; eight eager/captured
epoch cases; sixteen real two-table disk/lookup/consumer cases with resident
scales both enabled and disabled, zero/padded rows, staging reuse, a cold GEMM,
and an injected short read; all 29 compressor-ring regressions. The injected
short read reports an error and leaves its epoch unpublished. These gates do
not yet prove TP4 fault propagation or full-model numerical qualification.

Distribution uses the existing uncompressed Docker archive procedure on the
switched 200G fabric. The launch contract stays K7, 600K, utilization 0.85,
disk Engram overlap enabled, deterministic MoE enabled and dense turbo disabled.
GLM is untouched. No promotion claim is made.

## Full-model checks and residual, 2026-09-25

The candidate completed its first boot on all four nodes. Receipt
`receipts/engram-repair-20260925T182128Z/` contains six cold salted trials each
at 385, 128 and 16384 tokens: every answer correct, with exactly identical
full response and top-20 logprob signatures within each length. The first
385-token request after boot is included. Startup allocation warnings remain
recorded; this is not a claim that memory-pressure warnings were fixed.

The three meadow-to-filler transition cycles also pass, including equality of
the original prompt's logprob signature across cycles. The subsequent
28-length, 168-request sweep did not pass: one 255-token trial differs in
logprobs from its five peers, starting at the first output token, despite
correct complete answers throughout. Largest shared top-20 logprob difference
for that pair is about 1.75. All other lengths passed their six repeats.
Receipt: `receipts/engram-full-qualification-20260925/`. Its driver correctly
stopped before semantics, historical 524K, long context, concurrency and grid.

Follow-up without restarting reproduced the residual in two of 100 cold
255-token requests and one of 100 each at 254 and 256 tokens. All answers
remain correct. The three anomalous 255-token signatures are distinct, so this
is not merely one stable alternate signature or tied top-k ordering. Receipts:
`receipts/engram-repair-255-stress-20260925/` and
`receipts/engram-repair-boundary-stress-20260925/`.

The remaining cause is not established. In particular, the adjacent-length
controls contradict a 255-only boundary attribution. One-output-token trials
are the next discriminator for whether subsequent speculative/asynchronous
steps are required. Their expected truncated answer is not a semantic pass.
The repair is not qualified or promoted, and no performance claim is made.
# Residual investigation update, 2026-09-25

## First activation localization

Diagnostic image `0b5c65a81b2544b2ebe4d1450076f654e21feb43703931601778a645c829c5ba`
adds only the reviewed activation helper/model hook to `ee03505d`. Its
source/native inventory and GB10 digest/async transport gate passed
(40 records, five tensors each). All four loaded image IDs match. The
disarmed boot reproduced a 254-token logprob divergence before tracing.

`receipts/activation-localization-20260925/probe/` records 200 cold
requests, 100 traced and 100 untraced. All traced steps have complete,
epoch-aligned four-rank coverage, with zero drops or overflow. One traced
254-token response diverged at epoch 474. Its first differing checkpoint
is `layers.0:0` on every rank. Layer-zero tuple members 1 through 4 match
their modal digests exactly on all ranks; member 0 is the output of the
feed-forward module. The anomalous member-zero digest is itself identical
across all four ranks. Response/trace time correlation identifies the same
request. This localizes the first observed change to the first-layer FFN
path, including its collective, but does not identify the offending kernel
or originating rank. It precedes the first Engram layer in this event.

The traced anomalous GPU step took approximately 244 ms against a 247 ms
clean median. Thus the previously observed slow-window correlation is not
necessary under instrumentation. Do not infer a specific race mechanism
from timing alone. A same-boot narrowed module-hook run is in progress in
`receipts/activation-layer0-20260925/`; no rebuild or serving-contract
change is needed for that refinement.

The completed `engram-repair-interleaved-20260925/probe` diagnostic contains
1,100 cold requests: 50 seeded random permutations of 22 frozen lengths.
All answers were correct, but seven requests differed from their length's
modal full response/logprob signature. The affected lengths were 248, 250,
254, 257, 258, 271 and 1023, one event each. The pooled rate is 0.64 percent
(Wilson 95 percent interval 0.31 to 1.31 percent, under an independent-trial
assumption). This is not enough data to compare per-length rates.

Every divergent request had server prefill time at least 1.76 times its
own length's clean median; the divergent median ratio was 2.25. However,
259 clean requests also exceeded 1.5 times their length's clean median.
Slow prefill is correlated with divergence, not a sufficient trigger and
not proof of a race. Two divergent requests were consecutive, at
18:46:25.876 and 18:46:26.864 UTC. No unexplained TTFT gap reached five
seconds in the diagnostic.

The separate one-output-token diagnostic also completed: three of 100
cold 255-token requests had different first-token logprobs (repeats 9,
96 and 99). These requests deliberately truncate the answer, so they are
not semantic qualification. Later output tokens are unnecessary for the
variation; speculative preparation is still configured and has not been
universally excluded.

The candidate remains unqualified. The next localization instrument must
verify actual module coverage and cross-rank request alignment before its
activation digests are interpreted. Added launches can mask or amplify
the failure, so traced and untraced blocks must be compared on one boot.

## Moving FFN divergence, same diagnostic boot

The layer-zero refinement completed in
`receipts/activation-layer0-20260925/`. At epoch 1230, a 256-token
request had modal layer-zero inputs, router, shared experts and outputs,
but divergent final model outputs. The subsequent all-layer run in
`receipts/activation-all-layers-20260925/` captured epoch 2955 at
254 tokens: the first differing checkpoint was `layers.13.ffn` on all
four ranks, with modal FFN inputs and earlier checkpoints. Its GPU step
was about 254 ms versus a 253 ms clean median. All 250 traced requests
had aligned four-rank coverage, with no dropped or overflowed records.

Together with epoch 474 at layer zero, this is evidence against a fixed
layer attribution. It supports investigating the common FFN execution
path, without identifying the router, shared branch, routed branch or
collective as the cause. The branch-level follow-up records router and
shared-expert outputs in every one of the checkpoint's 40 target layers.
The model geometry is hidden size 5120, 384 routed experts and top-k 6;
communication size calculations using hidden size 4096 are inapplicable.

The first all-FFN-branch experiment completed 250 traced and 250 untraced
requests without a nonmodal signature. Its coverage includes every layer's
FFN input, router output and shared-expert output on every rank. This clean
interval does not invalidate the earlier same-boot failures. An extended
alternating run is retained separately under
`receipts/activation-ffn-branches-extended-20260925/`.

An additional source lead, not attribution: pinned B12X `dynamic.py` has
unfenced `ml_pipeline.consumer_release`, `up_pipeline.consumer_release` and
`phase2_pipeline.consumer_release` sites. Some follow shared-fragment loads,
resembling the previously demonstrated dense-GEMM stage-reuse fault. Which
specialization executes, whether it reproduces in isolation, and whether an
ordering change affects this residual remain unverified. Do not apply the
dense fix mechanically or describe this as the routed-MoE root cause.

The extended alternating run finished with zero divergent responses in 1,000
traced requests and six in 1,000 untraced requests (254 once, 256 four times,
1023 once). All answers remained correct. Traced records covered all four
ranks, all 40 layers and all required branch hooks without drops or overflow.
Kernel journals for the run had no NVRM/Xid/OOM warnings. Because tracing
adds work and changes timing, these clean traces cannot exonerate a branch.

All four diagnostic containers were stopped cleanly and retained under
`activation-before-routed-isolation-20260925t195027z`. The helper-only
MoE-seam derivative `14abc3e66a45eacbd7478c1c1f6a69ecd345df6bffb8cd81ea972a5c4b304d7f`
built on idle dusty and passed the exact source/native inventory and GPU
digest/transport gates. Its 19 CPU seam tests plus the original 29 tests also
passed independently. It is not deployed or qualified. The next experiment
is isolated routed MoE with the retained serving selection cache, to avoid
the whole-model instrumentation that may be suppressing the residual.

## Isolated routed MoE and executed-path review

The isolated tests completed on dusty using the retained boot's cached
selection and compiled programs, including its `sm_121a` environment and
the model's SwiGLU clamp of 10. Each arm ran 1,000 repetitions at each of
128, 254, 255 and 256 rows: baseline, scratch poison, auxiliary-stream
matmul load, and poison plus auxiliary-stream load. All 16,000 calls were
bitwise stable with no nonfinite output. Receipts are the
`routed-isolation-dusty-20260925T200153Z`, `T200304Z`, `T200447Z` and
`T201108Z` directories under `receipts/` (each has the full timestamp prefix).
These use synthetic weights and routing. Auxiliary work was enqueued on a
separate stream, but actual GPU overlap was not measured. These passes do
not clear the full-model execution path.

Claude's completed source review corrects the earlier release-site lead:
none of the nine `consumer_release` sites executes in the selected W4A8
M16 N64 split-materialized specialization. Its front end runs routing and
input preparation, followed by separate FC1, FC2 and fixed-order top-k
reduction kernels. Adding fences to those inactive sites would not be a
discriminator. No cause was established by this review.

The helper-only MoE-seam image is now frozen as the diagnostic candidate
and distribution has started. Its purpose is to distinguish rank-local
shared/routed output, combination, and all-reduce. The moving first
divergence remains evidence against a defect confined to a fixed layer.

## Full-model MoE boundary evidence on the new boot

Image `14abc3e6` was verified on all four nodes and started worker-first at
20:19 UTC. Its 18 initial completion/repeat probes passed. Startup journals
retained in `receipts/moe-seams-20260925T201902Z/` show NV_ERR_NO_MEMORY
warnings: dusty 159, toby 40, rusty 44, kirby 58, during load/capture and
the initial probes. There are no Xid messages in those journals. These
warnings remain a separate health caveat, not an attributed cause.

The completed run `receipts/moe-seams-localization-20260925/` contains 500
traced and 500 untraced cold requests. All 500 traced steps have complete
four-rank coverage of the 160 MoE seams. Three traced steps diverged:

| Epoch | Rows | First local difference | Other ranks first differ |
| --- | --- | --- | --- |
| 906 | 254 | toby, layer 12 routed output | layer 12 post-reduce |
| 1491 | 256 | toby, layer 11 routed output | layer 11 post-reduce |
| 1914 | 254 | toby, layer 12 routed output | layer 12 post-reduce |

The shared branch remains modal at each first divergent layer. The other
three ranks' local partials and pre-reduce tensors remain modal there.
This localizes these events before the MoE collective; it does not yet
prove that routed inputs or router logits match. The untraced block has
zero divergent signatures, illustrating the intermittent rate rather
than establishing an instrumentation effect. The isolated tests were on
dusty, not the now-localized rank toby, and used synthetic tensors.

The same-boot `moe-inputs` diagnostic adds each layer's FFN input and
router output hashes. It keeps all four MoE seams and requires complete
coverage, including 40 seam runners, on every rank. Ten local coverage
tests pass. This is diagnostic instrumentation, not a numerical fix or
qualification result.

The `moe-inputs-localization-20260925` comparison completed with 500 traced
and 500 untraced requests, all stable; its input/router coverage was complete.
The lighter `moe-seams-repeat-20260925` then completed another 250 traced
and 250 untraced requests without divergence. Both runs have clean kernel
journals. Neither clean interval disproves the three captured events.

The four diagnostic containers were stopped cleanly and retained under
`moe-seams-before-toby-isolation-20260925t204447z`. Toby's isolated baseline
and combined poison/auxiliary-stream-load arms each ran 4,000 calls with
zero mismatches. Receipts: `routed-isolation-toby-20260925T204733Z` and
`routed-isolation-toby-20260925T204843Z`. They used the cached production
dynamic/internal/grouped/tile16/max-active-clusters48 variant. Synthetic
data and unmeasured GPU overlap remain the same limitations as on dusty.

The reviewed capture derivative
`02b046d3ac02641f741fe731edb0c13f3373201526f64d43b2f475de82d2d0e6`
built on idle dusty from the helper-only seam image. It passed source/native
confinement, the 40-record CUDA transport test and the real-GB10 capture
gate: an injected mismatch saved exact input, logits, selected weights/IDs
and routed output at epoch 4; a delayed generation-2 record did not ready
generation 3, whose own epoch-11 failure saved correctly. Build receipts:
`receipts/moe-capture-build-20260925T205005Z/`. This gate validates the
instrument, not the model. Distribution completed with the same image ID
on all four nodes. The worker-first boot completed and the 18 initial
repeatability requests passed (`moe-capture-20260925T210024Z`). Startup
allocation warnings remain recorded separately from the numerical defect.

Independent review caught a diagnostic flush issue: 16 tokens could replay
a captured graph without calling the Python latch-drain hook. The driver
now uses 64 tokens, above capture size 32 and below trace minimum 100.
The corrected delta was reviewed, and the 26 driver/offline-validator CPU
tests pass. `moe-capture-localization-20260925` is running the interleaved
500 traced / 500 untraced request comparison. No determinism fix or
qualification result is claimed from these instrument gates.

The first 500-request capture block localized three divergent steps with
complete rank coverage and no incomplete captures. The earlier routed-output
lead is now narrowed further upstream:

| Epoch | Rows | First recorded difference | Other ranks first differ |
| --- | --- | --- | --- |
| 882 | 256 | dusty, layer 8 router logits | layer 8 post-reduce |
| 906 | 254 | dusty, layer 6 router logits | layer 6 post-reduce |
| 1590 | 254 | rusty, layer 11 router logits | layer 11 post-reduce |

All four downloaded captures pass the offline identity, layout and tensor
digest checks. Dusty's epoch-882 layer-8 capture is nonmodal: input and
input-after match the modal digests, shared output and selected expert IDs
match, but router logits and selected weights differ. The other three
captures are downstream at layer 9. The first capture is therefore useful
for a real-input router-projection replay, not evidence of an expert-kernel
failure with identical routing. No input mutation was observed. Results
are in `moe-capture-localization-20260925/probe/traces-00/`, including
`capture-verification.json`. The untraced comparison completed with two
nonmodal responses in 500 requests, one each at 254 and 256 tokens. The
kernel journals for the entire comparison are clean. All four containers
were cleanly stopped and retained in
`stop-moe-capture-before-router-replay-20260925T211426Z` for isolated replay.

Independent CPU forensics against the pinned public layer-8 BF16 gate
weight (SHA256 `b98a45639ac40ddb1aba726b58fdc94300011e3cd84eaea210795095669294b0`)
reproduced the stage-reuse signature: rows 192 through 194 and columns 320
through 383 have maximum absolute error 0.156321, while p99 error overall
is 2.80e-7. Replacing activation K48:64 with K176:192 explains each row's
error with residual about 6.2e-7, whereas the next hypothesis leaves
0.26 to 0.28. This is the next occupant of the same two-stage buffer.
Receipt: `traces-00/router-forensics-independent.json` under the capture run.

The implicated router uses `bf16_gemv/_prefill.py`, not the previously
repaired `dense_gemm.py`. M256 has an exact prepared plan; M254 falls back
to capacity 8192. A one-site fence-before-release diagnostic is generated
by `prepare_router_release_probe.py`, with pinned source and patch hashes.
It has not yet passed a causal GPU replay or model qualification. No
serving repair is claimed from the static patch or CPU forensics alone.

### Router isolated replay and next full-model comparison

The captured input and real gate weight replayed with the boot's cached
`prefill` selections: M256 exact plan and M254 capacity plan share program
`31b8df526546e7d9ed35334ccc1e4f7a1dce3bd71f3a833c688d654b1165df89`.
The first 40,000 and then 400,000 unfenced calls were stable. The latter
used 100,000 calls per row count and idle/side-load arm. The one-line
fenced source also passed 400,000 matched calls, with bitwise fidelity to
the captured logits outside their corrupted rows. Both are clean: this
isolated workload does not reproduce the full-model trigger, so it cannot
establish that the fence fixes the defect.

Receipts: `router-isolation-baseline-20260925T212346Z`,
`router-isolation-baseline-20260925T212450Z`, and
`router-isolation-fenced-20260925T212904Z`. Fenced compilation used a fresh
task-local B12X/CuTe cache, with production cache mounted read-only and
the remote capture SHA matched against the verified download. Timing
interval containment exceeded 99.99%; this is not direct proof of
concurrent CTA residency. No kernel errors were reported by either arm.

The instrument-free candidate kit is prepared from `ee03505df7a3` with
unchanged vLLM tree `ce1b9bcdb91706251314d3c1ea955887c779901c` and B12X
tree `03a4e0b363d17c291516678689ae72d3e266749e`. Claude's completed
artifact review cleared the build after the input, native-inventory,
allowlist and candidate-label corrections. The build subsequently passed
after the parent control completed, as recorded below. The full-model red-capable test remains required:
matched parent/fenced boots, compared tuning configurations, concentrated
254/256-token exposure with a 258-token comparison, and a return parent
arm to distinguish a fix from a quiet boot. No promotion has occurred.

`verify_router_sass.py` checked the actual isolated replay cubins and
retained both disassemblies plus `receipts/router-stage-release-sass.json`.
The baseline has no `MEMBAR.ALL.CTA`; the candidate has one between the
last observed `LDSM` and consumer `SYNCS.ARRIVE`, with only a NOP between
the fence and arrival. This is emitted instruction-order evidence, not
proof of full-model correctness or of cross-warp scheduling guarantees.

The clean parent A1 run (`router-fullmodel-parent-a1-20260925`) produced
nonmodal signatures at all three
lengths, including 258. Therefore 258 is not a demonstrated negative
control. The test remains a comparison of event rates, with full response
and logprob receipts, rather than an assumption that a shape is immune.

This parent boot logged 160 cached and 20 bind-stage candidate measurements
(eight new sparse-MLA selections, not twenty choices).
Consequently a claim of entirely frozen boot tuning would be incorrect.
Per-rank selection snapshots are retained under
`router-parent-a1-tuning-20260925`; compare the candidate's actual winners
and distinguish new KV-geometry-dependent bind queries from changed
router or mHC selections before attributing a full-model result.

The parent A1 control completed all 3,000 cold requests: 14/1,000 nonmodal
at 254 tokens, 12/1,000 at 256, and 1/1,000 at 258. No request failed the
answer/cold-cache gate, and no unexplained five-second TTFT gap occurred.
Container and kernel log collection succeeded on all ranks, with no Xid,
NV_ERR, OOM or traceback in the request window. The mid-run and final
selection snapshots are byte-identical on every node. The run did have
thin unified-memory margin: minimum MemAvailable was 1.620 GiB on dusty,
0.566 on toby, 0.334 on rusty and 0.790 on kirby. These observations do
not prove that memory pressure is unrelated; the fenced arm retains the
same utilization and will record its own memory envelope.

The four A1 containers exited cleanly and were retained under
`stop-router-parent-a1-before-build-20260925T220715Z`. On idle dusty,
`router-release-build-20260925T220934Z` produced image
`e06df11a8ca18fa514d9f28f67cc691aef296da2eeb22b113a734519853bccd7`.
All build gates passed: native op and loader linkage, four router shapes,
eight Engram epoch cases, sixteen native enqueue cases, and twenty-nine
compressor cases. The loader directory remains
`bd733e8470bf9f229d319169`, with native SHA256
`0dbc9263cc6f09d3cdce6779533ceecd80766ee9f844f512769db544421c3b62`,
identical to the parent. The image is a candidate, not a qualified fix.

The candidate was distributed unchanged to all four nodes and booted as
`router-fence-20260925T221700Z`. Its initial eighteen cold probes passed.
The 3,000-request B1 comparison is running under
`router-fullmodel-fenced-b1-20260925`; interim clean counts are not a final
verdict. The startup journal contains NV_ERR_NO_MEMORY warnings (dusty
208, toby 38, rusty 38, kirby 60), retained separately from the request
window. This boot must not be described as kernel-warning-free.

The post-boot selection comparison retains all 1,082 parent configurations
and assignments, including 31 router and 54 mHC plans, and adds eight
KV-capacity-dependent sparse-MLA queries. KV blocks changed from 81,375 to
80,927. The comparison is therefore conditional on that disclosed capacity
difference, not a claim of identical complete execution. Cached selection
records preserve their original `programs` metadata: on cache hits B12X
reads the configuration and assignment, recompiles using a source-derived
key, and does not rewrite the record. Those historical program IDs are not
evidence of which candidate object executes; actual compilation manifests
and object hashes require a separate provenance check.

The captured first-divergence locations moved across layers 6, 8 and 11
and ranks 0 and 2. The hypothesis is a shared router kernel stage-reuse
defect, not a fixed-layer weight or state defect. Isolated replay remains
clean in both arms; only the full-model comparison can validate this repair.

`router-b1-actual-artifacts-20260925` now anchors the actual on-disk router
objects on all four nodes, independently of stale selection-record program
IDs. The collector found exactly one matching 384x5120 BF16-to-FP32 router
artifact per arm per node, with matching object digest and size. All parent
objects carry package fingerprint `bf3d4440fb35e9efd0e1c3d2d709f0bb302e146fd7afea044d89de866b0f34a4`;
all candidate objects carry `177bacb78261325fe8ba0bc56d9dd05e9adcad584ae5d364cff9ccf4cc3cc3ba`.
Device identity, toolchain, options, compile specification and non-path
environment match within each node's pair. Offline extraction/disassembly
passes on all eight objects: each candidate has one CTA fence between the
last shared-memory load and its sole consumer-release instruction; each
parent has none. This is verified disk-object provenance and instruction
order, not process-level loaded-module tracing or full-model acceptance.

Cross-image modal response signatures are not identical at 254, 256 or
258 tokens, despite identical requests apart from the cold-cache salt.
The answer text and generated token sequence agree. Across the three
output positions, maximum differences among shared top-20 logprobs reach
2.50, 3.125 and 1.75 respectively. B1's within-boot repeatability must not
be described as bitwise fidelity to A1. Return boots must retain this
comparison separately from the within-boot nondeterminism event counts;
neither the cause of the cross-boot shift nor its acceptability is proven.

B1 completed all 3,000 cold requests with zero nonmodal signatures at each
of 254, 256 and 258 tokens, and zero unexplained five-second TTFT gaps.
`check_router_repeatability.py` independently recomputes each signature
from its raw response, validates the complete cycle/length inventory, and
matches the recorded counts. All four request-window kernel logs were
collected successfully and contain no Xid, NV_ERR or OOM; container logs
contain no traceback. Minimum MemAvailable was 0.827 GiB on dusty, 1.648
on toby, 1.631 on rusty and 1.889 on kirby. This is still thin headroom.

The attention-selection confound is substantive: A1's eight newly measured
sparse-MLA choices include two BF16 and six FP8 configurations, whereas
B1's include four of each. Roles are not yet decoded, so these counts do
not prove which numerical path caused the modal shift. The router fence
is not yet causally qualified. The unchanged parent A2 return will retain
its winners and modal signatures; a controlled-capacity, same-selection
comparison may still be needed rather than assuming ABAB resolves this.

### Parent return and controlled-follow-up preparation

After retaining B1, the exact parent was restored using the recorded
candidate amendment, with all four containers stopped worker-first and
exit codes zero. A2 boot `engram-repair-20260926T001941Z` passed eighteen
initial cold probes and allocated 81,606 serving blocks. Its 3,000-request
run is `router-fullmodel-parent-a2-20260926`. The first nonmodal response
appeared at cycle 61, length 254, by request 186. The run is still in
progress; do not treat interim counts as its final result.

The offline sparse-MLA decoder labels records only on exact choice-key
hash matches. Across six earlier boots, five target precision regimes
correspond to five modal response triples. The capture boot and B1 share
both the target regime and modal triple; the unfenced capture boot still
diverged. This supports the router hypothesis but is not a matched control
because the capture image was instrumented. A2 introduces another regime:
ratio-1/2 extend FP8, their decode BF16, SWA extend BF16/h16 and decode
FP8/h8. Its modal is different again.

`prepare_router_fixed_control.py` prepares an optional 79,000-block
comparison profile without modifying active runtime files. Its five tests
prove the ordinary launch is unchanged, the two arms receive identical
controls, and other block counts are rejected. This is below the smallest
historical profiled capacity, but changes memory conditions and is not a
production serving limit. No generated patch has been applied. The full
serving qualification explicitly rejects this diagnostic environment.

If a fixed-capacity comparison is needed, first calibrate that capacity,
then use cache-hit scored boots and exact measured selections in both
namespaces, with preserved backups. Require the same eight attention
choices, no new bind measurements and matched modal triples; do not pool
its event rates with A1/B1/A2. Before spending those additional boots, the
next fenced return will run the broader original correctness battery to
look for residual defects. No causal or promotion claim is made yet.

### Completed parent return A2 and broader B2 gate

A2 completed all 3,000 cold requests. Independent raw-receipt validation
found eleven nonmodal responses: five at 254 tokens, four at 256, and two
at 258. There were no unexplained five-second gaps. All eight final log
collections succeeded, with no Xid, NV_ERR, OOM or traceback during the
request window. This does not erase the separate startup allocation
warnings. Minimum MemAvailable in the request window was 0.875 GiB on
dusty, 0.777 on toby, 0.977 on rusty, and 0.707 on kirby.

The final selection snapshot is
`receipts/router-parent-a2-final-tuning-20260926`. The comparison remains
A1 27/3000, B1 0/3000, A2 11/3000, with the previously documented attention
selection confound. B2 uses the ordinary serving envelope, not the
unapplied fixed-capacity diagnostic.

The broader battery now records eight cold repeatability trials at
131,072 tokens and four at 524,288, separately by length. The three-trial
original 524K prompt gate is unchanged. Claude reviewed the battery and
found no blocker; the separate Engram fault-injection recipe still needs
rebasing and integration before it can exercise this exact candidate.

B2 boot `router-fence-20260926T005102Z` completed on the same `e06df11a`
image and passed eighteen initial cold probes. It allocated 81,592 blocks
and reported 11,398,183 KV tokens. The four-rank selection gate passed
conditionally on capacity differences. Exact choice-key decoding resolves
all 64 retained attention records per node. This boot's eight attention
configurations match B1, including draft configurations, and its initial
254/256/258 response signatures equal B1's modal signatures. Receipts:
`router-b2-tuning-20260926`, `router-b2-selections-20260926.json`, and
`router-b2-regimes-20260926.json`. The broader battery is running under
`router-b2-full-qualification-20260926`; interim success is not qualification.

B2's content-transition gate and the complete 22-length, 1,100-request
residual stress passed with zero nonmodal responses and no unexplained
five-second gaps. The driver advanced to the further boundary, semantic
and long-context gates. This is additional short-context evidence, not a
completed model qualification.

The additional 28-length, 168-request repeatability matrix passed, as did
the semantic, vision, tool-round-trip and concurrent checks. The first
original 524K cold trial then failed answer correctness: it returned the
archive identity followed by `739184`, not `739184, 482617`. It took
210.44 seconds with zero cached tokens. The second trial took 183.43
seconds, was also cold and wrong, but its full response/logprobs exactly
matched the first. The third trial is still running. Do not infer numerical
correctness or change the historical gate from this interim stability.

The third original trial finished in 181.81 seconds: also wrong, cold,
and bitwise-identical. Independently recomputing the full response and
top-20 logprob signature gives
`236fea27c34c43a4d745984b15af265f8a8ca2a84ea34fa77e03e5b7071cbdc8`
for all three. The full qualification exited 1 as required. Every final
container/kernel log collection succeeded; there were no request-window
Xid, NV_ERR, OOM or engine tracebacks on any rank. Python tracebacks in
the local driver report the intentional failed correctness gate, not an
engine crash.

This establishes observed repeatability for these three cold 524K trials,
not absence of every rare race or proof of numerical correctness. The
historical answer failure remains a promotion blocker. The separately
labelled `diagnose_router_long.py` continuation checks the already frozen
single-needle prompts at 131K and 524K, without altering or replacing the
failed qualification. No benchmark has started.
