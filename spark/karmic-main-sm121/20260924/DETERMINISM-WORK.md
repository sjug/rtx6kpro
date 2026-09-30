# DS4.1 determinism investigation, September 24/25

User authorization: continue on dusty/toby/rusty/kirby through root-cause fix,
review, rebuilt image, deployment and qualification. GLM remains untouched.
No permission to post externally is inferred. No determinism result alone
qualifies an incorrect retrieval answer.

## Feedback loops established

### September 25 diagnostic capture integration repair

The unarmed diagnostic image `5eb584a63115` reproduced the same-content-length
predecessor effect in two cold cycles: after the meadow/code-826493 predecessor,
the original 385-token prompt produced `3a9bf9330d15`, then `a5e03661674c` twice.
All answers were correct. See `receipts/stale-probe-unarmed-20260925/`.
This is evidence of history dependence, not completed determinism qualification.
The new cache namespace retuned 112 of 152 dense assignments, so comparisons
must remain within this boot. The earlier 296-case dense gate covers the earlier
boot's winners, not these newly selected assignments.

Arming the capture then exposed a diagnostic-helper bug, not a new GPU fault:
the helper treated `UniformTypeKVCacheSpecs` as an individual layer geometry.
The resulting Python exception terminated the engine before any snapshot.
All controls were disarmed and all four failed containers retained; see
`receipts/stale-capture-observed-20260925/` and
`receipts/stop-stale-probe-group-spec-failure-20260925T152204Z/`.

Claude's subsequent source audit identified three further geometry errors:
page-size padding is not record payload, manager block size need not equal
kernel block size, and the per-layer metadata table is the kernel's actual
table. The repaired helper resolves individual specs, uses their
`state_content_size_bytes`, requires one head and exact row payload size, and
uses metadata block size and table. The group table is retained separately.
Ten CPU Torch tests pass inside the pinned image, including actual vLLM
page-size unification and cache-view creation for padded shared-layer storage
and subdivided manager blocks. These are diagnostic integration tests, not
proof that the serving determinism defect is fixed. Review and rebuild remain
pending at this entry.

Follow-up: build `47caaa211ce1` passed the installer/native-preservation gate
and all twelve CPU Torch tests on idle dusty. Lock SHA is
`c584b99194d993a81d2df70df18d82c6793c5479a64fb95c753d3f73682457b5`.
The additional tests cover real ring and MLA storage-block geometry, slot
bounds, all-layer validation before writes, and a read-only capture failure
that leaves serving alive while emitting an explicit failed receipt. The
capture driver's seven tests pass after freezing the new image identity.
The earlier pre-freeze driver run correctly rejected modified inputs against
the old lock; it is not a passing receipt. Distribution is in progress.
Observe now fails the diagnostic without aborting the request on Python
validation errors; write modes still raise. Device errors are re-raised.
The helper's introductory fail-closed prose predates this narrower behavior;
the executable behavior and the failure-path test are authoritative.

The prompt-logprobs control answered but its receipt was rejected because
the standard cache-counter gate expected queries. Pinned `sampling_params.py`
intentionally disables cache reads for prompt logprobs. A separately tested
expected-zero-query accounting path now covers this diagnostic only; the
ordinary cold-request gate is unchanged. That control still needs replay.

- Current-image stock oracle: all selected sets valid, tied-score sets vary.
  Receipt: `receipts/topk-baseline-20260925.json`.
- New deterministic reference regression fails stock at width 513, equal scores.
  Receipt: `receipts/topk-stock-regression.log`.
- Three-pass threshold repair passes 26 GPU cases, including graph replay,
  524288-wide rows, short/empty lengths, overflow tie masses, BF16 scores,
  and mapped candidates with k512/k1024. It changes no input scores.
  Receipt: `receipts/topk-repair-regression-mapped.log`.
- Extended test's initial k32 case was invalid for B12X and failed explicitly.
  It was corrected to supported k512/k1024; the failed receipt is retained.
- Full-server stock baseline: frozen prompts at 256, 513, 514 and 1024 tokens,
  four cold repeats each, all lengths vary in full returned logprobs. Cache
  counters verify zero hits. Receipts: `receipts/determinism-baseline/`.

## Current execution

The initial MoE-only deterministic control attempted the same original image,
TP4, K7, stock NCCL, 600K context and utilization 0.85. The only behavioral
change is `B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1`. The optional launcher input
is frozen in `determinism-profile-amendment.json`; model arguments are unchanged.
The four baseline containers were gracefully stopped worker-first, verified
exit zero/no OOM, renamed and retained. Driver receipt:
`receipts/moe-control-20260925T013827Z/`.

That boot failed in B12X preparation: the dynamic MoE declares its main kernels
but omits the ordered reduction subsequently executed when deterministic output
is enabled. The isolated `test_moe_dependency.py` reproduces the missing real
program identity on stock. The declaration-only repair in `moe-dependency.patch`
passes that GPU regression for both ordinary and FP32 output variants.

The failed diagnostic containers are retained as
`ds41-flash-karmic-main-tp4-moe-preparation-failed-20260925`. Dusty's failed head
did not terminate within `podman stop -t 60`; Podman used SIGKILL and the head
eventually exited 137, OOMKilled=false. No serving request ran in this boot.
All original observers ended; there were no systemd units to remove.

Dependency-only image `d880b1297eda5d788bd45c2f85a0b42e9dbcdde856117a10ec6ec3f92d7b5937`
was built on idle dusty, passed source/native-preservation and GPU dependency
gates, and was distributed as one plain Docker archive over switched 200G.
All three receiver image IDs match. The repaired MoE control booted and ran
the short cold-repeat matrix automatically. All 30 answers were 739184, but
all five lengths varied in full-response logprobs. At 513 tokens the first-token
top-20 payload was identical across six repeats, while later tokens varied;
at 256 tokens even the first-token payload varied. This is not the selector
fix and is not promoted. Receipt: `receipts/moe-control-repaired-20260925T015720Z/`
(the exact directory is also printed in the driver log). The driver's finally
block stopped its four observers and retained container/kernel logs.

`restart_determinism.py` watches startup and immediately runs six cold repeats
at 256/513/514/1024/16384, without waiting for an interactive continuation.
Read-only one-second telemetry is active for all four nodes. No systemd unit
was created. Observers must be stopped when this arm completes.

## Patch development

`deterministic_topk.py` is a diagnostic repair, not yet a deployed fix. It takes
the exact threshold from the production selector and deterministically picks
the lowest logical candidate positions among equal scores. Candidate positions
are then remapped through the original gather table. This is deterministic
but is not a claim of lowest physical/global index ordering for arbitrary maps.
The existing MXFP4 sort retains final ordering responsibility.

`prepare_deterministic_diagnostic.py` generates the two-file source overlay and
independently reconstructs the baseline and modified Git trees without writes
to the B12X checkout. The diagnostic Dockerfile and installer are prepared,
not built or qualified. Native artifacts remain untouched.

Claude is independently reviewing the diagnostic and developing an efficient
in-kernel tie-break with tests. Its first source review recommends covering the
overflow path and global-index mapping and identifies dense split-K atomics
and the unchecked Engram timeout as separate issues to test, not established
causes of this specific failure. Actual DS4.1 prefill dispatch must be traced;
generic tiled-indexer behavior must not be mistaken for the active MXFP4 path.

Claude's first kernel patch passed its isolated GPU quick suite, 1100 checks,
zero failures (`receipts/claude-topk-gpu-first.log`). It is not ready to ship:
review found `_exact_overflow_fallback` gained two required arguments but its
caller in `fused_indexer.py:1461` was not updated. Claude's revised patch keeps
that shared helper byte-identical and gives the tested kernel a separate stable
helper. Local identity and unchanged-helper checks pass. Its revised GPU gate
is still pending. The corrected source dispatch is:
`dsa_indexer/api.py:218-228` directly routes `select` to
`select_mxfp4`, including the serving call at DS4.1 `attention.py:1482`.
The check script now always owns its temporary directory instead of accepting
an arbitrary environment-provided path that it would recursively delete.

Revised kernel passed 9095 isolated GPU checks, then the expanded in-image
matrix passed 10598 checks (including mapped width 16384 and block width 75000).
No failures. The combined image is
`30ef9c2d529e9df0a3ea348c773d8ff55284389a90a40922ac1b0d93ccffdd9b`,
B12X tracked tree `33ecd859e9ab087449b230409c835c5552c5336f`.
Its installer checks the 1215 tracked baseline files with the declared MoE
exception, installs the one selector file and preserves native objects.
Build receipts: `receipts/determinism-build/`.

All four image IDs match after plain Docker archive distribution over f0
switched fabric. Archive SHA256:
`703a90c8cba02d9decb6af6172646e0509758038eb6851430a550fb04ab2b66c`.
All completed-control containers were stopped with exit zero and retained
worker-first. The combined profile is starting with deterministic MoE enabled;
split-K turbo and Engram overlap are still upstream settings, to isolate the
selector change before further discriminators. Driver:
`receipts/combined-determinism-driver.log`. This image has build-gate approval,
not serving qualification or promotion.

## Combined-image serving evidence

The combined boot completed at approximately 02:31 UTC September 25, with
11,191,012 KV tokens. Six cold repeats at 513, 514, 1024 and 16384 tokens
returned byte-equal full response/logprob payloads. The initial 256-token
series varied for its first two requests, then its remaining four matched.
All answers were correct. This is progress, not a full determinism pass.

Without restarting, a second six-repeat sweep was exact at 128, 255, 256,
511, 512 and 513, but varied at 257. Further probes were variable at every
tested length 257, 258, 259, 260, 263, 264, 265, 272, 288, 320 and 384;
448, 510, 768 and 1025 were exact (four repeats for the wider sweep).
The same frozen messages and zero-cache-hit counters are retained in
`receipts/combined-short-boundary/`, `receipts/combined-near257/` and
`receipts/combined-wide-boundary/`. No root cause is assigned to this range
yet. Long cold-repeat probes are running before the next configuration arm.

The actual checkpoint configuration is retained in
`receipts/pinned-model-config.json`: hidden width 5120, intermediate width
2304 (576 per TP4 shard), 384 experts and top-k **6**, not 8. Any reasoning
using eight selected experts for this checkpoint is invalid.

Three cold 131072-token requests were exact; the first two cold 524288-token
requests also match in their full response/logprob records. The third is still
running. These use the frozen single-code instruction, not the historical
ambiguous dual-needle instruction, whose unchanged replay remains required.

The first 524K request took 413 seconds and the second 424 seconds. A period
of low GPU power and unchanged logger counters initially suggested a stall,
but completion disproved a permanent hang. Host memory was thin (approximately
1 GiB MemAvailable on dusty with approximately 5 GiB swap in use). Native
debugger snapshots were attached/detached during the first request and a CUDA
snapshot around its completion/next request; these are not clean performance
receipts. No kernel fault was observed. The snapshots did not identify a
faulting kernel. Do not infer a selector deadlock from this episode.

Claude's completed source review found an exact live-row prefetch cutoff at
256 but no proven cause of the upper boundary. Its claim that logprob spread
alone excludes accumulated rounding noise is too strong: the multi-layer
model can amplify small changes. Both a timing-dependent race and another
non-deterministic operation remain hypotheses. The next controls preserve
that distinction: upper-edge and first-token-only probes on this boot,
followed by one explicit prefetch-off arm. Dense split-K is not currently
the leading explanation for first-token variation at these larger row counts.

A separate fixed-input PyNCCL discriminator is prepared with the serving
2.30.7 library and transport environment, intra-rank repeat checks, cross-rank
hash equality and interleaved message sizes. It will run only after serving
stops, avoiding additional communicator allocations in the thin live margin.
A pass cannot exclude every interaction in the serving process.

The long arm completed: three of three cold 131K and three of three cold 524K
records were bitwise equal. The third 524K request took 227 seconds, reinforcing
that the earlier latency is not a stable throughput estimate. The unchanged
historical dual-needle input is still outstanding.

Automatic follow-ups then ran without a restart. At 385 tokens four repeats
varied; 400, 416 and 447 were exact. At 300 tokens, ten `max_tokens=1` requests
varied, and another ten after one 256-token request also varied. This puts the
remaining effect in prefill, without needing draft/decode execution to explain
the first output record. Receipts: `combined-upper-edge`,
`combined-prefill-only`, `combined-cross-middle`, `combined-cross-after`.

Claude reviewed the next control kit. The missing verbs-device access and
library-path replacement in the standalone NCCL probe were corrected to mirror
serving. `freeze_short_controls.py` verified all four previous launcher hashes,
preserved their exact text and prior manifest in
`receipts/short-controls-amendment.json`, then updated only the launcher hash.
The optional prefetch-off input is binary-validated and remains absent by
default. The completed serving arm is being stopped and retained before these
tests. GLM remains untouched.

The isolated communication test passed on all four ranks. It used the exact
candidate, stock NCCL 2.30.7 with the declared library SHA, both f0 verbs HCAs
(NET/IB confirmed), the serving rank order and no symmetric-memory path.
There were 64 measured repeats per shape, widths 5120 and 6144, twelve row
counts from 128 through 512, plus interleaved 256/257/384/448 sequences.
All outputs were repeatable and all rank baselines matched. Receipts:
`receipts/nccl-repeatability-stock/`. Probe containers exited normally and
remain retained. This clears this isolated test, not every serving interaction.

The prefetch-off control completed under the same image and deterministic
MoE setting, changing only `VLLM_DS41_L2_PREFETCH=0`. Six cold repeats each
at 128,255,256,400,448,513,16384 were exact; 257,300,320,384,385 remained
variable. The failing band is unchanged, so disabling prefetch does not fix
the residual. This does not exclude every timing interaction. The automatic
driver finished with REPEATABILITY-FAIL and cleaned up its observers; the
four serving containers remain available for diagnosis. Receipts:
`receipts/combined-prefetch-off-20260925T030643Z/repeatability/report.json`
and `receipts/combined-prefetch-off-driver.log`.

First-token-only probes sharpen this result: eight repeats at 256 were exact;
257,383,384 varied; 385 and 386 were exact. The earlier 385-token failure was
in an eight-output-token comparison and must not be treated as a first-token
boundary. Four repeats at 386,388,390,392,393,396,399 were also exact. Changing
only the filler to ` paper` in a separate frozen corpus retained variation at
300 and 384, while 256,385,386,400 were exact (six repeats). Receipts are
`prefetch-off-first-token-edge`, `prefetch-off-edge`, and `prefetch-off-paper`.

The pinned vLLM `b12x_layers.py` includes CED capacities 128,256,384,512,... in
`_execution_capacities`; `_block32_linear` selects with `bisect_left`. Thus
257 through 384 select the 384-row block-FP8 plan. This corrects the earlier
claim that no plan boundary matched the band. It is a source-level lead, not
an attribution. Claude is checking the corresponding lowering independently.

A separate attention-boundary diagnostic is prepared and reviewed. It changes
only the existing opaque attention wrapper to clone inputs, positions, and
outputs, with a runtime control file enabling one named 300-row capture.
There is no per-layer host synchronization; snapshots copy to CPU after layer
39. Full bytes are compared per rank, not an unweighted checksum that could
miss permutations. Clones can perturb allocation and timing, so the same
repeatability failure must survive before interpreting localization. The
diagnostic uses its own cache namespace and refuses snapshot activation if
CUDA-reported free memory is below twice the conservative snapshot footprint.
The completed prefetch-off containers are being stopped and retained first.

The tracing build initially failed on dusty's inherited context ignore file;
an explicit small allowlist corrected that. Independent local builds then
produced different image IDs despite matching source and labels. That gate
failed before launch. The build script now builds only on dusty and uses the
existing unchanged Docker-archive transport. All four nodes verified canonical
image `335a15d1ffed84e0203d0105032c143d68fdb1ab85b56e480ebcd813afcb31fe`.
Failed build receipts are retained. The new cache namespace triggers a fresh
tuning run, so its decisions must be compared as well; do not attribute a
changed result to instrumentation alone. The actual combined-image selection
cache is retained as `receipts/combined-tuning.json`, separately from the older
MoE-control cache Claude initially decoded.

While the archive was prepared and distributed, the isolated block-32 probe
ran on idle rusty. Shapes 5120->1792, 5120->1152, and 6144->25600 each ran six
row counts (64,65,257,300,320,384) under four capacity/tactic configurations,
64 repeats per case. Every result was exact, including swapped tile_n64 and
the unswapped/tile_n128 controls. Repeating the entire matrix with scratch
poison cycling through 0,127,255 also produced identical logical outputs.
This is 4,608 calls per full matrix, and does not reproduce the serving defect.
Receipts: `block32-tactic-quick.log`, `block32-tactic-full.log`, and
`block32-tactic-poison.log`. These are synthetic fixed-input tests, not proof
that the real model's operator inputs or cross-operator lifetimes are safe.

## Attention boundary trace, 2026-09-25

On the diagnostic image's fresh tuning cache, 256-token first outputs differ
while 300-token first outputs match (later decoded tokens still vary). This
supersedes a fixed 257-384 shape interpretation: the failure depends on more
than token count alone. The current tuning record is preserved in
`receipts/trace-tuning-before-ops.json`.

Three cold 300-token forwards have identical attention inputs and outputs at
all 40 layers on all four ranks. Two cold 256-token forwards first differ at
the input to attention layer 3, identically on every rank: 82,426 differing
elements, maximum absolute difference 1.12371826171875. Attention layers 0-2
match exactly, but the initial trace did not capture their residual branches.
It therefore does not exclude an earlier residual-only divergence.

The next diagnostic retains the same image and tuning cache and adds a
hash-pinned, read-only helper overlay. It captures mHC boundaries at layers
0-3, and the routed and shared MoE outputs after their stream join. Predictions:
identical mHC inputs with differing outputs implicate mixing; identical FFN
inputs with differing routed outputs implicate routed execution; differing
shared outputs isolate the shared branch. MoE arguments captured after return
are labeled accordingly, not claimed to be original inputs if mutated.
The original cold-response reproducer remains necessary because tracing can
perturb memory layout and timing. This is diagnostic instrumentation, not a fix.

Receipts: `receipts/attention-trace-requests/`. Additional dedicated 1 Hz
observers started near the end of collection; do not claim full telemetry
coverage for every traced request.

The first operator-overlay boot retained 168 tuning choices, with zero
measurements and compilations, and reproduced the 256-token variation. Its
snapshot guard refused all four ranks: CUDA free bytes were 1.09-1.58 GB,
below the conservative 1.68 GB requirement. No tensor trace was saved and
this is not a passing diagnostic. Receipts are in
`receipts/combined-prefetch-off-20260925T040551Z/` and
`receipts/ops-trace-requests/`. Driver allocation warnings recurred during
startup, recorded in the final per-node logs, without a model process failure.

The reviewed replacement helper saves only attention layers 0-4 plus mHC/MoE
windows 0-3, with an explicit engagement check and actual snapshot byte count.
The 256-token snapshot bound is 503 MB and requires twice that free. No
serving utilization, model length, tuning selection or arithmetic is changed.
The full refused helper and manifest are retained as `receipts/ops-full-*`.
The narrowed helper SHA is
`2f25546f9ec0dac9bbf58f5431ab2b7f81b675349e0360dc65fbb6902c0f88f3`.
Current upstream-derived runtime JIT monitoring is `warn`, as logged; do not
describe these current boots as the historical R38 strict-JIT profile.

## Engram boundary localized, 2026-09-25

The narrowed 256-token trace was still refused by its memory guard. On the
same boot, smaller requests exposed the failure without reducing utilization
or weakening the guard. The 160-token trace used 215,950,944 snapshot bytes
against a 314,572,800-byte bound on rusty, with 824,012,800 CUDA bytes free
before capture. Receipts: `ops-small-trace-requests/`, `small-ops-observed/`,
`short-bucket-repeats/`, and `short-bucket-observed/` under `receipts/`.

Across three cold 160-token forwards, rusty first differs at the input to
`run_pre` before attention layer 1: 1,256 residual elements, maximum absolute
difference 0.001953125. The preceding `run_post` output and both MoE layer 0
outputs are exact. Other ranks first differ at attention 1's output. At 129
tokens the corresponding first residual difference is on dusty (97 elements,
same maximum difference). This places the first captured divergence across
Engram layer 1, before attention propagates it through the TP collective.
It does not yet distinguish lookup/communication, projection, or mix.

The next hash-pinned helper captures Engram all-reduce input after the existing
lookup wait, all-reduce output, replicated projection input/output, mix
inputs/output, and lookup epochs after completion. No pre-wait staging reads
are added. The prior helper and manifests are retained as `receipts/pre-engram-*`.
The small-input serving reproducer remains the regression signal; instrumentation
can perturb timing, so these traces alone will not validate a fix.

## Projection isolated in serving, 2026-09-25

The extended trace (`receipts/engram-ops-trace-requests/`) captured 160x3,
129x2 and 192x3 cold forwards. All Engram all-reduce results are bitwise
identical across ranks and repeats. At 192 rows, toby's first captured
projection output differs from its next two and from the other three ranks:
496 BF16 elements, max abs 0.0386962890625. The changed region is rows 96-111,
columns 656-671 and 720-735 (31 changed values per row). The projection source
is exact; mixing receives those same changed values. The first captured
request is not the first execution: six untraced 192-token requests preceded it.
Prefetch remains disabled throughout this arm.

The lookup ready and expected epochs match in each capture, but the sticky
failed word is 1 on every rank in every request. Its first transition was not
captured, so this cannot certify timeout-free lookup. It does not account for
different projection outputs from identical captured rows in this case.

The isolated capacity-256 projection test covers the measured swapped 64x128
tactic and an unswapped 128x64 control. Synthetic inputs/weights at lengths
128,129,160,192,224,255,256 are exact over 64 repeats each. The real captured
192-row input with actual checkpoint layer-1 Engram weights reproduces the
good serving output exactly in all 64 calls for both tactics, including a
scratch-poison cycle of 0/127/255. Logical quantized values/scales are exact;
MMA-layout logical scale bytes equal compact scale bytes.

Checkpoint tensor SHA-256: weight
`44d799d6444a3f728ae96c873c8199bdff49cafbc0b5d703a0bf07be948d0447`,
scale `2c395fc76e65fad2f33e5a3d8f532f40dc232754a0e637e42ca89a674e3b44e0`.
Good output SHA-256:
`29b5a61994edf5da3c6c7ba4c70a22b444fb42caf6a3a87fdad7bf1856dbaabb`.
This localizes the observed failure to the serving projection path but does
not yet identify its internal mechanism. Receipts: `engram256-*.log`.

The next helper adds quantized-value and both scale-layout byte snapshots
after the projection call, with no arithmetic change. Its SHA is
`6971fd0ed8b8223fff78062d50817e44377d88b2c25db9399db57dbf122ab7b1`.
Physical padding bytes may vary without numerical relevance; compare logical
MMA scale entries separately before attributing a failure to those bytes.

Unfiltered Compute Sanitizer racecheck with tensor-operation checks enabled
completed three actual-input/actual-weight calls with zero reported hazards;
all three matched the good serving output. Receipt:
`receipts/engram256-trace-checkpoint-racecheck-all.log`. This isolated pass does
not exclude a serving-only race. The earlier invalid filter invocation and
filtered run are retained, not counted as equivalent coverage.

The quantized-trace boot again fails first-token repeatability at both 160 and
192 tokens (five of five comparisons differ at each length). Eight instrumented
requests completed; per-rank CPU tensor comparisons are in progress. Offline
comparison now uses one Torch CPU thread to avoid thread-launch overhead; the
serving process and compilation parallelism are unchanged.

The eight quantized captures completed (`receipts/quantized-engram-trace-requests/`).
At 129 rows, toby differs by 962 projection elements (max abs 0.50390625), rusty
by 512 (max abs 0.078125). Both have identical projection source, quantized values,
compact scales and complete physical MMA-scale buffers across their two repeats.
Logical MMA scales match compact scales on all four ranks. Thus neither changing
quantizer output nor changing physical padding bytes accounts for these within-rank
differences. Physical padding does differ across ranks, which is not itself evidence
of incorrect consumption. Pointers are unchanged between repeats. On toby the
differences occupy rows 96-127 and two 16-column pairs separated by 64 columns;
this strengthens a tile-local execution hypothesis but is not a proved mechanism.
The 160/192 captures have exact first-five-layer boundaries in this boot, despite
untraced first-token variation. No NVRM, Xid, OOM or traceback appears in the
retained trace-window logs. Owned observers stopped on completion.

## In-process dense replay, 2026-09-25 05:36 UTC boot

Helper `0b769e53487cf794aaecb0231385580d5a1e49c4044b3a218e579bfa784f81a9`
adds two diagnostic executions of the prepared Engram projection without replacing
the model's output: immediate replay and replay after device synchronization.
It records the compiler program key, weight prefix and complete weight-scale
plane, and resets the lookup failure flag before the traced request's wait.
Claude reviewed the helper and byte bound. Synchronization changes scheduling of
later operators, so causal comparisons here are confined to this projection.

Untraced 129/160/192-token repeats all fail repeatability. All nine traced
requests completed and retained tensors on every rank. The full-trace CPU
comparison exceeded its 120-second timeout on dusty/192; its exact orphaned CPU
process was terminated. No inference was rerun. The focused memory-mapped
Engram comparison recovered all twelve rank/length reports in
`receipts/replay-engram-trace-requests/*-engram-analysis.json`. Its scope is
explicitly not a full-layer comparison.

All four ranks have differing direct dense replays, including after device
synchronization. On dusty all four synchronized M129 replays differ from the
original, with 379-1140 changed BF16 elements and max absolute difference
0.04296875-0.265625. Original M129 output on dusty is identical in all four
requests. Captured operands and weight prefix/scales are stable. All 36 traced
rank/request lookup failure flags are zero, with ready epoch equal to expected.
The prior sticky failure flag was one on the first request, so older snapshots
must not be described as proving request-local timeouts.

Observed dense tactic remains swapped 64x128x128, splitK1, expected_m=256,
live_m=129/160/192. Dusty's program key is
`68f67353dbff5c4b5760c3c24ce3a50624366e2e4735b23440b6c751a1b82069`.
This strengthens a kernel-execution hypothesis; it does not prove the exact
ordering defect. Earlier isolated passes did not match the full compiler
environment and cannot be used to exonerate this exact binary.

Offline fitting of dusty M129 replay 0, faulty CTA (1,14), verifies the checkpoint
weight prefix and every logical scale. The strongest fit is a scaled final K32
group, but residual 0.528 exceeds rounding floor 0.128 substantially. No stale
stage or missing-group explanation is established. Receipt:
`receipts/replay-engram-129-0-dusty-cpu-fit-checkpoint.log`.

All four servers stopped worker-first with exit zero and were retained under
`replay-engram-captured` for an idle-GPU exact-binary reproducer. GLM untouched.
Stop receipt: `receipts/stop-replay-engram-captured-20260925T055036Z/`.

## Exact standalone dense reproducer, September 25

All four DS4.1 serving containers were stopped cleanly and retained before
these tests. Only an isolated GPU container on dusty runs; GLM is untouched.
`run_engram_dense_replay.py` uses captured M129 operands, the checkpoint
Engram weights, capacity/expected M256, and the unanimous four-rank original
output as reference. No lookup, quantizer, collective, or model runs inside
the replay loop. Operand bytes are checked again afterward.

The original program key is `68f67353dbff5c4b5760c3c24ce3a50624366e2e4735b23440b6c751a1b82069`;
object SHA is `96f193593e9474f686e1f6dc41c7d19d148d7768ea84c8751aebf99beb6d04de`.
Fresh compilation reproduced that object byte for byte and reproduced the
fault. Stale cached code is not required. Except the first baseline, calls
are separated by 50 ms. Receipts are `receipts/engram-dense-isolated-dusty-*`.

| Independent arm | Incorrect / calls |
| --- | --- |
| Original, no delay | 1 / 64 |
| Original, 50 ms | 26 / 256 |
| Fresh same-source compile | 24 / 256 |
| Warp barrier before releases | 13 / 512 |
| Always wait, no peek bypass | 1 / 256 |
| Unroll two rather than four | 16 / 256 |
| Release final stage after final MMA | 84 / 512 |
| Duplicate constant-zero initialization after initial wait | 56 / 512 |
| Remove register reallocation calls | 4 / 512 |
| MMA-group barrier after initial wait | 37 / 512 |
| Runtime-derived zero after initial wait | 0 / 512 |
| Same runtime-derived-zero object, repeat | 0 / 2048 |
| Runtime-derived zero before initial wait | 0 / 512 |
| Original binary return control | 61 / 512 |
| Negative literal zero initialization | 71 / 512 |
| Output columns 0:2048 only | 0 / 512 |
| One work tile per CTA, otherwise identical SASS | 0 / 512 |
| Grid 300, two work tiles per CTA | 61 / 512 |
| All stage releases moved after MMA, initial run | 0 / 512 |
| All stage releases moved after MMA, confirmation | 1 / 2048 |
| MMA-group barrier before each stage release | 0 / 1024 |
| Same group-before-release object, confirmation | 0 / 4096 |
| Group barrier at main-loop release only | 24 / 1024 |
| Group barrier at final-stage release only | 0 / 1024 |
| CTA acquire-release fence before both releases | 0 / 1024 |
| Same fence after both releases, negative control | 39 / 1024 |
| Epilogue barrier | 84 / 512 |
| Original binary, NaN-initialized output | 1 / 512 |

The NaN-output arm reproduced a numerical mismatch at repeat 233 with zero
NaNs, including on the failing call. This weakens the proposed missing-store
explanation; it is not evidence that output initialization fixes the defect.
The source warp-sync diagnostic lowered to a NOP, not a WARPSYNC instruction,
so it did not test an actual warp synchronization. The 4096-call confirmation
of the group-before-release barrier passed. The output-only replacement fit
on the NaN-arm failure found no other-tile match (best residual 0.858).
SASS for the group barrier contains `BAR.SYNC.DEFER_BLOCKING 0x1, 0x100`
before release. The final-stage-only result is consistent with a work-tile
transition hazard but does not yet identify the racing accesses. Fence-before
passed while fence-after failed; their SASS retains the intended relative
`MEMBAR.ALL.CTA` / `SYNCS.ARRIVE` placement. A 4096-call confirmation on Kirby
is running. This supports an ordering defect at release but does not establish
which in-flight accesses supply the corrupt values. No fix is qualified.

The earlier runtime-derived-zero change is diagnostic, not a root-cause fix or release.
It uses `alpha_value * 0.0` and changes code generation from paired CS2R
zeroing to scalar MOVs, as well as register allocation and instruction
scheduling. The SASS has no SETMAXREG instruction in the baseline. A source
change that removes the corresponding calls therefore cannot on its own
attribute the defect to hardware register reallocation. No initialization
mechanism is established by those runtime-derived-zero results.

The first-MMA-overwrite diagnostic cannot compile: the installed MXF8
trait accepts SFA/SFB but not `WarpField.ACCUMULATE`. This is a rejected
diagnostic, not a numerical failure. The local DSL also defines a separate
trait supporting that field; inspecting only that trait was insufficient.

Baseline faults occupy first persistent CTA work tiles; the unroll-two arm
has one later-tile exception. Row 128 is valid, so the missing M tile 2 is
not automatically explained by masking. New receipts record exact faulty
rows and columns. Instrumented racecheck with the exact original object
passed eight calls and reported zero hazards, but did not reproduce the
fault; this is not proof of race freedom. Cubin/SASS extraction is retained
alongside the original host object. No variant is deployed to serving.

## Dense stage-release attribution, 2026-09-25

Two independent real captures now identify the overwritten input, rather than
only a passing timing perturbation. The H12 fitter substitutes the next
persistent work tile's activation values into the last K block while retaining
the current weights and scale bytes:

| Capture | Best replacement residual | Rounding floor | Next-best residual |
| --- | --- | --- | --- |
| dusty, 129 rows | 0.0333 | 0.1277 | 0.528 |
| rusty, 160 rows | 0.0456 | 0.2309 | 0.755 |

Both select K block 47.3, with activation K6112..6143 replaced by the next
work tile's K352..383. This is the producer refilling stage 2 before the
consumer's last shared-memory fragment load has completed. The original SASS
places the consumer release immediately after that LDSM, without a fence.
Fencing before both releases passes; fencing after them fails 39/1024.
This supports the kernel-level stage-reuse ordering defect. Attribution below
that level, to hardware, compiler lowering, or DSL semantics, remains open.

The proposed fix is exactly two `cute.arch.fence_acq_rel_cta()` calls, before
the main-loop and hoisted-tail consumer releases, without architecture or
shape gates. It changes neither arithmetic nor tactic selection. Kirby's
129-row confirmation passed 4096 calls, and Toby's 160-row NaN-output check
passed 1024. The two-node 129/160/192 matrix is still being completed.

The candidate will use a fresh compilation/tuning namespace. Autotuning stays
enabled; its selected winners must be compared explicitly against
`receipts/combined-tuning.json` before interpreting serving results. No old
timing selections are silently copied into the corrected source namespace.
Normal image tagging after build gates is not model promotion: the build
receipt and image label retain candidate-not-qualified status until the full
serving gates pass. No production deployment currently uses the DS4.1 tag.

## Built fence candidate and broader decode finding

The complete two-node matrix passed: six cells of 4096 calls, zero mismatches.
Image `3413799408e09ad3a1bb4ea2297384fa03927ca8b4e8e72f2af61aef8951d54c`
was built from the clean repair base. Its MoE declaration test, 10598 DSA
checks, another 4096-call dense replay and the five-arrival SASS gate passed.
Build receipts: `receipts/dense-release-build-20260925T132954Z/`.
An initial preflight attempt rejected Kirby's older receipt because it lacked
the subsequently added `reference_key` field. The compatibility case is
limited to that 129-row receipt and still requires its exact reference digest.

The broader shared-kernel gate is not a pass. Four-way BF16 atomic split-K
decode configurations change output bytes across identical calls. This is
separate from the repaired single-slice stage-reuse fault. Two-way atomic
Engram cases also exceed the harness's sampled FP64 error tolerance; that
tolerance finding is not by itself a new model correctness defect.

The existing `B12X_DENSE_SPLITK_TURBO=0` path stores FP32 partials and reduces
them in a fixed order. It supports at most two slices. The next control uses
that path, preserving each recorded decode geometry except that four-slice
winners explicitly become two-slice controls. This is not a new autotuning
receipt: `receipts/non-atomic-control-tuning.json` records the original and
control assignments and source digest. The candidate has not been deployed
or promoted while this additional determinism exposure is investigated.

The broad sweep completed all 259 cases with 31 failures: 27 repeatability
failures and four sampled-accuracy failures. The non-atomic control then
passed all 105 cases across 56 programs, with 64 repetitions per case.
Receipt: `receipts/dense-release-shared-3413799408e0-nonatomic.json`.
This establishes kernel-level repeatability and sampled accuracy for that
control, not yet full-model determinism or qualification. Fresh serving
autotuning with turbo disabled must still select and validate its own winners.

Claude's follow-up found 19 omitted records for the 15360-to-5120 draft
projection. Enumeration now includes that shape and rejects any unmatched
dense configuration instead of silently skipping it. The expanded fixed-order
control passed all 120 cases across 64 programs, 64 repetitions each:
`receipts/dense-release-shared-complete-3413799408e0-nonatomic.json`.
The earlier 105-case result remains valid only for its narrower coverage.
The 37,356,349,952-byte unchanged Docker archive was copied across the fabric
and image ID 3413799408e0 verified on all four nodes. The monitored boot is
`receipts/dense-release-20260925T140235Z/`.

The planned repaired serving profile explicitly sets deterministic MoE output
to 1 and dense split-K turbo to 0. The start driver clears remote-shell
prefetch/turbo overrides, preflights all four nodes before any launch, and
the qualification driver independently checks the effective container values.
It records and rechecks container ID, image, kit, start time and restart count.
The candidate pin now names the build-gated image without the trace overlay.

The original-needle request preserves the historical prompt but adds cold
cache salts and logprob reporting. At the pinned vLLM commit, sampler.py
computes logprobs out of place and greedy tokens by logits.argmax. In
rejection_sampler.py, apply_sampling_constraints returns logits unchanged for
all-greedy requests (lines 556-557), and the greedy rejection path uses their
argmax (lines 473-488); the logprob-dependent return_probs branch is not used
there. This supports unchanged token selection semantics, not a claim that
the extra GPU work could never expose a separate timing-sensitive bug.
The active V2 path was checked separately: gpu/sample/sampler.py disables the
FlashInfer sampling branch whenever a greedy request is present regardless
of return_logprobs (lines 300-307). gpu/spec_decode/rejection_sampler.py calls
the verification routine before _get_logprobs_tensors (lines 238-262);
max_num_logprobs is used for the returned scores, not passed to verification.
The earlier generic sampler references alone would not establish the V2 path.

## Live residual after the dense repair (2026-09-25)

The repaired image has not passed full-model repeatability. The initial short
sweep differed on the first 128-token request, only after the first token.
The expanded sweep (`receipts/dense-release-full-20260925/`) differed at
385 tokens and in one of six 16384-token requests. All answers were correct;
the 16K exception also changed the first-token top-20 logprobs. These failures
remain qualification blockers, not a permitted tolerance change.

The targeted repeat (`receipts/dense-release-residual-20260925/`) produced
20 identical cold 16K responses, but the first 385-token response differed
from the following 19. New filler and retrieval-code content reproduced the
385 pattern at about 0.3 seconds per request, without the initial boot's long
stall. Per-request speculative counters for the expanded sweep showed one
draft, seven verified draft tokens and two accepted tokens in all six 385 and
16K trials. Adaptive verification depth does not explain those differences.

A fast transition reproducer is now available:
`probe_length_transition.py --corpus receipts/determinism-corpus.json --out <new-path>`.
In three cycles, both 384-to-385 and 400-to-385 transitions produced a different
first 385 response followed by two matching responses (6/6 transitions).
The same-length 385-to-385 target controls matched in all three cycles.
Receipt: `receipts/dense-release-transition-20260925/summary.json`.
Every request uses a new cache salt and independently verifies zero cache hits.
This localizes a repeatable history-dependent residual; it does not yet name
the responsible kernel or metadata. No restart or launch change was made.

The same-length content control also fails reproducibly. With every request
at 385 tokens, two priming requests using filler `meadow` and code `826493`
are followed by three standard requests. Across three cycles the first
standard request has logprob signature `1b1f671559a5`, then the next two
return to `ce4159105f31`. Priming responses likewise have distinct first and
second signatures in every cycle. Receipt:
`receipts/dense-release-content-transition-20260925/summary.json`.
Prior content therefore affects the current request despite unchanged lengths.
This rejects a solely shape-dependent explanation; it does not distinguish
stale KV from other persistent content-bearing buffers without instrumentation.

The predecessor map covers 383, 386, 387, 388, 391, 392, 393, 408, 416 and
512 tokens, each twice. Every predecessor yields its own repeatable first-385
signature across both cycles; there is no simple shared-signature boundary
in that set. One additional exception occurs in cycle 1 after predecessor
392: the second 385 response differs at the first token as well, at 0.49 s.
The third returns to the common signature. This is important: a five-second
timeout cannot account for every first-token outlier either. Receipt:
`receipts/dense-release-predecessor-map-20260925/1-392-target/385-1.json`.

Current tuning receipts were copied from all four nodes as
`receipts/dense-release-live-tuning-final-{node}.json`. All 152 dense-program
assignments match the initial live snapshot on every node, with no missing or
changed dense winners. The whole-file hashes differ because these receipts
also contain other preparation records; they are not interchangeable by hash.
`run_dense_regression.py --live-winners --dry-run` now validates that complete
inventory without contacting nodes. Its pending idle-GPU run covers 296 cases,
64 repetitions each, with Turbo disabled. This is prepared, not yet executed.

Claude separately identified a silent five-second Engram wait timeout as a
source-level hazard and a plausible cause of the first boot's slow discrepancy.
That cannot by itself explain the fast length-transition reproducer. Both
observations need investigation rather than merging them into one attribution.

## Remaining gates

1. Finish MoE control and identify remaining variation at the same frozen inputs.
2. Review/install deterministic selection; rerun the identical short corpus.
3. If variation remains, isolate dense split-K and instrument Engram timeout
   handling rather than assuming the selector was the entire cause.
4. Repeat cold long-context inputs including the original 524K failure, with
   exact response/logprob comparison and independently measured cache misses.
5. Review final efficient patch, build with identities and native/graph gates,
   distribute the identical Podman image over the switched 200G fabric.
6. Full semantic, tool, context, concurrency/cache qualification and the standard
   benchmark harness without editing its repository; report correctness and
   determinism separately, including any unresolved original-input failure.
# Stale-state diagnostic build, 2026-09-25

The actual live-selected dense inventory passed all 296 cases / 152 programs,
64 repeats per case, with the cpasync check. This does not close the serving
repeatability failure. The preceding request's content still determines a
repeatable alternate logprob signature on the first subsequent request.

Claude reviewed the capture helper and packaging. The 27 standard-library
tests passed locally; all eight CPU Torch tests passed, without skips, inside
the pinned runtime and again in the diagnostic image. The new image is
`5eb584a63115549dd6066379d96dc1ac1ea7113f2878a7be84e03826320a0c6f`,
with lock `b45785d04f45cc9b4955aa71688468833226b8e946c2eb59dbf320cef8392684`.
It is diagnostic only, not a qualified release. Its native objects are unchanged.

The first build passed its eight tests but failed the marker gate because
stdout and stderr interleaved the marker into a test line. The corrected build
uses one unbuffered output stream; both build receipts are retained.

Before arming the probe, reproduce the content-transition failure on the new
boot with no control file and compare its tuning selections with the prior
live inventory. Do not require cross-boot logprob hash equality. Validate
controls offline and install atomically on all four ranks before requests;
require capture markers from every rank. Cache contents for the scheduled
verification positions have not yet been written by that forward, so raw
differences there alone do not establish an invalid read.

## Compressor-ring causal result, 2026-09-25

The corrected diagnostic image `47caaa211ce1` captured all 54 layers on each
rank at the first verification step after the 385-token prompt. Input tokens,
positions, CED/lookback state, and every non-ring cache payload matched between
the first-after-meadow request and its repeat. Only the compressor rings differed.
See `receipts/stale-capture-r4-20260925/`, including snapshot hashes and local
CPU comparisons. The full-prompt-logprobs control still reproduced the variation.

Both directions of ring-only transplant reproduced the predicted full logprob
signature twice: stable rings into first-after-meadow gave `62d694c9029c...`;
first-after-meadow rings into stable requests gave `8c505983c9fd...`.
Receipts are `ring-transplant-repeat-20260925/` and
`ring-transplant-first-20260925/`. Every rank engaged, requests were cold, and
controls were disarmed afterward. This establishes causality for the ring bytes,
not that either signature represents correct computation.

Claude traced the defect: CircularBufferSpec disables generic slot mapping,
which supplies -1; DS4.1 `_tokens` rejects those input slots, so the compressor
never saves its projected partial rows. Odd continuations still load those rows
through the block table. The proposed fix bypasses only that generic slot check
for circular groups, retaining all request, padding, null-block and writer guards.

The real GPU regression on clean base `3413799408e0` failed exactly as predicted:
the 385-token prefill stored nothing and the continuation loaded 512 NaNs instead
of its own last projection. `receipts/ring-fix-base-red-initial.log` records the
assertion failure, not an import or collection failure. The fix is not yet built
or serving-qualified. Even-length startup variation and the rare 16K outlier
remain open until retested.

An optional narrower transplant attempt failed because its layer regex omitted
the `.attn` path component. The helper selected no backed slot and aborted the
engine before a write. All controls were disarmed, logs retained under
`ring-transplant-slot0-observed-20260925/`, and all four diagnostic containers
were stopped and retained by `stop-ring-selector-failure-20260925T161801Z/`.
This is an experiment-selector error, not evidence of a new model fault.

The clean candidate is now built: `e7b273407022ae74726d6c7b7465aa567b2c5a0594d5e0b46002a742a11f5146`,
lock `6ac758e4db4567911ffec3a79c418e73603b1434fbb581f373a9619bf934b668`.
`receipts/ring-fix-build-20260925T162355Z/` retains the failed base assertion,
complete build log and inspection, and all 29 passing GPU cases on the candidate.
The cases include a NaN-poisoned ring, eager and graph replay, null blocks,
zero-length padded requests, and K7 verification/look-ahead slot preservation.
The installer inventories both source trees and native objects and verifies that
only the pinned sparse-MLA Python file changes. No stale-state or attention-trace
helper is installed. Model-level qualification is still pending.

The unchanged 37,359,880,192-byte Docker archive was distributed over the
200G switch and verified on toby, rusty and kirby. Archive SHA-256 is
`a75eed73f35e982d2859d0b92b3a339c319066f830ed3e8c2f86b7c83f8e1842`.
Receiver inspections and the archive receipt are retained locally. Worker-first
startup began at 16:31 UTC under `receipts/ring-fix-20260925T163121Z/`, with
per-node telemetry and a bounded startup watcher. The first cold probe lengths
are 128, 385 and 16,384, six repeats each. GLM remains untouched.
# Ring-fix serving result and remaining first-use issue (2026-09-25)

Image `e7b273407022` booted on all four ranks. The startup driver completed
with a failed repeatability verdict, not a failed answer: all 18 retrieval
answers were correct. Six cold salted requests each at 128 and 16,384 tokens
had identical full output/logprob payloads. At 385 tokens, the first request
differed from its five repeats, starting at output token zero. First request
latencies were 18.46 s (128), 7.17 s (385), and 6.78 s (16K). These timings
do not by themselves identify the remaining cause.

The original cross-request ring contamination discriminator now passes:
two cycles of two meadow/826493 requests followed by three filler/739184
requests, all 385 tokens, produced one exact target logprob signature across
all six target requests. This is evidence for the ring repair, not complete
model qualification. No kernel fault signature appeared in the retained
transition-window journals.

Evidence: `receipts/ring-fix-20260925T163121Z/repeatability/`,
`receipts/ring-fix-transition-20260925/`, and
`receipts/ring-fix-transition-observed-20260925/`.

Next diagnostic is Claude's reviewed Engram host-boundary timing kit over
the ring-fix image, explicitly not a serving release. Its logging preserves
CUDA call order but can perturb scheduling enough to change timeout behavior.
An instrumented pass cannot establish that the underlying issue is gone.
The four DS4.1 containers are retained during the worker-first stop; GLM is
unchanged. Full qualification and benchmarking remain pending.
