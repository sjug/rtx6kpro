# September 24 DS4.1 candidate execution

## Current checkpoint: September 27, clean-release qualification failed

### September 27 05:43 UTC: both expanded layers within reference bounds

Layer-14 capture completed three cold, identical correct trials, then all four
containers stopped cleanly and were retained in
`receipts/stop-mhc-expanded-layer14-passing-complete-20260927T053257Z/`.
The standalone comparison reproduced the captured serving outputs exactly.
Its initial `contract-violation` classification was a harness reference error:
upstream `tests/norm/test_mhc_lagged.py` returns BF16-rounded y, whereas the
comparator applied that output's absolute tolerance to unrounded FP64 y.
Both serving configurations had identical y; the 13 reported errors therefore
could not explain their selection-dependent answer difference.

The comparator now mirrors upstream's float32 collapse, intermediate BF16
rounding, float32 RMSNorm and final BF16 rounding for the y contract check.
No tolerances changed. Original unrounded FP64 errors remain in a separate
report field, with all 13 outlier coordinates and values (no truncation).
All 13 actual values equal both upstream and BF16-rounded FP64 values. For
example row 2019/column 2193 is -4.03125, against unrounded -4.0464190102.
This is ordinary final quantization, not an arithmetic defect. There are 943
other differences from rounded FP64; these are not asserted to be exact matches.
The upstream-reference maximum y error is 0.00048828125, with zero violations.
The correction has a test that fails before implementation and mirrors the
actual pinned upstream reference; all 35 local tests pass.

Corrected reports (old reports retained):
- `receipts/mhc-expanded-compare-mhc-expanded-passing-l14-20260927t051553z-reference-v2/report.json`
- `receipts/mhc-expanded-compare-mhc-expanded-passing-l1-20260927t044921z-reference-v2/report.json`

Both layers pass capture identity, selection identity, exact retained-output
faithfulness, repeatability and the complete 18-class population checks.
Classification is `within-valid-population` in both. This is evidence for these
captured final-chunk inputs only, not proof about all inputs or earlier chunks.
The one-record transplant still establishes sufficiency for the observed
answer flip; no mHC contract violation has been demonstrated. Do not promote
an arbitrary configuration merely because it answers the historical prompt.
Full clean-profile qualification remains failed. The four DS4.1 nodes are idle;
GLM was not touched. No commits, pushes or external filings were performed.

Claude's final read-only review completed after a waited prompt and confirmed
the corrected reference and both reports. No further population boots are
recommended: they would not resolve the serving policy choice. A reproducible
candidate would need versioned image/selection/geometry/NCCL identities, and
capacity-keyed plans mean ordinary per-boot KV profiling can select different
arithmetic. A proposed approach is to profile normally, require that the result
admits a declared KV block count, then allocate that count and reject cache misses
or new tuning. This is a proposed serving-contract change, NOT implemented or
authorized by this report; the user previously rejected fixed KV allocation.
An mHC-only accuracy filter would not establish cross-boot reproducibility.
Any approved manifest must be chosen independently of the needle's answer and
face the unchanged full qualification, without rerolling until it passes.
The goal is not complete; approval for a changed serving contract is requested.

Real-input layer-1 expanded-mHC comparison completed in the unchanged release
image `1989e16d`, on idle dusty after all four capture containers stopped
cleanly and were retained. Report:
`receipts/mhc-expanded-compare-mhc-expanded-passing-l1-20260927t044921z/report.json`.
Capture identity and selection gates passed. Standalone replay reproduced all
retained serving post/comb/pre_out elements and the y tail exactly, with matching
full-y integer digest; repeated operator calls were bit-identical. Both selected
configurations and all 18 reduction-order candidate classes stayed within the
upstream FP64 reference bounds. Passing/release maximum differences were
3.8743e-7 (post), 2.9802e-7 (comb), 7.7486e-7 (pre_out), and zero for y.
The fixed reference consumer changed 254 residual and 140 FFN-input BF16
elements in the captured tail, maximum difference 0.001953125. This demonstrates
rounding propagation, not an upstream contract violation or an actual downstream
serving trace. Layer 14 and earlier chunks are not covered by this capture.

The first standalone attempt stopped before GPU work because the harness rejected
Python multiprocessing's `__mp_main__` alias. It now accepts that name only when
it is the identical module object as `__main__`; an unrelated alias still fails.
The capture test suite passed 34 tests. The retained operator container was
restarted against that host-harness correction only; no image or kernel changed.

Claude reviewed the alias correction and capture evidence after a completed wait.
Its suggested downstream discriminator is sparse-indexer top-k membership, not
an established explanation. The existing `ds41_indexer_capture.py` captures
layer-2 projection inputs/outputs, not actual selected top-k IDs; direct membership
comparison would require additional reviewed instrumentation. First, the same
capture image/profile is rebooting for layer-14 coverage in
`receipts/mhc-expanded-capture-passing-chunk8192-blocks81389-20260927T051001Z/`.
All four ranks have passed the stock-NCCL and worker precision markers. No GLM
changes were made. The operator container remains stopped and retained on dusty.

That boot finished with 30/30 short repeatability requests passing. Layer-14
recording is active in `receipts/mhc-expanded-layer14-passing-20260927T0516Z/`
under the matching observed wrapper. The first, instrumented cold request passed
in 399.19 seconds with zero cached tokens and the exact passing token-0 record
(739 versus 510, margin 0.125). Both uninstrumented repeats passed in 352.25
and 236.63 seconds, with identical full responses and zero cached tokens.
All shards were gathered and SHA-verified; the recorder and observer exited
zero. All eight container/kernel log collections succeeded, with no kernel
Xid, NV_ERR or OOM matches. Graceful worker-first shutdown is in progress
before standalone comparison. This is not yet layer-14 operator validation.

Claude's second completed review agrees that the layer-1 evidence supports
contract-valid reduction-order rounding, not a demonstrated operator defect.
Its proposed classification is bounded by the sampled inputs: layer 14 and
earlier chunks cannot be exonerated by the final-chunk layer-1 report. More
population serving arms are optional, not required merely to seek a correct
answer. Full clean-profile qualification remains blocked and unchanged.

Capture boot `receipts/mhc-expanded-capture-passing-chunk8192-blocks81389-20260927T044033Z/`
passed all 30 short repeatability requests and worker precision markers.
The first recorder attempt (`mhc-expanded-layer1-passing-20260927T0446Z`)
sent no inference: it requested one repeat, which the unchanged original-needle
driver rejects. Only the owned waiting recorder was terminated; its observer
wrapper saved logs and cleaned up. The recorder now requests three cold trials
and rejects a missing/incomplete report before waiting for capture receipts.
The 34-test capture suite passes. Neither image nor serving settings changed.
The retry is `receipts/mhc-expanded-layer1-passing-20260927T0449Z/`, with
telemetry in the corresponding `-observed-20260927T0449Z/` directory.
The first, instrumented request completed correctly in 245.82 seconds with
zero cache hits and the exact passing first-token logprobs (margin 0.125).
All four ranks logged a saved layer-1 capture for token
`mhc-expanded-passing-l1-20260927t044921z`. Both uninstrumented cold repeats
also passed (292.46 and 203.43 seconds), with zero cached tokens and identical
full responses and first-token logprobs. The recorder gathered and SHA-verified
all four shards on dusty. The observer exited zero; all eight container/kernel
log collections succeeded, and retained kernel logs contain no Xid,
NV_ERR_NO_MEMORY or OOM matches. The operator comparison has not run yet.

The expanded-only boot stopped cleanly on all four nodes and was retained in
`receipts/stop-selection-mhc-expanded8192-complete-20260927T042128Z/`.
The revised capture diagnostic is built as
`bbe5d4966717a96a0c791a43a03fcdcc64ccf73ee05f52418d7be2d06c03343c`,
lock `aab479ea81fbdf5e96787e6a33493672190df50978d86d655ceb72dadf6ddf5b`.
Its build receipt is `receipts/mhc-expanded-build-20260927T043252Z/`.
Installation inventory, unchanged image environment, base-layer ancestry,
loaded module identities, GPU host registration/copy/unregistration and
CPU/GPU digest agreement passed. Live capture and repeated response stability
are now verified as above; standalone replay faithfulness is still pending.
The row-sharded one-layer capture needs approximately 95 to 103 MiB per rank,
including its estimated digest temporaries. Its two-lane integer checksums
are not collision-free proof of equality; exact comparisons of the retained
outputs and tails remain separate gates. No numerical conclusion follows from
checksum agreement alone. Independently run suites passed 30, 22, 8, 2 and 3
tests in separate processes. Running them together was invalid because the
source-isolation harness deliberately removes the kit from sys.path.
The host and in-container KV-block guards now both admit only the explicitly
pinned capture kind; the runtime kit digest is
`dc66ba82fed72fb81ec3996f9c6114a54a03341a2fb65f6a20e6e65cb0f8a4b4`.
The unchanged Docker archive is being distributed over the switched 200G
fabric. No capture image has been promoted and GLM is untouched.

Expanded-only replay completed: all three cold original-input trials are wrong
and full-response bit-identical, with zero cache hits (419.45, 422.91, 360.96 s).
Each token-0 record exactly matches the BF16 failing variant; wrong-token
margin is 4.125 nats. The full-response signature is
`6fe2620bd512d3543e3c8caa72229f8683f6efcb8be55a3bd3f2606051cdd6c3`.
Before/after identity and selection checks passed. The observer and all eight
final log collections exited zero; no Xid, NV_ERR or OOM matches appear in
the inference-window kernel logs. Only the expanded mHC 8192-row record was
changed, establishing its sufficiency for this first-token shift on the fixed
input and geometry, not a numerical-contract violation. Receipts:
`receipts/selection-mhc-expanded8192-20260927T0352Z/` and
`receipts/selection-mhc-expanded8192-observed-20260927T0352Z/`.

Capture kit note (Claude, local review): the capture source and lock stay frozen at
`aab479ea81fbdf5e96787e6a33493672190df50978d86d655ceb72dadf6ddf5b`, matching the gated
image `bbe5d496`. Its per-rank tensor digests are two wrapping int64 sums over 32-bit
words, one of them position-weighted. They are noncryptographic consistency checks. Equal
digests across ranks, or between the reassembled shards and the device record, are strong
evidence of identical bytes but not a collision-free proof. The staging files themselves
carry SHA-256 from the streaming write, checked again at gather and load. The comparator's
faithfulness gate does not rest on digests alone. It requires exact element equality of
post, comb and pre_out on every row and of the 128-row y tail. Only the full-size y is
compared by digest. The in-container KV-block kind addition in `runtime.py` is exactly
one entry: removing it reproduces the prior manifest hash `a826e97e`, and it matches the
host guard in `run_node.py`. The integration patch now carries that hunk, so base plus
patch equals the live tree. Each guard is tested alone, and a structural test keeps the
two kind lists identical. Every capture-kit suite must run in its own process, because
the harnesses remove their directory from `sys.path` on import.

Capture review checkpoint at 04:20 UTC: the expanded-only replay has completed
two cold trials, both wrong with identical full outputs and token-0 records
(419.45 and 422.91 seconds, zero cached tokens); the third remains active.
The initial real-input capture kit passes its 27 CPU tests independently.
It is not deployed. Review found two changes needed first: reduce the roughly
1 GiB per-rank pinned staging and full-buffer hashing copies, and ignore
ordinary fused `post_pre` calls intercepted through the same wrapped `pre`
method. The pinned B12xMHC implementation calls `self.pre` from `post_pre`,
so the original helper would abort on that legitimate call before reaching
the decision chunk. Claude is revising the capture kit and tests; live serving
files are unchanged. This is a diagnostic-kit defect, not evidence about the
serving numerical cause.

The 275-selection diagnostic completed and was stopped cleanly, workers first,
with all containers retained under
`receipts/stop-selection-transplant275-complete-20260927T031335Z/`.
The reviewed two-mHC-only patch is now applied. Seven local tests pass,
including unchanged outcomes for the eight existing orchestration suites and
exact per-rank verification that only the two mHC records differ from passing.
Runtime kit digest is
`99fcb4e5032e59d76718c81e8be41607937e830a2ea928068c6ef2e539a8ae67`.
The image, passing geometry and all other selection records remain fixed.
Seeding and startup completed; all 30 short repeatability requests passed.
Startup receipts are
`receipts/selection-transplant-mhc8192-chunk8192-blocks81389-20260927T031610Z/`.
Three cold original-input trials are running under
`receipts/selection-mhc8192-20260927T0325Z/`, with telemetry and final logs
under `receipts/selection-mhc8192-observed-20260927T0325Z/`.
All three cold long-input trials completed in 344.29, 289.36 and 340.35
seconds, wrong and bit-identical within the boot, with zero cache hits and
the exact BF16 failing-control first-token record (wrong-token margin
4.125 nats). Final identity and selection checks passed, the observer and
all eight log collections exited zero, and kernel logs have no matched
Xid, NV_ERR or OOM. The two mHC records alone reproduce that first-token
shift on this input and geometry. This establishes sufficiency, not
necessity, numerical error or a general precision policy.
The hardened operator-reference harness passes 21 local tests; GPU execution
is pending a graceful stop. Tokenization returned exactly 524288 IDs with
SHA256 `af119436430644d8ba24ca157fbfe93af077ccb67e5f7ab3d5dda7f48d8b7b41`.
The isolated GPU comparison is complete under
`receipts/mhc-layer0-replay-20260927T0345Z-r4/`. The real layer-0 faithfulness
gate passed and both native configurations are bit-identical on all four
outputs, with no reference propagation differences. The expanded-residual
synthetic comparison differs within all tested operator/parity tolerances:
maximum post difference 1.073e-6; maximum post error versus FP64 is 4.025e-7
for passing and 9.654e-7 for release. Synthetic inputs cannot settle numerical
correctness on the actual serving forward. Earlier attempts stopped on
harness plumbing errors (Torch registry pseudo-files, nested text config,
and reference kwargs); these were corrected and covered by 22 local tests.
The reviewed one-key patch is now applied, with eight tests passing before
and after. Expanded-mHC-only seeding/startup is running, keeping layer-0
and all other passing records unchanged. Claude is reviewing the operator
evidence and preparing a real expanded-input diagnostic, local-only.
The one-key kit digest is
`32fb99102fcefa0392d241b4ee9a08538cb4d3816ff6f1333fd7d429b65524ca`;
startup is recorded in
`receipts/selection-transplant-mhc-expanded8192-chunk8192-blocks81389-20260927T035133Z/`.
A supervised continuation launches the three cold original-input trials only
after startup and the short repeatability driver exit successfully. Its target
receipt directories are `receipts/selection-mhc-expanded8192-20260927T0352Z/`
and `receipts/selection-mhc-expanded8192-observed-20260927T0352Z/`.
Startup and all 30 short repeats passed. The first expanded-only cold 524K
trial completed in 419.45 seconds, wrong, with zero cache hits and the exact
BF16 failing-control token-0 record (margin 4.125). Only the expanded mHC
8192-row selection was replaced, so it is sufficient for this first-token
shift in this trial. Two repetitions and final gates remain pending. Live
health checks found all four containers running, not OOM-killed, and no
kernel journal entries since 04:00 UTC.
Claude is independently preparing a real-input layer-0 mHC reference test,
local-only, without modifying these live launch helpers.

At 02:19 UTC the clean release `1989e16d` began a full-selection replay on
dusty/toby/rusty/kirby. This restores all 1162 saved selection records per
rank from the passing standard-communication capture, with its 81389 KV
blocks and 8192 prefill threshold. It is a transfer test, not a one-variable
causal test or qualification. Only selection JSON was seeded into a separate
cache under the existing remote task tree; the original caches are untouched.
Startup is recorded in
`receipts/selection-replay-chunk8192-blocks81389-20260927T021939Z/`.
Startup and all 30 short repeats passed. The all-rank replay preflight passed,
including exact equality of the 1162 seeded selection records. The first cold
524K replay completed correctly in 450.25 seconds with zero cache hits. Its
full response signature is `fba6114819113ce773e8bae15429a81db022eadd4217347d8848ac23be828430`,
exactly matching the earlier passing capture, not just its first token.
The margin remains only 0.125 nats. All three trials completed correctly,
bit-identical to the anchor, in 450.25, 246.98 and 306.43 seconds under
`receipts/selection-replay-20260927T0228Z/`, observed by
`receipts/selection-replay-observed-20260927T0228Z/`.
The recorder returned `identical-to-passing`; all three first-token records
also match. Before/after identity and exact selection equality passed on
every rank, and the observer wrapper exited zero. This is a completed
transfer test on the uninstrumented release, not full qualification. It
jointly restored selections and geometry and does not identify an individual
kernel or establish a numerical precision policy. A selection-only transplant
at this geometry is the next discriminator. Its reviewed patch is now applied;
10 tests passed before and after application. Kit digest is
`85a8f5cb806697a85bebc1531bd88511677e2657457da7eb796d0787a9a22373`.
All four successful replay containers stopped cleanly and were retained under
`receipts/stop-selection-replay-passed-20260927T024710Z/`. The transplant
seeds exactly the 275 changed shared records from the failing release snapshot
into a separate task-local cache root, preserving passing geometry and
capacity-specific attention plans. Every other record is unchanged. All four
nodes confirmed seeding and received only the reviewed runtime-file updates.
Startup is beginning. The BF16 ratio-1 variant is its primary failing
first-token reference; original FP8 control and passing anchor are retained
as additional references. The separate fixed-verification patch remains
unapplied and must be rebased before use because it shares the launch files.
Transplant startup is recorded in
`receipts/selection-transplant-chunk8192-blocks81389-20260927T025132Z/`.
The 275 shared changes classify as 124 BF16 GEMV, 97 dense/block-FP8 GEMM,
26 mHC, 23 MoE, two vocabulary projections, two sparse MLA and one varlen.
The two shared MLA keys map to `swa.extend` and `draft.extend` with 32
temporary profiling blocks, not the serving plans. Key reconstruction of all
eight serving attention plans at 81389 blocks preserves the passing modes,
including BF16 ratio-1 extend. This is source/key evidence, not an execution
trace. The varlen call sites are vision/non-causal, not text-only DS4.1 MLA.
The mHC 8192-row pair is confirmed by query mapping; its changed reduction
geometry is a candidate for a narrower follow-up, not an established cause.
Transplant startup and all 30 short repeats passed. Its monitored replay is
`receipts/selection-transplant-20260927T0251Z/`, with telemetry and final logs
under `receipts/selection-transplant-observed-20260927T0251Z/`. The first cold
trial completed in 220.53 seconds, wrong, with token `510` beating `739` by
4.125 nats. Its complete first-token logprob record equals the BF16 ratio-1
failing variant exactly. All three trials are now complete, wrong and
within-boot bit-identical, in 220.53, 224.85 and 272.85 seconds. All three
full first-token records equal the BF16 variant. Before/after identity and
selection checks passed; the observer and all eight final log collections
exited zero, with no matched Xid, NV_ERR or OOM in the inference-window
kernel logs. This supports a shared-selection effect at fixed geometry; it
does not yet select an individual kernel or prescribe a precision policy.
The latency versus the anchor's 187 seconds remains unexplained; live checks found running
containers, no OOM kills and no new kernel journal entries during that request.
Sixteen replay-kit tests and 51 existing launcher/orchestration tests passed
after applying the reviewed patch. A previous eight-test integration claim
below included three vacuous comparisons of failed renders; do not treat
those three as evidence. The new tests require the base render to succeed.

Claude's completed source review identifies adaptive speculative verification
as a testable explanation for later-token cross-boot drift: it profiles cost
curves at boot, chooses verification depth using those curves and confidences,
and reaches different numerical backends at different row counts. Different
draft totals support investigating this path but do not prove cost changes
caused the drift; changed confidences can also affect depth. It cannot explain
the first-token retrieval failure. A fixed-depth diagnostic is being prepared
as an unapplied patch while the selection replay runs. GLM remains untouched.

### Earlier ratio-1 sequence

The next diagnostic is built, not yet booted: `a463239fbccb805de30abfac77d827e6b57d64390544c3c76d8b1342afc4c2ef`.
It derives from `1989e16d` and changes only the ratio-1 main-model extend
attention declaration to use the supported complete BF16 configuration pin.
The native/source inventory and GPU-visible pin smoke passed. Fifteen artifact
tests and eight integration tests passed independently before the build.
No claim is made that the pin fixes the failure yet.

All four failed-release containers stopped cleanly, worker-first, and are
retained under the `precision-release-failed-original` suffix. Their selections
were snapshotted in `receipts/ratio1-namespace-snapshot-20260927T010531Z/`;
the separate diagnostic namespace was copied without overwriting the release
namespace and verified in `receipts/ratio1-namespace-seed-20260927T010712Z/`.
The launch kit now admits exactly 80022 blocks for this diagnostic's control,
variant and return, never as a normal serving default. The recorder verifies
the current kit digest and actual boot allocation and streams replay results.

Distribution attempt one failed before copying because the generic wrapper's
derived tag omitted the builder's `-diagnostic` suffix. The wrapper now imports
the actual ratio-1 builder tag; three transfer tests pass. Retry logs are
numbered so the first failed attempt remains intact. Attempt two completed:
all receivers verified image `a463239fbccb805de30abfac77d827e6b57d64390544c3c76d8b1342afc4c2ef`.
The matched release control started worker-first on all four nodes at
01:17 UTC, with 80022 blocks pinned. Startup and short-repeatability monitoring
are active under `receipts/precision-release-blocks80022-20260927T011706Z/`.
The control completed: three cold original 524K requests were wrong and
bit-identical in their recorded response signatures (184.85, 175.02, 176.97 s).
The first result also matches the failed clean-release signature exactly.
All 1050 selection records remained unchanged on every rank, with zero additions.
Inference-window kernel logs contain no matched NV_ERR, Xid or OOM errors.
Receipts: `receipts/ratio1-control-20260927T0122Z/` and
`receipts/ratio1-control-observed-20260927T0122Z/`.
The four control containers stopped cleanly and were retained under
`receipts/stop-ratio1-control-20260927T013152Z/`. BF16-only variant startup is
next at that checkpoint. The variant has now completed as well:
three cold original 524K trials were wrong and bit-identical (203.56, 196.02,
222.44 s). Its wrong-token margin was 4.125 nats, versus control 3.375.
All 1050 saved selections remained unchanged on all ranks, zero additions;
the inference-window kernel logs contain no matched NV_ERR, Xid or OOM errors.
Receipts: `receipts/ratio1-variant-20260927T0141Z/` and
`receipts/ratio1-variant-observed-20260927T0141Z/`.
This precision change is not a sufficient fix. The variant was stopped cleanly and retained under
`receipts/stop-ratio1-variant-20260927T015214Z/`.

The return control completed three cold wrong trials (212.24, 247.31, 239.66 s),
identical within its boot, with the original 3.375-nat wrong-token margin.
However, the full-signature comparison is **invalid**: return signature
`a06771f4...` differs from control `76a91395...`. Answer text and finish reason
match; the first four token-logprob records are exactly equal, and the first
difference is at output row 4 (the fifth token), in alternative-token logprobs.
For example, the EOS alternative changes from -17.8125 to -17.0625 there.
The first-trial speculative counters also differ: control drafted 33 tokens,
return 34, each accepting 20, with different per-position acceptance counts.
That is a measured association, not proof of the cause of the later-logprob drift.
Do not describe this as a successful full-response A/B/return causal gate.
The comparison is `receipts/ratio1-comparison-20260927T0213Z.json`; return
receipts are `receipts/ratio1-return-20260927T0201Z/` and
`receipts/ratio1-return-observed-20260927T0201Z/`.
The original-input accuracy failure remains unresolved, as does cross-boot
later-token logprob reproducibility. A separate full-selection replay proposal
is locally tested and reviewed but remains unapplied at this checkpoint.

The full driver is terminal with exit 1. All three original 524K trials were
wrong and identical, with zero cache hits and the same wrong-token margin
of 3.375 nats. Times were 261.75, 219.86 and 264.63 seconds. The failure is
accuracy under this boot, not observed within-boot nondeterminism. The
driver stopped before the later long-repeatability, maximum-context,
conversation and final gates. Benchmarking and promotion remain blocked.
All four inference-window kernel/container collections succeeded, with no
NV_ERR, Xid, OOM or runtime traceback matches; owned observers were cleaned
up. Containers were not restarted.

The second selection snapshot (`post-first-replay-selections/`) contains
no new, removed or changed records versus `initial-selections/` on any rank.
The next diagnostic is being prepared to isolate `ratio1.extend` arithmetic
without changing other selections or destroying the existing cache. The
frozen-input mixed-load gate has been authored and independently passed its
13 local tests (including the margin helper); live execution is held until
the current accuracy failure is understood.

### Earlier observations during this run

Promotion was blocked again when the first original 524K cold trial on the clean
release returned the wrong archive identity (`510...`), after 261.75 seconds,
with zero cache hits. Its first-token margin over the correct `739` token is
3.375 nats. The remaining two trials are still running; no conclusion about
within-boot repeatability is made from one request. The six passing matched
diagnostic trials therefore did not establish correctness of this normal boot.

Read-only selection comparison identifies a concrete confound: the passing
standard-NCCL diagnostic used 81389 KV blocks and BF16 `ratio1.extend`; the
clean boot uses its independently profiled 80022 blocks and FP8 for that plan.
The other seven decoded attention plans are identical. Across the 1042 shared
selection keys, 275 configuration choices differ, identically on all four
ranks. This is a hypothesis for the transfer failure, not causal attribution.
Sources are the passing `decision-row-matched8192-nccl-standard-upstream-capture-20260926T233036Z/*-selection.json`
and this run's `initial-selections/*.json`. No cache, runtime or image change
has been made in response. Benchmarking is held.

The clean release has finished booting on all four nodes. Per-rank precision
markers and 18 short repeatability requests passed. Full qualification is
running under `run_observed.py` in
`receipts/precision-release-full-qualification-20260927-r2/`.
Content-transition checks and the complete interleaved stress passed: 1100
requests, 22 lengths, 50 trials each, one signature per length, zero nonmodal
trials and zero unexplained five-second TTFT gaps. The subsequent 28-length,
six-repeat gate and semantic/vision/tool/concurrent gate passed. The original
524K cold replay gate is now running. This is partial progress, not a
completed qualification. The live kernel-selection snapshot is retained under
that receipt's `initial-selections/` directory.

The first driver invocation stopped before inference because its host-side
environment gate incorrectly rejected `CUDA_MODULE_LOADING=LAZY`, which both
the parent image and release image declare. The gate now requires LAZY and
rejects missing or EAGER values; four regression tests pass. Only the driver
was restarted. No model container, image or serving setting changed.

Independent second-boot retrieval, completion of the full correctness suite,
fault-gate evidence audit and standard benchmarking remain pending. The
historical benchmark wrapper pins the old R38 container and failed campaign;
it cannot be reused verbatim for this release. Any new wrapper must retain the
standard harness and validate this release's own completed gates first.

The independent completion audit found that the existing conversation gate
randomizes the archive identity for its mixed-load needle. It does not prove
the original frozen failure under concurrency. A separate unchanged-input,
cache-salted mixed-load gate is being authored and must pass before promotion;
the running suite is not being edited. The earlier Engram fault-path test is
inherited evidence by source identity, not a re-execution on this image. The
audit also retains its operational caveat: workers required explicit teardown
after rank zero rejected the injected fault.

Historical entries below describe earlier stages, not the current live state.
The precision diagnostic completed both 8192/4096 chunk arms with ordinary
NCCL: three cold original 524K retrievals passed per arm, with identical full
response signatures within each boot. Cross-grid logprobs still differ; the
earliest captured difference is layer 0 after output-projection all-reduce.
See [the precision record](PRECISION-ARM-20260926.md) for evidence and limits.

All four DS4.1 nodes were stopped gracefully and retained after those captures.
The uninstrumented release candidate `1989e16daf38` passed all build gates on
idle dusty, was distributed over the switched 200G fabric, and has the same
verified ID on all four nodes. The clean release is selected and booting on the
normal profile, with a fresh recorded cache namespace. Its startup receipt is
`receipts/precision-release-20260927T001624Z/`. No full model qualification or
promotion of this release has occurred. GLM and nous have not been touched.

Next: finish clean-release startup on the normal 600K/u0.85 profile,
qualify the original retrieval and concurrent/mixed-load cases, complete the
existing correctness/stress/context suite, verify a second independent boot,
then benchmark. No fixed diagnostic KV or NCCL geometry belongs in that profile.

The user approved stock NVIDIA NCCL 2.30.7, commit
`73cf112295c33aee2b895f329f592f2a9b4b0f97`, for the switched 200G fabric.
Claude independently reviewed the source payload and component driver and found
no component-build blocker. Runtime and model qualification are pending.

## Qwen rollback captured before the component build

Endpoint metrics: zero running and zero waiting requests.

Container name on both hosts:
`qwen38-flash-next-nvfp4-karmic-main-hcbase-qsa865-tp2`.

Image: `a25bedd43581dffecca2f9eefb40e60b170af6ae19124fef1c292850b2a0483e`.

- dusty container: `082691f6285ceb9c215cb51a7a08e8ca45f5f57e672362b4750c2a5cba242dc6`
- kirby container: `280538ed0dbc8e80ef02c49ada1aef665fc7cb34284d3c235811f3b4a7a89e75`

Keep both containers. Rollback starts the retained kirby container, then dusty,
and requires a real chat completion. No replacement or deletion is authorized.
DS4 Vision and GLM remain serving during the component build.

Remote kit: `/home/jugs/git/bld-jj-r38-spark/karmic-main-sm121/ds41-20260924`.
Component receipts live under its `receipts/` directory. The build is offline,
uses 20 make jobs, and protects the component with a non-serving retention tag.

## Component result

Stock NCCL component passed at
`886976fe71ebc1ab0defe88edc7ffa283cda2230e11036a9309b2c91e6401939`.
Library SHA256:
`044746e929293451ce47c59d255437a6b42403565a23acd55f28392ea8b7fe8c`.
Receipt: `receipts/nccl-20260924T182247Z-2495162/` (retained locally and on dusty).
ARM64 ELF, SM121 SASS, NCCL version 23007, linkage and checksums passed.
No distributed or model qualification is implied.

Both retained Qwen containers stopped with exit 0 and OOMKilled false.

## First runtime assembly

Stopped before publication on the LMCache installed-file identity gate. The
base wheel normalizes one Git executable script to 0644 and dereferences one
documentation symlink. Direct inventory comparison found no other package-byte
differences. The preparer now derives these two path-scoped wheel transformations
from the independently verified upstream tree; it does not relax other file
checks. Regression tests cover both transformations and reject unknown links.
The NCCL component needs no rebuild.

## Runtime build passed

Candidate `eb3adc203432f2e54738c5977b65bc6a618ba4fdb76f23dd510df5910365e08d`
passed the inherited native, FlashKDA, regression, workspace and draft-head gates,
plus the stock NCCL single-object identity, four-rank CLI, loader linkage and
nine upstream Engram tests. Receipt:
`receipts/runtime-20260924T183825Z-2609630/`. Build success is not model qualification.
The frozen runtime manifest identifies the adapter separately from the image.

Before distribution, DS4 Vision reported zero running and waiting requests.
Retained containers `ds4-vision-jj-r38p-tp2` were stopped worker first (toby), then
head (rusty), both exit 0 and OOMKilled false. Their complete inspections are
`receipts/previous-toby.json` and `receipts/previous-rusty.json`.
Rollback starts those same containers on toby then rusty and requires a real
completion. GLM remains untouched.

The first admission remains upstream's 500,000-token context. A proposed 600,000
context-only follow-up for the exact historical 524K input has been reviewed by
Claude but is pending user approval. No shorter or reworded probe resolves that
historical failure. The old full checkpoint hash receipts are inherited; this
deployment repeats metadata, file-size and symlink-containment checks, not all
510 GB of weight hashing per node.

Archive: Docker archive, 37,344,512,512 bytes, SHA256
`b5b082b3fd821e778d9121f82c8ad2b9131f2a781ba64fdb5f4999e1ddf81fe6`.
Saved without conversion or compression. Receiver addresses are kirby
`10.11.11.8`, rusty `.5`, toby `.6`; routes from dusty use `enp1s0f0np0`
with source `10.11.11.7`. Control SSH continues to use hostnames.
All four hosts report nofile 500000. The existing host cache directory is reused;
the image's new fingerprint isolates its JIT entries from R38.

## First live admission blocked before weight loading

All three transferred archives matched the source checksum. Loaded image IDs,
Docker-v2 manifest type, configuration/history, labels and uncompressed layer
identities match the build. Podman load changed the transport manifest digest,
not the image configuration ID or filesystem content; both inspections are kept.
All four host preflights and in-container native/stock-NCCL/io_uring byte checks
passed.

At 18:57:45 UTC the head exited on `check_shm_free_space`: 160 MiB required,
0 MiB free. Private `--shm-size=0` is an unlimited tmpfs, but reports zero space
through statvfs/shutil.disk_usage. The pinned vLLM check treats that as no capacity.
The build resource gate checked unlimited semantics but did not exercise this
actual engine allocation check. No weights or model qualification ran.

Head exit 1, OOMKilled false. All workers were stopped gracefully, exit 0 and
OOMKilled false. Containers and admission logs/telemetry are retained. No IPC
recipe correction has been applied; Claude review and user approval are required.
GLM remains serving. Qwen and DS4 Vision remain retained and stopped for the
approved four-node window.

## Approved IPC correction

The user approved restoring upstream's private `--shm-size=64g` ceiling. This
does not reserve 64 GiB. The strengthened gate calls the actual pinned vLLM
`check_shm_free_space(160 MiB)` and requires the expected tmpfs ceiling. It passed
with native NCCL, CLI and loader checks in `receipts/shm64-gate.log`.
The image is unchanged. `shm-amendment.json` preserves the prior runtime manifest
and binds the new host-adapter manifest to the successful gate receipt.

The user also approved a context-only 600,000-token follow-up, conditional on
500K admission passing, to replay the exact historical 524K input.

## Corrected 500K boot and partial qualification

The corrected boot reached API readiness at 22:41 UTC. The head admitted
10,526,248 KV tokens; KV allocations were about 17.4 to 17.7 GiB per rank.
Weights occupied 74.59 GiB per rank. Full cold startup took about 15 minutes,
including weight and state preparation, full-pool binding, FlashInfer tuning,
and target/draft graph capture. The silent interval before binding resumed
without intervention; no hang was established.

`receipts/qualification-500k.log` records passing arithmetic, thinking and
non-thinking responses, vision plus continuation, tool call/round trip,
concurrent requests and exact 16,384-token dual retrieval. The original harness
then failed on null `prompt_tokens_details`, not on model output. Upstream does
not enable the optional usage-detail flag. The task-local `qualify_upstream.py`
keeps prompts and correctness assertions unchanged and uses isolated Prometheus
query/hit/success/preemption deltas instead. Its helper has five focused tests;
the historical R38 harness and benchmark repository remain unchanged.

Long tests are held because idle MemAvailable fell to about 1 to 2 GiB per node,
with about 3 to 6 GiB swap used. Allocation warnings recurred during loading,
pool allocation, capture and first completion. No Xid or OOM-killed process was
observed. Host /dev/shm uses only hundreds of KiB and current private shm about
100 MiB, excluding a large stale host-IPC segment as the explanation. Large
worker mappings and the auto-sized KV pool contribute to the current footprint;
their causal shares are not established by a matched test.

Claude reviewed the pressure and recommends not running 499K in this state.
Options discussed are upstream's fixed 10 GiB KV or utilization 0.80 with auto
KV. The latter preserves the user's no-fixed-KV preference and changes only one
setting, freeing approximately 6.1 GiB of budget per rank, subject to re-admission.
No utilization, KV, resident-scale, or context change has been applied. The
candidate remains idle and unqualified; GLM is untouched. Full 500K admission,
600K historical replay and benchmarking remain pending.

## Resumed at the approved utilization

The user directed qualification to resume without reducing utilization. The
0.85 profile is unchanged. Low MemAvailable and accumulated swap use alone do
not establish unsafe operation; monitor active swapping, allocation failures,
and serving health during the tests. The corrected harness is running with
receipts in `receipts/qualification-500k-resumed/` and its sibling `.log`.
The 600K context-only follow-up remains conditional on 500K admission passing.

The resumed run exited zero with DS41-QUALIFICATION-PASS. Exact dual retrieval
passed at 16,384 (4.36 s), 131,000 (66.18 s), 262,000 (158.06 s), and 499,000
tokens (350.71 s), each with zero measured cache hits. Short semantic, visual,
tool and concurrent probes also passed. Proceeding to the approved context-only
600K re-admission, preserving utilization 0.85 and all other serving settings.

## 600K historical replay window

Claude reviewed the context-only change before launch. The prior adapter bytes
are retained as `receipts/context-prior-{launch_contract,gate_ds41}.py`.
`context-amendment.json` binds the prior and updated manifests to the unchanged
image and independent successful native/CLI gate in `receipts/context600-gate.log`.
All four 500K containers stopped with exit 0 and OOMKilled=false and remain
retained with the `-500k-20260924` suffix. No image rebuild occurred.

600K startup completed and short semantics, vision, tool and concurrent checks
passed. Cold dual retrieval passed at 16,384 (5.89 s), 131,000 (57.07 s), and
262,000 tokens (138.88 s). These are correctness probes, not a benchmark A/B.
The engine reported 11,615,413 KV tokens and theoretical concurrency 19.36x
at 600K. Startup allocation warnings recurred on all nodes and are retained;
startup and these requests completed successfully.

The unchanged historical diagnostic probe is running three salted cold replays
of the original 524,288-token failing input, followed by the resuffix control.
The original input file SHA256 is
`1b89c7f4d1e1047f6cf7ebe6000d6aff242981cbdbe8d8ce2a742ff61a7d6db6`.
Receipts are under `receipts/historical-524k-600k/`; independent server cache
counters are retained in `receipts/historical-524k-metrics.jsonl` because the
upstream launcher omits optional API cache usage details. The original probe's
`cold_valid` field is therefore not independently dispositive. No promotion
or benchmark claim is made while that replay remains pending.

## Final replay verdict

The replay completed: original input correct in 1 of 3 cold trials, wrong archive
identity in the other two. Code-minus-identity first-token logprob margins were
+1.75, -3.75, -6.50; elapsed times 682.83, 470.37, 522.25 s. Aggregate counters
verify three times 524288 new queried tokens, three completions, zero cache hits,
zero preemptions and no overlapping requests. Inter-request sampling missed two
individual completion boundaries; this is recorded rather than editing receipts.
The final-instruction-only control returned the correct code in 540.32 s at
524274 tokens. It does not resolve the original failure.

Claude independently reviewed the original trials and agreed the historical
failure persists. No proven numerical defect or runtime cause is asserted.
An unchecked Engram overlap timeout was found in source, but there is no evidence
that its failure flag fired during these requests. The initial claim that the
historical input was random text was wrong; both inputs use repeated filler.
The read-only thread/I/O snapshots and unexplained timing variation are retained.

Qualification and benchmarking are blocked on the original-input correctness
gate. `ISSUE-DRAFT.md` is local only. No image was promoted or external issue
posted. Utilization remained 0.85 throughout; GLM was untouched.

Final inference-window kernel receipts (since 00:12 UTC) contain no NVIDIA
allocation warnings, Xids or OOM events. All four containers remain running
with OOMKilled=false; the endpoint is idle. Startup warnings are not erased by
that result. The full four-request aggregate verifies 2097138 queried tokens,
four completions, zero cache hits and zero preemptions. Temporary telemetry
SSH sessions were stopped after saving the completed run; no systemd units
were created. The model remains loaded, not promoted.
