# Focused layer 0-1 capture

Status: first image's 8192 capture audited successfully; its 4096 window
helper aborted on prepared-versus-live row capacity. Both arms are stopped
and retained. A narrow instrumentation correction passed independent review
and build gates and is verified on all four nodes. Both corrected arms passed
capture audits. The first observed cross-grid divergence is layer 0's TP
reduction, exactly reproduced by different BF16 addition orders. Parent
restoration passed its 12 short repeatability probes; no whole-model fix is established. Neither
arm is model qualification. User approval is in
`window-capture-approval-20260926.json`.

## Frozen identities

- Parent: `e06df11a8ca18fa514d9f28f67cc691aef296da2eeb22b113a734519853bccd7`.
- Capture image: `866a04f5661068f27fc007a0040bbbffc38bec005ee1c1916ecaad16aea0621d`.
- Lock: `4018f06783a2b391b0aa2e5e014ad90e181f1930949ae64dc3fac61c01cac7e3`.
- vLLM tree: `3eb4846cfcece230c55c988a8b9ec18c2080201c`.
- B12X unchanged: `03a4e0b363d17c291516678689ae72d3e266749e`.
- Unchanged Docker archive SHA-256: `96c1b596999354d7ffd01caaeffb40ece0b28dc3943c0c3c21ff4744317a96d8`.
- Build receipt: `receipts/window-build-20260926T170931Z/`.
- Clean worker-first stop and retained parent containers:
  `receipts/stop-window-capture-build-20260926T170655Z/`.

## Scope and gates

Same frozen 524288-token input, TP4/K7, 600000 maximum context, utilization
0.85, deterministic MoE, turbo off, prepared capacity 8192, and 81389 KV
blocks. Compare explicit 8192 and 4096 prefill thresholds. GLM is untouched.
No numerical kernel changes. No production promotion is implied.

New capture records the last128 rows of layers0-1: incoming hidden states,
normalized KV, rotated KV and write slots, attention Q/output and the 255
distinct cache records needed for those windows, and WO before/after TP
reduction. It brackets, but does not instrument, the intervening FFN/mixing
operations. The old decision-row helper is byte-identical and writes a
separate sibling capture. Token IDs are not in the new device capture; the
frozen input and tokenizer receipts remain the input authority. Same-boot
decision-row records carry the attention plans.

Claude reviewed the helper and independent orchestration changes before the
build. Main closed the remaining host-kind gate, pinned the reviewed lock on
both launch paths, and refreshed the three affected runtime manifest entries.
Local results: 23 helper tests, 69 orchestration tests, 12 comparator tests.
The build checks inherited file inventory, exact declared source changes,
native preservation, and GPU-visible loaded-module paths/hashes, schemas and
helper identity. All passed. The inherited shell logged `PS1: unbound variable`
during RUN; the installer and import gates still passed.

Each grid requires two identical cold unarmed controls, one armed replay,
zero cache hits, matching numerical plans and complete four-rank captures.
Compare old/new decision-row observations with `compare_window_baseline.py`:
physical page addresses may differ across boots. Use the window comparator's
slot-aware decision consistency only on sibling captures from the same boot.
Run CPU audits only after stopping the diagnostic serving pair/group. Restore
the diagnostic parent after the experiments; do not call it qualified merely
because short completion probes pass.

## 8192 arm

Boot: `receipts/window-chunk8192-blocks81389-20260926T171631Z/`.
Capture: `receipts/decision-row-matched8192-capture-20260926T172150Z/`.
Stop: `receipts/stop-window8192-complete-20260926T173335Z/`.

Two cold unarmed controls and the armed request returned `739184, 482617`
with identical response signatures, zero prefix hits and the pinned plans.
The four window receipts contain no helper problems. CPU audits ran after
all four containers stopped successfully, not alongside serving.

The 40-layer decision-row Q, output, sink and chronological SWA records are
bit-identical to the earlier 8192 capture on every rank. The window's last
row also agrees with its same-boot decision-row sibling. All 128 captured
KV records at both layers match the reference encoder byte for byte;
slot/position checks pass. The largest per-row attention relative L2 error
against the same-input reference is 0.00311153, within the established
conformance envelope. These are operand and local-reference checks, not an
independent whole-model oracle.

Claude's completed review confirms these checks. The `window-self-*` files
compare each window against itself to run its local-reference and sibling
consistency checks: their cross-grid fields are placeholders, not evidence
of equality between chunk schedules. The preceding 127 records read by the
earlier window rows have no captured producers in this probe. Their reference
use checks attention arithmetic, not how those preceding records were made.

The full response signature differs from the older 8192 boot: chosen tokens
are unchanged; all top-20 records at output positions 0 through 4 are exact;
logprob records at positions 5 and 6 differ slightly. Thus the captures show
no observed first-decision perturbation, but do not establish cross-boot
identity of every decode step.

Request-window journals contain no NVRM/Xid. Headroom was thin, with sampled
MemAvailable around 1.0 to 1.4 GiB and kirby's minimum 875 MiB. Dusty showed
direct reclaim; kirby recorded a software-power-cap sample. These receipts
must not be presented as a performance benchmark or as proof of generous
memory headroom.

## 4096 instrumentation defect and correction

The first image's 4096 receipt is
`receipts/decision-row-matched4096-capture-20260926T174639Z/`.
All three cold requests returned the same wrong answer and full signature
`c77d73dd87ac509808bc55ac95172f11c7d31fd8526f5fde7331ccc57dae9e4a`.
The old decision-row helper saved on all ranks, but the window helper aborted
on every rank at layer 0: `swa_indices has 8192 rows, expected 4096`.
There is no usable 4096 window capture from this image and no cross-grid
window conclusion. The stop receipt is
`receipts/stop-window4096-helper-abort-20260926T180501Z/`.

The capture call passed the full prepared-capacity metadata, unlike
`mla.bind`, which passes its active `[:rows]` views. The correction adds those
same two views to the window call, leaving the helper's strict shape checks,
all numerical kernels, all other capture calls, and serving settings intact.
The test now executes the actual generated hook with 8192 metadata rows and
4096 live rows, poisoning unused capacity. It failed with the exact observed
abort before the change and passed afterwards, including cache-record checks.
The complete helper suite passes 24 cases, including both chunk schedules.
The original recipe is preserved under the first build receipt's `inputs/`.

Corrected lock: `59100837bbd6fdee5a9c2d689ae44ebd8b1d4b01838f5067dc3cc08706cea63b`.
Corrected vLLM tree: `9e339e969493cded736d3cbb53aad2d1c1e7143b`.
B12X and both helper modules are unchanged. This is a repair of the approved
measurement, not a claimed model fix. Repeating both arms avoids comparing
window captures from different instrumentation trees.

Claude independently replayed the red test against the retained original
call, confirmed the green tests and scope, and cleared the corrected build.
The corrected image is
`c4a51be481c1c8130a691e72c880c3fa3b751dbddc9b2da7aa6702cc8a0039f9`,
with build receipt `receipts/window-build-20260926T181235Z/`. Installation,
native inventory and GPU-visible import gates passed. The old-style 4096
decision-row audit also passed its same-input conformance envelope, despite
the separate window-helper abort; it does not supply missing window evidence.
The corrected Docker archive SHA-256 is
`09f64f4d974e98393d554dc55d936e9542959bcb1c1e331631747bde6a4deb91`;
distribution completed and every receiver verified image ID `c4a51be481c1...`.

Kernel journals for both complete boot windows are now retained in their
`window-chunk*/` receipts. Counts of `NV_ERR_NO_MEMORY` on dusty/toby/rusty/kirby
are 115/60/43/76 for the first 8192 boot and 56/57/45/66 for the 4096 boot.
No Xid appears. The warnings precede the long capture request windows; do not
describe the boots as warning-free. Request elapsed times are not performance
evidence: the first image's 4096 controls/capture took 292.8, 323.6 and 384.8
seconds and included low-power intervals without debugger intervention.

## Corrected image, 8192 arm

Boot: `receipts/window-chunk8192-blocks81389-20260926T182000Z/`.
Capture: `receipts/decision-row-matched8192-capture-20260926T182527Z/`.
Stop: `receipts/stop-window-fixed8192-complete-20260926T184108Z/`.

Two cold controls and the armed request all returned `739184, 482617`, stop,
with signature `56e100713bb2eade04dc7e1f16ebd73a289d4d77bd17923b07aee309df37c074`.
All four window captures were collected without helper errors. The decision-row
audit passed; all nine subsequent CPU analysis commands passed after serving
stopped. All 40-layer observations match the original 152805Z capture exactly.
The window's final rows match their same-boot decision-row captures, both layers'
128 packed records match the reference, slots match positions, and replicated
tensors match across ranks. The largest per-row attention relative L2 remains
0.00311153. These checks do not imply whole-model correctness.

Full response top-20 records differ from the first instrumented 8192 boot after
output position 1, although all seven chosen tokens match. Within this boot the
three full signatures match. Request-window kernel collectors report no
NV_ERR_NO_MEMORY or Xid. MemAvailable minima were 638.8, 1093.8, 823.8 and
657.5 MiB on dusty, toby, rusty and kirby. This is thin headroom, not a production
capacity qualification. Elapsed request times are not benchmark results.

The corrected 4096 boot is
`receipts/window-chunk4096-blocks81389-20260926T184505Z/`; its capture driver
uses `receipts/decision-row-matched4096-capture-20260926T185004Z/`.

## Corrected image, 4096 arm and cross-grid result

All three cold requests returned the same incorrect first code, with full
signature `e65787f226671f7f4314e2481a31c6ce24b83fd8120656cb22a5ec5ac83c2020`.
Elapsed times were 408.95, 429.41 and 578.94 seconds, not benchmark results.
All four window captures saved without errors. The four containers stopped
cleanly and were retained under
`receipts/stop-window-fixed4096-complete-20260926T191458Z/`.
Decision-row, window reference, packing, slots, same-boot consistency,
old-baseline and cross-rank audits passed. The cross-grid comparison guard now
checks the actual baseline/packing/slot/replication verdicts, not only an
invocation marker. Nine wrapper tests and three baseline tests pass.

`window-cross-{node}.json` in the 185004Z receipt gives the same first boundary
on every rank: layer 0 `wo_reduced`. Everything before that boundary in the
captured 128 rows, including each rank's `wo_partial`, is identical across
grids. Reduced outputs differ in rows 16 through 111: 199111 of 655360 BF16
values, relative L2 0.0030658833, maximum absolute difference 0.03125. The
reduced outputs agree across ranks within each grid. Layer 1 hidden input and
KV first differ on those same 96 rows; its attention spreads the difference
through all 128 output rows. This localizes the earliest observed difference
without claiming every earlier model operation or every prompt row was traced.

### Arithmetic explanation

`analyze_window_reduction.py` reads all eight trusted captures, requires each
rank's local partial to match across grids and the reduced output to agree
within each grid, and enumerates BF16-rounded sums and a double-precision sum
rounded once. Two synthetic arithmetic tests pass. Its CPU-only invocation,
source digest, input digests and output are retained as `window-reduction*`
in the 185004Z receipt.

Every one of the 128 reduced rows in both grids matches a sequential BF16
addition order exactly. In the 8192 grid, consecutive 32-row stripes match
rank orders 1230, 2301, 3012, 0123. In the 4096 grid, the same cycle uses
16-row stripes and repeats after 64 rows. Swapping the first two operands
gives equivalent references. This accounts exactly for why rows 0-15 and
112-127 match and the intervening 96 differ. Neither grid matches any entire
row of the sum rounded only once; relative L2 to that reference is about
0.0029 for both. There are no unexplained values at this captured boundary.

This is strong evidence for ordinary BF16 reduction association, not a wrong
operand or unexplained memory corruption. Arithmetic matching does not alone
identify the backend, prove the whole-model answer distribution, or establish
a fix. The logged RoCEnante cap is 2 MB; these layer-0 BF16 messages are
8192x5120x2 = 80 MiB and 4096x5120x2 = 40 MiB. Per-call dispatch and a future
invariant-reduction experiment must be checked before attributing globally.
The eight frozen attention plans are not an inventory of every model kernel.

Request-window collectors all exited zero with no NV_ERR_NO_MEMORY or Xid.
4096 MemAvailable minima were 489.5/1341.5/1330.8/1339.3 MiB across
dusty/toby/rusty/kirby. Complete corrected boot-window NV_ERR_NO_MEMORY counts
are 186/43/54/74 for 8192 and 55/47/40/67 for 4096; no Xid. This remains a
thin-memory diagnostic, not a qualified production envelope.

## Review and restoration

Claude independently reviewed the corrected results and the arithmetic
explanation. Accepted limits: equality before the reduction is layer 0 only;
valid repeated BF16 additions can deviate by more than one final-output ULP
under cancellation; cross-boot captured-row equality does not establish full
decode-logprob equality. No library-wide or precision-changing patch was made.

Restored parent `e06df11a...` using the saved 182000Z amendment. Boot receipt:
`receipts/router-fence-20260926T192348Z/`. All 12 short probes passed. The
diagnostic block-count and chunk-threshold overrides are removed. Boot-window
allocation-warning counts are 61/63/49/66 on dusty/toby/rusty/kirby, with no
Xid in the collected logs. This is the diagnostic parent, not a promoted image.

The next proposed discriminator is the NCCL-only subset of upstream vLLM's
batch-invariance settings: tree all-reduce, Simple protocol and one channel,
on the matched capture grids. This changes the launcher contract and the
rounding order and may lower bandwidth; seek direct approval before applying.
Do not enable the full `VLLM_BATCH_INVARIANT` mode: it also changes compilation,
Torch operations and attention configuration. FP32 reduction remains a separate
precision experiment, not an assumed fix. Neither proposal changes the unresolved
524K qualification verdict until tested.

The user subsequently approved the matched NCCL-only diagnostic. The parent
was stopped and retained worker-first in
`receipts/stop-pre-nccl-tree-20260926T194340Z/`, all exits clean.
While the nodes were idle, `run_window_fp32.py` checked the saved corrected
8192 and 4096 captures. Both receipts now contain `window-fp32.json` and a
source/input-hashed invocation record. For layers 0 and 1 on each grid, all
24 sequential and three pair-tree FP32 addition orders agree bitwise after
one BF16 rounding, and all agree with the FP64 reference. The sufficient
FP64 exactness bound passes (maximum operand exponent spans 20 through 22).
The stronger sufficient FP32 significand bound does not pass, so the FP32
result is measured agreement for these captured operands, not a general proof.
No serving precision was changed.

## Approved geometry diagnostic results

Selection amendment: `window-20260926T194810Z-amendment.json`. The existing
capture image is unchanged. The narrow `DS41_NCCL_GEOMETRY=tree-simple-1ch`
selector adds only the approved NCCL settings to the matched capture profile;
normal rendering remains unchanged. Restore this amendment after both arms.

The first preflight at 195306Z stopped before inference because its warning
filter matched the known NIC speed-query warning, `ibv_query_port_speed ...
Protocol not supported errno 93`. The identical warning is in the corrected
baseline boot at 182000Z. Added that exact line as a failing test, excluded
only that warning from the geometry-setting classifier, and passed the tests.
No live setting changed and no restart was needed for that retry.

8192 arm: `receipts/decision-row-matched8192-geometry-tree-simple-1ch-capture-20260926T195418Z/`.
All three cold requests returned `739184, 482617`, stop, with signature
`f6510535dbd394d0126b668976164c15f34503329ccb5fc4bd5048e13c456184`.
Elapsed times were 179.31, 179.42 and 180.36 seconds, diagnostic timings only.
All four decision-row and window captures collected without helper problems;
the eight attention plans match the pinned control. Request-window collectors
all exited zero with no NV_ERR_NO_MEMORY or Xid. The arm was stopped and
retained in `receipts/stop-nccl-tree8192-complete-20260926T200419Z/`.

4096 boot: `receipts/window-geometry-tree-simple-1ch-chunk4096-blocks81389-20260926T200508Z/`.
Startup and 12 short probes passed. Long replay completed in
`receipts/decision-row-matched4096-geometry-tree-simple-1ch-capture-20260926T200900Z/`.
All three cold requests returned the same wrong first value,
`510c94b1bb4f42d2998093bd6b9c3d99, 482617`, stop, signature
`df7f7025773c602c4f190bda8ea353e565faa6f59e645bf5c090e9abf721676d`.
Times: 187.27, 186.09 and 186.82 seconds, not benchmark measurements.
All captures and pinned plans passed. The arm was stopped and retained in
`receipts/stop-nccl-tree4096-complete-20260926T201932Z/`.

Both decision-row audits passed their per-operator conformance checks. All
22 geometry window-analysis commands exited zero, with structural checks for
packing, slots, sibling captures and replicated ranks passing. Across the
geometry grids, every captured boundary in layers 0 and 1 is now equal on all
four ranks, including WO partials and reductions. The earlier first observed
reduction divergence is eliminated in this measured pair, but the final
answer still differs. This is a partial mechanism result, not a retrieval fix.

The full-depth decision-row comparison first differs at layer 2 on every rank.
At that layer, Q, sink, active SWA records, indexer query and index-key-page
hashes agree. Six of 32 indexer-weight values differ, max absolute difference
0.00006103515625; the top-512 sets intersect in 490 entries. Attention output
relative L2 differences across ranks are 0.01158/0.00896/0.00849/0.00994.
Query and SWA differences first appear at layer 3. The layer-2 projection
inputs and raw projection outputs were not captured: equal downstream queries
do not prove equal hidden input. Indexer weights are the next observed lead,
not an attributed root cause or proof that no earlier uncaptured value differs.
Evidence: `geometry-decision-cross-*.json` and its source-hashed invocation
record in the 4096 geometry receipt.

Request-window journals show zero NV_ERR_NO_MEMORY and zero Xid on both arms;
all collectors exited zero. Minimum MemAvailable in MiB (dusty/toby/rusty/kirby)
was 8381/10272/10119/10357 at 8192 and 8160/10235/10129/10222 at 4096.
These are request windows, not claims that boot journals are warning-free.

The saved parent selection was restored with the 194810Z amendment; startup
and all 12 short completion/repeatability probes passed in
`receipts/router-fence-20260926T202436Z/`. Restoration boot-window logs contain
87/51/40/54 NV_ERR_NO_MEMORY warnings on dusty/toby/rusty/kirby, with no Xid
or traceback. The first 256-token probe took 17.04 seconds and the second
2.57 seconds; subsequent probes were below a second. These restoration
observations are retained, not described as a warning-free boot or benchmark.
No qualification or promotion claim follows from this diagnostic.

Independent review completed after a long Herdr wait. Claude verified the
receipt-level conclusions and identified the indexer weights projection as
the next source lead: a replicated unquantized 5120-to-32 BF16 projection,
with the pinned B12X policy resolving this large-M geometry through Torch.
Different cuBLAS arithmetic by M is a hypothesis, not an observed kernel
selection or proof of a faulty result. A small earlier hidden-state difference
masked by the other quantized projections remains an alternative. Likewise,
the top-k difference is correlated with the weight difference, not an isolated
causal replay. No precision or workspace policy has been changed.

Recommended next approved scope: extend instrumentation to layer 2's final
128 rows, including hidden input, raw weights-projection output before scaling,
scaled index weights and relevant Q/K boundaries, under the same matched tree
geometry. This distinguishes equal-input projection differences from input
differences inherited from the layer-1 FFN/MoE/mixing interval. It requires a
new helper lock/image and explicit approval for the diagnostic selector to
admit that image. Do not change numerical policy before that distinction is
measured. All four live parent image IDs and absence of diagnostic overrides
were independently checked after restoration. GLM was untouched.
