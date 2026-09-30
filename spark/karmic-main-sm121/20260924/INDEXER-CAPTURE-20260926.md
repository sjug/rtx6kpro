# Layer-2 index-weight observation

Status: build and matched diagnostic run approved by the user on September 26.
Build and GPU-visible import gates passed; image verified on all four nodes.
Both captures and the offline structural comparison completed.
No numerical fix or promotion is implied.

Built image: `25c92dde8801c9601164dd9144ec5c9aa453581d670b212e60a91d4973b96248`.
Lock: `20ea0e6e920b2f5a7d78cc4ae0c3632b8ab4f140536ba6a58c09427b7bc2520f`.
vLLM tree: `c959c4f44c7c76fc4e71e3ac59fe1073ce0838d4`.
B12X tree remains `03a4e0b363d17c291516678689ae72d3e266749e`.

Claude's final integrated review found no blockers. Local helper, installer,
comparator, selection, distribution and orchestration tests passed (49 selected
tests; a further 23-test contract selection also passed). The inherited helpers
retain byte-identical Python; their provenance read paths now contain explicit
child-tree stubs, with the original metadata retained beside them.

The preceding tree/Simple/one-channel experiment made the captured layer-0 and
layer-1 boundaries equal across the 8192 and 4096 chunk arms. Retrieval still
passed only in the 8192 arm. The first observed decision-row difference moved to
layer 2: six of 32 index weights differ, with 490 of 512 selected indices shared.
This does not prove that index weights cause the answer difference.

## Predictions, ranked for discrimination rather than claimed likelihood

1. Shape-dependent index-weight projection: identical captured hidden inputs and
   projection weights produce differing raw projection outputs.
2. An earlier difference in the compiled interval after layer-1 attention:
   layer-2 hidden inputs already differ, despite equal rounded downstream Q/K.
3. The scaling or capture boundary is responsible for the observed discrepancy:
   raw projection outputs agree, while scaled outputs differ.

Capture the final 128 token positions at layer 2: hidden input, normalized KV,
rotated Q, rotated pre-quantization index query, raw index weights, scaled index
weights, and the projection weight. Record the actual selected projection plan,
including whether lookup used the live row count or the capacity fallback.
`B12xLinearMethod.apply` tries the exact `(dtype, rows)` entry first, so equal
capacity does not establish equal plans. TF32 configuration does not establish
the arithmetic of this BF16 projection.

## Boundaries

- Derive from the exact previously gated window image c4a51be481c1c8130a691e72c880c3fa3b751dbddc9b2da7aa6702cc8a0039f9.
- Preserve old layer-0/1 and full-depth decision capture, model math, precision,
  loader, speculation, and numerical-plan family.
- Same fixed 81389 KV blocks, preparation capacity 8192, and explicit chunk
  thresholds 8192 versus 4096, under the approved tree/Simple/one-channel arm.
- No graph capture, blanket device synchronization, or buffer mutation by the
  observation helper. New helper has its own completion lifecycle.
- Require two unarmed cold controls and the armed request to agree within each
  boot. Check image, lock, plans, positions, captured file digests, and health.
- Readiness and a successful helper receipt are not retrieval qualification.
- Keep GLM and nous untouched. Do not edit the benchmark repository.
- Restore the recorded parent after the diagnostic pair. Further numerical
  experiments remain separate from this instrumentation approval.

The local comparator rejects changed projection weights, wrong positions,
nonfinite tensors, and inconsistent shapes. It compares bytes as well as values,
so signed-zero differences are not silently called bit-identical. Its report is
descriptive and does not substitute for runtime identity and control gates.

## Executed 8192 arm

Boot: `receipts/indexer-geometry-tree-simple-1ch-chunk8192-blocks81389-20260926T210629Z/`.
Capture: `receipts/decision-row-matched8192-geometry-tree-simple-1ch-capture-20260926T211037Z/`.
Twelve short repeatability requests passed. The two unarmed cold 524K controls
and armed request all returned `739184, 482617`, finish `stop`, signature
`9230292faaab608b938278952ada116cfac92ec524b103830a7dfc910fba370f`.
Elapsed times were 179.382, 179.399, and 179.646 seconds; these are diagnostic
request times, not a benchmark. Cache accounting reported no reused tokens.
All four ranks produced valid decision, window, and indexer receipts; gathered
file hashes verified. The decision audit is `within-conformance-envelope`.

Request-window logs contained no Xid, NV_ERR_NO_MEMORY, or OOM traceback.
Minimum MemAvailable in GiB: dusty 6.916, toby 10.178, rusty 9.863, kirby 10.111.
This is a request-window observation, not a claim that startup had no warnings.
The boot was stopped cleanly and retained before the next arm.

## Executed 4096 arm and comparison

Capture: `receipts/decision-row-matched4096-geometry-tree-simple-1ch-capture-20260926T212615Z/`.
The twelve short repeatability requests passed. All three cold 524K requests
returned the same wrong answer, `510c94b1bb4f42d2998093bd6b9c3d99, 739184`,
with finish `stop`, no cached tokens, and signature
`8e14506d88b0b174165c02754e724dc273e4f303b4cc77843372c60333a508f5`.
Request times were 186.523, 186.743 and 189.459 seconds, not benchmark results.
All four capture sets passed validation and checksum checks. Request-window
logs had no Xid, NV_ERR_NO_MEMORY or OOM traceback. Minimum MemAvailable in GiB:
dusty 7.367, toby 10.019, rusty 9.436, kirby 9.794.

The full offline comparison passed its structural gates. Every captured
layer-0/1 boundary agrees across grids on all four ranks. At layer 2:

- The captured hidden inputs, projection weights, normalized KV, rotated Q and
  rotated index query are bit-identical.
- Raw index weights differ at 1022 of 4096 values across all 128 captured rows,
  maximum absolute difference 0.0078125, identically on every rank.
- Scaled weights have the same changed positions and differences divided by 64.
- The actual plans use exact row counts 8192 and 4096, respectively, not a
  capacity fallback. Both select `backend=torch, rows_per_tile=8`.

This localizes an observed shape-dependent projection difference on identical
captured operands. The actual cuBLAS kernel and reduction precision are not
traced, and final-answer causality is not established. Within-boot repeatability
does not imply cross-boot equality: these full response signatures differ from
the preceding geometry experiment, and that experiment's wrong 4096 answer
ended with `482617`, not `739184`.

Local CPU reference analysis is saved in `indexer-cpu-reference.json` in the
4096 receipt. FP64 BLAS and a separate ordered FP64 sum agree after BF16 rounding.
Against that rounded reference, the 8192 arm differs at 849 of 4096 values and
4096 at 577. Mean absolute errors to the unrounded reference are 0.000718607 and
0.000656104, respectively. FP32 partial-sum variants of 16 through 1024 products
do not reproduce either arm. The wrong-answer arm is closer by these local
metrics, so it would be unsound to equate the correct retrieval answer with the
more accurate projection. This is not a reference for whole-model correctness.

Both diagnostic boots were stopped and retained. The recorded parent
`e06df11a8ca18fa514d9f28f67cc691aef296da2eeb22b113a734519853bccd7`
was restored under `receipts/router-fence-20260926T214229Z/` and passed six
repeatability requests each at 256 and 1024 tokens. All four live image IDs
were checked, all containers are running without OOM kill, and the temporary
NCCL algorithm/channel overrides are absent. Diagnostic observers were stopped
by the driver's cleanup path. This remains a diagnostic service, not promotion.

The restored baseline boot logged NV_ERR_NO_MEMORY allocation warnings:
dusty 62, toby 44, rusty 32, kirby 59. No Xid, Python traceback or
OutOfMemoryError appeared in its retained final logs. Successful startup and
short probes do not resolve those warnings. They are separate from the clean
request windows of the two capture arms.

Claude reviewed the comparison and CPU reference after completed waits. Both
reviews agree on the observation boundary, not final-answer causality. His
additional CPU accumulation variants also failed to reproduce the captured
outputs; this suggests an accumulation-precision mechanism but does not identify
the actual GPU implementation. No shape-pinning derivative was implemented.

Recommended next authorization: an isolated GPU replay on one idle node, with
the same image and captured tensors, no model loaded. First reproduce both
captured outputs using tail-positioned rows in 8192- and 4096-row inputs. Only
then vary reduction precision or placement and record the actual library
dispatch where available. This separates a small operator reproducer from
another whole-model numerical intervention. Any resulting serving change still
needs its own reviewed scope and qualification.

The local replay harness is prepared as `replay_indexer_projection.py`, reviewed
by Claude, and has not run on a GPU. Eleven comparator/harness tests pass and
four harness tests also pass under `python -O`. It checks input digests before
deserialization, requires the captured Torch backend and visible SM121, records
workspace/allocator environment, strides, alignment and per-sample hashes, and
restores its optional reduction flag on exceptions. A baseline-return arm follows
the optional precision variant. The approved image ID must be verified externally
and recorded in the invocation receipt; library logging must be retained even
if it does not expose a tactic. A cuBLASLt-only log can be empty if Torch selects
another cuBLAS API, so an empty log cannot prove an unchanged algorithm.

The user approved the isolated replay. The four-node diagnostic service was
stopped worker-first, cleanly and with containers retained, under
`receipts/stop-pre-indexer-replay-20260926T215353Z/`.

## Isolated GPU replay

Receipt: `receipts/indexer-gpu-replay-20260926T215626Z/`.
The exact image `25c92dde...` ran on idle dusty with no model loaded. Capture and
script digests verified; image inspection and exact invocation are retained.
All variants repeated three times per shape with identical within-shape hashes.

| Variant | Differences between grids | Differences from serving, 8192 / 4096 | Differences from rounded FP64, 8192 / 4096 |
| --- | ---: | ---: | ---: |
| Default BF16 reduction | 1022 | 0 / 0 | 849 / 577 |
| Reduced-precision reduction disabled | 0 | 848 / 576 | 2 / 2 |
| Default restored | 1022 | 0 / 0 | 849 / 577 |

Library logs now show the actual dispatch. Default and return use cuBLASLt
algorithm 21, tile 32x32, in-place reduction, with three K splits at M8192
and two at M4096. Disabling reduced-precision reduction introduces the
`REDUCTION_SCHEME_COMPUTE_TYPE` heuristic mask and selects algorithm 67, with
different tile configurations for the two shapes. Those two configurations
produce identical observed outputs, despite not being the same tactic.

This is a controlled operator-level reproduction and intervention: the
default reproduces both serving captures exactly, the flag removes all observed
shape differences, and the return restores the original results. It is not
proof of general batch invariance or final-answer correctness. The two remaining
FP64-reference mismatches are retained rather than hidden by a tolerance.
No process-wide precision setting was changed in the serving workers.

The isolated container exited successfully and was removed by `--rm`; all four
nodes were checked idle afterward. Parent restoration completed under
`receipts/router-fence-20260926T215703Z/` with 12/12 short repeatability probes.
All four running image IDs were independently rechecked against `e06df11a...`;
none was OOM-killed. Startup allocation warnings remain: dusty 82, toby 40,
rusty 31, kirby 56 NV_ERR_NO_MEMORY lines, with no Xid, Python traceback or
OutOfMemoryError in retained final logs. GLM was untouched.

The isolated image's allocator environment variables were unset, unlike the
serving runner's expandable-segments override. Exact reproduction still held
for both shapes; that override is therefore not necessary for this isolated
projection discrepancy. This does not establish equivalence of allocator
behavior for the full model.

Claude's completed replay review independently confirmed hashes, reversibility
and the library trace. Options discussed were a fixed SIMT backend for the
index-weight projection, a process-wide reduction flag set once before worker
initialization/capture, or a per-call global toggle. The per-call toggle is not
recommended because it mutates shared process state on the hot path.

Do not overextend this result: two FP64-reference mismatches have not been
individually traced to rounding boundaries; zero cross-grid differences on
these 128 rows does not prove invariance at every row count. Likewise, a later
full-model mismatch under the flag would not by itself exclude all BF16 GEMM
contributions. The prospective worker-wide arm needs an explicit scope inventory
and approval because it may change other Torch BF16 GEMMs, not just this indexer.

## Read-only scope inventory before the proposed worker-wide arm

`inventory_bf16_scope.py` scanned the exact running parent image without importing
Torch or performing GPU work. Its source hashes and lexical matches are in
`receipts/indexer-gpu-replay-20260926T215626Z/bf16-scope-inventory.json`.
All four requested source roots existed; 27 files matched, including comments
and reference/test implementations. This is not a complete dynamic call graph.

Relevant additional implementation surfaces are the B12X BF16 vocabulary
projection's Torch backend and fused-MoE fallback routing when callers omit
router logits. The DS4 shared attention code also contains compressor matmuls
with FP32 output, and fallback output projections contain batched matmuls.
Their execution and operand dtype in this exact DS4.1 profile are not proven
by a lexical match. Reference-only FP32 attention calculations are not evidence
that the precision switch would change attention kernels themselves.

The runtime wrapper uses `os.execv` to start vLLM. Setting a Torch Python flag
in that wrapper would not configure the replacement process or its spawned
workers. A proposed worker-wide arm must set and report the flag inside each
GPU worker before model preparation and graph capture, such as the verified
`gpu_worker.py:init_device` boundary, and not toggle it on the inference path.
That was the state at the scope review. The user subsequently approved the
worker-wide diagnostic. It is implemented and build-gated as image `58b0720238b8`
and is being distributed; full-model results remain pending. See
`PRECISION-ARM-20260926.md` for its exact identity and acceptance boundary.
