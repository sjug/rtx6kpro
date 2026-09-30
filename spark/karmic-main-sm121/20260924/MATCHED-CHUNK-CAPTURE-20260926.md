# Matched 8192/4096 decision-row captures

Status: approved experiment completed; no model promotion or new numerical fix.
Independent final review is recorded below. GLM was not touched.

## Identity and controls

Image `4d39defb3d52c9d12665fde0d5567a5cda09adc6abfb5ea90942b12e65536cb8`,
capture lock `3eb16ccca36e78ebbf133fb238ed8b0ffc261316c71ada958ab810ef8f07e8b4`,
kit `a0acb8e8b159b31382eb7528cea0ae44598c89fc1a6586d170e0ff1abb0ae474`.
Both arms use 81389 KV blocks, prepared capacity 8192, the same frozen
524288-token prompt and the same eight pinned sparse-attention plans.
Only the long-prefill threshold changes. This does not mean all operators
execute identical shapes: changing the chunk schedule necessarily changes them.

Each arm has two cold unarmed controls and one armed replay. All three full
response signatures match within each arm. Every request has zero prefix hits;
the driver checks counters rather than relying on nullable usage details.
Capture metadata, source trees, plan identities and all four capture hashes
were checked. Each boot passed twelve short repeatability probes.

| Chunk threshold | All three answers | First-token margin, code minus identity |
| --- | --- | --- |
| 8192 | `739184, 482617`, correct | +0.75 |
| 4096 | `510c94b1bb4f42d2998093bd6b9c3d99, 482617`, wrong | 0, exact reported tie |

Signatures: 8192 `60756062c20115c6b4b10e044cfb95bf4aadb2b3ea4b91514a0f7c40242f852d`;
4096 `803fc580a62d51ec51339f0481a44cf72517653bd90650a691fa362d4207da9b`.
These outcomes also match prior fixed-grid/fixed-regime controls. They establish
repeatability of this grid effect, not chunk-invariant arithmetic or model accuracy.

4096 control 1 took 544.9 seconds and included debugger attachments during a
low-power interval; it is not timing evidence. Same-user GDB located a host
yield inside libcuda, but CUDA-GDB reported no contexts/kernels and identified
no GPU kernel. Both detached. Control 2 took 203.84 seconds and the armed
replay 177.60 seconds without debugger attachments. No timing cause is proven.

## Per-side operator audits

Both audits return `within-conformance-envelope`: 160 attention rows per arm,
eight exact indexer selections, no mapper or replicated-input mismatch.

| Compute mode | Worst per-head relative L2, 8192 | Worst, 4096 | Envelope |
| --- | --- | --- | --- |
| BF16 | 0.002585 | 0.002585 | 0.006968 |
| FP8 | 0.043152 | 0.037441 | 0.539488 |

The FP8 bound is broad and provides weak conformance evidence. The references
consume captured cache bytes; they do not validate how those bytes were
produced or provide an independent end-to-end model oracle.

## Cross-grid observations

All four ranks agree on the first observed locations:

- Layer 0: captured query, output, sink and decoded active SWA cache values match.
- Layer 1: query and sink match, but cached values and attention output differ.
- Layer 2: first differing captured query and first indexed selection differences.

Layer 1 SWA differences occupy row offsets 16 through 111, inclusive, of the
128 active records. On every rank, 13154 of 65536 decoded values differ;
relative L2 is 0.01783248 and maximum absolute difference is 0.125. Rank 0's
raw row multisets differ too, so this is not merely a permutation of the same
records. The comparison excludes physical slot addresses and unused tail rows.
Read-order mapping remains subject to source review.

Rank 0's layer 1 output relative L2 across grids is 0.00439819. Its layer 2
query relative L2 is 0.02265414. The layer 2 top-512 sets share 404 entries.
Both retrieval-code spans remain selected at all eight inspected indexer layers
on both grids. Therefore a missing needle in these selections is not supported.

The earliest captured difference is an operand difference before layer 1
attention. It does not prove a defect in that attention kernel or identify the
producer: earlier rows and the path between layer 0 and the layer 1 cache write
were not captured. Cross-grid differences are descriptive, never judged using
the same-input conformance envelope.

## Evidence and health

- `receipts/decision-row-matched8192-capture-20260926T152805Z/`
- `receipts/decision-row-matched4096-capture-20260926T155201Z/`
- In each: controls, armed response, zero-hit counter receipts, identities,
  numerical plans, hashes, telemetry, kernel-journal collection and `audit.json`.
- In the 4096 directory: `comparison-{node}.json`,
  `comparison-swa-{node}.json`, `comparison-swa-multiset-dusty.json`, debugger logs.
- Comparison helpers retained on dusty in the same receipt directory:
  original SHA `76b48579f0babf1188ac581acebb504b0cb564ef3b8b25b81cbe99af67836686`;
  SWA SHA `1088bfb1e4d5802df8e6cc050d5820f3bb510f7ef49866f65c20e27dae2abda1`;
  multiset SHA `5f9e8ec432089e5103226d49b439e611bc1505fc3c12380ca88908314f08b840`.
  Nine comparator tests pass, including active-row bounds and permutation checks.

Startup NV_ERR_NO_MEMORY warnings remain recorded separately. Both capture
request windows have successful kernel-journal queries with no entries and
zero collector exit codes. Observers were cleaned up by the drivers. Both
diagnostic boots were stopped worker-first, exited zero, and were retained.
No image deletion, prune, commit, push or benchmark-repository edit occurred.

## Restored state and review boundaries

The non-instrumented parent `e06df11a8ca18fa514d9f28f67cc691aef296da2eeb22b113a734519853bccd7`
is restored on all four nodes with automatic KV sizing and no capture/block/
chunk-threshold overrides. `receipts/router-fence-20260926T161958Z/` records
81663 serving blocks, 11408102 reported KV tokens, and twelve correct identical
short probes. Every container is running and not OOM-killed; no observer
process remains. Boot/first-probe NV_ERR_NO_MEMORY counts are 44/43/47/67 on
dusty/toby/rusty/kirby, with no Xid. This is a restored diagnostic baseline,
not a newly qualified production image. No long-context gate was rerun on it.

Claude independently confirmed receipt identities, cold-cache accounting,
within-grid response equality, audit results and comparator tests. Main
retains narrower interpretation than several initial review inferences:

- SWA is FP8 E4M3 with per-32 UE8M0 scales, not the indexed MXFP4 format.
- Mapping must be checked against the pinned `deepseek_v4_1` implementation,
  not an unrelated current-checkout `deepseek_v41` path.
- Equal final-row outputs do not prove earlier-row inputs were equal.
- Shape-dependent GEMM arithmetic need not change every row, so unchanged
  rows cannot exclude it.
- Identical differences on four ranks support reproducibility of this
  observation, not universal exclusion of races.

126 combined local tests pass after the offline comparator additions.

Claude accepted the narrower interpretation after a completed follow-up
review. Next proposal, not yet authorized: an instrumentation-only derivative
capturing the final 128 rows through layers 0 and 1 on the same matched grids.
Include layer 0's query rows and 255 KV records needed for their 128-token
windows, then existing no-compile/custom-op boundaries through the layer 1
pre-quantization KV and cache write. Preserve the decision-row snapshots;
check output signatures and their prior tensors for observed perturbation.
Do not insert instrumentation inside compiled fusion regions. Capture absolute
positions, token IDs and relevant plan identities, and bracket TP reductions
where the selected boundary requires it. No numerical kernel or serving-default
change is proposed. The request for build/replay approval has been sent; no
new derivative has been authored, built or launched.

Subsequent approval: the user answered "yes" to the focused capture request.
`window-capture-approval-20260926.json` records that new scope. The original
capture above remains immutable evidence. New helper preparation and local
orchestration tests are in progress; a new build and replay must not be
reported complete until their separate receipts exist.

## Pinned-source mapping follow-up

The initial read-only check used older vLLM `57a80980`. The follow-up checked
the actual upstream pin `1794dcf18454900263e0c66711af8ea4a1283ac1` from
`runtime.lock.json`, plus the local `ring-fix.patch`. The sparse metadata file
is unchanged between those two upstream commits. This is not an additional
live capture or proof that every runtime metadata value obeyed the mapping.

- `deepseek_v4_1/sparse_mla.py::_chunk` emits SWA indices in chronological
  order: `logical = max(position + 1 - WINDOW, replay_start) + column`,
  then `physical = block_table[logical // PAGE] * PAGE + logical % PAGE`.
  For the full 128-row window ending at 524287, offsets 16 through 111
  consequently denote positions 524176 through 524271. Physical page IDs
  need not match between boots; the comparator gathers through each arm's
  own slots before comparing records.
- The local ring patch changes validity checks for circular metadata only.
  It does not change `_chunk`'s chronological read ordering. This does not
  by itself exonerate the write metadata.
- `attention.py::_forward` computes fused QR/KV, normalizes KV, then calls
  `_cache_context_kv`. That existing eager boundary resolves live metadata;
  `insert_context_kv` rotates KV and passes it with live slots to
  `mla.write_cache`. These are distinct input, rotation and packed-write
  boundaries to distinguish, rather than treating cache bytes as the
  direct output of the projection.
- `_forward` is entered through `vllm::dsv41_b12x_attention`, an existing
  custom op. `_cache_context_kv` and `forward_mqa` have existing
  `eager_break_during_capture` decorators. A future capture must preserve
  those boundaries and validate its patch against the actual composed
  runtime tree, not assume that the base blob is the deployed blob.

The next probe should distinguish three possibilities: earlier-row hidden
states already differ; equal hidden states produce different projected or
rotated KV; or equal rotated KV produces different packed records/addresses.
The existing final-row capture cannot choose among these. No new runtime
experiment was performed during this source-only follow-up.

The existing local `claude-decision-row-attention.py` hashes to its lock's
`8594274f59b3d7340690ae092f4efeb2986754b685c7bd4c964f851f9768e749`.
An AST comparison against the actual upstream pin confirms identical
`_attention`, `_forward`, `_rotated`, `_cache_context_kv`,
`insert_context_kv` and `_o_proj` functions. Thus these specific producer
boundaries are checked against the frozen capture source, not just the older
base used initially.

The actual pin adds L2-prefetch calls after query projection and before
attention's TP all-reduce compared with `57a80980`. The shared prefetcher
returns immediately when `num_tokens` exceeds
`VLLM_GLM53_L2_PREFETCH_MAX_TOKENS` (default 256). Consequently these calls
do not launch on the 4096/8192-row encoder forwards under that default.
This is a conditional source conclusion, not a newly inspected worker-env
receipt or blanket exclusion of prefetch elsewhere, including 128-row CED
decoder forwards.

For a future capture, the attention custom-op entry can preserve the last
128 hidden-state rows without adding a new outer compile boundary. Inside
that existing op, retain the normalized KV before rotation and the rotated
KV immediately before packing, along with positions and actual write slots.
Capturing attention output must distinguish the sparse attention result from
the WO projection and its TP-reduced result: `_attention` returns the latter,
while the current `forward_mqa` hook captures the former. Equal current
layer-0 snapshots therefore do not establish equal layer-0 block outputs.
The existing no-compile boundary is a suitable probe location, not evidence
that adding copies cannot perturb execution. The proposed capture still
requires matched unarmed controls and comparison to the existing snapshots.
