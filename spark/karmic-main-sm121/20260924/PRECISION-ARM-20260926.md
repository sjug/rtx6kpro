# Worker-wide BF16 reduction diagnostic

User approved preparation, build and matched full-model tests on September 26.
This is a diagnostic numerical change, not a promoted image or a completed fix.

## Evidence and prediction

The isolated replay in `receipts/indexer-gpu-replay-20260926T215626Z/`
reproduced both serving projection outputs exactly. Disabling Torch's BF16
reduced-precision reduction changed 1022 cross-grid differences to zero;
restoring the default restored both original hashes. cuBLASLt recorded in-place
split-K reductions with three and two splits, respectively, for the defaults.

Prediction: setting the same flag once in each serving worker will eliminate
the captured layer-2 index-weight discrepancy. Whether later layers agree or
the historical retrieval answer becomes correct remains an experiment, not an
assumption. Full-model differences could remain even with this flag.

## Scope

- Derive from the exact indexer-capture image `25c92dde...`, retaining all three
  observation helpers, native artifacts, checkpoint and B12X code.
- Set the flag inside each GPU worker before preparation and graph capture,
  not in the launcher that subsequently execs vLLM, and never toggle it per call.
- Report and require each worker's rank and effective setting in boot logs.
- Preserve the 600K matched diagnostic envelope, 81389 KV blocks, preparation
  capacity 8192, explicit 8192/4096 chunk thresholds, K7, deterministic MoE,
  turbo off, and the approved tree/Simple/one-channel NCCL diagnostic geometry.
- Keep GLM and nous untouched. Stop DS4.1 workers first; retain containers.
- Use the established Docker archive transfer without conversion or compression.

The setting can affect every eligible Torch BF16 GEMM in the worker, including
other BF16 projection backends. The read-only lexical inventory is recorded in
the isolated-replay receipt; it is not a dynamic call graph or a claim that every
matched call executes. The new arm is intentionally broader than one indexer.

## Execution and verdict

Installation gates passed on image
`58b0720238b8d03abb8d11b32dcac8ad66daec75aa1cddc83bf2869c94339c09`,
lock `541ba7997b38b3e9ba2afb286df3f3796beed99def29582200adb3764f198974`.
The vLLM tree is `1ac17c6371991c04acd212673fbbeced0fbe4003`;
B12X remains `03a4e0b363d17c291516678689ae72d3e266749e`.
Build evidence is in `receipts/precision-build-20260926T223502Z/`.
The GPU-visible helper reported the flag changing from True to False.
Sixty local tests passed. Claude reviewed the integration and found no blockers.
Full-model acceptance remains pending; these are not qualification results.

The 8192-token arm completed three cold historical-input trials, all correct
(`739184, 482617`, finish `stop`) with identical response signature
`9d0a2bad9a933f85b4badc8a533777eda25d0dad9fd4434b4e8ee8fa7c79083e`.
Elapsed times were 181.49, 180.78 and 184.12 seconds. The third trial captured
all three observation types on all four ranks with no helper-reported problems.
The kernel plans stayed at the pinned matched regime. Evidence:
`receipts/decision-row-matched8192-geometry-tree-simple-1ch-capture-20260926T224649Z/`.
The 4096-token arm then completed three cold trials, all correct with signature
`c2b1e17bd003fc1f22a16794bb1729e396863ae0c349aa740391ecb8ccd23abf`.
Elapsed times were 185.89, 187.36 and 186.18 seconds. Evidence:
`receipts/decision-row-matched4096-geometry-tree-simple-1ch-capture-20260926T230221Z/`.
The baseline 4096 arm on image `25c92dde` was wrong in all three trials.

All six precision trials have exactly equal reported logprob entries for tokens
0 and 1, including their top-20 alternatives. Differences between chunk grids
start at token 2; the chosen output tokens remain equal. This does not prove
full-logit or earlier-row equality. Within each boot, all complete response
signatures match. Both arms accepted 4 of 13 drafted tokens per trial, but their
per-position acceptance/verification patterns differ. That is consistent with,
not proof of, a speculative-path explanation of the later logprob differences.
The inference-window final logs contain no NV_ERR_NO_MEMORY, Xid, Python
traceback or OutOfMemoryError on any rank. Startup evidence remains separate.

Cross-grid tensor audits completed on idle nodes. All layer-2 captured fields
agree bit-for-bit on every rank: hidden input, normalized KV, rotated queries,
raw index weights and scaled index weights. The layer-0/1 window structural
checks pass. Full-depth comparison reports no differing captured layer across
40 layers on any rank, including decision-row queries, attention outputs and
sliding-window values. The matched baseline first differed at layer 2's output
and layer 3's query/sliding-window values on every rank. These observations
cover the captured rows only, not every prefill row or the entire KV cache.
The reports are `indexer-comparison-*.json` and
`geometry-decision-cross-*.json` in the 4096 capture receipt above.

A further CPU-only rank-0 cross-image check compared the baseline 8192 capture
(`20260926T211037Z`) against the precision 8192 capture. Hidden inputs, projection
weights, KV normalization and rotated query tensors are bit-identical. The raw
projection output changes from SHA256 `c7903da91307f56e4ccd97a30d0445bf6c47083cf1bd6b8351a6cbf5eb88c9fe`
to `e48b5773928f30a2bae29d65bfb32c950c93c9c2e3a2e42eef3aa85110b8a6cc`,
exactly the isolated replay's default and reduced-precision-off hashes.
There are 848 changed raw values, maximum absolute difference 0.0078125.
This strengthens the projection-level causal link without identifying that
projection as the only class-wide flag effect on the final answer.

Testing without diagnostic NCCL geometry remains required. The class-wide
flag does not isolate the indexer projection as the sole cause of the answer
change. These matched tree-geometry passes are not qualification of the
ordinary serving profile.

### Standard-NCCL follow-up, in progress

The standard-upstream 8192 arm started at 23:25 UTC, retaining the same image,
81389-block matched allocation, K7, and precision setting but removing the
tree/Simple/one-channel overrides. All 12 short repeatability trials passed.
The first two cold original 524K requests returned `739184, 482617`, stopped
normally, and had identical response signature
`fba6114819113ce773e8bae15429a81db022eadd4217347d8848ac23be828430`.
Elapsed times were 187.23 and 170.83 seconds. The captured third trial was also
correct with the identical signature (233.53 seconds, including instrumentation).
All three capture families were retained from all four ranks, without reported
capture problems. The idle audit passed `within-conformance-envelope`; it
checks the captured computation and does not itself judge answer correctness.
The 4096 arm also passed all three cold historical requests, with signature
`e4c50374c6e9d2067cc7bf0129a56d1bdd0386e0ff9b3bf3fc3950fd12f27067`
identical within the arm. Elapsed times were 219.41, 236.72 and 291.62 seconds;
the third includes capture instrumentation. All four decision captures are
valid, and the pinned kernel selections are unchanged. Its receipt is
`receipts/decision-row-matched4096-nccl-standard-upstream-capture-20260926T234907Z/`.
The idle tensor comparisons and structural checks passed. All ranks first
differ at layer 0 `wo_reduced`, with equal captured `wo_partial` values. The
decision-row comparison first differs at layer 1 attention output and stored
sliding-window values, and layer 2 queries. This localizes the earliest observed
standard-NCCL discrepancy to the reduction boundary, not to the index-weight
projection. It does not prove the exact NCCL algorithm or that no other
shape-dependent arithmetic exists. This is not a completed normal-profile
qualification or a performance comparison.
Receipt: `receipts/decision-row-matched8192-nccl-standard-upstream-capture-20260926T233036Z/`.

Unlike the tree/Simple/one-channel arms, the standard-NCCL arms have different
reported logprobs across chunk grids. Correct first token `739` versus `510`
has margin +0.125 nats at 8192 and +2.75 at 4096. Same-boot repeatability holds,
but cross-chunk bitwise equality does not. The narrow 8192 margin strengthens
the need for the existing concurrent retrieval and independent clean-boot
checks. No full-model numerical oracle exists here, and a correct answer alone
does not establish numerical correctness. NCCL reduction order is a candidate
for the residual cross-grid differences, not a conclusion from logprobs alone.

The clean release artifacts derive from uninstrumented router image `e06df11a`
and preserve its repairs and native inventory. Only the worker initialization
hook and precision helper are added. Fourteen release artifact tests passed
locally after independent review. The integration proposal passed 17 tests on
a patched copy, was reviewed and applied, and 73 combined tests passed in the
Torch-equipped CPU environment. A first combined invocation using system
Python failed only because it lacked Torch; the build did not start in that
invocation.

The release build completed on idle dusty at 00:10 UTC on September 27:
`1989e16daf38d2966b03a2e2abcb878263fb7d17183c8d6c5b084a51fbed7e8f`.
Lock SHA256: `4b1afffe455d634cbadc82353a64beb3c96bbaf89b595c948c00dbfa3b1f1c32`.
vLLM tree: `0a446838b6cc8d5ad3de0d4e2020e117813d1d57`; B12X unchanged.
Native, router projection (4), Engram epoch (8), Engram enqueue (16), compressor
ring mapping (29), and precision smoke gates passed. The image inventory gate
permits exactly two vLLM source changes over the uninstrumented router parent.
The build receipt is `receipts/precision-release-build-receipt.json`.
Distribution completed with identical image IDs on all four nodes, using archive
SHA256 `f90540cafe3de2389253aa16b4590b18b473a45890d51ffe0a632afaccc8ad29`.
The clean release boot started at 00:16 UTC; receipt
`receipts/precision-release-20260927T001624Z/`. Full model qualification and
promotion are pending.

Review artifact composition and integration before building. Require immutable
source identities, preserved native inventory, exact worker source delta, helper
import/flag tests, and consistent child provenance in all captured tensors.

For each matched chunk grid: verify all ranks, run short repeatability probes,
two unarmed cold historical 524K requests and one captured request. Preserve
all responses and logprobs, cache accounting, effective plans and health logs.
Require controls to agree within the boot; do not substitute new prompt wording.

Collect decision, layer-0/1 window and layer-2 indexer captures. While nodes are
idle, run structural checks and same-image cross-grid comparisons at full depth.
Compare raw/scaled index weights, subsequent attention outputs and final answers.
No promotion on operator equality alone. Any failed original retrieval remains
visible and blocks full qualification. If the result supports a repair, remove
diagnostic instrumentation in a separately gated image and perform the complete
correctness, context, concurrency and benchmark qualification before promotion.
