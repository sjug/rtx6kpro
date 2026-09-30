# Source attribution of the historical 524K retrieval failure

Read-only analysis, September 25 UTC. No serving change or new GPU experiment.
Pinned vLLM `1794dcf18454900263e0c66711af8ea4a1283ac1`, B12X
`a7d7d29b2ef8869086e0ceaa787321f17544e3c9`.

## What is established

The exact historical input returned the requested codes once and the archive
identity twice across three cold greedy requests on one boot. Both the repeated
input and aggregate cold-cache accounting were verified. The first-token
code-minus-identity margins were +1.75, -3.75 and -6.50. This establishes unstable
logits and a failed answer gate, not a particular erroneous arithmetic operation.

The earlier R38 investigation separately isolated two sources of variation:
deterministic MoE accumulation made short requests bitwise reproducible, while
variation returned as the DSA indexer began selecting among more than 512
candidates. The standalone production selector returned mathematically valid
top-512 sets but different token sets when scores tied at the boundary.

## Source chain: unstable attention membership

At the pinned B12X source:

1. `b12x/attention/dsa_indexer/mxfp4.py:1136-1137` allocates BF16 scores
   and FP32 logits. Widening an already rounded score does not undo its ties.
2. `select_mxfp4`, lines 1318-1337, prepares logits, calls `run_row_topk`,
   then sorts the selected results. Selection precedes that final sort.
3. `tiled_topk.py:791` bypasses radix selection when the candidate count
   does not exceed top-k. This is consistent with the earlier threshold tests.
4. `tiled_topk.py:1160-1165` resolves the final equal-score bucket using an
   atomic decrement of the remaining-slot counter. The threads arriving first
   get the remaining slots. Token index is not a secondary comparison key.
5. The overflow path at `tiled_topk.py:476-479` likewise atomically assigns
   slots to equal-pivot entries and keeps only those below the top-k limit.

Equal indexer scores do not imply equal attention keys or values. Exchanging a
selected historical token for a different equal-scoring token can therefore
change attention output, subsequent hidden states, and next-token logits.
Sorting the retained indices afterward cannot recover excluded candidates.

An AST comparison against R38 B12X `ce419b52` found these definitions identical:
`DSATiledTopkKernel`, `_exact_overflow_fallback`, `_convert_to_uint32`, and
`_smem_xadd`. This is stronger than assuming the behavior survived the wrapper
refactor. It does not identify which production attention rows drove the final
answer change; those rows were not captured.

## Separate source of variation: MoE reduction

`moe/_shared/kernels/w4a8_phase2.py:725-763` chooses either one-writer stores
followed by fixed-order reduction, or BF16 atomic accumulation. In
`moe/fused_moe/_impl.py:1410`, `B12X_DYNAMIC_DETERMINISTIC_OUTPUT` defaults off
for eligible dynamic paths. The earlier deterministic-MoE arm did not resolve
the historical answer failure. MoE is therefore not a sufficient explanation
on its own, although it can perturb scores upstream of the indexer.

## Real additional defect, not attributed to these answers

vLLM `common/engram.py:620-636` spins for a ready epoch, then writes a failure
flag and returns after 5000 ms. `Engram.forward`, lines 900-903, subsequently
consumes staged rows without checking that flag. All uses of `overlap_epochs`
and `_engram_epochs` in the pinned vLLM tree were inspected: none reads the
third failure element. `nvidia/model.py:508-530` publishes readiness from a
side-stream lookup job; `_finish_engram_job` waits on the Python future, not a
failure flag. This is a fail-open timeout path if readiness is not achieved.

Whether this path timed out in our run is unknown. High GPU utilization, low
power and long latency do not establish that it did. R38 lacked this overlap
path and reproduced the answer failure, so it cannot be the sole explanation
of the historical defect. It deserves a separate regression/report.

## Missing causal test and proposed order

The existing B12X `_reference_topk_indices_from_logits` in
`dsa_indexer/_impl.py:1419-1445` uses a stable descending argsort. It provides a
diagnostic reference for deterministic membership, not a drop-in production
performance fix. Candidate-to-physical-index mapping, sentinels, output
buffers and graph capture must be preserved in any injected control.

Before another expensive replay:

1. Prepare a non-serving selector test that verifies stable membership under
   ties, including the overflow path and real MXFP4 selector mapping. Retain
   the old score-validity oracle, but add identity stability assertions.
2. In a separately approved diagnostic image/profile, observe the Engram
   failure flag rather than infer it from timing. A detected timeout invalidates
   an attribution to the selector until overlap is controlled separately.
3. Hold deterministic MoE accumulation fixed in both comparison arms; compare
   the current selector with deterministic tie-breaking at the short threshold
   and then on the original 524K input. Do not combine a new overlap policy and
   a new selector in the same attribution arm.
4. If logits become repeatable but the wrong answer remains, reproducibility
   is fixed but retrieval is not. A non-B12X/reference forward or captured
   layerwise parity test is then needed to separate model behavior from bias.

Changing the selector/profile or instrumenting a worker requires the user's
explicit approval under the existing launcher contract. No such change was
made by this source investigation. There is no basis yet to promise that
deterministic tie-breaking makes the model select the correct retrieval code.
