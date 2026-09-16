# Qwen R32 to R38 prefill investigation

Local source analysis, 2026-09-15. No node access, GPU execution, image
builds, serving changes, source-repository edits or external filings are
authorized for this investigation. Qwen stays on its qualified R32 profile.

## Measured symptom and limits

The existing qualification established a repeated Qwen prefill throughput
loss on GB10 at equal SM clocks. The final R38 return arm, with R32 run in
between, compared as follows:

| Fresh prompt | R32 tok/s | R38 tok/s | Change |
| --- | ---: | ---: | ---: |
| 8K | 3032 | 2957 | -2.47% |
| 16K | 2995 | 2904 | -3.04% |
| 32K | 2908 | 2839 | -2.37% |
| 64K | 2748 | 2677 | -2.58% |
| 128K | 2483 | 2422 | -2.46% |

The long-context deficit repeated over three R38 boots. No material decode
regression was established. See `fresh-r32-vs-r38-r03.json` and the previous
review receipts for the other boots. The existing comparison script was
re-executed locally against that file's two raw input paths, returning
`grids_valid: true` and the numbers above. This rechecks archived measurements;
it is not a new performance experiment or a causal bisection.

GLM R38 prefill was within 1% of its R32 baseline. DSv4 Vision did not show
the same repeatable Qwen pattern. Those are different workloads and shapes,
not controls that categorically exonerate shared libraries.

## Frozen source and runtime evidence

The release lock and patches, not current upstream branch tips, define the
comparison. Read-only local object stores are under `../.compose/repos/`.

| Source | R32 runtime tree | R38 runtime tree |
| --- | --- | --- |
| vLLM | `80a18accc688cdee2974f4c5e03e9416b2896087` | `077347fdeab296404d0b0cb297d316176acc635a` |
| B12X | `c4bfeee9f3c9457400d191c870eb2e44fbcd5c2e` | `6abad73444018f7ce1f0bdc3f0992649a4c21722` |

Upstream commits are vLLM `5576927057cf71b6ec61d120932338b333efa089`
and `66c293578412417476f842c1da5805d3a3d959a8`; B12X
`3edbcbce70f491741b82f5eab9c1b30b39447228` and
`ce419b52681b7922bb0972d4b58b590a3fd005b2`. These are composed, divergent
lineages. Commit ancestry alone does not identify runtime changes; verify
each candidate against the two frozen trees before designing a reversal.

Saved head container inspections under `contrast-r32/` and
`contrast-r38-return/` show identical node command arguments. Relevant
VLLM/B12X/NCCL/CUDA/Torch/FlashInfer environment differences are the
fingerprinted cache directories. The launcher diff changes release text,
alias and identity preflight, not performance settings. Saved boot logs
show the same attention block size, 2,848 tokens, and 4,096-token budget.

FlashInfer did change, from `1ac6942776b383c6b03c7a5805a22e72a3e3349f`
to `803c4664f4771ddc418f20a57f752469a237a825`. Torch 2.13.0,
NCCL 2.31.2 and the FlashKDA native identity were retained. Dependency
Python patches remain part of the R38 delta; equal version strings alone
do not establish equal behavior.

## Confirmed backend change, not yet a confirmed cause

Current-boot lines in the saved logs show:

- R32, 14:24 EDT: `Using FlashInfer GDN prefill kernel (requested=auto, head_k_dim=128)`.
- R38 return, 14:47 EDT: `Using b12x CuTeDSL GDN prefill kernel (requested=b12x, head_k_dim=128)`.
- Both retain B12X GDN decode. Filter logs by boot timestamp because reused
  container names also expose historical lines.

vLLM commit `60b7b7191b` changes `_resolve_gdn_backend_selection` in
`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`: selecting
B12X decode also selects B12X prefill. Thus the unchanged launcher option
`--gdn-decode-kernel b12x` changes the prefill execution contract in R38.

A CPU-only extraction and execution of that exact selector verified:

| Explicit selection | R38 resolver result |
| --- | --- |
| decode=b12x | prefill=b12x, decode=b12x |
| prefill=flashinfer, decode=b12x | ValueError, conflicting backends |
| prefill=flashinfer, decode=cuda | prefill=flashinfer, decode=cuda |
| Neither, Qwen model type | prefill=b12x, decode=b12x |

This checks selector behavior only. It does not validate CUDA-backend
support or numerical/performance behavior of any proposed control.

The B12X operator is introduced in `75ffee6`. R38's GB10 profile has no
`sequence.gdn_prefill` entry. Under the recorded auto policy, the operator
therefore resolves its generic sequential heuristic, with v_split=64,
k_split=1 and stages=3. The existing `attention.gdn` profile is a separate
component and is unchanged; it must not be cited as tuning for this new
prefill operator.

The pinned `docs/gdn-prefill.md` explicitly labels the API research-only.
It reports Max-Q coverage for segment-parallel execution, while GB10
qualification of that algorithm and measured embedded profile integration
remain incomplete. That does not establish that the active sequential path
is incorrect, nor does it supersede the passing R38 serving correctness
receipts. It does establish that a measured GB10 prefill profile was not
the basis for this default selection. Commit `b188c9c` removed heuristic
fallback warnings, so absence of those warnings is not evidence of profile
coverage.

The new integration also routes pure prefill through the mixed-batch GDN
wrapper. That path stages inputs and metadata, gathers/scatters output and
visits speculative handling whenever state-index columns exceed one.
Some kernels can do no live work; source-level launch counting is not a
timing measurement. Recurrence scheduling and wrapper overhead are separate
sub-hypotheses to discriminate later.

This is a strong workload-specific lead: GLM uses KDA, and DSv4 uses MLA,
not this Qwen GDN prefill path. It remains a hypothesis until a matched
GB10 experiment recovers the measured loss.

## Narrowed alternatives

The GB10 `moe.decode` profile is not evidence of a Qwen W4A16 tuning
regression. A recursive comparison of condition paths and selected configs
found 8,449 leaves in R32 and 8,079 in R38. All 370 changed or removed
leaves select `w4a8_mx`; no NVFP4 or W4A16 leaf changed. This excludes that
specific profile-drift explanation, not changes in the MoE implementation.

There is a narrower W4A16 implementation delta in B12X `5dfba5e`:
`W4A16TopKSumKernel` gains an optional FP32 output and widens one flattened
FC2 address calculation to Int64. The compile-spec version changes from
3 to 4; that is a cache identity version, not a pipeline-stage count.
Reachability of the changed branch and its timing contribution need proving
before this becomes a performance explanation. Do not undo overflow-safety
changes speculatively.

The shared-expert lifetime correction in vLLM `0627ffd885` adds
`output.record_stream(current_stream())` after the producer-stream wait.
Its stated contract is allocation lifetime, not changed arithmetic or an
added global synchronization. Keep this correctness fix. If later traces
implicate allocation overhead, test a safe alternative lifetime scheme,
not a blind reversal that restores the demonstrated use-after-reuse risk.

PLE metadata and graph-integration changes in vLLM `41ea64ae68`,
`8770c10708`, `4b276a363c` and `60b7b7191b` are a lower-priority,
Qwen-specific alternative. The table placement did not change under the
recorded environment, so new CPU/NVMe offload is not the explanation.

The FlashInfer source-tree delta is exactly
`csrc/sparse_mla_sm120_prefill.cu` and its test. That does not explain a
changed GDN prefill implementation inside FlashInfer, and R38 uses B12X
for this operation anyway. This narrow source exclusion does not prove
equivalence of all build products or dependency integration.

## Reviewed evidence limits

Claude's completed report was reviewed through Herdr against the pinned
sources and archived receipts. The supervisor retained the ranked GDN,
W4A16, PLE and allocation-lifetime leads with these qualifications:

- Equal graph-capture inventory counts exclude a changed count, not a
  change in graph contents, eligibility or mixed-batch metadata. The GDN
  integration changes these contracts.
- The router change restricted to at most 128 rows is inactive on full
  2,848-row intermediate chunks. Short final chunks and decode can still
  reach it, so it is not an absolute model-wide exclusion.
- A source-derived estimate of about 110 MB additional wrapper traffic
  per GDN layer per chunk, divided by assumed bandwidth, is not a bound
  or measured contribution. Cache residency, eliminated work, overlap,
  launch overhead and compiler choices remain unmeasured. No percentage
  of the observed regression is attributed to the wrapper here.
- Two stable historical R32 boots do not establish a statistical
  resolution of 0.5%. Future repeats must report their actual spread.
- The missing `sequence.gdn_prefill` profile proves heuristic selection,
  not that the heuristic is slower. Existing correctness receipts remain
  valid; no correctness regression was identified by this analysis.

## Original discriminator plan, subsequently partly exercised

This plan predates the serving screen below. It is retained as investigation
history, not an instruction to resume experiments or take more controls.

GPU work requires a separate window. Preserve R32 serving until then.
No image, launcher or source modification was made in this investigation.

1. Start with a stateful, exact-shape GDN comparison on an available GB10.
   Use the recorded rank-local head geometry (8 key heads, 24 value heads),
   the actual 2,848-token intermediate chunks, and remainder lengths
   derived from the frozen benchmark token counts. Include chained
   chunks with nonzero initial recurrent state, final-state checks and
   the production checkpoint/export settings. Compare both raw
   recurrence and complete integration, including gather, staging,
   convolution, normalization, scatter and state updates. The upstream
   generic 2,048/4,096-token benchmark is supplementary, not a faithful
   substitute. Validate numerical outputs and states before timing.
2. If useful, run a low-cost whole-stack serving screen: R38 stock versus
   explicit FlashInfer prefill with the B12X decode flag removed and no
   environment override selecting it. First verify the exact rendered
   command, selected decode fallback, backend logs and correctness.
   This also changes decode and graph/metadata handling. Even when
   measuring prefill, a recovery implicates the broader GDN stack, not
   the recurrence kernel alone. Client TTFT includes first-output work.
3. For a causal prefill-only control, prepare a narrowly reviewed R38
   Python variant restoring independent prefill selection while keeping
   B12X decode. Do not revert all of `60b7b7191b`. Selector decoupling
   alone is not sufficient evidence of compatibility: inspect and test
   metadata construction, state ownership, mixed batches and graph
   eligibility, including the tests that currently reject this backend
   combination. Preserve unrelated correctness fixes. Native ABI reuse
   does not replace these execution-contract tests.
4. Compare the resulting valid control against stock R38 with the
   standard `run_bench.sh` campaign and exact token corpus. Use repeated,
   interleaved boots, warm measured cells, zero cached prompt tokens,
   clock receipts and no foreign inference. Compare against the banked
   R32 reference. Record prefill, engine steps, acceptance and correctness;
   report spread rather than inferring a precision threshold in advance.
5. A recovered whole-stack result plus operator timing/trace evidence can
   separate recurrence tuning from integration overhead. Kernel parity
   alone does not uniquely prove wrapper causality because serving also
   changes overlap and compiler tactics. If the GDN controls do not
   recover the loss, test reachable PLE and W4A16 changes next. Do not
   remove allocation-lifetime or overflow-safety fixes to chase speed.

## Outcome

The local source narrowing is complete. The backend flip is confirmed and
is the strongest workload-specific explanation to test. The cause of the
measured 2.4-2.6% long-context loss is not yet proven. All serving nodes
were left untouched. This document records an investigation and future
test plan, not a completed GPU bisect or a qualified fix.

### Subsequent serving screen and pause

See `stock-r38-replay-20260915/RESULTS.md` for the successful original-container
replay and `flashinfer-direct-20260915/RESULTS.md` for the failed FlashInfer
prefill plus CUDA decode arm. All three greedy checks returned corrupted
text; the correctness gate stopped execution before any timing. The cause
is not isolated to prefill. No new R38 control measurement was taken.

R32 was restored with a correct first completion at 21:08:58 EDT on
September 15. The user then explicitly chose to stay on R32 for Qwen and
pause experiments. The proposed remaining tests are deferred, not queued.
