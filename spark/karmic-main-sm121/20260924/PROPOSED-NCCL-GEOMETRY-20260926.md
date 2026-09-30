# Proposed NCCL geometry diagnostic

Status: USER APPROVED AND COMPLETED; parent restored and short probes pass.
This is not a serving recommendation. Results are in `WINDOW-CAPTURE-20260926.md`.
The measured layer-0/1 cross-grid divergence disappeared, but the 4096-grid
retrieval failure persists. No promotion follows.

## Evidence and purpose

The corrected captures in `WINDOW-CAPTURE-20260926.md` show identical layer-0
local partials but different reduced outputs between the matched 8192 and
4096 prefill grids. Enumerating BF16 additions reproduces every captured row
exactly. This explains the first observed divergence as reduction association;
it does not establish whole-model correctness or that reduction invariance
will resolve the 524K answer.

The next test should distinguish reduction geometry from precision without
enabling the broad upstream batch-invariance mode. The pinned vLLM source is
`1794dcf18454900263e0c66711af8ea4a1283ac1`; its
`vllm/model_executor/determinism/batch_invariant.py` function
`override_envs_for_invariance` supplies a precedent for these settings:

```text
NCCL_ALGO=allreduce:tree
NCCL_PROTO=Simple
NCCL_MIN_NCHANNELS=1
NCCL_MAX_NCHANNELS=1
```

The function-scoped algorithm avoids forcing Tree onto unrelated collectives.
The pinned NCCL is stock 2.30.7. Verify parsing and actual geometry during the
diagnostic; the precedent is not a runtime receipt. Do not copy the function's
other settings blindly: it also changes Torch operators, compilation, cuBLAS
and attention behavior. `VLLM_BATCH_INVARIANT` remains off. RoCEnante limits,
model precision, source/image identities and attention plans stay unchanged.

This is four explicit environment overrides on every rank, not a generic env
passthrough. It changes the current contract, which unsets protocol and channel
pins. Obtain direct user approval before editing/deploying that contract.
Single-channel Tree may reduce collective bandwidth. A successful diagnostic
is not grounds to promote it without performance and correctness qualification.

## Execution after approval

1. Preserve the current candidate/manifest and verify all four ranks idle before
   the worker-first stop. Keep GLM and every other deployment untouched.
2. Select the already-built corrected capture image `c4a51be481c1...`, lock
   `59100837...`, with 81389 diagnostic blocks, preparation capacity 8192 and
   the same TP4/K7 source/checkpoint. Require explicit 8192 or 4096 threshold.
3. Add a narrowly scoped diagnostic selector, reject it outside that exact
   capture profile, and render/test all four ranks. Normal launch output must
   remain byte-identical. Record the new manifest and selector on both arms.
4. Verify effective env on all ranks. Retain communication initialization and
   tuning evidence; do not claim INFO logs identify every individual collective
   unless they actually do. Fail rather than silently accept unsupported settings.
5. Repeat two cold unarmed controls and the armed unchanged 524K request on each
   grid. Preserve failures as results. Require complete captures, zero prefix
   hits and the same eight attention-plan selections. Do not demand equality
   with baseline output signatures: altered reduction order can change them.
6. Stop and retain before CPU audits. Compare layer-0 partials and reduced
   outputs, then layer-1 input/cache, then the final response and logprobs. The
   two grids use separate boots; Tree topology can differ across boots and
   complementary trees can still create offset-dependent association. Neither
   setting the variables nor their acceptance guarantees invariance. Report
   observed equality, stripe patterns and missing per-call geometry evidence.
   A
   remaining later difference is evidence for the next boundary, not a failed
   excuse to weaken the retrieval gate.
7. Restore the saved parent selection/manifest and verify real completions.
   Nothing is promoted. Report the measured invariance and any unresolved
   answer-quality or numerical differences separately.

Do not compare changed numerical outputs to the old cross-boot baseline with
`compare_window_baseline.py` as a pass requirement: that gate was valid for an
instrumentation-only change. Within-arm captured/unarmed consistency, reference
packing, slot correctness and cross-rank consistency remain required. Preserve
the existing baseline gate rather than loosening it globally; the geometry
experiment needs its own explicitly identified comparison path.

## Offline precision check

`analyze_window_fp32.py` enumerates 24 sequential and three pair-tree FP32 sums
of the four captured BF16 partials, with one final BF16 rounding. It separately
checks order agreement and agreement with a double-precision reference. The
reference's exactness has a sufficient exponent-span bound, not an assumption.
Six synthetic tests pass, including overflow and FP32 order dependence at wide exponent
spread and a case where FP64 itself cannot be called exact under that bound.
The saved corrected 8192 and 4096 captures were checked while all nodes were
idle: all 27 orders agree after BF16 rounding for both captured layers in each
grid and match the FP64 reference, whose sufficient bound passes. This study does not
require, imply or authorize changing serving precision.
