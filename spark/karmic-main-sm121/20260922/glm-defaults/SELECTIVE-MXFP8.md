# Selective MXFP8 output projection experiment

Status: withdrawn by the user's subsequent scope clarification on 2026-09-24:
do not introduce changes that R38 did not use. R38 kept these projections BF16.
The first full correctness and grid receipts remain historical evidence, not a
promotion candidate. The same-boot repeat was interrupted after three cells;
its incomplete grid was correctly rejected. No activation-quantized arm was run.

## Question and parent

Does reducing selected BF16 target weight traffic improve decode without failing
the semantic, tool, vision, cache, concurrency and native-1M gates? Profiling the
no-prefetch K3 parent put BF16 GEMMs at approximately 32% of summed C4 kernel time.
Kernel sums overlap streams and are not a prediction of end-to-end savings.

The parent is `no-prefetch`, not the shared-head or K2 arm. Keep the same image,
checkpoint, K3 window, sampling, graphs, request-boundary retention, RoCEnante and
memory envelope. The only numerical intervention is selective target MXFP8 with
BF16 activations. Model inspection logging is added as an engagement receipt.

## Exact initial scope

`experiment.py` declares 56 exact runtime names, with no wildcard:

- `language_model.model.layers.0..44.self_attn.o_proj` (45 projections).
- `language_model.model.layers.{3,7,11,15,19,23,27,31,35,39,43}.self_attn.q_b_proj`
  (11 MLA query projections).

The header-only checkpoint inventory independently contains these 56 BF16
matrices, totaling 4,311,744,512 checkpoint bytes. This is not per-rank allocation
or measured DRAM traffic. Runtime names differ from checkpoint names.

Excluded: all target and draft experts, routers, shared experts, dense MLPs,
recurrent input and gate projections, fused MLA inputs, indexer, vocabulary heads,
embeddings, vision and MTP layer 45. `kv_b_proj` is excluded because MLA absorbs
and retains dequantized BF16 weights; quantizing it is not an ordinary linear
bandwidth experiment.

## Source contract

Pinned vLLM `e77be22511ab91ecf217760524b7579c366cca2a` and B12X
`10a553ef980571f23a073f930cb386fbc8a77e07`, with the existing image overlays:

- `quantization/base_config.py:resolve_quant_method` composes online quantization
  over unquantized checkpoint linears. Unmatched layers retain their checkpoint
  method. A target matching an already-quantized layer raises rather than silently
  replacing the checkpoint method.
- `online/mxfp8.py:Mxfp8OnlineLinearMethod` quantizes weights at loading, then
  dispatches through the selected MXFP8 linear kernel.
- Explicit `VLLM_B12X_MXFP8_ACTIVATION_MODE=a16` limits this first arm to weight
  quantization. B12X validates the prepared mode against that request. Its auto
  policy can instead choose quantized activations for larger token counts; that
  is a different numerical experiment, not part of this arm.
- GLM KDA calls `o_proj` normally. The MLA wrapper receives `q_b_proj` and
  `o_proj` as modules. `kv_b_proj` follows a separate absorbed-weight path.
- `B12xBlockscaledLinear.ensure_plan` declares exact capture-M regimes plus the
  batch-capacity regime via `plan_regimes`. Other row counts use that capacity
  regime. This differs from the earlier DS4 per-row output-projection preparation
  issue; runtime logs still must be checked for actual late JIT/preparation.

## Gates and execution

1. Local finite-profile tests require the exact 56 targets and preserve all
   other arguments and policies. Independent source review precedes launch.
2. Finish and retain the shared-head control first. Do not change helpers mounted
   into its live containers while the benchmark runs.
3. Gracefully stop GLM workers, then head. Sync the reviewed kit, start workers
   before head, and retain previous containers. Qwen and DS4 are untouched.
4. Every rank must report exactly the intended online target set and select
   `B12xMxfp8LinearKernel`. Retain model inspection output to review unchanged
   expert, head and recurrent-input methods. The existing four-rank command,
   environment, image and CUDA gates remain mandatory.
5. Run `qualify.sh` with scope `full`, not just the screen: semantic, short-pool,
   vision, tool round trips, concurrent/repeated requests, frozen cache pairs and
   triples, then native retrieval through 1,048,000 tokens. Inspect health and
   meaningful complete outputs before any timing. Passing these probes does not
   establish general quantization equivalence or a broad quality score.
6. `benchmark.sh` refuses a screen-only receipt for this arm. Use the unchanged
   standard harness and retain profile identity, acceptance, engine steps,
   throughput, prefill, per-rank KV and health. Numerical drift means acceptance
   changes must not be presented as pure kernel speedups.
7. Repeat a promising result before promotion. Failed correctness or unexplained
   health changes block timing/promotion; retain receipts and the prior profile.

## Independent review before launch

Claude confirmed the pinned composition path, target names, kernel selection,
alignment and explicit A16 policy. The review found that target and draft loads
may print identical accumulated online-quantization summaries. The gate now
accepts repeated identical declarations but rejects any different or expanded
set; tests cover both. All 18 local tests pass normally and under `-O`.

The subsequent screen-only timing refusal test brings the local suite to 19;
all 19 also pass under `-O`.

The proposed expansion to shared experts and dense MLPs is deferred. This first
arm retains the approved experts-unchanged scope. No claim is made that only
recurrent-input quantization could affect long-context results: altered residuals
also propagate to later recurrent inputs, so every quantized arm needs the full
native-context battery. Header byte counts and projected savings are hypotheses,
not measured bandwidth or guaranteed performance.

## First boot and correctness

Receipt: `qualification/selective-mxfp8-output-a16-20260924-run1/`.
Effective profile `3801fdb71cf032381bd994d3d23adb457bcd130317681793aa817d5e4afe2956`.
Four ranks passed exact online target and backend checks, with the expected
identical summary printed on target and draft creation. The inspection shows
target NVFP4 experts, MTP MXFP8 experts, and unchanged unquantized recurrent
inputs and MTP linears. KV admission: 6,668,943 tokens.

Seven short-pool, 15 reasoning/vision/tool checks, repeated/concurrent requests,
frozen prefix pairs and short triples passed. Native retrieval passed exactly
at 2,048 / 2,049 / 262,000 / 1,048,000 tokens with normal stop. The 262K request
was cold and took 99.735 seconds; the 1M request took 413.048 seconds and reused
253,952 tokens. This is not a fresh-token prefill benchmark. The short native
requests reused their earlier short-pool inputs; those earlier checks are kept.

Health caveat: startup allocation warnings occurred on all ranks, and buddy had
five more at 13:09:50 UTC during early concurrent checks. No Xid, OOM kill,
exception or foreign inference was found through full correctness. No further
driver warning occurred during the long-context phase. Host pressure remains
tracked; this is not a claim that the 0.85 envelope is pressure-free. First-use
Triton warnings occurred before long-context completion, not inside a performance
grid. Four-rank journals and complete logs are retained before timing.

## First full grid

Raw result: `20260924T092437-0400__karmic-main-no-prefetch-mxfp8-output-a16-sm121-tp4-dcp1-mtp3__r01.json`
in the `2026-09-karmic-sm121-qualification` campaign. All 15 cells validate;
the wrapper completed normally. `comparison.json` pins both raw input hashes.

| Concurrency | Parent tok/s | Selective tok/s | Parent steps/s | Selective steps/s |
| --- | ---: | ---: | ---: | ---: |
| 1 | 49.752 | 54.056 | 20.232 | 21.323 |
| 2 | 76.994 | 80.652 | 30.592 | 31.771 |
| 4 | 114.117 | 118.575 | 44.821 | 46.068 |

These are geometric means across five contexts. Step gains are 5.39%, 3.85%
and 2.78%; output gains are 8.65%, 4.75% and 3.91%. Acceptance increased too.
Changed quantization can change routing and token streams, so these are profile
comparisons, not isolated kernel speedups. Against saved R38, output remains
about 4.4% lower at C2 and 3.6% lower at C4, with C1 about 0.8% higher.

Prefill at 8K/16K/32K/64K/128K is 2578/2696/2707/2684/2632 tok/s,
7.23% to 7.90% below the parent. This tradeoff prevents calling the profile an
overall winner. Explicit A16 was deliberately conservative about activation
precision; changing it would be another numerical arm requiring qualification.

During the benchmark, all four retained journals were free of NVRM/Xid/OOM
events, and logs showed no foreign inference, exception or late JIT warning.
Telemetry shows zero clock-event samples, swap growth, allocation stalls,
compaction stalls or direct reclaim. Minimum available GiB were
2.717/3.521/3.138/5.324 on sparky/buddy/rocky/lucky. Startup and earlier
correctness warnings remain recorded above and are not reclassified as clean.

Claude reviewed the completed comparison. The subsequent user constraint
withdraws this numerical experiment; retained R38 serving was restored and
verified at 13:49 UTC. Future performance investigation must preserve R38 precision and
numerical policies, not compensate for implementation costs with quantization.
