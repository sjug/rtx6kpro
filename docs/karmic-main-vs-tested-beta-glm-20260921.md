# GLM: Karmic main versus our tested beta, 2026-09-21

## Scope and conclusion

Read-only comparison of existing Git objects, not a new build or qualification.
vLLM main is `af9e4dca109e0348323c0182e98a3aaf7282bfc3`; tested beta is
`57a80980bbf4b40398de7ed851b23e55a3a4c50e`. B12X main is
`0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68`; tested beta is
`e9ce547767ff9ee6509faf294fa1b4e2380dfbf5`.
The tested composition is recorded in [source-selection.json](../spark/karmic-beta-sm121/source-selection.json).
This does not compare main against a newer, untested beta tip.

Main would remove real GLM memory-management work present in our beta. It is
not established as a fix for the tool-response failure. Neither changed
request-boundary scheduling nor namespace serialization offers an obvious
active explanation for that particular plain-name tool test with aligned caching.

## Active and conditional differences

| Area | Beta implementation relative to main | Effect of selecting main for our profile |
| --- | --- | --- |
| GLM sparse pooled indexer | One `Glm5NextIndexerScratch` instance is passed across target layers instead of retaining separate large query, score and table buffers per layer. KV views and recurrent tails remain separately owned. | Loses this reduction in persistent scratch storage. DCP1 still uses query/table buffers even though DCP pool-score reduction is inactive. |
| Target/MTP storage | `share_target_indexer_storage` rebinds target/draft temporary selection buffers and scratch by identity. V2 loader invokes it with PP1 and no DBO, subject to single MTP layer and matching geometry. | Loses eligible target/draft scratch sharing. MTP3 means three speculative tokens, not necessarily three MTP layers; the actual layer-count guard matters. |
| GLM KDA projections | Returns the output projection directly rather than preallocating and copying a full output tensor; releases projection owners before output GEMM. Shared Kimi KDA machinery replaces persistent preparation probes with call-local tensors and permits borrowing spare workspace for eligible B12X prefill transients. | Restores older allocation lifetimes and preparation footprint. Workspace borrowing remains conditional on backend, dtype and available capacity. |
| Vision | Above 4096 rows, chunks token-local normalization/QKV/output projections and MLP while attention still sees the complete image. Borrows QKV scratch only with unquantized projection, FlashAttention and sufficient workspace. Chunked merger reuses owned storage only for matching geometry. | Loses large-image memory optimizations, not vision support itself. Small images need not exercise the new chunked path. |
| Vision admission/profile | Honors `max_image_tokens` when deriving pixel budget and searches aligned rectangular canvases within RoPE bounds for maximum features. | Loses more complete worst-case feature profiling. This is not proof that the tested image reached the affected maximum. |
| Sparse selection preparation | Clears bound index cache on unbind, resets physical-selection plans on cache binding and validates declaration before DCP1 decode work. | Loses stricter preparation/cache-lifetime handling in the active DCP1 path. No measured correctness attribution follows from this diff alone. |
| Worker preparation | Reserves scratch by target/draft workspace lane; registers native top-k/top-p warmups even when unseeded warmup chooses FlashInfer. | Loses preparation coverage and lane-specific reservation changes, not a demonstrated answer-format fix. |

Primary source anchors, inspected as pinned local Git objects:

- [Target scratch construction](https://github.com/local-inference-lab/vllm/blob/57a80980bbf4b40398de7ed851b23e55a3a4c50e/vllm/models/glm5next/nvidia/model.py#L962), [scratch implementation](https://github.com/local-inference-lab/vllm/blob/57a80980bbf4b40398de7ed851b23e55a3a4c50e/vllm/models/glm5next/nvidia/pooled_indexer.py#L54).
- [MTP sharing guards and identity rebinding](https://github.com/local-inference-lab/vllm/blob/57a80980bbf4b40398de7ed851b23e55a3a4c50e/vllm/models/glm5next/nvidia/mtp.py#L339), [V2 loader invocation](https://github.com/local-inference-lab/vllm/blob/57a80980bbf4b40398de7ed851b23e55a3a4c50e/vllm/v1/worker/gpu/spec_decode/eagle/utils.py#L185).
- [GLM KDA forward](https://github.com/local-inference-lab/vllm/blob/57a80980bbf4b40398de7ed851b23e55a3a4c50e/vllm/models/glm5next/nvidia/kda.py#L140), [shared KDA workspace/probes](https://github.com/local-inference-lab/vllm/blob/57a80980bbf4b40398de7ed851b23e55a3a4c50e/vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py#L651).
- [Vision chunks](https://github.com/local-inference-lab/vllm/blob/57a80980bbf4b40398de7ed851b23e55a3a4c50e/vllm/models/glm5next/common/multimodal.py#L259), [vision budget/profile](https://github.com/local-inference-lab/vllm/blob/57a80980bbf4b40398de7ed851b23e55a3a4c50e/vllm/models/glm5next/common/multimodal.py#L783).

## Changes not active as advertised fixes for this profile

The [launcher](../spark/karmic-beta-sm121/run-glm-tp4-node.sh) pins TP4, DCP1,
aligned recurrent checkpoints and BF16 MTP head by default, and the qualification
driver verifies those settings from container receipts.

- Beta adds a `prefill_tail` request-boundary checkpoint and a fourth checkpoint
  slot. Scheduler stopping and checkpoint publication are guarded by the
  boundary-checkpoint path. This is not an aligned-cache feature. See
  [scheduler](https://github.com/local-inference-lab/vllm/blob/57a80980bbf4b40398de7ed851b23e55a3a4c50e/vllm/v1/core/sched/scheduler.py#L514).
- Beta's chunked FP32 scale calculation for an NVFP4 draft vocabulary head does
  not apply to our BF16 head. The local SM121 draft-head overlay likewise only
  relaxes the capability check for the optional NVFP4 head.
- GLM DCP query-gather profiling changes do not apply at DCP1.
- B12X repacked W4A8 resident-grid tuning requires `w4a8_mx` and
  `fp4_e8m0_k32`, so it should not be advertised as a native ModelOpt NVFP4 GLM
  fix. B12X also changes general small-batch tuning routes to sample 0/20/40/60/80
  percent sharing rather than four realizations of 40 percent. That can affect
  selected plans, but does not establish a GLM numerical regression. See
  [tuning predicates](https://github.com/local-inference-lab/b12x/blob/e9ce547767ff9ee6509faf294fa1b4e2380dfbf5/b12x/moe/fused_moe/_tuning.py#L190)
  and [routing workload](https://github.com/local-inference-lab/b12x/blob/e9ce547767ff9ee6509faf294fa1b4e2380dfbf5/b12x/moe/fused_moe/workloads.py#L83).

## Tool serialization versus observed failure

Beta normalizes explicitly namespaced tool identities into `namespace::name`,
including tool definitions, choices and history. It also handles tuple histories,
avoids mutating input dictionaries, and restricts which tool-call iterables are
materialized. Inputs without namespace metadata preserve their function names.
Main loses these API compatibility changes, but our test uses an ordinary
`multiply` tool, with no namespace metadata. See
[protocol normalization](https://github.com/local-inference-lab/vllm/blob/57a80980bbf4b40398de7ed851b23e55a3a4c50e/vllm/entrypoints/openai/chat_completion/protocol.py#L546)
and [identity helper](https://github.com/local-inference-lab/vllm/blob/57a80980bbf4b40398de7ed851b23e55a3a4c50e/vllm/utils/tool_names.py#L25).

Both [initial allocator retry](../spark/karmic-beta-sm121/qualification/glm-allocator-20260921/aligned-mtp3/driver.log)
and [confirmation](../spark/karmic-beta-sm121/qualification/glm-allocator-confirmation-20260921/aligned-mtp3/driver.log)
stop on the same final answer: `17 × 23 = 391\n\nFINAL: 391`. Their semantic
receipts contain two successful tool round trips before that failure. The
arithmetic is correct; the strict requested final-answer format is not.
Five separate [format diagnostic repetitions](../spark/karmic-beta-sm121/qualification/glm-allocator-20260921/tool-format-diagnostics.jsonl)
returned exactly `FINAL: 391`, so isolated reproduction and the admission sequence
must not be conflated.

This evidence does not identify the responsible commit, prove that main passes,
or justify calling all beta changes unstable. A main/beta comparison would still
need the same Spark overlay, allocator choice, model revision, launch profile,
prompt order and correctness gate before any performance interpretation.
