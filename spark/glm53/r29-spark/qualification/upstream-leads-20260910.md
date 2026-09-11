# Non-release upstream developments, 2026-09-10

Read-only investigation against `rtx6kpro/master` at
`59f01d1c3ab25c7a6d39c4e56de75c9df2f63944`, with current primary-source
GitHub and Hugging Face API checks. No source fetch, node operation, build,
deployment, or external post was performed. Release-lock analysis is separate.

## 1. GLM QAD results now support equivalence, not a checkpoint upgrade

Commits `2a763eb0de595cd6a432971781dbaf1634f858bc` and
`59f01d1c3ab25c7a6d39c4e56de75c9df2f63944` publish and align the matched
three-checkpoint behavioral evaluation. Each checkpoint answered the same
7,168 generated tasks three times on R30 at temperature 1, top-p 0.95,
maximum reasoning, and a 32,768-token completion ceiling.

| Checkpoint | Semantic score |
|---|---:|
| Published Local Inference Lab NVFP4 | 94.5386% |
| QAD step 2,500 | 94.5698% |
| QAD TV-nucleus step 2,500 | 94.5989% |

All three pairwise primary-endpoint comparisons are practically equivalent
within the predeclared one-percentage-point margin. The narrower earlier
greedy program-execution advantage does not transfer into a demonstrated
programming or agentic advantage under this sampling contract. QAD has better
teacher-forced distribution fidelity, but that does not establish better
served answers. On the stricter criterion that all three repeats must be
exact, published NVFP4 outperforms ordinary QAD step 2,500.

The report and public JSON agree on these scores, sample counts and decisions.
The upstream report states that its independent verifier checked all 64,512
receipts; those underlying retained receipts were not independently replayed
here. This is a served-system evaluation with MTP disabled, not our Spark MTP3
qualification. Uneven memory-clock offsets exclude a valid throughput claim.

Practical consequence: no demonstrated reason to replace our current GLM
NVFP4 weights with QAD. Preserve the distinction between sampling changes,
checkpoint changes and runtime changes.

Sources:

- [Matched QAD report](https://github.com/voipmonitor/rtx6kpro/blob/59f01d1c3ab25c7a6d39c4e56de75c9df2f63944/models/glm-5.3-flash/qad-step2500-verifier-backed-behavioral-fidelity.md)
- [Machine-readable summary](https://github.com/voipmonitor/rtx6kpro/blob/59f01d1c3ab25c7a6d39c4e56de75c9df2f63944/models/glm-5.3-flash/validation/r30-top-p095-vbf-three-checkpoint-20260910.json)

## 2. Qwen PLE disk offload is real, but the summary's mmap flag is obsolete

Current `local-inference-lab/vllm` JJ tip is `4b276a363c`. Three new source
commits implement the feature and change its public interface:

- `41ea64ae68`: loader-independent mmap PLE selection.
- `8770c10708d02ac56a8bada02249c8b26f18da08`: replaces mmap with bounded
  io_uring reads and prepares embeddings outside CUDA graph replay.
- `4b276a363c`: exposes the public environment values `ram` and `disk`.

The matching B12X commit is
`01ac763bef7503d934c4c2874bfd005799ca24a1`. It adds native reader code in
`b12x/loader/_ple_reader.c` and changes the embedding contracts. This is not
a one-variable experiment against our current R29 B12X package.

The current public disk selector is `VLLM_PLE_TABLE_MEMORY=disk`, not `mmap`
or `io_uring`. It works with InstantTensor and other supported loaders. Reads
deduplicate 4 KiB blocks, coalesce at most 64 KiB, and use registered io_uring
buffers with O_DIRECT. At a 4,096-token budget and 16-head NVFP4 geometry,
the implementation documents about 28 MiB of reader/planner storage, excluding
existing embedding output and scratch. CUDA graphs consume prepared outputs;
they do not issue host I/O.

Requirements include immutable checkpoint files throughout serving,
O_DIRECT-capable storage, liburing development files, pkg-config, and kernel
and container permission for io_uring. Unsupported disk configurations fail
explicitly, without a silent fallback. Both owning commit messages explicitly
say Spark serving was not tested. The daily summary's no-performance-penalty
claim is therefore not evidence for our pair.

Practical consequence: disk-backed PLE is a plausible later capacity
experiment on GB10 UMA; pinned host RAM is still part of the same physical
pool. It should not be folded into a correctness-focused R32 refresh, whose
published B12X package remains `d76de6c4ceaa9c2feb49edaf8c2f77d5a29ab5cb`.

Sources:

- [Current vLLM storage contract](https://github.com/local-inference-lab/vllm/blob/4b276a363c/docs/design/streaming_weight_loading.md)
- [Graph-safe preparation change](https://github.com/local-inference-lab/vllm/commit/8770c10708d02ac56a8bada02249c8b26f18da08)
- [Matching B12X implementation](https://github.com/local-inference-lab/b12x/commit/01ac763bef7503d934c4c2874bfd005799ca24a1)
- [R32 source lock](https://github.com/voipmonitor/rtx6kpro/blob/59f01d1c3ab25c7a6d39c4e56de75c9df2f63944/models/glm-5.3-flash/validation/concurrent-checkpoints-r32.source.lock)

## 3. DeepSeek V4.1 Flash is released, not a drop-in DS4 Vision update

The official Hugging Face API returned HTTP 200 for the public model at
revision `dba1be0a40aa45a94ad051997016db3960a90277`, architecture
`DeepseekV41ForCausalLM`, model type `deepseek_v41`. The primary model card
confirms text and image input, a one-million-token context, 552B backbone
parameters plus 196B Engram conditional-memory parameters, a 20-layer causal
encoder followed by a 20-layer decoder, CSA2 shared/compressed attention,
single-pass mHC, and integer reasoning effort from 1 through 100. Its global
KV footprint claim is 890 bytes per token. These are publisher descriptions,
not independent local measurements.

Upstream vLLM model support PR #56214 was open and unmerged at the check,
head `e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba`. Separate open work covers
kernel integration, SWA bounded replay, Engram microbatch lookback and tool
encoding. No corresponding V4.1 result was returned by the local-inference-lab
vLLM issue search. This does not prove no unpublished work exists.

Practical consequence: track as a new-model program, not a replacement model
name in nous's existing DS4 Vision recipe. Its new architecture, weights,
encoding and runtime support need their own compatibility and capacity work.
Do not infer two-GPU admission from its much smaller KV cache.

Sources:

- [Official pinned model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/README.md)
- [Official model API](https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4.1-Flash)
- [vLLM model-support PR #56214](https://github.com/vllm-project/vllm/pull/56214)
- [vLLM kernel integration tracking #56217](https://github.com/vllm-project/vllm/issues/56217)

## Summary-only leads kept out of the qualified conclusions

The 2026-09-10 daily-summary file describes the previous day's discussion.
Its NVIDIA GLM quant comparison, TrellisMX speed figures, Spark-ring tuning
claims and Qwen quant acceptance comparison are community-report leads here.
Their original execution receipts were not inspected. None justifies changing
our current checkpoint, switched-fabric RoCEnante settings or sampling policy.
The mmap selector is already contradicted by today's owning source contract,
which illustrates why these summaries are discovery material, not release
specifications.

Source: [daily summary at the inspected master](https://github.com/voipmonitor/rtx6kpro/blob/59f01d1c3ab25c7a6d39c4e56de75c9df2f63944/daily-summaries/2026-09/2026-09-10.md).
