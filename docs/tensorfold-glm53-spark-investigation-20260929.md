# TensorFold GLM-5.3-Flash on Spark: investigation, 2026-09-29

## Decision

Worth an isolated evaluation with **original, non-abliterated GLM weights**. Do not adopt the linked recipe unchanged or use its published speedups as a forecast for our service. Sebastian explicitly rejected abliterated models during this investigation. That excludes its default checkpoint, including as our proposed benchmark target.

The interesting proposition is satisfactory GLM service on **two Sparks instead of four**, with fast low-concurrency decode and persistent conversation reuse. It is not yet evidence of better quality, aggregate throughput, prefill, or operational reliability than our qualified TP4 service.

This investigation inspected public source, benchmark clients and receipts, local qualification records, and the running GLM head's identity. It ran no builds, benchmarks, downloads of weights, installs, fetches, container changes, or deployments. The only repository changes are this report and the linked [engine source investigation](tensorfold-engine-research-20260929.md). Associated repositories were read remotely because no TensorFold checkout was established.

## Identities and scope

| Component | Inspected identity |
| --- | --- |
| Recipe | `jayleaton/glm53-tensorfold-spark@80dba15d5db555b655fc81f34f04711a572de4ca` |
| Engine submodule | `ashhart/TensorFold@2f8e514b0b7d615df7c971627ce3c0fb7e55d93a`, v0.3.4 |
| Published recipe patch stack | 57 patches, numbered through 0410 |
| Container foundation | `nvcr.io/nvidia/pytorch:26.07-py3`, tag rather than digest |
| Rejected default target | `neko-legends/GLM-5.3-Flash-Uncensored-EXL3@07135ec082f8f11f7a71e4244a4e5167a0f96277` |
| Recipe's optional drafter | `incoai/GLM-5.3-Flash-DFlash2@7d74cdd881ed7e32c31175984a67823127b66cfe` |
| Live local GLM head | sparky container `glm53-flash-nvfp4-jj-r38-spark-tp4`, image `ea031e1d3d051033f077fc986bf6f8fce04cf9ab52483d5a719ba13114567fc5`; four-node arguments, aligned recurrent checkpoints |

Pins come from the [recipe tree](https://github.com/jayleaton/glm53-tensorfold-spark/tree/80dba15d5db555b655fc81f34f04711a572de4ca), [Dockerfile](https://github.com/jayleaton/glm53-tensorfold-spark/blob/80dba15d5db555b655fc81f34f04711a572de4ca/docker/Dockerfile), and [production configuration](https://github.com/jayleaton/glm53-tensorfold-spark/blob/80dba15d5db555b655fc81f34f04711a572de4ca/config/prod.env.example). Only the local head was inspected live; this is not a fresh four-node health certification.

## Original-model route

TensorFold's pinned GLM CUDA loader supports MLX and EXL3, with TP exactly two. Our NVFP4 checkpoint is not a drop-in input. The source investigation identifies two named, original-model candidates:

| Candidate | Pin | Assessment |
| --- | --- | --- |
| `Vontra/GLM-5.3-Flash-MLX-4bit-MTP` | `76add2a341a1cd90ad0e86bb69839ea9c35827c6` | Cleanest explicit provenance: weight-only quantization, no fine-tuning, original source `zai-org/GLM-5.3-Flash@3f1971b7b5f7a528c9c4ef6212c8785298a8c24a`; 4-bit/group 64. Preferred first compatibility candidate. |
| `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` | `9eaebb7c4e96d983dcd538e18624622ba5b820a8` | Model card identifies a byte-identical mirror of the original-model brandonmusic conversion. Closer to the recipe's EXL3 kernel path, but independently qualify provenance, arithmetic and memory. |

Sources: [Vontra pinned model card](https://huggingface.co/Vontra/GLM-5.3-Flash-MLX-4bit-MTP/blob/76add2a341a1cd90ad0e86bb69839ea9c35827c6/README.md), [Mia pinned model card](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw/blob/9eaebb7c4e96d983dcd538e18624622ba5b820a8/README.md), and the source references in the engine investigation.

Do not simply substitute `MODEL_PATH` and inherit the recipe's qualification. Its nonexpert q4mse conversion and EXL3 expert kernels are checkpoint-specific; MLX uses a different weight path. Derive settings from the selected format. The recipe author's expectation that original-model drafting should improve acceptance remains an unmeasured hypothesis. Neither candidate establishes numerical equivalence to our NVFP4 conversion.

## Why it could be useful

This is a separate inference engine, not a vLLM backend switch. The Spark recipe substantially extends the pinned engine:

- Latent MLA KV storage, then FP8 storage and a shared paged pool, make long context practical. The advertised 1,048,576 tokens are a **shared total pool** across four active slots, including admission reservations. It does not provide four simultaneous 1M contexts.
- MTP, DFlash2 and suffix lookup propose tokens; measured-cost drafting and verification windows up to 16 rows target different workloads. Edit/copy tasks benefit especially from prompt lookup. A 16-row verification envelope is not equivalent to fixed K16 speculation.
- Chunked and row-split prefill, EXL3 expert kernels and overlapping compute/communication improve execution. Some algorithms derive from B12X and existing Spark EXL3 kits; the gains are not all unique discoveries by TensorFold.
- RAM/NVMe session snapshots and shared system-prefix reuse can avoid large repeat prefills. Prepared per-rank weights reduce subsequent startup time.
- Its small-message RoCE transport adapts B12X's RoCEnante design. Our R38 runner already enables RoCE all-reduce, so this is not evidence that our image lacks that whole class of optimization.

See the [patch inventory](https://github.com/jayleaton/glm53-tensorfold-spark/blob/80dba15d5db555b655fc81f34f04711a572de4ca/docs/PATCHES.md), [production settings](https://github.com/jayleaton/glm53-tensorfold-spark/blob/80dba15d5db555b655fc81f34f04711a572de4ca/config/prod.env.example), [attributions](https://github.com/jayleaton/glm53-tensorfold-spark/blob/80dba15d5db555b655fc81f34f04711a572de4ca/NOTICE), and our [R38 runner](../spark/glm53/r38-spark/run-glm53-flash-jj-r38-spark-tp4-node.sh).

## Performance evidence and what it means

I recomputed the ten overlapping decode medians in the author's older vLLM versus TensorFold comparison from each receipt's completion tokens and decode duration. They match the reported medians. Selected rows:

| Workload | Author's vLLM TP2 | TensorFold R16 TP2 | Ratio |
| --- | ---: | ---: | ---: |
| Sampled code | 35.20 | 42.30 | 1.20x |
| Sampled chat | 27.06 | 41.03 | 1.52x |
| Greedy code | 41.89 | 77.56 | 1.85x |
| Greedy chat | 22.83 | 44.60 | 1.95x |
| Hashmap task | 29.99 | 53.16 | 1.77x |
| Structured counting | 72.74 | 100.58 | 1.38x |

Units are output tokens/s excluding prefill and TTFT. This uses the same source checkpoint and client on the author's pair, but TensorFold additionally changes nonexpert quantization and arithmetic. vLLM has five repetitions; TensorFold R16 has three for these rows. This is useful evidence of an appliance-level gain, not an engine-only experiment. Both targets are the rejected abliterated model. Sources: [vLLM receipt](https://github.com/jayleaton/glm53-tensorfold-spark/blob/80dba15d5db555b655fc81f34f04711a572de4ca/results/E1-A-vllm-kit.json), [R16 receipt](https://github.com/jayleaton/glm53-tensorfold-spark/blob/80dba15d5db555b655fc81f34f04711a572de4ca/results/W10/glmbench-R16.json), [measurement client](https://github.com/jayleaton/glm53-tensorfold-spark/blob/80dba15d5db555b655fc81f34f04711a572de4ca/bench/glmbench.py).

The newer RigMark receipt reports code/prose/structured decode of 68.6/43.2/88.2 tokens/s and C1/C2/C4 aggregate end-to-end throughput of 54.3/65.5/82.2. Its published comparison also shows slower cold prefill, slower immediate prefix replay, and C4 TTFT of 1.9 seconds against about 0.8 for the comparison vLLM systems. Different weights, quantization, protocol revisions, machines and context limits prevent attribution to TensorFold. The comparison requires `--allow-mismatch`. Source: [RigMark comparison and caveats](https://github.com/jayleaton/glm53-tensorfold-spark/blob/80dba15d5db555b655fc81f34f04711a572de4ca/results/rigmark/README.md).

For local context only, our saved R38 standard-grid geometric means were **53.63/84.37/123.05** output tokens/s at C1/C2/C4 and roughly **2,775-2,955** prefill tokens/s over 8K-128K. Those use four nodes and our different fixed-duration/context-grid harness. They neither prove that TensorFold loses nor support its advertised speedup against us. The relevant decision requires matched tasks and quality, and both two-node resource efficiency and absolute service performance. Source: [local qualification comparison](../spark/karmic-main-sm121/20260922/glm-defaults/QUALIFICATION.md#performance).

## Specific corrections and qualification gaps

**Sampling/API blocker for general client use:** I independently checked the pinned GLM sampler: nonzero-temperature `top_k=0` selects only eight candidates per rank before the generic sampler interprets zero as unlimited. `top_k=-1` is also mishandled. Recipe batch patches retain this calculation. Fix it or reject those values explicitly before using ordinary clients; serial/speculative equality cannot detect a distribution error shared by both paths. The published positive-top-k cells are not invalidated by this finding. Unsupported `response_format` and `tool_choice` are likewise not enforced by the inspected CUDA adapter. See the [source-level finding](tensorfold-engine-research-20260929.md#verified-sampling-defect-disabled-top-k-is-not-honored).

1. **Newest results are ahead of published source.** RigMark ran image `b5` with patches through **0490**. The inspected public recipe stops at **0410**; its RigMark README explicitly says `/tokenize` patch 0490 and other inputs will arrive in another update. Do not attach those numbers to an image built from this pinned main or call it reproducible yet. [Receipt provenance](https://github.com/jayleaton/glm53-tensorfold-spark/blob/80dba15d5db555b655fc81f34f04711a572de4ca/results/rigmark/README.md#recipe).
2. **Edit repetition counts are overstated.** W10 documentation labels final edit cells as three repetitions. `suite_edit` actually uses `max(1, reps // 2)`. R16's `reps=3` produces one recorded run per edit cell; FIN's `reps=5` produces two. The medians themselves recompute correctly, but the repetition claim does not. [Client](https://github.com/jayleaton/glm53-tensorfold-spark/blob/80dba15d5db555b655fc81f34f04711a572de4ca/bench/glmbench.py), [FIN raw receipt](https://github.com/jayleaton/glm53-tensorfold-spark/blob/80dba15d5db555b655fc81f34f04711a572de4ca/results/W10/glmbench-FIN.json).
3. **Exactness has a limited scope.** Final raw receipts do report ten drafted-versus-serial matches and four batched-versus-alone matches. These test the same TensorFold configuration. They do not prove equivalence to BF16 weights, BF16 KV, vLLM, or every sampling distribution. The engine investigation explains its sampler. [Exact receipt](https://github.com/jayleaton/glm53-tensorfold-spark/blob/80dba15d5db555b655fc81f34f04711a572de4ca/results/W10/exact-FIN2.json), [batch receipt](https://github.com/jayleaton/glm53-tensorfold-spark/blob/80dba15d5db555b655fc81f34f04711a572de4ca/results/W10/batchexact-FIN2.json).
4. **Quality is not established by 200 questions.** FIN reports 176/200 MMLU answers correct. A small thinking-off subset on a changed checkpoint is insufficient for our coding, reasoning and tool workloads. Requantization changed 13/200 answers in an earlier comparison. Published needle checks reach roughly 314K-358K, not the full advertised 1M context. [Quality receipt](https://github.com/jayleaton/glm53-tensorfold-spark/blob/80dba15d5db555b655fc81f34f04711a572de4ca/results/W10/quality-FIN.json), [W10 results](https://github.com/jayleaton/glm53-tensorfold-spark/blob/80dba15d5db555b655fc81f34f04711a572de4ca/docs/RESULTS.md).
5. **Memory qualification has an acknowledged open end.** Four roughly 250K sessions passed the author's 8 GiB margin, but a subsequent needle after stress and MMLU dropped the worker to 7.39 GiB. No OOM occurred there; this still fails a universal claim that the configuration maintains that margin. Larger 8,192-row prefill buffers were rejected for memory. [W10 final gates and caveat](https://github.com/jayleaton/glm53-tensorfold-spark/blob/80dba15d5db555b655fc81f34f04711a572de4ca/docs/RESULTS.md).
6. **Source pinning is incomplete for reproducible images.** The Dockerfile pins neither the base digest nor exact versions of several pip dependencies and falls back from `git apply` to `patch`. Our candidate would need a resolved ARM64 base digest, exact dependency inputs, patch/tree hashes and installed-source verification. Calling the build reproducible solely because the submodule is pinned overstates its guarantee. [Dockerfile](https://github.com/jayleaton/glm53-tensorfold-spark/blob/80dba15d5db555b655fc81f34f04711a572de4ca/docker/Dockerfile).

## Operational fit

The primary launcher assumes Docker and two nodes with a direct CX7 link. It has useful preflight, readiness, canary, memory-gate and watchdog machinery, but it needs adaptation to our Podman/CDI setup and actual interface mapping. Its production memory gate enables host page-cache drops using sudo on both nodes. `start`/stop paths remove named containers. Do not run these scripts as discovery tools on serving nodes. Even the nominal preflight can start a temporary non-GPU Docker container to inspect a cache marker. Source: [serve.sh](https://github.com/jayleaton/glm53-tensorfold-spark/blob/80dba15d5db555b655fc81f34f04711a572de4ca/scripts/serve.sh).

The Compose file explicitly says it omits the production latent-KV, batching, pool and RoCE environment and is only a minimal debugging alternative. Supplying the production context alone through Compose does not reproduce production and may not fit. Prepared weight storage is approximately 83 GB per rank per key in the recipe's comments, plus the original checkpoint, compiler caches and the configured 64 GiB/rank session tier. Establish disk budgets for the selected original checkpoint rather than inheriting these estimates. Source: [Compose configuration](https://github.com/jayleaton/glm53-tensorfold-spark/blob/80dba15d5db555b655fc81f34f04711a572de4ca/docker/compose.yaml).

API compatibility is partial: no images or logprobs, `n>1` rejected, unsupported fields can be silently accepted, and single-stream stop/cancel behavior can leave hidden engine work. Tool parsing success does not establish constrained JSON-schema generation or forced tool selection. A text-only service could be useful, but cannot transparently replace every current GLM capability. See the engine investigation for source-level details.

The engine is MIT and the recipe's own work is Apache-2.0 with third-party notices. The optional DFlash2 checkpoint is documented as CC BY-NC-ND 4.0, a separate constraint from the engine license. Preserve actual model licenses in any candidate manifest; engine licensing does not determine weight permissions. An MTP-only evaluation avoids depending on that optional drafter but cannot inherit its measured speed. [Notices](https://github.com/jayleaton/glm53-tensorfold-spark/blob/80dba15d5db555b655fc81f34f04711a572de4ca/NOTICE).

## Proposed evaluation, not execution authorization

1. **Prepare a source-locked original-model candidate.** Prefer the pinned Vontra checkpoint for explicit provenance; consider the original EXL3 mirror as a separate candidate if its format is a better fit. Check config/tensor layouts, tokenizer/template, MTP and model lineage before weights are staged. Pin all image and patch inputs. Either use published 0410 and label it accurately, or wait for a complete source publication for the newer RigMark image.
2. **Establish correctness before timing.** Start with text-only TP2, modest context and one active slot; use the selected model's appropriate quantization path. Compare serial, MTP and optional DFlash on fixed-token/seed tests. Exercise tool round trips, reasoning fields, JSON requirements, cancellation, stop sequences, unsupported options, and a meaningful original-model coding/reasoning battery. Keep our production model service name separate from the experiment.
3. **Measure useful service behavior.** Use identical prompts, sampler parameters, templates, context sizes and output budgets across candidates. Run C1/C2/C4, cold and reused prefixes, and mixed long-prefill/short-decode traffic. Record TTFT, inter-token gaps, aggregate and per-stream output rate, accepted tokens per verification round, elapsed task time and answer quality. Keep RigMark with one pinned protocol as a secondary common workload. For engine attribution, compare the same original checkpoint in both engines; for deployment decisions also compare the complete candidate against our NVFP4 TP4 service and disclose the quantization/topology difference.
4. **Expand one feature at a time.** Separate MTP-only from DFlash/lookup, NCCL from RoCE, BF16 from FP8 KV, batch slots, RAM/NVMe persistence and prefix sharing. Increase to 262K, 524K and finally 1M only after memory and correctness gates pass. Include near-boundary needles, four long sessions, capacity admission, cache eviction, restart/resume and cancellation recovery.
5. **Require repeatability and rollback evidence.** At least two independent boots and repeated grids, followed by a mixed-traffic soak with minimum MemAvailable, swap, kernel errors, restarts and decode stalls tracked. Preserve the existing GLM containers and image until a separately authorized cutover passes these gates.

Recommended disposition: **keep R38 serving; investigate TensorFold as an original-model TP2 alternative.** Its best potential win is freeing two Sparks while retaining acceptable service quality. Published evidence warrants testing that hypothesis, not a production replacement or a promised speed multiplier.
