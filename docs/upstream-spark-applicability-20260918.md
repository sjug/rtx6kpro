# September 18 upstream changes: Spark applicability

Read-only source assessment. No serving nodes, images, source branches or benchmark repositories changed. Local documentation baseline: `master ecdc481`. Frozen serving recipes were read with `git show spark:<path>`, not inferred from the incomplete directories visible on master.

## Decision

Prepare a new Karmic Kraken ARM64/SM121 foundation, not an R38 Python-only refresh. Keep the qualified serving deployments unchanged while that candidate is prepared. No inspected change establishes a fix for the retired DS4.1 524K retrieval failure or the Qwen R38 prefill cost.

## GLM KDA scratch overlap: include in a new candidate, not an emergency R38 backport

Upstream [cdea8220c2](https://github.com/local-inference-lab/vllm/commit/cdea8220c24dedece15dce9b3cd5f3ffc2d0819b) reserves disjoint projection workspaces before forking the KDA gate stream and binds each branch to its reservation. It also reserves those buffers during serialized warmup before capture locks allocation. This is a real correctness fix, not an optimization claim.

The affected new prepared blockscaled linear implementation obtains shared scratch through `current_workspace_manager().get_simultaneous` unless a branch-specific workspace is bound. See [b12x_blockscaled.py at the fixed source](https://github.com/local-inference-lab/vllm/blob/cdea8220c24dedece15dce9b3cd5f3ffc2d0819b/vllm/model_executor/kernels/linear/b12x_blockscaled.py#L231-L260).

Our frozen R38 Spark vLLM tree is `077347fdeab296404d0b0cb297d316176acc635a`, declared by `spark:spark/glm53/r38-spark/source.lock.json`. Its KDA gate also overlaps streams, but its linear kernels and quantization methods have no `get_workspace_size` or `current_workspace_manager` use. Unquantized projection dispatch in `layers/linear.py:212-221` calls `dispatch_unquantized_gemm`; on CUDA, `layers/utils.py:394-401` selects `default_unquantized_gemm`, which calls `torch.nn.functional.linear` at lines 86-92. The compressed-tensors method selects that unquantized method when no quantization scheme applies. The qualified NVFP4 GLM profile uses BF16 target dense projections.

Therefore the specific shared projection-scratch ownership defect fixed here is not established in R38 and the inspected R38 call path lacks its mechanism. Do not disable the qualified gate-stream overlap or cherry-pick the new workspace integration into R38 based on the headline. This does not certify absence of every possible stream bug.

For a new candidate, retain the fix and run its stream/workspace regression on GB10, including repeated graph replay. Upstream's serving validation cited MTP disabled; it does not replace our TP4 MTP3 semantic, concurrency, prefix-cache and standard benchmark gates.

## Qwen positional overrides: defer unless extending context

[PR 777](https://github.com/local-inference-lab/vllm/pull/777), inspected at `e0ad58ef5873fbd0cf4ed3970c0c8008baf65e03`, remains open. It propagates dictionary target HF overrides into the internally constructed MTP draft while preserving the MTP architecture discriminator. Its example is YaRN factor 4 with a 1,048,576-token positional limit instead of 262,144.

Our R32 launcher, read from `spark:spark/glm53/r32-spark/launchers/serve-qwen38-flash-next-jj-r32-spark.sh`, defaults to 262,144 and passes no HF positional overrides. This patch offers no demonstrated benefit for that native-context contract and is not a fix for the earlier GDN prefill backend change. Keep Qwen on R32. If context extension becomes a goal, pin the merged fix or explicit overlay and qualify target/draft positional agreement, retrieval, MTP acceptance and memory separately.

## Build and Vision assessment

See [the companion primary-source build assessment](karmic-kraken-spark-feasibility-20260918.md). The new upstream Vision loader fixes belong in that candidate. Do not change the qualified R38 InstantTensor profile solely because newer loader infrastructure needed allocation/finalization fixes.

## Execution order after candidate authoring

1. Pin one Karmic Kraken release and its complete dependency manifest. Rebuild ARM64/SM121 native artifacts for its CUDA/Torch ABI, rather than copying R38 objects.
2. Preserve current model-specific contracts. Do not import workstation tuning defaults with the source refresh. Include the KDA race regression and loader ownership/finalization tests.
3. Obtain a build/serving window before using any occupied Spark. No such interruption is performed by this research.
4. Qualify GLM and DS4 Vision independently against their current R38 profiles, correctness before timing. Reuse the standard benchmark harness without changing that repository.
5. Qwen remains R32 unless a separate candidate demonstrates correctness and removes the confirmed prefill cost. DS4.1 remains stopped; a new Spark launcher is not evidence that its unresolved long-context result is fixed.
