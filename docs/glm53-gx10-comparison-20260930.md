# GLM-5.3-Flash GX10 TP4 comparison, 2026-09-30

## Verdict

This is a credible performance candidate, especially for cold prefill and workloads with strong DFlash2 acceptance. The posted table does not establish that it beats our latest completed beta qualification overall. It uses a different benchmark, and the recipe changes both the serving implementation and target-model precision. Keep the qualified baseline until a matched comparison and correctness gates pass.

Scope: local receipts and public source inspection only. No nodes, containers, builds, or benchmarks were touched. The public repository has no existing checkout used for this investigation; its files were read through GitHub, without cloning or fetching.

## Source identity and what the recipe actually runs

The inspected recipe commit is **4e63b6404e8a4058b8e93cc2826cf41d022afd2c**. Its build script defaults to `spark-glm53:v9`; the documented base is `vllm/vllm-openai:nightly-ddd6fbca148a867aad1fcab7ec72f582b9977db4`, with mentat 0.17.1. This is a recipe plus runtime overlays, not an identified digest of the image behind the quoted September 29 run. The original checkpoint is `nvidia/GLM-5.3-Flash-NVFP4`; there is no abliterated target configured. [Pinned build script](https://github.com/kindlingai/glm-5.3-flash-gx10/blob/4e63b6404e8a4058b8e93cc2826cf41d022afd2c/image/build.sh), [pinned README](https://github.com/kindlingai/glm-5.3-flash-gx10/blob/4e63b6404e8a4058b8e93cc2826cf41d022afd2c/README.md), [model descriptor](https://github.com/kindlingai/glm-5.3-flash-gx10/blob/4e63b6404e8a4058b8e93cc2826cf41d022afd2c/model.yaml).

The important differences from our B12X/MTP3 composition are:

- A separate `incoai/GLM-5.3-Flash-DFlash2` drafter, about 2.2 GiB, with up to seven proposals. Adaptive scheduling chooses how many proposals to verify from recent acceptance and measured step costs. This offers a way to accept more tokens per target verification pass; the gain over our MTP3 is workload-sensitive and has not been measured here.
- Custom GB10 MoE kernels, Triton sparse MLA, custom small-message RDMA collectives, and sequence-parallel prefill with quantized gathers and RDMA reduce-scatter. These provide plausible speedups beyond changing the drafter. Their published gains against stock vLLM do not transfer directly to our already optimized B12X and RoCEnante paths.
- Additional quantization of dense target layers that NVIDIA's checkpoint leaves BF16: FP8 generally, NVFP4 for selected KDA projections, attention output projections and shared experts. The target lm_head defaults to FP8. The drafter is also quantized.
- RecoverSSM and a separate drafter KV pool increase concurrency and capacity. Processed-weight snapshots affect boot time and available memory when measuring.

These are implemented by eight ordered compose overlays: arx, snapshot, adaptive-k, fp8, megamoe, fixes, sp, recoverssm. Several replace complete installed vLLM or FlashInfer files. An image digest alone cannot reproduce the advertised configuration; record the recipe commit, overlay hashes/order, effective environment, checkpoint revisions, image digest and snapshots. The overlays explicitly target this pinned nightly and should not be copied into our beta wholesale. [Experimental implementation guide](https://github.com/kindlingai/glm-5.3-flash-gx10/blob/4e63b6404e8a4058b8e93cc2826cf41d022afd2c/experimental/README.md), [actual FP8/NVFP4 overlay](https://github.com/kindlingai/glm-5.3-flash-gx10/blob/4e63b6404e8a4058b8e93cc2826cf41d022afd2c/experimental/compose/fp8.yaml), [entrypoint](https://github.com/kindlingai/glm-5.3-flash-gx10/blob/4e63b6404e8a4058b8e93cc2826cf41d022afd2c/image/entrypoint.sh).

## What can be compared today

The latest completed grid at the end of this investigation is the September 30 `20260930T140024` QAD qualification, which finished while this report was being prepared: all 15 cells, zero errors, beta image 500ae05b, checkpoint revision 175ae8ce, MTP3, effort Max and top_p 0.95. Its checkpoint differs from the earlier morning local-inference-lab 46aaae8a run, so preserve that identity in every comparison. This investigation did not verify live serving state. [Local execution record](../spark/karmic-beta-sm121/20260929/glm/EXECUTION.md), [benchmark campaign](../runs/glm-5.3-flash/nvfp4/2026-09-qad-175ae8ce-qualification/throughput/20260930T140024-0400__karmic-beta-20260929-qad-175ae8ce-max-topp095-seq4-tp4-mtp3-native1m__r01.json).

| Latest completed local grid, aggregate output tok/s | C1 | C2 | C4 |
|---|---:|---:|---:|
| Geometric mean over contexts | 53.89 | 82.81 | 128.03 |
| Zero added context | 54.79 | 80.51 | 126.29 |
| 32K context | 53.82 | 86.10 | 130.78 |
| 128K context | 52.48 | 82.03 | 129.38 |

Local first-scout prefill measured approximately 2,735 tok/s at 8K, 2,977 at 32K, and 2,919 at 128K; later fresh warmed measurements are around 3,000 tok/s.

The pinned upstream README reports TP4 cold prefill **4,934 tok/s at 32K and 4,776 at 128K**, roughly 1.6 times our warmed prefill. This is the strongest reason to evaluate it, although the inputs and timing methods remain different. It reports RigMark code/prose/structured decode **109.3 / 60.8 / 159.9 tok/s**, and separate code concurrency throughput **131 / 141 / 209 / 254 tok/s at C1/C2/C4/C8**. Those are different tests, not internally interchangeable measurements. The README specifies temperature zero, reasoning effort low, 512 output tokens per concurrent code prompt, and TP4 medians over four snapshot-restored boots. Our benchmark is a sustained approximately 30-second decode measurement with different prompts and sampling defaults. [Pinned measured table and method](https://github.com/kindlingai/glm-5.3-flash-gx10/blob/4e63b6404e8a4058b8e93cc2826cf41d022afd2c/README.md).

The user-supplied September 29 table has only 128 generated tokens per labeled test and reports 79.5/110.0/152.4 at d0. Compared numerically with our zero-context row, those are higher, but that ratio is not a measured speedup. I could not locate its raw run, harness implementation, image digest, overlay list, sampling settings, or exact meaning of `tg t/s`, `pp t/s`, and `d`. Do not assume its concurrency metric is our aggregate steady-decode metric. The 32K C2/C4 values of 25.9/23.3 are a reason to examine methodology and scheduling, not proof of a recipe regression. Short outputs, staggered prefill completion, prefix reuse and decode timing conventions could materially alter the comparison.

The posted label says 1M context. Current source defaults TP4 to **524,288 maximum tokens per request**, and documents needle tests through about 507K. A 4.40M-token shared KV pool is not a demonstrated 1M single-request qualification. The table might have used an override or another configuration; the quoted text is insufficient to identify it. The checked-in RigMark metadata is itself an older 50-sequence configuration with checkpoint revision `09b04e5e`, while current defaults with RecoverSSM use 64. [Entrypoint](https://github.com/kindlingai/glm-5.3-flash-gx10/blob/4e63b6404e8a4058b8e93cc2826cf41d022afd2c/image/entrypoint.sh), [measurement metadata](https://github.com/kindlingai/glm-5.3-flash-gx10/blob/4e63b6404e8a4058b8e93cc2826cf41d022afd2c/gate/rigmark-metadata.json).

## Quality and adoption boundaries

The additional dense quantization has a documented cost of about 1% prose NLL. NLL is a loss metric, not a one-percentage-point accuracy loss. Small published task checks remain equal: GSM8K first 250 at 97.2%, HumanEval 156/164, with counting and tool-call probes. That is useful evidence, but it does not establish identical output distributions or parity on our workload. Quantizing target layers changes the distribution that speculative decoding verifies against. Disabling the dense quantization is an important control, but then the fully optimized speed claims no longer apply. [Quality measurements and quantization explanation](https://github.com/kindlingai/glm-5.3-flash-gx10/blob/4e63b6404e8a4058b8e93cc2826cf41d022afd2c/experimental/README.md).

The recipe explicitly identifies the DFlash2 checkpoint as **CC BY-NC-ND 4.0, non-commercial**. Native MTP is an option without that separate download, but the reported DFlash2 speedup cannot be assumed for MTP. This is a concrete candidate dependency to resolve for the intended service, not an abliterated-model issue. [Checkpoint requirement and fallback](https://github.com/kindlingai/glm-5.3-flash-gx10/blob/4e63b6404e8a4058b8e93cc2826cf41d022afd2c/README.md).

## Minimal useful evaluation

An inexpensive first isolation experiment is already prepared in our local runner: `SPECULATOR=dflash2` with K7 versus MTP3 on the same target and beta image. That option is not a qualified result; its local-inference-lab drafter is distinct from the Kindling drafter and does not reproduce their adaptive scheduler or target quantization. It can tell us how much of the opportunity comes from changing drafters before replacing the entire serving stack. [Local runner](../spark/karmic-beta-sm121/20260929/glm/run-glm-tp4-node.sh).

1. Preserve the current qualified baseline and pin a separate complete candidate identity. Obtain the posted run's raw benchmark/configuration before interpreting its exact ratios. No deployment is authorized by this investigation.
2. On the same four machines in an authorized evaluation window, compare both engines with the same original checkpoint revision, rendered prompts, explicit reasoning effort Max, temperature 1 and top_p 0.95, FP8 KV, topology and client method. Run the normal C1/C2/C4 grid, fresh-prefix prefill at 32K and 128K, and the posted short-burst harness if available. Warm both configurations and measure a restored-snapshot candidate boot as its authors specify.
3. Record spec acceptance, engine step rate, actual generated tokens, time-window definitions, scheduling delays and peak memory. Repeat the leading cells on another boot before claiming a durable gain.
4. Run a dense-quantization-disabled control to distinguish kernel/drafter benefit from changed target precision. Separate DFlash2 from native MTP if attribution is needed; do not predict the combined result by adding individual percentages.
5. Before cutover, qualify our tool/semantic/counting/prefix/padded-transition batteries and requested long contexts, including 1M if preserving our current advertised capability. Include concurrent long-prefill interference and health checks. A faster short-context C1 result alone is not a sufficient replacement criterion.
