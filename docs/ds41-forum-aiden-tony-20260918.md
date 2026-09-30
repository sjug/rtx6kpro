# DS4.1 Spark recipes: Aiden and Tony artifact review

Read-only review on 2026-09-18. No nodes, images, model files or benchmark repository were changed. This note distinguishes published recipe claims from independently inspected artifacts, and context admission from retrieval qualification.

## Sources and pins

- [Tony/Kai repository](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark), HEAD `8f7acfbb29047c1c1100618f47719b0d52004fb3`.
- Its [vLLM fetch script](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/blob/8f7acfbb29047c1c1100618f47719b0d52004fb3/build/fetch_vllm_branch.sh) pins official `vllm-project/vllm` commit `e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba`, the former `dsv41-feat` branch. This is not our JJ R38 tree.
- [Aiden vLLM recipe](https://aidenle.com/recipes/deepseek-v4-1-flash-4x-dgx-spark/) and [Aiden SGLang replacement](https://aidenle.com/recipes/deepseek-v4-1-flash-sglang-4x-dgx-spark/). The initial HTML only exposes Overview. I read the complete tab content, including commands, from its [published JS bundle](https://aidenle.com/assets/index-CwypHSr1.js).
- Aiden publishes image tags and describes an MIT overlay repository but the inspected pages/bundle did not supply a direct overlay repository URL or image digest. The public `aidendle94/vllm` repository reports a June last push, so it is not evidence for the September overlay. Aiden's additional patch sets therefore remain recipe-described, not independently source-replayed here.
- Local comparison authority: `spark/ds41/r38/contract.py`, image `ea031e1d3d051033f077fc986bf6f8fce04cf9ab52483d5a719ba13114567fc5`, official model revision `fb2764a5cf321eaa5070ca8f9e892818f477c16d`.

## What is materially different

| Setting/path | Our retired R38 investigation | Tony original boot10 / final speedrun | Aiden vLLM production recipe |
|---|---|---|---|
| Model | Official checkpoint | boot10 official; final speedrun EXL3 3.5bpw Pollard **abliterated** model | Official FP8-Engram production; NVFP4-Engram separately tested, not promoted |
| Engine source | JJ R38, all B12X attention/linear/MoE | Official day-0 branch plus mounted patches; final EXL3 expert path, B12X dense trial rejected | Same day-0 lineage plus B12X 1.3.0 MoE and custom DCP port |
| Speculation | K7 greedy draft, standard rejection, adaptive verification on | K5 probabilistic draft, block rejection, adaptive off | K5 probabilistic/block, adaptive off |
| Parallelism | TP4/DCP1 | TP4; no DCP flag | Narrative says DCP4; pasted command still says DCP2 |
| Context and chunks | 1M envelope, 4096-token chunks | final receipt 300K, later note500K; chunks8192 | 500K, chunks8192 |
| Requests | 4 | 8 | 8 |
| Graphs | FULL_AND_PIECEWISE; breakable graphs off | FULL_AND_PIECEWISE; explicit K/K+1 capture ladder; breakable on | Same explicit ladder, breakable on |
| Memory/KV | u0.80 during final investigation; auto sizing | u0.80; auto sizing | u0.80 plus6GiB/rank fixed KV |
| Transport | PyNCCL LL,Simple; RoCEnante off | original NCCL; final RoCEnante plus NCCL cap8 | RoCEnante, NCCL fallback; LL,LL128,Simple |
| Engram | JJ native disk table, no resident scales/prefetch/projection TP | custom row-reader/prestage patches; final memmap gather128 threads | memmap gather, local row shards, next-chunk prefetch |
| Allocator | expandable_segments=True | final speedrun False | True |
| Thinking | true/max default | false for benchmark and final serving receipt | now true/max default, performance probes thinking off |

Neither Tony's nor Aiden's inspected V4.1 launch command explicitly sets `--kv-cache-dtype`; do not infer cache precision from the model name or their older V4 recipes. Our contract likewise has no explicit KV dtype flag. Exact effective cache layouts need runtime receipts/source, not a comparison of tag names.

Tony's [final args receipt](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/blob/8f7acfbb29047c1c1100618f47719b0d52004fb3/runs/2026-09-14-speedrun/results/final-best/args.json) confirms the changed model, K5, 300K and8192 settings. His [indexer patch](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/blob/8f7acfbb29047c1c1100618f47719b0d52004fb3/patch/sm12x-indexer-topk/sparse_attn_indexer-sm12x-topk.diff) routes SM12x away from persistent top-k to `top_k_per_row_decode`; it is a different indexer implementation from our B12X atomic tie-selection path. [Kernel results](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/blob/8f7acfbb29047c1c1100618f47719b0d52004fb3/patch/sm12x-indexer-topk/RESULTS.md) report torch.topk set equivalence at widths600 to300000. That does not prove stable tie selection on our524K case.

## Qualification evidence, including important limits

Tony boot10 has concrete seven-case vision/tool receipts. Final speedrun has a five-part gate: counting, JSON shape, executable prime-check code, arithmetic, and nondegenerate prose. These are useful correctness checks, not arbitrary long-context qualification. The [quality script](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/blob/8f7acfbb29047c1c1100618f47719b0d52004fb3/runs/2026-09-14-speedrun/tools/sr_quality.py) does not test repeated-logprob reproducibility.

The strongest inspected Tony long receipt is [b1-ctx1m needle.json](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/blob/8f7acfbb29047c1c1100618f47719b0d52004fb3/runs/2026-09-14-speedrun/results/b1-ctx1m/needle.json): **520921 actual prompt tokens**, one pass at50% depth, TTFT383.9s,1356.9tok/s. The [needle harness](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/blob/8f7acfbb29047c1c1100618f47719b0d52004fb3/bench/v41needle.py) asks one unambiguous passphrase question, temperature0/thinkingfalse, accepts answer containment, and makes one trial per target. It is not our original competing-identity/code prompt, not a repeated cold-salted distribution, and not a1M-token retrieval run.

Earlier [boot7 '1M proof'](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/blob/8f7acfbb29047c1c1100618f47719b0d52004fb3/results/boot7-proof.txt) demonstrates1M admission and short arithmetic/counting, but its32K needle **fails**, followed by scheduler dump. Do not present this receipt as successful1M retrieval.

Aiden vLLM reports40 teacher-forced completions/8345 tokens, DCP agreement97.4-97.6% against its own-output floor97.5%. It explicitly acknowledges B12X batch-dependent near-tie token flips. It reports540 uniquely tagged requests/20min/eight workers with zero errors or failed output checks; subsequent DCP/RoCE shorter soaks passed. Largest described retrieval is **487264 actual prompt tokens**,310s, within500K envelope. No inspected Aiden evidence repeats our original524K prompt, demonstrates raw-logit determinism, or retrieves through1M. Its separately fixed DeepSeek reference `act_quant` TileLang race is explicitly not used by its vLLM stack; reference retrieval was only3.3K/9.8K.

## Performance numbers, not cross-harness comparisons

Tony [speedrun report](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/blob/8f7acfbb29047c1c1100618f47719b0d52004fb3/runs/2026-09-14-speedrun/README.md), abliterated EXL3 final: C1/C2/C4/C6 aggregate58.27/96.00/152.44/189.97tok/s; C1 per-stream63.74; short TTFT0.221s; cold prefill2032tok/s at93335 tokens. Prompt categories and acceptance differ from our standard harness. Counting alone112.9tok/s is not representative mean decode. Original official-checkpoint boot10 averages C1/C4/C6 aggregate37.95/85.72/131.86, with cold prefill902-1539tok/s.

Aiden vLLM RoCE reports counting/code/prose94.5/74.5/35.9tok/s, mean acceptance3.56 vs NCCL3.10. It attributes part of gain to acceptance variation, not all to communication.32K/128K prefill2094/2152tok/s;3.418M KV-token pool at500K envelope,6GiB/rank. These are reported recipe measurements, not independently replayed.

## Aiden's newer SGLang production lane

The [September17 replacement](https://aidenle.com/recipes/deepseek-v4-1-flash-sglang-4x-dgx-spark/) uses `aidendle94/sparkrun-sglang-dsv41-gb10:production-1.0`, based on `lmsysorg/sglang:dev-dsv41`, with eight import-hook features: disk Engram C row gather inside graph callbacks; next-chunk prefetch; B12X MXFP8 dense routing with CUTLASS K576 fallback; indexer planner correction; ratio2 extra-page reblocking; prefill allocator flush; RoCEnante; aliases. It uses **TP4/EP4**, `dsv4` attention, **FlashInfer MXFP4 MoE**, DSpark block5, context524288, prefill chunks2048, static fraction0.80, maxrequests8, seed0. This offers a meaningfully different attention/MoE reference from our all-B12X JJ stack, but is not fully non-B12X because dense kernels and collectives still use B12X.

Reported final results: unique-text prefill3068/2580tok/s at32K/128K; counting/code/prose94.4/67.8/32.4; prose aggregate C1/C2/C3/C4=30.0/49.6/62.2/72.3;4.219M KV tokens;11-14GB host headroom during128K. Correctness examples are32K/128K needles, counting, images and tools. **524288 is configured capacity, not a published524K retrieval gate.** Earlier128K boot exhausted host memory and killed two ranks, leading to allocator flushing/smaller chunks. Moving upstream tag changed the hook point on September17; recipe says its base is digest-pinned in an overlay repository but does not expose that pin in the inspected command. Obtain the actual overlay URL and manifest before attempting a reproducible build.

## Conclusion

These are real functioning deployments with meaningful short/medium-context and some near500K evidence. Their claim is not that every long prompt is deterministic or that our specific dual-needle failure is solved. The most useful next reference would be the identical original524K input on Aiden's official-checkpoint SGLang lane, with repeated cold trials and both instruction variants. Tony's final altered checkpoint cannot isolate runtime differences. None of this evidence alone warrants reversing the decision to keep our DS4.1 deployment stopped.
