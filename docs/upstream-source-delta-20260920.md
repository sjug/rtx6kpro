# Upstream source delta, 2026-09-20

Read-only inspection of already-fetched source refs. No GPU, build, benchmark, or node operations were performed. This is source and published-receipt evidence, not local qualification.

## Anchors

| Repository | Previous check | Current ordinary development ref | Current integration ref |
| --- | --- | --- | --- |
| local-inference-lab/vllm | `cdea8220c2` | `dev/karmic-kraken` at `af9e4dca109e0348323c0182e98a3aaf7282bfc3` | `integration/karmic-kraken-beta` at `57a80980bb` |
| B12X / local-inference-lab/sparkinfer | `a8333658` | Both local `upstream/master` and `lil/master` resolve to `0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68` | `lil/integration/karmic-kraken-beta` at `e9ce547767ff9ee6509faf294fa1b4e2380dfbf5` |

Both ordinary development refs are ancestors of their corresponding integration refs. Each ordinary development line has six commits beyond the previous check. The beta contains substantial additional code, not merely release labels. Integration history contains duplicated/cherry-picked changes, so raw commit counts are not independent feature counts.

## 1. Qwen now has a concrete GB10 prefill improvement lead, not a proven fix for our R38 deficit

B12X [d5139035](https://github.com/local-inference-lab/sparkinfer/commit/d5139035) fuses exact stable QSA winner construction. Its [public receipt](https://github.com/local-inference-lab/sparkinfer/blob/d5139035/benchmarks/evidence/qsa_stable_selection/README.md) reports DGX Spark TP4 serving prefill changes of +0.15%, +2.23%, and +3.96% at 8K, 64K, and 128K. Complete-QSA component latency falls roughly 9-11% at 64K/128K. These are different measurement scopes.

Important limits: TP2 has candidate-only serving numbers, not a matched control; the measured source composition is not the standalone master combination; the receipt is explicitly research-only. Twenty-six exact shared-prefix/history-edit checks passed at 64K/C4 and 128K/C8 per topology. An initial partial TP4 run had an unexplained slowdown and is retained separately. Decode output-rate deltas include acceptance differences.

Separately, beta vLLM [938f7e4778](https://github.com/local-inference-lab/vllm/commit/938f7e4778) removes GDN prefill activation staging copies. The actual implementation changes `B12xGdnPrefill.run` from copying mixed QKV, a, and b into persistent layer buffers and copying its output back, to binding caller tensors directly. Sequence metadata still has staging. This is relevant to the wrapper-overhead hypothesis from our R38 investigation, but it does not establish the cause of the R38-versus-R32 gap or supply a matching local benchmark. It is a beta-only delta relative to today's ordinary development ref.

Action: these justify a later Qwen TP2 candidate comparison at 32K-128K with our correctness battery first. They do not justify replacing qualified R32 now.

## 2. GLM MTP graph work was measurably over-sized at low concurrency

Ordinary vLLM development [ed6f2c5853](https://github.com/local-inference-lab/vllm/commit/ed6f2c5853) moves `on_prefill_begin` into `_prefill(num_reqs)`, so capture binds compaction outputs for each captured request capacity instead of binding them once at `max_num_reqs`. The previous one-request graph on a four-slot server ran the first draft MLP for four rows.

The commit records GLM-5.3-Flash GB10 TP2/MTP3 checks and matched four-step profiles: first-draft graph rank-maximum time 2.979 to 2.148 ms and rank-0 MoE 1.130 to 0.333 ms. These are draft-component timings, not a model throughput claim, and the topology differs from our GLM TP4 service. The code is a general autoregressive speculator change, not an SM120-only branch.

Action: especially relevant to C1 with a larger configured sequence capacity. Keep full concurrency qualification; do not extrapolate a percentage serving gain.

## 3. Startup-memory ownership and distributed tuning fixes directly address Spark admission risk

Ordinary vLLM [298f265c67](https://github.com/local-inference-lab/vllm/commit/298f265c67) tears down temporary profiling/speculator context, collects cycles, synchronizes, and returns freed allocator blocks before the serving KV allocation on every rank, including explicit KV-budget configurations. Its author explicitly says the reported admission failure was not reproduced, so this is preventative ownership cleanup, not a proven explanation of our historical NVRM events.

[af9e4dca10](https://github.com/local-inference-lab/vllm/commit/af9e4dca10) removes trial tensors from `PreparedCall.owners`, which serving preparation plans retain, while retaining them through the transient producer closure during races. Bounded-storage regressions at 1/4/6/8/64 tokens failed before and passed after the change.

vLLM [0f60770e85](https://github.com/local-inference-lab/vllm/commit/0f60770e85) and B12X [06809d53](https://github.com/local-inference-lab/sparkinfer/commit/06809d53) coordinate cached tuning requirements across startup ranks. This is a paired API/behavior change; do not independently replace only one repository without compatibility checks.

Action: carry the coherent source composition, record startup memory and KV admission per rank, and preserve the no-building-while-serving rule. Source cleanup is not permission to raise utilization.

## 4. Beta recurrent-cache work is meaningful, but conditional on policy

Beta [eb4ec03df6](https://github.com/local-inference-lab/vllm/commit/eb4ec03df6) adds a fourth optional recurrent checkpoint before the final portion of a long prompt. The position is `(prompt_tokens // chunk_budget - 1) * chunk_budget`, if beyond the instruction boundary, leaving one or two chunks after the recovery point. It also allocates only endpoints a request can publish.

The capacity accounting changes in [02dc2740c7](https://github.com/local-inference-lab/vllm/commit/02dc2740c7) and [1c90e4a7fb](https://github.com/local-inference-lab/vllm/commit/1c90e4a7fb) reserve endpoint blocks and an immutable restore source. This prevents treating those states as free capacity. The helper returns zero when `use_request_boundary_checkpoints` is false. Our explicitly aligned profiles therefore do not automatically receive the new request-boundary retention behavior.

Beta also carries GLM scratch/projection/indexer lifetime reductions, including caller-owned DCP storage, shared sequential target/MTP indexer storage, and releasing preparation probes. They are not all equivalent benefits for our DCP1 profile.

Action: keep aligned as the comparison control; qualify an auto arm separately with exact repeats, divergent suffixes, concurrent admission, and long-prefix reuse. Do not revive a previous auto recommendation merely because new checkpoints exist.

## 5. DS4.1 has compatibility and preparation work, but no identified fix for our 524K result

New ordinary B12X [0f3a8cbf](https://github.com/local-inference-lab/sparkinfer/commit/0f3a8cbf) removes a DS4.1 special case that flattened sparse-MLA head partitions, preserving TP3 partitioning. That is not a TP4 nondeterminism fix.

Beta vLLM [101b033ebb](https://github.com/local-inference-lab/vllm/commit/101b033ebb) preserves tool namespaces and reminder messages. Beta also reduces DS4/DS4.1 graph-activation and prepared-scratch ownership and improves shared-expert-aware tuning. These can matter to agentic compatibility, memory, and performance, but do not explain our fixed-token greedy retrieval distribution.

Neither B12X master nor the inspected beta delta from `a8333658` changes files under `dsa_indexer`. The new stable selection implementation is under `attention/qsa`, the Qwen path, and must not be advertised as fixing DS4.1's DSA top-512 tie behavior. The vLLM DeepSeek indexer delta inspected here clears pool-derived views/plans on unbind; it does not alter selection arithmetic.

Action: retain the unresolved DS4.1 qualification finding. A new source composition still needs the original frozen 524K input and an independent numerical/runtime reference; no commit inspected here establishes that the previous answer distribution has been corrected.

## Additional tuning caveat

B12X [57f35723](https://github.com/local-inference-lab/sparkinfer/commit/57f35723) restores native NVFP4 A16 precision-tuning candidates and default direct routes at capacities 1-8 on SM120/SM121, bumps the candidate contract to invalidate prior tuning decisions, and prefers A16 at equal measured latency. Explicit precision selections remain valid. GPU oracle/graph tests were added but not run in that commit's validation. Do not assume every existing GLM/Qwen launcher switches precision merely because this commit is present; resolve each workload's requested precision and selected prepared plan.
