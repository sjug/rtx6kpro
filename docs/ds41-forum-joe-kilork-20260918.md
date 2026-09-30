# DS4.1 forum reference audit: Joseph Rose and Kilork

Read-only source inspection on 2026-09-18. No recipes executed, images built, nodes touched, or benchmark repository modified. Published measurements below are author reports, not independently rerun results.

## Conclusion

Both authors have materially different, plausible working DS4.1 serving stacks. They are not simply running our R38 with another environment flag. Both derive from upstream `vllm-project/vllm` PR #56214 and use FlashInfer sparse MLA plus upstream indexer/top-k code. Their baseline MoE is DeepGEMM; B12X MoE is optional or subsequently enabled. Our failed candidate explicitly selected B12X attention, dense linear and MoE kernels in the JJ R38 lineage.

This makes their non-B12X variants useful reference candidates for our unresolved 524K test. It does **not** establish that either passes that exact test. Joseph explicitly reports no prompt above 262K and no B12X needle test. Kilork's published 512K envelope and tool benchmark are not a 512K retrieval qualification.

## Sources and pins

- [Joseph Rose repository](https://github.com/josephdrose/joe-spark-patches/tree/9d592115b721bd9ca5b353af4d86dd6a975b8628/dsv41), main at inspection: `9d592115b721bd9ca5b353af4d86dd6a975b8628`.
- [Kilork gist](https://gist.github.com/kilork/1ba224bfe28779c571d096cf954fff31/47c6ec89fd5a5218d3fb8768c5a10863fa4cb9f9), latest history revision `47c6ec89fd5a5218d3fb8768c5a10863fa4cb9f9`, updated September 11 19:59:01Z. All included README, recipe, Dockerfile, install/run scripts, versions and relevant sparse-attention code inspected.
- Kilork's linked [sm121-launcher](https://git.kilork.org/sm121/sm121-launcher) returned HTTP 404 on September 18. The gist remains accessible.
- Both trace to [tonyd2wild's reference repository](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark); Kilork additionally credits [Aiden's recipe](https://aidenle.com/recipes/deepseek-v4-1-flash-4x-dgx-spark/).

## Joseph Rose: actual configuration

The [launcher](https://github.com/josephdrose/joe-spark-patches/blob/9d592115b721bd9ca5b353af4d86dd6a975b8628/dsv41/launch/vlspeed-tp4-4node-up.sh) and [build script](https://github.com/josephdrose/joe-spark-patches/blob/9d592115b721bd9ca5b353af4d86dd6a975b8628/dsv41/build/vl41-build-image.sh) describe:

- Four GB10, TP4, model revision `df42c109f1defefcbfcedbe7d905718a12266e40`.
- Base `eugr/spark-vllm-b12x:latest`, Torch 2.13/CUDA 13.0, FlashInfer 0.6.18, TileLang 0.1.12. The recipe asks operators to check equal base-image digests but does not pin a digest in the build script.
- Official upstream aarch64 wheel at parent `29af8bd672d5a780abd7399c0cc624078202e89d`, with 87 changed Python files from PR head `e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba` overlaid. Native/Rust code stays at the parent except a separately compiled SM121 `vl41_ops.so` supplying the widened `apply_q_norm` signatures.
- 1,048,576 maximum context, utilization 0.78, 8 sequences, 8,192 batched tokens, DSpark K5 in the documented measured invocation. The bare launcher does not set K5: its `DSPARK` default is empty, and the quickstart supplies `DSPARK=5` explicitly.
- B12X BF16-activation MoE via `--moe-backend b12x` and `VLLM_B12X_MOE_FP4_FORCE_A16=1`. W4A8 and DeepGEMM measurements are also reported.
- Exact multiples of K and K+1 as capture sizes, FULL_AND_PIECEWISE, breakable graphs enabled, adaptive verification disabled. NCCL RoCE rather than a declared RoCEnante path.
- Custom disk-row Engram reader, 12 threads, O_DIRECT, rank-local row files, pre-stage in `prepare_inputs` for both Engram layers.
- SM12x indexer decode selects upstream `top_k_per_row_decode`, not B12X `run_row_topk`. Their [top-k note](https://github.com/josephdrose/joe-spark-patches/blob/9d592115b721bd9ca5b353af4d86dd6a975b8628/dsv41/docs/topk-swap.md) reports index-set agreement with `torch.topk` at tested widths 600 through 300,000. It explicitly says the persistent-top-k failure was not reproduced on this fleet.

### Performance and qualification

Reported in the pinned [README](https://github.com/josephdrose/joe-spark-patches/blob/9d592115b721bd9ca5b353af4d86dd6a975b8628/dsv41/README.md):

| Measurement | Result |
|---|---|
| Single-stream BF16 MoE, counting/code/thinking-code/prose | 96.41 / 74.54 / 67.79 / 38.59 decode tok/s |
| Mixed aggregate, sequences=8, c1/c2/c4/c6/c8 | 51.58 / 81.26 / 116.06 / 141.74 / 159.56 tok/s |
| Same mixed TTFT | 0.313 / 0.379 / 0.496 / 0.573 / 0.660 s |
| Counting-only c8 | 311.52 aggregate tok/s, not representative mixed throughput |
| Head KV, sequences=8 | 12.84 GiB, 2,900,475 reported tokens at 1M envelope |
| Tool-eval-bench 2.6.1, 88 scenarios, three trials, seed42 | W4A8 91/89/93; BF16 90/91/90; DeepGEMM 90/90/91 |

Single-stream figures use temperature 0.4, reasoning effort 75, median of three; decode excludes TTFT. They are not directly comparable with our standard harness.

DeepGEMM-only needle runs: seven passes, actual prompt tokens 32,330 to 260,119. Three needle depths at approximately 32K and 130K; one at 260,119 tokens. Largest run: 164.3 s TTFT and reported 1,583.4 prefill tok/s. The needle was a single code, `COPPER-LANTERN-8315`, not our dual-needle input.

The README explicitly limits the evidence: no prompt above 262K, no B12X needle, no vision test, no long-context quality or greedy-reference/garble gate. Admission with a 1M envelope is not 1M correctness.

Engram validation is relatively strong but narrow: 144 seam-focused rows across eight shards against checkpoint bytes, plus 2,000 in-serving staged-lookup checks per rank and 41,698 token-rows with no mismatch. This checks fetched bytes and staging, not model hash-id semantics. Detailed test notes lag the newer README: they still disclaim rank-offset/concurrency coverage that the README later reports. Preserve dates and scope rather than combining them into a universal qualification claim.

### Reproduction caveats found in source

1. README says `MOE_BACKEND=` selects DeepGEMM, but `${MOE_BACKEND:-b12x}` resets an empty value to b12x. The exact script must be corrected or an explicitly supported backend selected before using it as a non-B12X reference.
2. Removing `B12X_A16` does not select W4A8: `${B12X_A16:-1}` restores 1. Explicit `B12X_A16=0` is required.
3. Build script resolves patch/shim files under its own `build/` directory although repository files are under `patch/`. As published, those paths need correction or a staged layout reproducing the author's build directory.
4. Base `:latest` and a live PR-diff download are not immutable inputs despite the source-head pin. Freeze all inputs for any local reproduction.
5. Launcher force-removes containers, drops host caches and compacts memory by default; readiness polls only `/v1/models`. Do not adopt those operations into our managed serving workflow.

## Kilork: actual configuration and limitations

The [pinned gist](https://gist.github.com/kilork/1ba224bfe28779c571d096cf954fff31/47c6ec89fd5a5218d3fb8768c5a10863fa4cb9f9) builds upstream PR head `e47aa780...`, tree `8bf2b4034505699b829f9fcc1a8cc8e041c49384`, on base image digest `sha256:a551e05307cd2e0092139d84db32af9c97e67d2eeeff072d21e429131d8c23f0`. It rebuilds `_C_stable_libtorch` for SM121, uses Torch 2.12/CUDA13.3 per its README, FlashInfer `07869c61ba581e6d6b8ad8d142f4a6c89b707cc1`, and carries Tony's boot-10 replacements.

Recipe: model revision `dba1be0a40aa45a94ad051997016db3960a90277`, TP4, 524,288 context, utilization 0.85, 8 sequences, 8,192 batched tokens, block128, K5 probabilistic draft/block rejection, adaptive verification off, exact capture ladder, FULL_AND_PIECEWISE. Disk Engram uses 32 threads and per-rank sparse local copies; checkpoint weights may come from NFS. Default thinking is on. No explicit B12X backend appears in the base recipe; README adds `b12x==1.3.0` and `--moe-backend b12x` for its subsequently served configuration.

The vendored `flashinfer_sparse.py` calls FlashInfer TRTLLM sparse MLA; `sparse_attn_indexer.py` calls upstream scoring/top-k with the SM12x persistent-top-k bypass. Therefore omitting the optional B12X MoE overlay is an independent-kernel reference candidate, not just another JJ/B12X configuration.

Reported TEB hardmode seed42, 88 scenarios: base 158/176 (rounded90), B12X167/176 (rounded95). Reported d0/c1 prefill/decode/TTFT: 1,387 tok/s /39.9 tok/s /1,886 ms versus 1,642 /47.0 /1,738. Counting decode71.5 versus95.4. No repeated-trial variance or raw run receipts are included in the gist. A separate claim, “128K prefill1.64s (pp)”, is internally ambiguous and conflicts with interpreting 1,642 as tokens/s; do not repeat it as a validated 128K latency.

The pinned export is not ready to execute literally: Dockerfile contains `COPY  /tmp/` without a source, then reads `/tmp/patches/mounts.txt` even though gist files are flat; install.sh does not repair that layout. run.sh sets Engram-local env but omits the required Engram mount and JIT mounts declared in recipe.toml. Those are packaging defects, not proof the author's original running deployment failed. The source package is useful, but needs our own checked build/runner translation.

## Most consequential differences from our R38 test

Our [contract](../spark/ds41/r38/contract.py) fixes JJ R38, explicit B12X attention/linear/MoE, InstantTensor BUFFERED, disk scales not resident, DCP1, 4 sequences, 4,096 batched tokens, K7 greedy/standard rejection with adaptive verification, breakable graphs off, strict JIT monitoring. The comparison recipes instead use upstream-vLLM DS41 integration, FlashInfer attention and upstream indexer, different Engram implementations, optional DeepGEMM or B12X BF16 MoE, K5, 8 sequences/8,192 batches, and breakable graph handling.

This is sufficient source-level difference to justify a separately qualified reference image if authorized. It is not a controlled attribution experiment by itself: tokenizer revisions, Engram implementation, score precision/selection, attention, activation format and speculative policy can all change outputs. Replay the exact frozen inputs and record prompt token IDs before interpreting answer differences. First match non-speculative target output if feasible, then compare the original and revised 524K instructions with repeated cold requests and retained outputs/logprobs. Do not substitute TEB or a declared context capacity for that gate.
