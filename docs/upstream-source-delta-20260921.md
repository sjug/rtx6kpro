# Upstream source delta, 2026-09-21

Read-only inspection of existing, already-fetched vLLM and B12X checkouts. No fetches, builds, GPU tests, benchmarks, node operations, launcher changes, or benchmark-repository edits were performed for this source review. Source presence and added tests are not model qualification.

## Frozen comparison

| Ref | September 20 | September 21 |
| --- | --- | --- |
| vLLM `lil/dev/karmic-kraken` | `af9e4dca109e0348323c0182e98a3aaf7282bfc3` | Unchanged |
| vLLM `lil/integration/karmic-kraken-beta` | `57a80980bbf4b40398de7ed851b23e55a3a4c50e` | `22476af54c637cbb7c7d8193addd160da83a5ce3` |
| B12X `lil/master` | `0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68` | Unchanged |
| B12X `lil/integration/karmic-kraken-beta` | `e9ce547767ff9ee6509faf294fa1b4e2380dfbf5` | `f6d8b8eb94cdeb4e652652f925a494c6fc86f101` |
| vLLM `lil/integration/beta` | `8eca34a0` | `074707f39a330c31201356858b7194d3d70d9562` |
| B12X `lil/integration/beta` | `b9f44def` | `39e0fa7506974f7bd7fdf261e26b6df8ced4de7e` |

Each Jovian beta delta changes only `.lil/README.md` and `AGENTS.md` to require changelog fragments: no serving-code delta. Karmic beta adds the three functional changes below, plus release-documentation changes. Mainline Karmic sources have not advanced since the previous report. These branch tips do not establish the source composition of any published image; consult the separate publication report.

## Qwen QSA DCP is new capability, not a DCP1 prefill fix

vLLM [PR 816 implementation](https://github.com/local-inference-lab/vllm/commit/d730eb3f04bbd7055817db340c31aeb07f3592fa), merged at `22476af54c`, enables decode context parallelism for Qwen3.8-Flash-Next QSA. B12X [PR 402 implementation](https://github.com/local-inference-lab/sparkinfer/commit/0e582e1eaa4d29a5efae3c32ce39cb132b8f519b), merged at `f6d8b8eb`, supplies rank-local compressed-cache geometry, global candidate IDs, 64-bit cache addressing and sparse-attention LSE output. The [vLLM release fragment](https://github.com/local-inference-lab/vllm/blob/22476af54c637cbb7c7d8193addd160da83a5ce3/.lil/changes/vllm-816.json) explicitly requires both B12X 401 and 402.

The actual vLLM path gathers Q/K/V heads, selects rank-local candidates, merges top-k globally, then combines partial attention. The small decode-row branch applies FP32 LSE weights before reduce-scatter; larger batches use `MLADCPManager`. MTP selection reuse receives a corresponding distributed attention path. A Qwen-specific config capability permits full-TP DCP with KV gathering even where the old generic GQA validation rejected it.

Important limits visible in [the implementation](https://github.com/local-inference-lab/vllm/blob/22476af54c637cbb7c7d8193addd160da83a5ce3/vllm/models/qwen4_exp/nvidia/b12x_qsa.py):

- DCP must be explicitly enabled. TP2 alone does not enable DCP2. With DCP1, the main execution path still calls `qsa.run` without the new distributed collectives.
- Interleave must be divisible by the QSA compression ratio, four. Prefill context parallelism remains rejected, and external QSA KV transfer remains unsupported.
- DCP disables the input-projection overlap option. Treat changing DCP as a separate experiment, not a transparent source-only comparison.
- Hybrid cache sizing now accounts for materialized DCP KV heads. Sharding token positions is not a blanket claim of halving every cache allocation.

Added tests cover ownership round trips and partial stripes, DCP1 geometry, a simulated DCP4 attention/LSE merge, KV-head deduplication, hybrid-cache alignment, FP32 weighting, and CUDA pack/unpack. These were inspected, not run. No new matched GB10 TP2 end-to-end serving receipt was added in the inspected delta. This is a possible future Qwen TP2/DCP2 experiment, not evidence that the measured DCP1 prefill deficit is fixed. Shared QSA internals also change, so DCP1 still needs regression qualification if adopting the source.

## Speculative graph cache identity now includes method and depth

vLLM [PR 815 implementation](https://github.com/local-inference-lab/vllm/commit/ca3609cd9ab1eb62f8681a776669f9439fcf4368) changes `SpeculativeConfig.compute_hash()` from an initially empty factor list to `[self.method, self.num_speculative_tokens]`. Its regression asserts different hashes for ngram depth one versus three and ngram versus ngram_gpu at depth three.

This is applicable across speculative models, including Qwen, GLM MTP and DS4 Vision configurations, when method/depth changes would otherwise reuse a graph cache identity. It is not a parser, allocator, loading or attention-arithmetic fix. A fixed-depth run does not establish that the observed GLM exact-format failure resulted from cache collision; no reproducer tying these together is supplied here.

## Prepared DCP executable identity is now retained

B12X [PR 401 implementation](https://github.com/local-inference-lab/sparkinfer/commit/19235190b99866159e29f5cf00a282a6eee6d7ed) attaches compiled program carriers to four PCIe DCP launcher closures: LSE reduce-scatter, head gather, pair gather and Kimi top-k16. Previously they returned a bare closure. The added mocked regression checks that the LSE factory preserves its exact `ProgramKey` for preparation/frozen-resolution validation.

This changes compile-plan observability, not the numerical collective or its launch API. It matters when these DCP programs are used. The scoped GLM TP4/DCP1 and DS4 Vision TP2/DCP1 profiles do not acquire DCP data-path benefits merely from upgrading. This is also not a change to the multi-node NCCL transport configuration.

## No demonstrated closure of the local qualification failures

The [local qualification record](../spark/karmic-beta-sm121/QUALIFICATION.md) distinguishes the initial GLM load stall from the allocator-only retry: adding `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to the same candidate allowed loading, but two semantic batteries retained the third tool-round-trip exact-format failure. Five isolated repetitions passed. The record supports the setting as sufficient on that trial, not a proven allocator mechanism. It also records that DS4 Vision already exported that setting.

The complete Karmic beta source delta has no model-loader, tool-parser, GLM-model, DS4-model, DSA-indexer or MoE implementation changes. The new graph hash is a credible general correctness improvement, not evidence of causation for the exact-format failure. QSA DCP affects Qwen, not DS4's DSA path. No inspected new change or receipt establishes a fix for the DS4 Vision prefill gap, GLM exact-format gate, or earlier DS4.1 524K instability. Do not waive those gates or alter the benchmark solely on this delta.

## Mainstream vLLM leads: tracked, not shipped in these Karmic sources

`git merge-base --is-ancestor` returns false for each of the following commits against both September 20 candidate `57a80980bb` and September 21 beta `22476af54c`:

- [d2983f2f16, dummy draft KV protection](https://github.com/vllm-project/vllm/commit/d2983f2f163249acbe3053c073cb9830290984a0) propagates `dummy_run` into draft input preparation, fills dummy request mappings with `-1`, and masks negative mappings before block-table reads and slot generation. This is a relevant correctness lead, not a demonstrated explanation of our gate failure. The fork's V2 speculator is customized: it has fused multistep draft execution, and its slot kernel additionally reads `group_num_blocks_ptr + req_state_idx` and checks block bounds. Current beta still copies active request mappings without a dummy-run branch and lacks the upstream negative-index guard. Therefore even evaluating a port requires adapting and testing all affected fork paths, including that extra metadata read; do not equate upstream merge with a local fix.
- [9b49f92344, small-batch DS4.1 mHC overlap](https://github.com/vllm-project/vllm/commit/9b49f92344312c41ad61e05282c8e6a2d9bafb7f) and [9a70c233cd, fused all-reduce/mHC preparation](https://github.com/vllm-project/vllm/commit/9a70c233cd5d3ef3b14d79c582a6f9ddfa03cf80) change the upstream DS4.1 implementation and native/runtime paths. They are not evidence of improvement for our DS4 Vision B12X/DGLIN service.
- [67513c8b67, restrict mHC overlap to full CUDA graphs](https://github.com/vllm-project/vllm/commit/67513c8b67e2439934ce7c0ba5a18c2e859f5c53) follows that overlap work, disabling it for eager execution as well as piecewise graphs and preserving Mega mHC warmup. The inspected Karmic DS4.1 tree has none of the new `MHC_OVERLAP_MAX_TOKENS`, `mhc_pre_delayed_overlap` or `all_reduce_mhc` symbols. This is not evidence of a missing fix for an identical active overlap path in our candidate.

No ports or local tests were performed. These are follow-up source leads, not authorization to patch or rebuild the active candidate.
