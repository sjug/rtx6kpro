# Upstream check, 2026-09-21

## Scope

Fetched every configured remote without pruning for the ten existing checkouts:
rtx6kpro, vllm, b12x, blackwell-llm-docker, LMCache, flashinfer, sglang,
cutlass, spark-vllm-docker and dgx-spark-infra. All fetches succeeded. Also
refreshed the canonical Local Inference Lab build, LMCache and FlashInfer
branches into their existing lil remote-ref namespaces, without adding remotes.
No checkout, rebase, merge, build, node action, commit or push. The separate
benchmark repository and its results were left alone.

Comparison anchor is the [September 20 check](upstream-check-20260920.md),
including its exact composition pins, not just commit timestamps. Broad
mainline dependency observations below use commits since September 20 20:00 UTC
where that report did not record a tip; this is a time-bounded scan, not a
claim to have compared an unrecorded prior hash.

Local master and fetched upstream/master both resolve to
57ded9e53cd25e1ebd928f6fc18180f168426c37. The only documentation commit beyond
cfe30c0 is the daily summary. Its community reports are leads, not verified
Spark qualification. No new numbered R39 release was identified.

## Findings relevant to our fleet

See [publication identities](upstream-publication-20260921.md) and
[source applicability](upstream-source-delta-20260921.md) for the detailed
checks. Karmic beta advanced; the ordinary Karmic source line did not. The
substantive beta changes are Qwen QSA decode-context parallelism, speculative
graph-cache configuration hashing, and prepared DCP executable identity
retention. Qwen DCP is a new optional configuration, not an improvement
automatically exercised by our TP2/DCP1 profile.

The new build recipe at 23d674e includes
[5710a5f](https://github.com/local-inference-lab/blackwell-llm-docker/commit/5710a5fee924602de1205fd60d7dfe5dab12d889),
which enables `--enable-prompt-tokens-details` in the common profile. This
exposes cached-token usage for Qwen/GLM as well as DeepSeek, useful for our
prefix-cache investigation. It changes observability, not retention itself.
Our custom launchers do not inherit this merely by fetching the recipe.

Recipe commit
[53be00a](https://github.com/local-inference-lab/blackwell-llm-docker/commit/53be00a)
adds generated component changelogs to container releases. This makes the
published changes easier to audit; it does not expand the qualification scope.

No new inspected release establishes a fix for our Karmic GLM exact-format
failure, DS4 Vision prefill regression, or retired DS4.1 long-context finding.
Keep the successful allocator-only GLM loading result separate from those.

## Other fetched repositories

| Repository | Current relevant ref | Development / applicability |
| --- | --- | --- |
| vLLM upstream | main 6b858751f6 | New speculative dummy-KV protection and DS4.1 work; see source report for fork applicability. |
| FlashInfer upstream | main 6870e3ff | Blackwell gather-GEMM partial-K bounds fix; not included in the pinned LiL wheel. |
| LiL FlashInfer | community/jovian-judgement-cu134-sm120 2206a14e | Unchanged assembly source. |
| LMCache upstream | dev 34361cd5 | Native connector tile balancing, code organization and nightly pins. |
| LiL LMCache | integration/local-inference-lab 688bee14 | Unchanged assembly source; LMCache remains disabled in our candidate profiles. |
| SGLang | main 14e9c40a72 | DS4/DS4.1 router rendering parity, YaRN cache-scaling preservation, offload staging fix and frontend identity reporting. Different runtime; no change to our vLLM plan. |
| CUTLASS | origin/HEAD f614dc40 | Latest commit September 17; no newer tip in this check. |
| spark-vllm-docker | upstream/main a33f4b5 | September 18 MoE trial-buffer ownership fix, not new since the last check; analogous ownership cleanup already exists in our Karmic source. |
| dgx-spark-infra | origin/HEAD 3326b9c | Latest commit July 30; no new development in this check. |

FlashInfer [6870e3ff](https://github.com/flashinfer-ai/flashinfer/commit/6870e3fff46b1e768ad423ea48d286e6f3e250fe)
adds logical-K bounds to activation and scale-factor copies in its gather GEMM.
The author records partial-K failures and passing corrected tests on B300.
That does not establish a defect in our B12X MoE path or a GB10-tested fix.
Its other new KDA routing change,
[c6ec2cce](https://github.com/flashinfer-ai/flashinfer/commit/c6ec2cce64d70f28dce7d5d4c3ec6e4316965dcf),
explicitly selects sm_100a/sm_103a, not SM121.

LMCache [0cf5efc7](https://github.com/LMCache/LMCache/commit/0cf5efc7e0e96f153d63ce1479e085a86845d668)
balances native connector batch tiles. It is not an ancestor of the pinned LiL
integration branch. Neither this nor the FlashInfer mainline change enters our
images automatically.

## Recommendation

Do not schedule another fleet rollout merely because a newer beta exists.
First inspect the speculative dummy-KV fix against our fork, and retain the
new graph-cache key fix in the next deliberately selected candidate. Treat
Qwen DCP as a separate capacity/long-context experiment requiring paired vLLM
and B12X changes and its own correctness evidence. The cached-token reporting
flag is a small future diagnostics improvement, not a reason to rebuild alone.

Production was not inspected or altered during this upstream check.
