# Upstream check, 2026-09-20

## Scope and fetch completion

Fetched all configured remotes, without pruning, for the existing inference/build/infrastructure checkouts: rtx6kpro, vllm, b12x, blackwell-llm-docker, LMCache, flashinfer, sglang, cutlass, spark-vllm-docker, and dgx-spark-infra. Also fetched the canonical Local Inference Lab LMCache, FlashInfer and build-repository forks into `refs/remotes/lil/*`, without adding remote configuration or changing working branches.

InstantTensor, nccl-canonical and FlashKDA had no standalone checkout in the inspected locations. Their canonical sources were cloned into persistent bare repositories under `/home/jugs/git/rtx6kpro-artifacts/upstream-mirrors/`. Fetch logs for the ten existing repositories are preserved under `/home/jugs/git/rtx6kpro-artifacts/upstream-check-20260920/`. No benchmark repository, serving node, container, model cache or image was changed. No rebase, merge, checkout, commit or push was performed.

The working branch is `spark`; the user-synced `master` and fetched `upstream/master` both resolve to `cfe30c0e25f5b0e3d211fb86d12b5cb0497e0837`. Comparison baseline is the previous recorded check at `ecdc481` on September 18, not the maintenance commits at the tip of spark.

## Conclusion

Publication has not stopped. The numbered community line remains R38, but current publication uses source-addressed date/hash snapshots and moving main/beta aliases for both Jovian Judgement and Karmic Kraken. Pin image digests, not merely tag names. Read the [publication inventory](upstream-publication-20260920.md) for exact registry identities and qualification scope, and the [source review](upstream-source-delta-20260920.md) for the model-specific changes.

Today's Karmic beta assembly uses the same six component source commits as the earlier September 20 image cited by master's serving results. Its recipe changes defaults, including GLM MTP3 and Qwen vision. A newer image timestamp is therefore not by itself evidence of new engine code or a new performance gain.

## New measured evidence in master

[Karmic serving results](https://github.com/voipmonitor/rtx6kpro/blob/cfe30c0/benchmarks/karmic-kraken-serving.md) now include six model/mode comparisons, five warmed windows per decode cell, acceptance and verifier rates, prefix checks, and raw records. Selected C1 verifier rates:

| Model | Saved JJ | Karmic | Change |
| --- | ---: | ---: | ---: |
| GLM Flash MTP3 | 99.39 | 112.14 | +12.84% |
| DS4 Vision K3 | 79.29 | 87.18 | +9.96% |
| Qwen Flash Next MTP3 | 76.06 | 81.78 | +7.52% |
| DS4.1 K7 | 89.87 | 98.58 | +9.69% |

These are RTX PRO 6000 Max-Q measurements with a +6000 memory-clock offset, not DGX Spark results. Qwen is TP1; DS4.1 uses RAM Engram and a 131K envelope. Saved comparison servers were not rebooted for these measurements. The timed source-composition image and registry image are distinguished explicitly in the report; the newest assembly was not the timed image. Do not transfer these percentages to our deployments.

The [startup study](https://github.com/voipmonitor/rtx6kpro/blob/cfe30c0/benchmarks/glm-startup-consistency.md) retains same-image GLM MTP3 verifier rates of 110.90 versus 105.74 steps/s across restarts. It does not establish the cause or recommend clearing caches. The [memory-control study](https://github.com/voipmonitor/rtx6kpro/blob/cfe30c0/benchmarks/tp4-memory-controls.md) tests allocator, NCCL and cuBLAS settings separately and promotes no combined preset. Neither justifies changing our qualified transport or memory settings wholesale.

## Recommended next candidate

Use one pinned Karmic beta composition as the next ARM64/SM121 build candidate, preserving model-specific serving settings. This needs a new CUDA 13.4/PyTorch 2.14 native foundation rather than copying R38's CUDA 13.3/PyTorch 2.13 objects. Published x86_64/SM120 wheels are not Spark binaries.

Priorities are startup-memory ownership, GLM MTP compaction, and the Qwen GDN/QSA prefill changes. Keep aligned checkpoint policy as the baseline and test request-boundary policy separately. DS4.1's original 524K qualification failure remains unresolved; new throughput, TP3 support, or tool-parser fixes do not close it.

This is research and fetch completion, not authorization or evidence for a production cutover.
