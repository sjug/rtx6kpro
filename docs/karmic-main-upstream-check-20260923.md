# Karmic main updates beyond the Spark candidate

Checked September 23, 2026 against the exact source pins in
`spark/karmic-main-sm121/20260922/source.lock.json`. Fetched existing vLLM and B12X `lil`
remotes and build-repository `origin`, with `--no-prune`. No checkout, worktree,
build or serving changes. LMCache fork comparison was read through GitHub because
the existing checkout has only its upstream remote, not the publisher fork.

## Source versus publication

| Component | Spark candidate | Current main |
| --- | --- | --- |
| vLLM | 6afb99982576a7a2eb53d667189e859629e22739 | e77be22511ab91ecf217760524b7579c366cca2a |
| B12X | 4f3028b19c1d8290dc72b6f483aba40de23eae5a | 10a553ef980571f23a073f930cb386fbc8a77e07 |
| Build recipe | 0ab249fe88f004c286f7100a2b33f66639901fb1 | ce668abd1d649907338a42df2a1f4abdeb0e925e |

Newest observed main publication:
`ghcr.io/local-inference-lab/vllm:karmic-kraken-20260923-f37d447a16799e72`,
published 16:55:34 UTC, digest
`sha256:751dc9c4803167e9887ffffaefa14ad524f3ef8b21bfb4029e364d403ed312c5`.
Its recipe is `459593df3dab9cc5861c0a66dfed709f1a5f2f72`, behind recipe tip.
The release pins the current vLLM/B12X main commits above, and LMCache
`413ac987336a91b5cb3899ce20720e9ad9f1fa47`. FlashInfer, InstantTensor and NCCL
source pins match our candidate. Publication's explicit qualification scope is
native GPU smoke and LMCache checkpoint/filesystem contracts, not full model
serving performance or GLM cache end-to-end qualification.

[Release and manifests](https://github.com/local-inference-lab/blackwell-llm-docker/releases/tag/karmic-kraken-f37d447a16799e7237d663e19069583ac2e811dd279c80aa9843a901212bd26a).

## Changes worth considering separately from QSA865

1. **DeepSeek prefill preparation:** vLLM `cff8aebfeb` plus B12X `0332cc50`
   prepare capacity plans in advance for block-FP8 and DeepSeek WO projections,
   covering intermediate token counts without first-use preparation. This directly
   addresses the previously observed WO tail-shape preparation warnings. It does
   not establish that the remaining long-context prefill regression is fixed.
   The commit records 12 GB10 checks including graph replay and a DS4.1 TP4 smoke.
2. **GLM/Kimi RecoverSSM correctness:** vLLM `e77be22511` (#860) keeps a
   speculative endpoint exactly on a block boundary in the correct block rather
   than selecting the following slot. Relevant to a later GLM candidate; not
   Qwen's GDN path and not a match for this QSA write fault.
3. **Loader overhaul:** vLLM `f92c576283` and B12X `1ec67ef6` remove loader-owned
   weight allocations, use ordinary PyTorch allocations and a bounded 16 MiB
   O_DIRECT io_uring ring with asynchronous CUDA copies. Upstream records DS4.1
   TP4 loading/capture/chat on four Sparks. B12X `ae431182` invalidates prior
   tuning decisions for the changed storage. This is a coupled loader/tuning
   experiment with native C changes, not something to mix into a narrow crash
   backport. Our Qwen profile uses InstantTensor, not the B12X loader.
4. **Deeper Qwen MTP:** vLLM `57fdda71b0` allows MTP beyond four draft tokens
   with ring-aware QSA alignment. MTP3's ring alignment remains eight; no claim
   of benefit for our existing MTP3 profile.

Remaining B12X main changes: representative DS4 0731 benchmark workloads,
neutral codebook names, and wheel CI for Karmic integration. No additional
Qwen fault fix is present on the inspected main tip.

Recipe changes add an optional prefill-token compute limit (requires the compute
share controller), bounded Qwen NVMe cache eviction, and a fail-closed Qwen DCP
external-cache gate requiring atomic QSA transfer support from vLLM #864.
Supervisor shutdown messages now identify model/cache exit status or external
signals. These affect the upstream lil-serve/cache workflow, not our retained
shell launchers. LMCache's two new commits isolate malformed MQ requests so its
RPC polling thread survives; LMCache remains disabled in our Qwen profile.

## Crash-fix status and decision

[PR865](https://github.com/local-inference-lab/vllm/pull/865) remained open at
head `26b42cf9eb715b118e55f91f53fa0ab412a79c09` when checked. It is **not** in
the observed main publication. The initial proposal was a narrow derivative.
The user subsequently approved integrating the backport into the new source
baseline. `spark/karmic-main-sm121/20260922/qsa865/` now composes that publication's
vLLM, B12X and LMCache updates plus PR865 over our existing ARM64 native image.
It retains InstantTensor with LMCache disabled, and capture size 32 rather than a
16-row avoidance workaround. GPU and model qualification remain pending.

The wider composition is not a one-variable causal comparison with the crashed
image. Qualification must establish the candidate's correctness without claiming
that a pass independently proves PR865 caused the recovery. DeepSeek and GLM
serving qualification remain separate from the Qwen window.
