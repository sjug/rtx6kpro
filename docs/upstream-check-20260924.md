# Upstream check, September 24, 2026

Read-only development/publication review after the GLM Karmic investigation.
Fetched every configured remote in the eleven existing checkouts below with
`git fetch <remote> --no-prune`; all fetches succeeded. No checkout, branch,
rebase, worktree, deployment, image build, benchmark-source edit or serving change.
The working branch remains `spark`. Local `master` is two documentation commits
behind fetched `upstream/master`; it was not moved.

## Comparison and publication

The relevant tested wider Karmic composition uses vLLM `e77be22511`, B12X
`10a553ef`, and LMCache `413ac987`, plus our QSA865/SM121 overlays. This is not
the older Qwen HC-off image that was retained for serving. The previous source
check is [September 23](karmic-main-upstream-check-20260923.md).

| Input | Previous checked/tested source | Current main |
| --- | --- | --- |
| vLLM | e77be22511ab91ecf217760524b7579c366cca2a | 1794dcf18454900263e0c66711af8ea4a1283ac1 |
| B12X | 10a553ef980571f23a073f930cb386fbc8a77e07 | a7d7d29b2ef8869086e0ceaa787321f17544e3c9 |
| LMCache publisher integration | 413ac987336a91b5cb3899ce20720e9ad9f1fa47 | 361a85e88698174e9fd3b52fe372ceb40bd054f4 |
| Recipe | ce668abd1d649907338a42df2a1f4abdeb0e925e | 626f7af3205cee128c65de44891fa04825ec3df6 |

There are five new vLLM main commits and five B12X main commits against those
source pins. Newest main publication by `published_at`, not API listing order:

- Image: `ghcr.io/local-inference-lab/vllm:karmic-kraken-20260924-291c78003da67f8d`.
- Published: **2026-09-24 14:21:21 UTC**.
- Digest: `sha256:6f811e9cde5130665bef4b9e336658e01118e60580b3faaec26f11d5b54e265a`.
- Recipe and vLLM/B12X/LMCache pins match the current-main table.
- FlashInfer `2206a14e46387a56c093860a46bbbdd00596b75b`, InstantTensor
  `95d4729b6d6a991bb8de61877147a9d9d9100b23`, NCCL
  `93fe05d9f9b6963ef841166a69cd0b30e4efe97b` are unchanged from the prior
  publication. NCCL remains 2.31.2. Foundation lock is unchanged.
- FlashKDA remains `b59532f1f464fbd536272780e30df5bf6a2ccc02`.
- Published manifest contains x86_64 wheels; foundation is linux/amd64,
  CUDA 13.4.1/PyTorch 2.14. This is not an ARM64/SM121 image for direct use.
- Release scope is native GPU smoke plus 110 passed/zero skipped LMCache
  contract tests. Full model-serving performance and GLM cache E2E are explicitly
  unqualified. The `qualified` status must not be read as our model qualification.

[Official release and manifests](https://github.com/local-inference-lab/blackwell-llm-docker/releases/tag/karmic-kraken-291c78003da67f8da63966f5e6001aef46d468d2a38a7ade55d2678299deb204).

## Most relevant developments

### GLM: narrower SM121 prefetch grid, not a demonstrated TP4 win

[vLLM d11d4f35a8](https://github.com/local-inference-lab/vllm/commit/d11d4f35a8e8360aa58d79a2987cd637b3e85e7d)
changes the shared GLM L2 prefetch issuer from a constant 16 CTAs to 4 on SM121,
overridable by `VLLM_L2_PREFETCH_GRID`. The stated purpose is to leave bandwidth
for concurrent latency-bound kernels. The measured development workload was
DS4.1, not our GLM TP4 deployment.

This is relevant to our finding that turning GLM prefetch off improved decode.
It does not change precision, and a future controlled comparison could distinguish
off, old 16-CTA and new 4-CTA behavior. It has no effect when prefetch is disabled.
It is not evidence that all GLM regressions are fixed. Preserve R38 numerical
policies; do not compensate with additional dense quantization.

### Reliability: validate CuTe cache contents

[B12X #418 / 2fca4df8](https://github.com/local-inference-lab/b12x/commit/2fca4df84e68e4c55af3fa957305aa5952c7587f)
checks cached object sizes and SHA-256 hashes, treats corrupt/missing pairs as
cache misses, and fsyncs atomic publication. This is a useful reliability change
given our prior cache-related hypotheses, not proof that a historical illegal
instruction was caused by a corrupt artifact.

### DS4.1: genuine four-Spark optimization, distinct from DS4 Vision

The same vLLM commit and [B12X 6e2090bc](https://github.com/local-inference-lab/b12x/commit/6e2090bc77dba59c279dc8a54c8b07921799d0b5)
add overlapping disk Engram reads, concurrent two-table lookup, shared per-group
decode metadata, and L2 prefetch. The commit reports a fixed eight-row verification
cycle improving 49.36 to 44.79 ms, adaptive serving 79 to 82 tok/s, and unchanged
116/128 GSM8K answers. Its detailed runlogs are explicitly untracked, so these are
author-reported results rather than independently replayed public receipts.

The initial patch enabled an NVFP4 Markov head, but the immediately following
[b99d0d4304](https://github.com/local-inference-lab/vllm/commit/b99d0d430427da7a77be8090eaf3bfe2ba323dc1)
turns it **off by default** because its acceptance effect was not isolated. The
separate quantized draft vocabulary head also remains opt-in. Do not cite the
combined performance result as a measurement of the final defaults alone.
These changes target DS4.1, not our DS4 Flash Vision model, and do not resolve
the earlier 524K retrieval/attribution issue.

### GB10 fragmentation: an investigation lead, not a serving fix

[B12X 555aaa83](https://github.com/local-inference-lab/b12x/commit/555aaa83791c4105267ea54f5b29e4fb1f06e1b8)
adds a benchmark helper that holds small-page device allocations before allocating
test weights. The author reports 2 to 4% lower streaming performance on fragmented
small pages and 1.3% improvement in paired microbenchmarks. The helper is called
by benchmark programs; it is not automatically used by vLLM serving. It reserves
host-shared memory, so it must not be transplanted into serving as a presumed
fix for our allocation-pressure warnings. The scale is interesting for the
remaining small performance gap, but causality remains unmeasured.

### New formats are out of our current scope

Main adds IQ2_XS/IQ2_XXS/Q8_0 and packed Super3 support. This is not a reason to
change our checkpoints. B12X also changes common dense/W4A16 implementation and
MoE tuning schema/candidate versions, so a wholesale refresh is not a tiny cache
patch and cannot assume identical autotuner selections. No vLLM native-source
changes were found in csrc, cmake, CMakeLists.txt, setup.py, requirements or rust
over this main delta; that alone is not a complete native-reuse approval.

## Beta is separate

Fetched beta tips: vLLM `95720e473f8a25313384662ba46d5e81f2a33c82`, B12X
`a4b0d1c44b86869678bc38edebe38930cbcf6d1e`. Latest beta publication is
`karmic-kraken-beta-89af7f860e39961a2bb67e866912134e12b8e68c13299d4c554df835767a323d`
at 16:08:19 UTC. Do not substitute its features for main's.

- QSA865 is in beta, but [PR865](https://github.com/local-inference-lab/vllm/pull/865)
  remains open against main. Main still requires our reviewed crash backport.
- GLM TP3 (#889) pads attention/shared-expert geometry, uses expert parallelism,
  and runs the vision tower data-parallel. Its optional FP8 dense path changes
  precision and is not within the current request. Its BF16 low-latency GEMM
  selection is explicitly TP3-only because shape tables collide at TP2/TP4.
  Thus neither is an immediate GLM TP4 performance fix.
- FlashInfer all-rank autotune persistence fixes warm-start EP hangs. External
  aligned-prefix/offload fixes and request-stall/timeline diagnostics also land
  in beta. External cache fixes are not active with LMCache disabled.
- Dummy-draft fix #885 was reverted by #890 after padded rows caused GLM startup
  assertions. The later changelog-fragment restoration does not reapply its code.

## Recipe/cache and broader dependency changes

LMCache publisher changes implement checkpoint-on-reuse, then supersession-aware
checkpoint-on-eviction, bounded admission waits, and restore-failure diagnostics.
The follow-up preserves a request's prompt checkpoint when its response endpoint
is published, important for templates that rewrite prior answers. Recipe switches
expose these optional policies and persist TileLang/TVM/FlashInfer caches. They
do not accelerate our LMCache-disabled deployments by themselves.

The publisher LMCache comparison was read through GitHub API; its configured local
remote is the separate LMCache/LMCache upstream. InstantTensor and NCCL publisher
refs were checked with `ls-remote`, without adding repositories or remotes.

Broader FlashInfer upstream now has native-layout W4A16 for SM12x and additional
KDA prefill/speculative-decode work. These are not in the unchanged FlashInfer pin
of this main image; adopting them requires a separate source/ABI/call-path review.
CUTLASS main advances to 4.8. SGLang moves too, but remains outside our serving
line. eugr's Spark image repo picks up B12X integrity protection and opt-in earlyoom
integration; no host daemon is installed or changed here.

## Fetched primary tips and completion

| Checkout | Inspected ref | Fetched tip |
| --- | --- | --- |
| rtx6kpro | upstream/master | ff1ffcf30ce0d0f0f5a5d1496da7458d50ff7877 |
| vllm | lil/dev/karmic-kraken | 1794dcf18454900263e0c66711af8ea4a1283ac1 |
| b12x | lil/master | a7d7d29b2ef8869086e0ceaa787321f17544e3c9 |
| blackwell-llm-docker | origin/main | 626f7af3205cee128c65de44891fa04825ec3df6 |
| flashinfer | upstream/main | 28ae778e49ab8f6104c59cc9d43efdca4343f91f |
| LMCache | origin/dev | 05fc77a0a7ababd9a7f2e343a771bc4bbc5b65cb |
| cutlass | origin/main | 0b55a2f691d69981583568fd9eb69687b1f0de8a |
| sglang | origin/main | 5bb24e399d471dbb4f3d461972f7614a12a1484e |
| spark-vllm-docker | upstream/main | 00837c8b1b4b38cb09d7887f4bb8f24b4be9231a |
| dgx-spark-infra | origin/master | 3326b9cebb55571957a02cc549374b1c5d2e2c1e |
| llm-inference-bench | upstream/main | ccd9ad8ced7e387794391bfb0ac6d99b1f66ba6f |

All configured remotes, not just these primary refs, were fetched successfully.
No fetch failures. `master..upstream/master` contains two daily-summary commits
(`2960b92`, `ff1ffcf`), not a new numbered R39 release.

Recommendation: prioritize the precision-preserving 4-CTA prefetch discriminator
and CuTe integrity hardening, not a blind whole-image rollout. Keep production
unchanged. A new Spark composition would retain QSA865 and require fresh model
qualification, including recording any changed MoE tuning selections.
