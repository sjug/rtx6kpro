# Upstream image publication check, 2026-09-20

Read-only release/API and source inspection. No node actions, builds or local repository fetches performed by this review.

## Conclusion

Publication has not stopped. Upstream now publishes source-locked, wheel-assembled snapshots under four channels rather than relying on the numbered JJ R38 naming convention. All four channels published images on September 20, and multiple snapshots were also published September 19. These are not broadly model-qualified releases: their receipts explicitly limit qualification to native GPU smoke and LMCache checkpoint/filesystem contracts.

The main-versus-beta distinction describes source selection, not production readiness. All four assembly releases remain GitHub prereleases. A historical `vllm-jovian-cu134-beta-` wheel name is an artifact format, not evidence that its source came from a beta integration branch. [Channel contract](https://github.com/local-inference-lab/blackwell-llm-docker/blob/8c5aa7f828689e6385fa309acd0cce168fb38ec7/docs/community-wheel-container-automation.md).

## Latest observed snapshots

Times are publication timestamps in UTC. All use recipe `8c5aa7f828689e6385fa309acd0cce168fb38ec7` and report 94 cache contract tests passed, zero skipped.

| Channel | Published | Image tag suffix | vLLM source | B12X source |
|---|---|---|---|---|
| JJ main | Sep 20 19:23:40 | `jovian-judgement-20260920-b28a393036050aa0` | `8e1f1e587f8d24faf606f334a1c4bdaaa6bd4368` | `0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68` |
| JJ beta | Sep 20 19:36:34 | `jovian-judgement-beta-20260920-2d02583e0f5ad1cc` | `8eca34a03304ee681cb8131f28d118f880efb697` | `b9f44defd3a871f8c81d51846f3949ac15ace73d` |
| Kraken main | Sep 20 19:47:15 | `karmic-kraken-20260920-c6f32f2d1095ef4b` | `af9e4dca109e0348323c0182e98a3aaf7282bfc3` | `0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68` |
| Kraken beta | Sep 20 20:00:22 | `karmic-kraken-beta-20260920-828a46ee4bbf1aa4` | `57a80980bbf4b40398de7ed851b23e55a3a4c50e` | `e9ce547767ff9ee6509faf294fa1b4e2380dfbf5` |

Image repository is `ghcr.io/local-inference-lab/vllm`. Exact image digests, in the same order:

- JJ main: `sha256:61107ae58b3d408f33600348e4196543b1f0ca3528082a808fac13d60d58d2b6`.
- JJ beta: `sha256:9a7ad66a6ca548b2166b45a636a837decffe531157a6e98c53dc26eb05c0653c`.
- Kraken main: `sha256:cf0fcf6fbb4dbf78716f6981cb984a363b1da6f2826769c877580bedb5ae5ab5`.
- Kraken beta: `sha256:b737367417a357a09088796553cf8d0ebe240cb3dc3a38f51c379a34a3c28081`.

Sources: [JJ main release](https://github.com/local-inference-lab/blackwell-llm-docker/releases/tag/jovian-judgement-b28a393036050aa06dce3677468ee0749eafa1126aeda050a0ba7c0e76086653), [JJ beta release](https://github.com/local-inference-lab/blackwell-llm-docker/releases/tag/jovian-judgement-beta-2d02583e0f5ad1cc9bcbce1c70cb22b22c7f00c8a0532c3436d9f1592d77feab), [Kraken main release](https://github.com/local-inference-lab/blackwell-llm-docker/releases/tag/karmic-kraken-c6f32f2d1095ef4bcce85c235a719f509706dad5fc3a8b934307d3873aae7fcd), [Kraken beta release](https://github.com/local-inference-lab/blackwell-llm-docker/releases/tag/karmic-kraken-beta-828a46ee4bbf1aa4fd0440d8ad930d0990977a4a1885ae0473eaa5a92749ae94). Their `container-release.json` assets supplied the identities; GitHub release metadata supplied publication timestamps. GitHub's immutable-release flag is false, so digest pinning is still required even for date/hash-named tags.

## What changed relative to the beta already documented on master

The September 20 03:08:42 Kraken beta `443d9f815c57d23b` and latest `828a46ee4bbf1aa4` have identical six component source commits. Their recipe changed from `f0c4f9fe04267d7a83a896f1c81d390b1ece5fee` to `8c5aa7f828689e6385fa309acd0cce168fb38ec7`. This newer image is not a new engine-source performance result.

The recipe update changes generic launcher defaults: GLM mode `off` becomes `mtp` with three proposals; Qwen `language-model-only: true` becomes `false`, enabling image input. Neither should silently replace our independently qualified Spark launch contracts. [Earlier beta release](https://github.com/local-inference-lab/blackwell-llm-docker/releases/tag/karmic-kraken-beta-443d9f815c57d23bfc3a5e780409fb26b0878d1c8b0a28c4c2818d6414af5454), [recipe commit](https://github.com/local-inference-lab/blackwell-llm-docker/commit/8c5aa7f828689e6385fa309acd0cce168fb38ec7).

## Spark applicability

The September 18 assessment still holds for these latest artifacts: they use CUDA 13.4.1, Python 3.12 and NVIDIA Torch `2.14.0a0+4fdf77b940.nv26.8.63802676`, with x86_64 InstantTensor, LMCache, NCCL and vLLM wheels and an explicitly SM120 FlashInfer JIT-cache wheel. The workflow uses an x64 builder and the foundation contract specifies Linux amd64. These particular releases are not native ARM64 Spark images. A Spark port needs a verified ARM64 foundation and rebuilt native components, not an R38 Python-only refresh. This is an inventory of these releases, not a claim about every possible registry image.

The runtime manifest itself says `research-only`; the assembly receipt says `qualified` with the narrow scope stated above. These are different scopes, not proof of full-model acceptance. [Runtime foundation](https://github.com/local-inference-lab/blackwell-llm-docker/blob/8c5aa7f828689e6385fa309acd0cce168fb38ec7/docs/jovian-cu134-wheel-runtime.md), [workflow](https://github.com/local-inference-lab/blackwell-llm-docker/blob/8c5aa7f828689e6385fa309acd0cce168fb38ec7/.github/workflows/community-container-release.yml), release manifest assets linked above.

## Dependency/source inventory

All four latest assemblies share:

- FlashInfer `https://github.com/local-inference-lab/flashinfer.git`, branch `community/jovian-judgement-cu134-sm120`, source `2206a14e46387a56c093860a46bbbdd00596b75b`.
- InstantTensor `https://github.com/local-inference-lab/InstantTensor.git`, branch `main`, source `95d4729b6d6a991bb8de61877147a9d9d9100b23`.
- LMCache `https://github.com/local-inference-lab/LMCache.git`, branch `integration/local-inference-lab`, source `688bee14e157b64623d93c07fc0d4db93470e12f`.
- NCCL `https://github.com/local-inference-lab/nccl-canonical.git`, branch `canonical/cu134-nccl2312-amd-turin`, source `93fe05d9f9b6963ef841166a69cd0b30e4efe97b`.

FlashKDA is a vLLM native dependency rather than a separate assembly wheel. Kraken main pins `https://github.com/vllm-project/FlashKDA.git` at `b59532f1f464fbd536272780e30df5bf6a2ccc02`, with its declared packed-checkpoint patch applied. [Pinned CMake source](https://github.com/local-inference-lab/vllm/blob/af9e4dca109e0348323c0182e98a3aaf7282bfc3/cmake/external_projects/flashkda.cmake).

## What to monitor going forward

Follow the four assembly channels and their component-source commits, not only new numbered release documents. The resolver checks wheel releases every five minutes, subject to Actions scheduling delays, and also responds to publication notifications. A source change waits for a complete compatible wheel before entering an assembly. Main uses development/master sources; beta uses separately composed integration branches. No automatic merge between them is implied. None of this requalifies our models or resolves the retired DS4.1 524K finding.
