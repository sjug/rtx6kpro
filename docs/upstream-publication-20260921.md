# Upstream image publication check, 2026-09-21

Read-only GitHub release/API and pinned-source inspection, checked approximately 15:37 UTC. Compared against [September 20's four-channel baseline](upstream-publication-20260920.md). No repository fetches, node actions, builds or benchmarks were performed by this publication review. Sources without a local checkout (InstantTensor, nccl-canonical and FlashKDA) were read remotely; no repositories, directories or remotes were added.

## Result

All four channels have newer publications than the previous report, still dated September 20 UTC. Only **Karmic beta** changes the shipped vLLM/B12X source pins. The other three are recipe republishes, not fresh engine improvements. All four remain GitHub prereleases with `immutable: false`. Digests below are from publication receipts, not an independent registry pull. Sort the release API by `published_at`: its returned order is not newest-publication-first. [Release API](https://api.github.com/repos/local-inference-lab/blackwell-llm-docker/releases?per_page=100).

All now use recipe `23d674e8f658dae2db75399c48430693c196b258`, replacing `8c5aa7f828689e6385fa309acd0cce168fb38ec7`.

| Channel | Previous image suffix | Latest image suffix | Latest publication UTC |
|---|---|---|---|
| JJ main | `jovian-judgement-20260920-b28a393036050aa0` | `jovian-judgement-20260920-2d28f356c192a540` | Sep 20 22:45:12 |
| JJ beta | `jovian-judgement-beta-20260920-2d02583e0f5ad1cc` | `jovian-judgement-beta-20260920-b1d2f0c73798b2e3` | Sep 20 22:55:54 |
| Kraken main | `karmic-kraken-20260920-c6f32f2d1095ef4b` | `karmic-kraken-20260920-919cdcb950d4c242` | Sep 20 23:06:35 |
| Kraken beta | `karmic-kraken-beta-20260920-828a46ee4bbf1aa4` | `karmic-kraken-beta-20260920-22f3f98b065ed8de` | Sep 20 23:17:12 |

Repository: `ghcr.io/local-inference-lab/vllm`. Latest digests:

- JJ main: `sha256:46faf9a4b1962c7b4a382e9f65e6cae91941dd366321c15a38953fab3267da8d`.
- JJ beta: `sha256:8553dc6824799907cbf58dc64a3d8f10ca367762815ca1fe03b25747318bf3b6`.
- Kraken main: `sha256:6988c9d764449e394b74b12a6728ff940dee5c60cbc12635fa0fd69b7113d8a0`.
- Kraken beta: `sha256:60dc178fe69015eb289db12b6183b8d6aec75613e0c659ce7e1f9ba05e8bbed1`.

Exact source pins from `container-release.json`:

| Channel | vLLM source, previous to latest | B12X source, previous to latest |
|---|---|---|
| JJ main | `8e1f1e587f8d24faf606f334a1c4bdaaa6bd4368`, unchanged | `0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68`, unchanged |
| JJ beta | `8eca34a03304ee681cb8131f28d118f880efb697`, unchanged | `b9f44defd3a871f8c81d51846f3949ac15ace73d`, unchanged |
| Kraken main | `af9e4dca109e0348323c0182e98a3aaf7282bfc3`, unchanged | `0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68`, unchanged |
| Kraken beta | `57a80980bbf4b40398de7ed851b23e55a3a4c50e` to `22476af54c637cbb7c7d8193addd160da83a5ce3` | `e9ce547767ff9ee6509faf294fa1b4e2380dfbf5` to `f6d8b8eb94cdeb4e652652f925a494c6fc86f101` |

Sources: [JJ main release](https://github.com/local-inference-lab/blackwell-llm-docker/releases/tag/jovian-judgement-2d28f356c192a54043c1039e0c37a37729f759eb391d1cbebeacc43a38829da1), [JJ beta release](https://github.com/local-inference-lab/blackwell-llm-docker/releases/tag/jovian-judgement-beta-b1d2f0c73798b2e3f5365d6f0fa16bb5c533050c5ffc98930807d0f1a9d0a063), [Kraken main release](https://github.com/local-inference-lab/blackwell-llm-docker/releases/tag/karmic-kraken-919cdcb950d4c24241eee64385fe6594c30cceea5b85f77acb9da89890600b61), [Kraken beta release](https://github.com/local-inference-lab/blackwell-llm-docker/releases/tag/karmic-kraken-beta-22f3f98b065ed8de60e78c8786ae2bc1b3956616f84b6e148e6393e95251f0bc). Corresponding receipt asset IDs are `577699610`, `577713834`, `577728669`, `577741670`.

JJ beta's receipt separately records observed branch commits `074707f39a330c31201356858b7194d3d70d9562` (vLLM) and `39e0fa7506974f7bd7fdf261e26b6df8ced4de7e` (B12X). These are not the wheel source pins. Channel mappings remain JJ main `dev/jovian-judgement`/`master`, JJ beta `integration/beta`/`integration/beta`, Kraken main `dev/karmic-kraken`/`master`, and Kraken beta `integration/karmic-kraken-beta` for both. [Pinned channel contract](https://github.com/local-inference-lab/blackwell-llm-docker/blob/23d674e8f658dae2db75399c48430693c196b258/tools/jovian_wheel_runtime/community-channel.json).

## Practical changes

The recipe adds generated component changelogs and defaults `enable-prompt-tokens-details: true` in the common launcher profile. DeepSeek already enabled this per-model, so moving it to common primarily exposes cached prompt-token usage for the other models. It does not change the underlying cache algorithm. Changelog fragments are derived from exact component revisions, published as `release-changelog.json`, embedded in the runtime manifest, and required for vLLM/B12X integration-source changes. [Recipe comparison](https://github.com/local-inference-lab/blackwell-llm-docker/compare/8c5aa7f828689e6385fa309acd0cce168fb38ec7...23d674e8f658dae2db75399c48430693c196b258), [fragment policy](https://github.com/local-inference-lab/blackwell-llm-docker/blob/23d674e8f658dae2db75399c48430693c196b258/docs/integration-release-changelog.md).

Kraken beta first shipped its new engine composition in intermediate image `karmic-kraken-beta-20260920-fc3922cee1c92350` at 22:29:38 UTC. Its generated release notes identify:

- Qwen QSA decode-context-parallel support: rank-local KV/selector ownership, globally merged candidates, and FP32-weighted distributed attention; external QSA KV-cache transfer remains unsupported. [vLLM #816](https://github.com/local-inference-lab/vllm/pull/816), [B12X #402](https://github.com/local-inference-lab/b12x/pull/402).
- Speculation method and draft depth separated in compiled graph cache identity. [vLLM #815](https://github.com/local-inference-lab/vllm/pull/815).
- Exact prepared DCP executable identities for frozen-resolution validation. [B12X #401](https://github.com/local-inference-lab/b12x/pull/401).

The final `22f3f98b` republish has the same six component source pins as `fc3922ce`. Its empty immediate-predecessor changelog therefore must not be read as “nothing changed since yesterday's report.” These are upstream release descriptions, not new local correctness or speed measurements. [Intermediate release and cumulative change list](https://github.com/local-inference-lab/blackwell-llm-docker/releases/tag/karmic-kraken-beta-fc3922cee1c92350621c02cc2a0cf626cdc637941b9e75c8430896e531111942).

## Architecture and gate boundaries

All four receipts still report 94 cache-contract tests passed, zero skipped, with qualification confined to native GPU smoke and LMCache checkpoint/filesystem contracts. Model-serving performance and full GLM cache E2E remain explicitly unqualified. The runtime manifest remains `research-only` despite the assembly receipt's narrower `qualified` status.

The newly inspected Kraken beta runtime manifest declares CUDA 13.4.1, Python 3.12, NVIDIA Torch `2.14.0a0+4fdf77b940.nv26.8.63802676`, and x86_64 vLLM/InstantTensor/LMCache/NCCL wheels, plus the SM120 FlashInfer JIT cache. This is still not a native ARM64/SM121 Spark artifact; an ARM64 foundation and native rebuild are required. The other channels retain their prior native component pins with the same recipe family. No claim here qualifies GB10, resolves Qwen's prefill deficit, or closes DS4.1 524K correctness. [Kraken beta runtime manifest](https://github.com/local-inference-lab/blackwell-llm-docker/releases/download/karmic-kraken-beta-22f3f98b065ed8de60e78c8786ae2bc1b3956616f84b6e148e6393e95251f0bc/manifest.json).

## Shared components and remote-only refs

All four still ship the same shared source pins:

| Component | Published source |
|---|---|
| FlashInfer | `2206a14e46387a56c093860a46bbbdd00596b75b` |
| LMCache | `688bee14e157b64623d93c07fc0d4db93470e12f` |
| InstantTensor | `95d4729b6d6a991bb8de61877147a9d9d9100b23` |
| NCCL | `93fe05d9f9b6963ef841166a69cd0b30e4efe97b` |

Live API checks confirm [InstantTensor main](https://api.github.com/repos/local-inference-lab/InstantTensor/commits/main) and [NCCL canonical branch](https://api.github.com/repos/local-inference-lab/nccl-canonical/commits/canonical%2Fcu134-nccl2312-amd-turin) remain at those published pins.

Kraken beta still pins FlashKDA `b59532f1f464fbd536272780e30df5bf6a2ccc02`, with packed-checkpoint and SM120 CTA-copy patches applied by its [CMake dependency declaration](https://github.com/local-inference-lab/vllm/blob/22476af54c637cbb7c7d8193addd160da83a5ce3/cmake/external_projects/flashkda.cmake). Live FlashKDA default branch is `master`, not `main`; `master` is `1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b`. Comparison is **diverged**, two commits ahead and twelve behind the recipe pin, rather than a simple two-commit upgrade. The master-side changes concern TMA proxy fences and sequence-length preparation scans. They are not incorporated merely by selecting the new container and cannot be recommended as a blind pin replacement. [Exact comparison](https://github.com/vllm-project/FlashKDA/compare/b59532f1f464fbd536272780e30df5bf6a2ccc02...1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b).
