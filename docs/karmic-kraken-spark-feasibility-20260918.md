# Karmic Kraken: Spark build feasibility and DS4 Vision applicability

Read-only assessment, 2026-09-18. No node actions, builds, source checkouts, or serving changes. Frozen local sources were read from the `spark` branch and its R38 composition Git tree, not the mostly absent files on `master`.

## Decision

Karmic Kraken is a plausible **new ARM64/SM121 build project**, not a published Spark image we can adopt and not an established Python-only refresh of R38. Keep the qualified R38 Vision deployment unchanged while preparing that port. The new Vision fix belongs in the new candidate, but does not establish a defect in our existing InstantTensor deployment.

## Published build contract

The main release `karmic-kraken-20260918-ddc62cd55417f6b8` pins vLLM `cdea8220c24dedece15dce9b3cd5f3ffc2d0819b`, B12X `a83336581a3a907076e60797df69ab66df5a2ff1`, and recipe `3adc1f885b1c9d99c2b31f5c65271fba79ecf690`. Its manifest contains:

- CUDA 13.4.1, Python 3.12, Torch `2.14.0a0+4fdf77b940.nv26.8.63802676`.
- Linux x86_64 wheels for vLLM, InstantTensor, LMCache, NCCL, and the FlashInfer JIT cache. FlashInfer is explicitly built for SM120.
- NVIDIA PyTorch 26.08 foundation digest `33ef5fc15e8937602d64022209cdb2777b32dadf742f41332023d946041b3c14`, documented as Linux amd64.
- Native-smoke and LMCache checkpoint/filesystem contract qualification only; model-serving performance and full GLM cache E2E are explicitly unqualified.

Sources: [published release and manifest assets](https://github.com/local-inference-lab/blackwell-llm-docker/releases/tag/karmic-kraken-ddc62cd55417f6b8d315c874f27fdd57f4758ee9fba2b5fb5f91a4037bd4724d), [foundation contract at the recipe revision](https://github.com/local-inference-lab/blackwell-llm-docker/blob/3adc1f885b1c9d99c2b31f5c65271fba79ecf690/docs/jovian-cu134-wheel-runtime.md), [x64 publishing workflow](https://github.com/local-inference-lab/blackwell-llm-docker/blob/3adc1f885b1c9d99c2b31f5c65271fba79ecf690/.github/workflows/community-container-release.yml).

This is positive evidence that **these published artifacts** are unsuitable as native ARM64 binaries, not a claim that every upstream registry tag or alternative build has been inventoried. The newer beta published at 09:37Z also needs to be distinguished from main, rather than silently substituting its source composition.

Our R38 lock instead pins CUDA 13.3, Torch 2.13.0, and architecture 12.1a. Therefore there is no validated binary-reuse contract across this foundation change. Porting requires selecting and verifying an ARM64 foundation, rebuilding native components for that foundation and SM121, and re-running import, architecture, native-op, RoCE, loader and model gates. The existing recipe/gate structure is reusable engineering, not evidence that existing `.so` files or x86 wheels can be reused.

Local evidence: `git show spark:spark/glm53/r38-spark/source.lock.json`, runtime section and declared native hashes. The new foundation's exact ARM64 digest, matching toolchain availability, driver compatibility, and full native dependency closure remain pre-build checks. No build-time or performance estimate is established here.

## DS4 Vision fix: relevant to the next candidate, not a proven R38 defect

[Commit da4cdd3f](https://github.com/local-inference-lab/vllm/commit/da4cdd3f372ce7d71bfff1fcccfa668bd32f966b) changes two contracts:

1. Vision RMSNorm weights, sentinel vectors and vision/aligner dtype conversions use `allocate_weights`, honoring the active loader's allocation policy.
2. The multimodal wrapper delegates language weights to `self.language_model.load_weights`, rather than a generic `AutoWeightsLoader`; its tests cover child finalization during load and deferred finalization, exactly once.

The commit adds a Spark Vision wrapper using our same checkpoint revision `6821d6ad3681a4b137b066b76094fa82ebd0a380` and DSpark K3. Its parent launcher uses the **B12X loader**, FP8 KV, a 500,000-token context, utilization 0.82 and an explicit 10 GiB KV allocation. Those are not our serving contract and must not be copied as a bundle. Sources: [Vision wrapper](https://github.com/local-inference-lab/vllm/blob/da4cdd3f372ce7d71bfff1fcccfa668bd32f966b/scripts/serve-ds4-flash-vision-dspark-tp2-rdma.sh), [parent Spark launcher](https://github.com/local-inference-lab/vllm/blob/da4cdd3f372ce7d71bfff1fcccfa668bd32f966b/scripts/serve-ds4-flash-dspark-tp2-rdma.sh).

Our R38 recorded head render uses `--load-format instanttensor`, FP8 KV, utilization 0.85, context 524,288. More importantly, the frozen R38 tree `077347fdeab296404d0b0cb297d316176acc635a` already calls `self.process_weights_after_loading()` explicitly at the end of the Vision wrapper load. That method delegates to the language model once; the language-model hook flushes transfers and finalizes mega-MoE, mHC broadcasts and B12X weights. Thus the new commit is **not evidence that R38 omitted those finalization steps**.

`allocate_weights` already exists in R38's `vllm/model_executor/weight_transfer.py`; this is not an entirely new interface. Its allocator is loader-dependent and falls back to the ordinary factory with no active policy. The new commit fixes additional Vision call sites and the newer child-loader contract. A complete loader-by-loader equivalence proof was not performed, so neither universal R38 immunity nor a required R38 backport is asserted.

Local evidence: `spark:spark/ds4-vision/r38/receipts/head-render.txt`; frozen R38 `vllm/models/deepseek_v4/nvidia/vl_model.py:314-343`, `nvidia/model.py:1980-1990`, and `vllm/model_executor/weight_transfer.py:57-65`.

## Practical next steps

1. Pin one Karmic Kraken composition and inventory the ARM64 foundation/native dependency closure before authoring a build lock. Do not reuse published x86_64 wheels or silently carry R38 natives across Torch versions.
2. Include the Vision fix in that composition. Retain our established serving envelope initially, keeping a B12X-loader experiment separate from an engine comparison.
3. Qualify real text, image/OCR, multi-image, prefix-reuse and concurrent requests, then compare the standard benchmark against R38. Upstream's short functional checks do not establish our context envelope or throughput.
4. Treat upstream's `Dockerfile.spark-io-uring` as a dependency overlay only: it installs liburing/pkg-config atop an externally supplied base, not an ARM64 vLLM build recipe. [Source](https://github.com/local-inference-lab/vllm/blob/cdea8220c24dedece15dce9b3cd5f3ffc2d0819b/docker/Dockerfile.spark-io-uring).

No evidence examined here resolves the retired DS4.1 524K retrieval finding, and this assessment does not reopen that deployment.
