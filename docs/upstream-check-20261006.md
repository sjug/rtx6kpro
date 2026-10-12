# Upstream refresh, October 6, 2026

## Result

All 23 configured remotes in the 11 existing checkouts fetched with `git fetch <remote> --no-prune`. The sjug forks were fast-forwarded to their GitHub parents on origin, with no force push:

| Fork | Branch | Commits |
| --- | --- | --- |
| rtx6kpro | master | 5 |
| vllm | main | 220 |
| blackwell-llm-docker | main | 732 |
| LMCache | dev | 7 |
| flashinfer | main | 632 |
| spark-vllm-docker | main | 15 |

Two forks have diverged and were not pushed:

- **b12x master:** the fork holds 834 commits of the pre-rewrite upstream history; upstream has 321 commits the fork lacks.
- **llm-inference-bench main:** two unmerged commits of our own from October 2 (TTFT probe headroom and `--chat-template-kwargs`); upstream is 5 ahead.

No local branch, build or serving state was touched.

**Karmic main** has no new release (latest is still `b174ad03`). `lil/dev/karmic-kraken` moved `ab86b707` → `eda1715e` with vLLM [#865](https://github.com/local-inference-lab/vllm/pull/865) (QSA capture metadata). Beta already carries #865 as merge `01f1b874`, and so does our production kit.

**Karmic beta** published seven images after our candidate `4102660a`. The most important change for us is in [`a0442f90`](https://github.com/local-inference-lab/blackwell-llm-docker/releases/tag/karmic-kraken-beta-a0442f909b0512f8eb0425a91e0cbec4e7a7ad250c59b560af4c4ad4c190f03f): upstream now builds an official DGX Spark linux/arm64 image for every beta release. The first such build (Actions run 37474677576, manual dispatch, 13:55 UTC) was still in progress when this report was written. No `-spark` tag has been confirmed yet, because `gh` lacks `read:packages`.

## Publication and source identities

| Component | Candidate `4102660a` (Oct 5) | Latest `7f1aba04` (Oct 6 13:06 UTC) |
| --- | --- | --- |
| b12x | `52640cb15d4ad1c7230f747e72c45d45dc2da681` | `236ddff04a5f5064084b46464800b70b34ec7d8c` |
| vllm | `f1c2508f1018f47ee8a78478a819d202f359887b` | `4a87c5884c2d640ae4514f90df455de584472cb9` |
| flashinfer | `dbd6238c6655b98195fdf77f04bba6facf5a38a4` | `34426e96c02cf25d5687405d74c8afeec1e20a10` |
| lmcache | `820af25ff630f4c00f7faefbcc31bc5ccd7bab71` | `3171cdb537de7af7d66d8e6ec7bf26bdcf3ba3bd` |
| nccl | `93fe05d9f9b6963ef841166a69cd0b30e4efe97b` | `300f3a73caba7196e436571d9ac34069e81edea4` |
| instanttensor | `95d4729b6d6a991bb8de61877147a9d9d9100b23` | `747390dc2bd1e0a18b0c7b13a09a38594edf342f` |

The FlashInfer, LMCache, NCCL and InstantTensor moves are each a single build-only merge that adds a linux/arm64 wheel target. They were read remotely with `gh api compare`, because `~/git/flashinfer` has no `lil` remote and NCCL and InstantTensor have no checkout. vLLM moved by 34 commits (9 PR merges) and B12X by 20 (4 PR merges).

Releases in order: `af322779` (docker #131), `2f1f4408` (vLLM #985), `ad0ffbca` (vLLM #986), `c8c8feea` (vLLM #987), `78703252` (docker #132, vLLM #993, b12x #483/#485), `a0442f90` (docker #133-135, vLLM #988/#989/#995/#996, b12x #488, LMCache #109), `7f1aba04` (docker #136, vLLM #994, b12x #487).

## What landed

### DGX Spark support upstream (docker #133-135, vLLM #996, b12x #488, LMCache #109, FlashInfer #5, NCCL #4, InstantTensor #3)

- New `dgx-spark` hardware profile (research-only). It sets `CUTE_DSL_ARCH=sm_121a`, `TORCH_CUDA_ARCH_LIST=12.1a`, `VLLM_ENABLE_PCIE_ALLREDUCE=0`, RoCE NCCL settings (`NCCL_IB_GID_INDEX=3`, `NCCL_CUMEM_ENABLE=0`, `NCCL_NVLS_ENABLE=0`), one compile worker per B12X stage, and `B12X_AUTOTUNE=0`.
- `PRESET=qwen38-dgx-spark-x2`: TP2 over two boxes, the `original` checkpoint, 16 slots, `gpu-memory-utilization` 0.75 and `VLLM_PLE_CPU_OFFLOAD=0`. Upstream reports 3.39M KV tokens and about 50 tok/s for one request.
- `PRESET=glm53-dgx-spark-x2`: GLM-5.3-Flash-NVFP4-Spark at TP2, MTP3, 8 slots, A16 prefill and `gpu-memory-utilization` 0.83. Upstream reports C1 32.6 and C2 52.0 tok/s, and 1,955 tok/s prefill at 8K.
- `NNODES`, `NODE_RANK`, `MASTER_ADDR` and `MASTER_PORT` options; ranks above 0 run headless.
- The image is published as `<channel>-spark-<date>-<id>`, with a `karmic-kraken-beta-spark` alias, from the same component commits as the amd64 image. Wheels are built under QEMU on the amd64 builder from the arm64 twin of NGC PyTorch 26.08.
- The bundled llm-inference-bench is v0.7.8 (lil-bench 1.4), which records GB10 unified memory and the node count.

### GB10 correctness (vLLM #989, #995)

- The DFlash split-KV draft attention kernel and the NVFP4 MTP draft head required compute capability exactly 12.0. Both now accept the SM120 family, which includes GB10 (12.1).

### All models (vLLM #988)

- The torch thread count is set before warmup. Before this, the first real request recompiled every torch.compile'd helper that warmup had compiled. Upstream measured GLM-5.3-Flash TP4 on four Sparks: cold first prefill went from +1.03 s to +0.22 s.

### GLM-5.3 744B on RTX PRO 6000 (vLLM #985-987, #993, #994; b12x #483, #485, #487; docker #132, #136)

- New models, presets and kernels for 8- and 6-GPU PCIe hosts.
- b12x #487 also changes the shared fused W4A16 MoE planning and launch code. Planned CTAs per SM now get a shared-memory carveout, and plans that cannot fit fail at planning time. It also fixes a K32 swizzle in the new K32 tiles. Upstream reports GLM-5.3-Flash TP2 serving unchanged. Qwen is not mentioned.

## Applicability to our fleet

- **Build path:** upstream's Spark image builds from the same component commits as amd64. We have been composing our own SM121 kits on blackwell-llm-docker for this. Before we use their image, we need to answer:
  - Is it SM121-native? `CUTE_DSL_ARCH=sm_121a` suggests yes for B12X, but the FlashInfer and NCCL arch lists are unverified.
  - Does it carry our kit's fixes, such as the NCCL `LL,Simple` both-rails contract (launcher-level) and any SM121 patches our kit applies?
  - Does it pass our gates?

  The image contents must be checked before we make any claim.
- **Qwen (dusty/kirby):** the upstream Spark preset differs from our qualified contract:
  - Checkpoint: `original` vs our QAD `7c4f1bc1`.
  - Slots: 16 vs our maximum of 4.
  - Autotune: `B12X_AUTOTUNE=0` vs our natural autotune.
  - MTP: no MTP mode override.

  It is a reference, not a drop-in. On the image side, #988 and the b12x #487 fused-MoE planning change are the only changes on our Qwen path. #988 should only cut first-request latency. #487 could move B12X plan selection and needs measuring.
- **GLM (sparky TP4):** our runner defaults to `VLLM_GLM53_MTP_DRAFT_HEAD=nvfp4`, which works today only because `spark/karmic-beta-sm121/spark-overlay.patch` relaxes the same exact-12.0 check in `mtp_draft_head.py`. vLLM #995 upstreams that fix, so the overlay hunk becomes redundant and will likely conflict on the next kit rebase. The overlay's CMakeLists hunk is separate. #989 (DFlash attention on GB10) matters only if we serve DFlash, which our runner does not do by default.
- **Qwen checkpoint:** `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` main moved today. `1134fa32` (12:57 UTC) is a config change, "Restore text_config.ple_embedding_dtype = nvfp4" (#7), and `6c01ac37` is README only. We serve `7c4f1bc1`. Upstream's `original` checkpoint has no revision pin, so it resolves to the new main.
- **DS4 Vision (rusty/toby):** nothing new.
- **b12x#463** (Qwen C1 autotune rule) is still open with no comments.
- **Our candidate `4102660a`:** none of the seven newer releases fixes something it needs. Moving the kit to `7f1aba04` would add #988 and #487 (Qwen path), at the cost of a rebuild and a fresh qualification.

## Fetch audit

| Checkout | Principal ref moves |
| --- | --- |
| `vllm` | `lil/integration/karmic-kraken-beta` `f1c2508f` → `4a87c588`; `lil/dev/karmic-kraken` `ab86b707` → `eda1715e`; `upstream/main` → `b267fe48` |
| `b12x` | `upstream` beta source `52640cb1` → `236ddff0` |
| `blackwell-llm-docker` | `upstream/main` `e881183c` → `bd2f416` |
| `LMCache` | `lil` integration `820af25f` → `3171cdb5`; `upstream/dev` → `8c77a6f7` |
| `rtx6kpro` | `upstream/master` at `58291d64` (no move since October 5) |
| `llm-inference-bench` | `upstream/main` → `12c2a73` (lil-bench 1.4, tool-eval-bench) |
| `flashinfer`, `sglang`, `cutlass`, `spark-vllm-docker`, `dgx-spark-infra` | upstream mainlines only; not in our image path |

## Evening pass (20:47 EDT)

All 23 remotes fetched again. Fork sync on origin, no force push:

- **Fast-forwarded:** vllm main +32, blackwell-llm-docker main +13, LMCache dev +4, flashinfer main +23.
- **Already current:** rtx6kpro and spark-vllm-docker.
- **Still diverged and not pushed:** b12x master (origin +834 / upstream +359) and llm-inference-bench main (+2 / +5).

### New beta releases

- [`43fff4d7`](https://github.com/local-inference-lab/blackwell-llm-docker/releases/tag/karmic-kraken-beta-43fff4d750947a1d5d11f400d9aa34e726930479bcf6c9520ba6352acd0c7e19) (20:26 UTC), vLLM `bf2b1d10`:
  - vLLM [#998](https://github.com/local-inference-lab/vllm/pull/998): opt-in CKV-gather prefill for GLM-5.3 744B with DCP>1. GLM-5.3-Flash and DCP1 are unchanged.
  - vLLM #997: precompiled-wheel packaging fix that unblocks the Spark build.
  - docker #137: the Spark build follows every completed publisher run.
- [`ff997f2f`](https://github.com/local-inference-lab/blackwell-llm-docker/releases/tag/karmic-kraken-beta-ff997f2fa04e5addda828af5bf3c3ec9f2f53e043524bec5de312a790dd3dc31) (22:16 UTC), vLLM `abdf960a`:
  - vLLM #999: GLM-5.3 DCP with online quantization.
  - vLLM #1000: opt-in NVFP4 draft head for GLM-5.3 744B. GLM-5.3-Flash is unchanged.

B12X, FlashInfer, LMCache, NCCL and InstantTensor are unchanged from `7f1aba04`. None of these releases touches Qwen, GLM-5.3-Flash or DS4 Vision.

### First upstream Spark image

`ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-spark-20261006-ff997f2fa04e5add` (alias `karmic-kraken-beta-spark`), digest `sha256:c06f3a89da21e7702649fab0b07095f7326e94d4f77c688c09867b3fc86f8caf`:

- **Build:** linux/arm64, 68 layers, 13.4 GB, recipe `966c341b`, built 2026-10-07 00:28 UTC (Actions run 37549159898). The 13:55 dispatch failed at the vLLM packaging step that #997 fixes, and the 22:16 run also failed.
- **Status:** `research-only`, with `qualified_serving_receipts: []`.
- **Components (from the `container-release-linux-arm64.json` receipt):** source commits are identical to the amd64 `ff997f2f` release. The arm64 bundles were built from same-tree commits: FlashInfer `7c925499`, NCCL `dfc31c36` (2.31.2), InstantTensor `b5fc1e43`.
- **vLLM native code:** reused from the arm64 build at `67626c51` ("target sm_121a for linux/arm64 wheels", `cuda.arch-list=12.1a`). No `csrc`, `cmake` or `CMakeLists.txt` changes exist between `67626c51` and `abdf960a`, so the reused objects match the source.
- **Base:** NGC PyTorch 26.08 arm64, CUDA 13.4.1, torch 2.14.0a0, base `CUTLASS_DSL_VERSION=4.6.2`. B12X master `143debed` aligns Spark wheels to CUTLASS DSL 4.7.1; which DSL ends up in `/opt/venv` is unverified.
- **Open check (needs a pull):** at CUDA 13.4, vLLM's `CMakeLists.txt` lists `CUDA_SUPPORTED_ARCHS` without 12.1. Our `spark-overlay.patch` adds 12.1 there. Upstream builds with `TORCH_CUDA_ARCH_LIST=12.1a` and no such patch. The build's FATAL guard against an empty arch set did not fire, and most kernel lists use `12.0f` family entries, which match `12.1a`. Kernels gated on a plain `12.0` entry may still be missing on SM121. Confirm with `cuobjdump --list-elf` on `vllm/_C*.so` after a pull.

### Unreleased upstream work to watch

- **`lil/dev/karmic-kraken` (Karmic main):**
  - `7e847a8f` and `e0734571` allow the NVFP4 MTP draft head and DFlash draft attention on SM120-family GPUs. This is #995/#989 arriving on main.
  - `f4cba696` invalidates tuning caches and startup plans with revision 22.
  - No main release has been cut.
- **B12X master:** `ffabbade` "Invalidate built-in tuning decisions with schema 22". When it ships, B12X selection caches (including our `~/.cache/vllm-jj-qwen38-4p89/jit/.../b12x/preparation` records) will be re-planned once. First boots on that image will retune, and selections may move. This is relevant to b12x#463 and to the 4102660a selection-cache diffs.

### Applicability

- Production `500ae05b` and candidate `4102660a` are unaffected by the evening releases. Nothing in them warrants a rebuild.
- The Spark image is now concrete enough to evaluate. The SM121 kernel coverage question above is the first gate, before any boot.
