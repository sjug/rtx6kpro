# Karmic beta source refresh, October 1, 2026

Checked 2026-10-01T13:03:22.950946+00:00. Fetched all 22 configured remotes in the 11 existing checkouts with `git fetch <remote> --no-prune`. All succeeded. Every checkout HEAD and porcelain status matched its pre-fetch state. No checkout, merge, rebase, reset, new remote, repository, worktree, build, benchmark or serving change was made. No new directories were created. Temporary API responses and fetch results were held as files in `/tmp`; durable findings are inline here.

Baseline: [September 30 report](upstream-check-20260930.md) and our [September 29 Spark build](../spark/karmic-beta-sm121/20260929/README.md), image `500ae05b`. Local `master` was synced and pushed earlier in this session, separately from these fetches.

## Latest published beta

[Release](https://github.com/local-inference-lab/blackwell-llm-docker/releases/tag/karmic-kraken-beta-7e6bf494ee15c5c31225868f612091d48c85c22290d3a5de9f99064affe37402), published **2026-10-01T12:24:06Z**. Selected by publication time, not release API list order.

- Image: `ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20261001-7e6bf494ee15c5c3`
- Digest from publication receipt: `ghcr.io/local-inference-lab/vllm@sha256:0283bc84008d37853f7ccc2e16ec2ad299008fa6ce052a5a1c65c7fdd5255ee3`
- Recipe: `20e617110c509b8c97879d2e7c09f20057d83fc8`
- Digest was read from the release receipt, not independently checked by pulling the registry image.

| Component | Published source commit | Local source access |
| --- | --- | --- |
| b12x | `914921dad15d71ffc68ea329c344b71bf1ae7fa7` | `~/git/b12x`, `lil` (sparkinfer URL); published commit present |
| flashinfer | `dbd6238c6655b98195fdf77f04bba6facf5a38a4` | `~/git/flashinfer` fetched; no configured Local Inference Lab remote, published commit absent; verified via GitHub API |
| instanttensor | `95d4729b6d6a991bb8de61877147a9d9d9100b23` | No corresponding checkout found under `~/git`; verified via GitHub API |
| lmcache | `820af25ff630f4c00f7faefbcc31bc5ccd7bab71` | `~/git/LMCache` fetched; no configured Local Inference Lab remote, published commit absent; verified via GitHub API |
| nccl | `93fe05d9f9b6963ef841166a69cd0b30e4efe97b` | No corresponding source checkout found under `~/git` (`nccl-tests` is distinct); verified via GitHub API |
| vllm | `980d84efb8c36894374d8333c4c0b7f92d0fb825` | `~/git/vllm`, `lil`; published commit present |

Published FlashInfer, LMCache, InstantTensor and NCCL pins were individually verified through their Local Inference Lab commit APIs. No missing remotes or checkouts were added. The recipe and vLLM/B12X publication commits are available in the existing local checkouts.

## Existing checkout refresh

| Checkout under ~/git | Remotes fetched | Ref | Before fetch | After fetch |
| --- | --- | --- | --- | --- |
| rtx6kpro | origin, upstream | `upstream/master` | `6e1c9d787fabc026ffc9d2e37b1e77a041047e77` | `6e1c9d787fabc026ffc9d2e37b1e77a041047e77` |
| vllm | lil, origin, upstream | `lil/integration/karmic-kraken-beta` | `a2b44f520bd3b9ede9dc6f15d505ea67a390ad1f` | `980d84efb8c36894374d8333c4c0b7f92d0fb825` |
| b12x | lil, origin, upstream, voip | `lil/integration/karmic-kraken-beta` | `b6fada83d6f2c14f2c5776635d3263d469eb5a8a` | `914921dad15d71ffc68ea329c344b71bf1ae7fa7` |
| blackwell-llm-docker | fork, origin | `origin/main` | `20e617110c509b8c97879d2e7c09f20057d83fc8` | `20e617110c509b8c97879d2e7c09f20057d83fc8` |
| flashinfer | origin, upstream, voip | `upstream/main` | `9e9ec38922a25a30a331b0abdedf7af8fc559a5a` | `1cad38165e5fe9d3007a07f79b3d5a2d73d6fce4` |
| LMCache | origin | `origin/dev` | `c89bbe9daa5f2716c372d96d9dbf720920d7ae51` | `8d7d2c47e1edb8c107033ae709722b2c3a05dec4` |
| cutlass | origin | `origin/main` | `0b55a2f691d69981583568fd9eb69687b1f0de8a` | `0b55a2f691d69981583568fd9eb69687b1f0de8a` |
| sglang | origin | `origin/main` | `28c5e7f5cbac8bb1a4e645579dbf56d83f442cbc` | `41cbe65de05b209f3263942389267f79bd749d6a` |
| spark-vllm-docker | origin, upstream | `upstream/main` | `42f62e349c610641ac6677ac28275c57bebdf9c3` | `9296ced563e9996f1424149a62432ea752816b90` |
| dgx-spark-infra | origin | `origin/master` | `f52bf95a7b3dd9df01176fbcb3534f791ede03f1` | `f52bf95a7b3dd9df01176fbcb3534f791ede03f1` |
| llm-inference-bench | origin, upstream | `upstream/main` | `a50025a30f56fa7158c1042eafdfe6703fbea9e7` | `a50025a30f56fa7158c1042eafdfe6703fbea9e7` |

Some unrelated feature refs were force-updated by their remotes; the fetches did not force-update any local working branch.

## Spark applicability and remaining qualification

The manifest uses CUDA 13.4.1, Python 3.12 and NVIDIA Torch `2.14.0a0+4fdf77b940.nv26.8.63802676`. The FlashInfer JIT wheel is explicitly SM120/x86_64; InstantTensor, LMCache and NCCL wheels are x86_64. These published artifacts cannot serve as a native ARM64/SM121 Spark image.

The receipt qualification scope is: Native GPU smoke and LMCache checkpoint/filesystem contract tests; model-serving performance and full GLM cache E2E remain unqualified.

Compared with our September 29 build, the published vLLM includes scheduler/executor fixes, the Qwen GDN state-pool guard and B12X MXFP8 expert integration. The canonical DS4.1 ring-mapping merge is identical to beta's already included #943; it is not a new runtime fix relative to our build. Published B12X now includes CUTLASS DSL 4.7.1 and native MXFP8 W8A8 expert execution. FlashInfer also moved from the September 30 report pin; LMCache remains at that report’s published pin.

### Comparison against our exact built sources

- vLLM `99cbe782` → `980d84ef`: #946 lets decodes preempt when a prefill-sharing turn schedules no work under full KV capacity; #950 preserves replies to later collective RPCs (hybrid checkpoint timeout fix); #959/#960 prevent queued but inadmissible requests stealing prefill lanes when parallel prefills are enabled. Qwen GDN tuning retains its bound state pool; MiMo retains FP8 QKV weight/scale pairs across loader calls.
- B12X `1b6cd278` → `914921da`: CUTLASS DSL 4.7.1, block-quantized launch heuristics and tensor API compatibility; native MXFP8 W8A8 experts with per-rank intermediate sizes divisible by 32. vLLM #957 integrates this for ModelOpt MXFP8 exports. Qwen `qad-step5500-ple1000` can opt into B12X draft experts; the launcher still defaults to auto/Marlin. Upstream RTX PRO 6000 TP2 results show no decode speed gain over Marlin; these are not Spark results.
- FlashInfer `2206a14e` → `dbd6238c`: the remote compare has two commits, only compiler/build dependency files; it aligns CUTLASS DSL to 4.7.1, rather than adding new attention kernels in this published delta.
- Our actual LMCache remains `413ac987`, not the September 29 publication's `75f2b59d` (see the Spark runtime lock). Published `820af25f` is 74 commits ahead: checkpoint retention and reuse/eviction storage policies, bounded admission waits, restore diagnostics, safe cancellation of stalled storage lookups and preserving intact disk checkpoint listings when RAM allocation fails. LMCache is disabled in the built deployment contract.
- Recipe `f28aea04` → `20e61711`: CUTLASS DSL 4.7.1, QuACK 0.6.5, lil-bench 0.7.5 and MXFP8 draft-backend selection based on the drafter's own checkpoint/export layout. InstantTensor and NCCL published pins are unchanged.

The B12X compiler-migration document reports 113 passing selected correctness cases on both compilers, with production specialization coverage and timing gaps. It is SM120 evidence, not SM121 qualification or a measured serving speedup.

The initial check was a source refresh only. At the user's subsequent request, the [October 1 Spark build kit](../spark/karmic-beta-sm121/20261001/README.md) was prepared with pinned ARM64 CUTLASS 4.7.1 wheels, a source-archive FlashInfer rebuild and compiler regression gates. All 24 local artifact tests, input preflight, shell checks and build dry-run passed. It remains unbuilt and GPU/model-unqualified. This is not the former Python-only refresh over our CUTLASS DSL 4.6.2 foundation. No live host state was checked.

Source identities and qualification scope: [container receipt](https://github.com/local-inference-lab/blackwell-llm-docker/releases/download/karmic-kraken-beta-7e6bf494ee15c5c31225868f612091d48c85c22290d3a5de9f99064affe37402/container-release.json), [runtime manifest](https://github.com/local-inference-lab/blackwell-llm-docker/releases/download/karmic-kraken-beta-7e6bf494ee15c5c31225868f612091d48c85c22290d3a5de9f99064affe37402/manifest.json).


The functional-review fixes isolate FlashInfer compilation in the original NGC
component environment with a dependency precheck, verify the imported venv
compiler, and require a new numerical FlashInfer/AOT runtime gate before tagging.
Build sources and inputs use transient mounts in the serving stage. DS4 ships
its model manifest and covers its runner in integrity checks; GLM stop/removal is
restored. Candidate-specific distribution and qualification tools are prepared.
Runner comparisons use a frozen fixture, and locked source inputs regenerate
through Git-blob verification and canonical tar/gzip headers without rewriting
locks. These fixes have local artifact evidence only; image assembly and all
Spark GPU/model gates remain pending.


The second review added explicit permission to update externally managed Python
only in the disposable NGC component stage, direct AOT/compilation enforcement
and vLLM top-p/top-k sampler coverage. The locked source digest now covers the
uncompressed tar. GLM launch keeps NCCL_DEBUG=WARN; Qwen/GLM candidate grids
require a matched production baseline for image 500ae05b and the current QAD
revision. A separate baseline-capture tool is prepared but not run. Qwen rollback
guidance points to the September 29 beta containers. Image transfer/load allows
continued serving; the build/launch idle checks are unchanged. Inspection of the
verified dbd6238c archive confirms FLASHINFER_DISABLE_JIT is read in
flashinfer/jit/core.py, contrary to that specific review assertion.
