# Upstream refresh, October 2, 2026

## Result

All 22 configured remotes in the 11 existing checkouts fetched successfully with
`git fetch <remote> --no-prune`. Local `rtx6kpro/master` fast-forwarded two commits
from `6e1c9d787fabc026ffc9d2e37b1e77a041047e77` to
`a968efc0d561424596417246c14e69895b8a0b04`, exactly matching `upstream/master`.
The active branch remains `spark`; all pre-existing working-tree status entries
were preserved in every checkout. No checkout, worktree, remote addition, push,
build, or serving action was performed. `origin/master` remains at `6e1c9d7`.

A new Karmic beta is published, but **no confirmed fix for our SM121 Qwen C1
regression was found**. The relevant dense tuning and measurement code is
unchanged; other preparation and MoE fixes are not backed by a matched Spark
regression comparison. This is not a claim that those changes cannot affect
performance.

## Publication and source identities

Checked at 2026-10-02T14:26:04.014029+00:00. Latest observed publication: [October 2 beta](https://github.com/local-inference-lab/blackwell-llm-docker/releases/tag/karmic-kraken-beta-7b398c9e420c91eb4c04b383cb74764ff9049076ba063a2cfd82728c459c7c2c), published 2026-10-02 10:03:20 UTC.

Image: `ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20261002-7b398c9e420c91eb`.

Digest: `ghcr.io/local-inference-lab/vllm@sha256:2230db60afb4fd06dc2ef2b7f7f27dc7f78d7a50be70a08461b4df9fa5e90732`.

| Component | Investigated candidate | New publication |
| --- | --- | --- |
| b12x | `b557d87850cc836268fd327ccd46eaa6a033cdbf` | `a77b3f85e5e2a81315ff4a90088912f187708005` |
| flashinfer | `dbd6238c6655b98195fdf77f04bba6facf5a38a4` | `dbd6238c6655b98195fdf77f04bba6facf5a38a4` |
| instanttensor | `95d4729b6d6a991bb8de61877147a9d9d9100b23` | `95d4729b6d6a991bb8de61877147a9d9d9100b23` |
| lmcache | `820af25ff630f4c00f7faefbcc31bc5ccd7bab71` | `820af25ff630f4c00f7faefbcc31bc5ccd7bab71` |
| nccl | `93fe05d9f9b6963ef841166a69cd0b30e4efe97b` | `93fe05d9f9b6963ef841166a69cd0b30e4efe97b` |
| vllm | `4a379ed42881ee022aaf5d9ada554f9096e5acf3` | `93dabce32fd4d5355662608296e64d720711cdcc` |

Recipe remains `6c0e9843bb962f483409b0296b225cca03fe8567`. CUTLASS DSL and its four library packages remain pinned to 4.7.1. The fetched vLLM and B12X beta branch tips match these published pins.

## Applicability to the regression

Compared B12X `b557d878..a77b3f85` and vLLM `4a379ed4..93dabce32f` directly:

- B12X `gemm/blockscaled/`, `_lib/dense_gemm.py`,
  `preparation/_measurement.py`, `preparation/_cache.py`, `sequence/`, and
  `norm/` are byte-identical across the compared trees. vLLM's
  `model_executor/kernels/linear/b12x_blockscaled.py` is unchanged too.
  There is no dense selection-policy or measurement fix in these paths.
- [B12X #384](https://github.com/local-inference-lab/b12x/pull/384) retains
  prepared launcher ownership and coordinates collective priming. This changes
  preparation/collective handling, but its inclusion does not prove improved
  steady C1 steps/s or stable autotuning winners on our pair.
- [B12X #450](https://github.com/local-inference-lab/b12x/pull/450) repairs
  dynamic MoE route broadcasts and deterministic shared-input addressing, and
  adds CSF scale support. These are real correctness changes, including native
  paths; they are not a measured fix for the observed C1 slowdown.
- [B12X #456](https://github.com/local-inference-lab/b12x/pull/456) integrates
  CSF and changes MoE query/config schemas. The new candidate must not assume
  that the prior 93 selection keys and metadata can be transplanted unchanged.
- `6f5a1474` restricts Trellis codebook shared memory to Trellis weights. Our
  `b557d878` already contains that weight-layout guard in its older implementation;
  the fix repairs an intervening change, not a demonstrated defect in our image.
- [vLLM #846](https://github.com/local-inference-lab/vllm/pull/846) recovers
  interrupted **FlashInfer** autotune JSON and reuses verified native wheel
  payloads. It does not repair B12X selection variability; our launcher disables
  FlashInfer autotuning.
- New BF16/MXFP8 input accumulators are used by the DFlash model code, not our
  native MTP3 path. Shared PLE remains a separate opt-in feature.

The publication also adds canonical/online compressed FP4 scales and Kimi/DS4.1
support. None of this warrants silently changing our checkpoint, loader, or
serving profile to claim the old regression is resolved.

## Qualification boundary

The container manifest's scope is native GPU smoke and LMCache contract tests;
its `qualified` status does not mean our model profile is performance-qualified.
The attached `csf-serving-2230db60-20261002.json` is marked `research-only`.
Its Qwen native/online/stored comparisons use RTX PRO 6000 Blackwell GPUs,
checkpoint `b797d2e1160b9596b2570e56c1d3590faa09d4ed` for native/online and
`4656f502fa1a6aba3f05ff9ed14f6ecc828b3e9e` for stored CSF, TP1/TP2,
context 0 and C1/C8/C16. They compare CSF modes, not our old image against this
new image on GB10 with checkpoint `7c4f1bc1`, five contexts and C1/C2/C4.

The correct conclusion is a newer candidate with potentially relevant fixes,
not an already demonstrated solution. Our current build locks were not re-pinned.

## Fetch audit

Remote names below are as fetched. Later on October 2 the remotes were
standardized (`origin` is the user's fork, `upstream` the fork parent): b12x
dropped its duplicate `lil` remote, blackwell-llm-docker renamed `origin` to
`upstream` and `fork` to `origin`, and LMCache, cutlass and sglang renamed
`origin` to `upstream`. The current layout is 21 remotes.

| Checkout | Remotes fetched | Changed remote refs |
| --- | --- | --- |
| `rtx6kpro` | `origin`, `upstream` | 2 |
| `vllm` | `lil`, `origin`, `upstream` | 18 |
| `b12x` | `lil`, `origin`, `upstream`, `voip` | 12 |
| `blackwell-llm-docker` | `fork`, `origin` | 0 |
| `flashinfer` | `origin`, `upstream`, `voip` | 6 |
| `LMCache` | `origin` | 6 |
| `cutlass` | `origin` | 0 |
| `sglang` | `origin` | 22 |
| `spark-vllm-docker` | `origin`, `upstream` | 0 |
| `dgx-spark-infra` | `origin` | 0 |
| `llm-inference-bench` | `origin`, `upstream` | 2 |

Changed principal refs (before → after):

- `rtx6kpro` `refs/remotes/upstream/master`: `6e1c9d787fabc026ffc9d2e37b1e77a041047e77` → `a968efc0d561424596417246c14e69895b8a0b04`.
- `vllm` `refs/remotes/lil/integration/karmic-kraken-beta`: `4a379ed42881ee022aaf5d9ada554f9096e5acf3` → `93dabce32fd4d5355662608296e64d720711cdcc`.
- `vllm` `refs/remotes/upstream/main`: `3a6963664537ed21172e2ec12e96e3a2dcd3718c` → `592c6f3fbcefeb3a4e0c0b461531329bf4ee0489`.
- `b12x` `refs/remotes/lil/integration/karmic-kraken-beta`: `b557d87850cc836268fd327ccd46eaa6a033cdbf` → `a77b3f85e5e2a81315ff4a90088912f187708005`.
- `b12x` `refs/remotes/upstream/integration/karmic-kraken-beta`: `b557d87850cc836268fd327ccd46eaa6a033cdbf` → `a77b3f85e5e2a81315ff4a90088912f187708005`.
- `flashinfer` `refs/remotes/upstream/main`: `1cad38165e5fe9d3007a07f79b3d5a2d73d6fce4` → `1b578de52924c973bd229fe4fd147499ead2781d`.
- `sglang` `refs/remotes/origin/main`: `f03a183719c9e7fb2bd528ed5c7e3b627d9df92d` → `bb9a820f09f97480bfc6d07564fb2b691b8857a3`.
- `llm-inference-bench` `refs/remotes/upstream/main`: `a50025a30f56fa7158c1042eafdfe6703fbea9e7` → `7c142e830b2133b5c9680ea5f1b8a645e9bff6f4`.

Publisher refs without a matching configured remote were read using `git ls-remote` (no remotes added):

- `local-inference-lab/flashinfer` `community/jovian-judgement-cu134-sm120`: `dbd6238c6655b98195fdf77f04bba6facf5a38a4`.
- `local-inference-lab/LMCache` `integration/local-inference-lab`: `820af25ff630f4c00f7faefbcc31bc5ccd7bab71`.
- `local-inference-lab/InstantTensor` `main`: `95d4729b6d6a991bb8de61877147a9d9d9100b23`.
- `local-inference-lab/nccl-canonical` `canonical/cu134-nccl2312-amd-turin`: `93fe05d9f9b6963ef841166a69cd0b30e4efe97b`.
- `local-inference-lab/blackwell-llm-docker` `main`: `6c0e9843bb962f483409b0296b225cca03fe8567`.

## Evening refresh: October 2, 23:02 UTC / 19:02 EDT

Fetched all 21 configured remotes across the same 11 existing checkouts with
`--no-prune`. Every fetch succeeded. This refresh updates remote-tracking refs
only; branches, build locks and serving state were not changed. The sections
above retain the morning snapshot.

### New beta publication

[Beta assembly 764596cbb1634f24](https://github.com/local-inference-lab/blackwell-llm-docker/releases/tag/karmic-kraken-beta-764596cbb1634f242f903a8a60804a4ac1c661dce1dae678be241918d67b0343)
was published at **21:11:13 UTC / 17:11:13 EDT**:

- Image: `ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20261002-764596cbb1634f24`.
- Digest: `sha256:ef547bdcf146e2c9b183e2fa0b82a3a01ada680765720d01a25c258b45a801a7`.
- Recipe remains `6c0e9843bb962f483409b0296b225cca03fe8567`.
- vLLM advances from `93dabce32fd4d5355662608296e64d720711cdcc` to
  `58d05bc7626dd87ee401155e9b52e6c6fd92bd5d`.
- B12X remains `a77b3f85e5e2a81315ff4a90088912f187708005`; FlashInfer remains
  `dbd6238c6655b98195fdf77f04bba6facf5a38a4`. InstantTensor, LMCache and NCCL
  component pins also remain unchanged from the morning publication.

The addition is [vLLM #965](https://github.com/local-inference-lab/vllm/pull/965):
opt-in exact GLM projection selection and online quantization while loading a
QAD source checkpoint. The example converts 531 attention/shared projections
to MXFP8 and BF16 MTP routed experts to NVFP4 W4A16. It adds strict target
coverage, independent draft settings and bounded loader/encoder allocations.
Existing online quantization defaults remain unchanged; MXFP8 scale compression
is not enabled.

The accompanying validation used **two RTX PRO 6000 Blackwell 96 GB GPUs**,
not GB10 Sparks. Its measured main projection allocation saving was 3.524 GB
per rank; this is a loader allocation measurement, not an end-to-end server
memory saving. Short throughput and corpus quality results are explicitly
research-only. This does not qualify our existing GLM checkpoint/profile.
The container publication itself covers native GPU smoke and LMCache contract
tests (158 passed, zero skipped), not full model-serving performance.

### Other newly fetched changes

| Repository | Since the previous refresh | Relevance to our current image |
| --- | --- | --- |
| B12X | New branch `upstream/fix/moe-deterministic-preparation` at `bb92eda85fcfabd0944dad51469984ff5d4d2d88`, [PR #464](https://github.com/local-inference-lab/b12x/pull/464), open and unmerged at inspection | Resolves `B12X_DYNAMIC_DETERMINISTIC_OUTPUT` before kernel selection, excludes unsupported NVFP4 split-materialized choices when determinism is required, and freezes the reduction policy. Explicit routing overrides retain precedence. Not in the published beta; not evidence of a dense GEMM throughput fix. |
| Canonical vLLM | [#59753](https://github.com/vllm-project/vllm/pull/59753), commit `f590eb2448a51028f6044c6f60104ce0ef3189e9`: SM121 Qwen4Exp BF16 skinny-GEMM plans | Explicitly DGX Spark **TP1** plans. Commit is not an ancestor of LIL beta. It does not directly address our TP2 B12X MXFP8 dense-GEMM selections. |
| Canonical vLLM | [#59464](https://github.com/vllm-project/vllm/pull/59464): reuse sparse MLA index conversion across GLM layers | Different attention backend; also not an ancestor of LIL beta. The advertised 3.5–3.9x is a kernel improvement, not model throughput. |
| FlashInfer | [#5949](https://github.com/flashinfer-ai/flashinfer/pull/5949), `7026bd818f0d15b31a810974237b000688390f29`: GB10-aware split/head-tile planning; [#5959](https://github.com/flashinfer-ai/flashinfer/pull/5959): Cake NVFP4 sparse-MLA prefill | DeepSeek-V4 `backend="cake"` changes on upstream main. Beta still pins `dbd6238c`; these are not its kernels or our Qwen B12X path. |
| LMCache | [#5434](https://github.com/LMCache/LMCache/pull/5434): best-effort whole-chunk L2 loading; [#5416](https://github.com/LMCache/LMCache/pull/5416): ordered L1 write overflow and owner tags | Canonical development changes; published component pin unchanged. |
| SGLang | KDA opt-in decode parity, Qwen EAGLE3/DFLASH auxiliary capture and HiCache fallback work | Separate serving engine. |
| spark-vllm-docker | HF model management/download/backup work and API-key protection across endpoints | Separate recipe. |

`rtx6kpro`, `blackwell-llm-docker`, `cutlass`, `dgx-spark-infra` and
`llm-inference-bench` had no remote-ref changes in this fetch. The benchmark
harness upstream remains `7c142e830b2133b5c9680ea5f1b8a645e9bff6f4`.

Principal updated refs:

- vLLM `upstream/main`: `592c6f3fbcefeb3a4e0c0b461531329bf4ee0489` → `6e517b15c1833cf72a7f557ee32524d98682e617`.
- FlashInfer `upstream/main`: `1b578de52924c973bd229fe4fd147499ead2781d` → `9178cf06a97fedb4544ef5bb0bd57a4562458450`.
- LMCache `upstream/dev`: now `4abc421a55ccd965f48c00592a2513cbc95c01fb`.
- SGLang `upstream/main`: `bb9a820f09f97480bfc6d07564fb2b691b8857a3` → `cbe070b6bfce18936620f406b9b0d71bdc17452b`.
- spark-vllm-docker `upstream/main`: `9296ced563e9996f1424149a62432ea752816b90` → `a8f4d693121f6276bb01c18b93870c9ed1d2d6e0`.

There is still **no verified fix for our TP2 C1 regression** in these changes.
The newer beta is principally a GLM quantization feature update since the
morning snapshot. No re-pin, build, GPU tests or deployment were performed.
