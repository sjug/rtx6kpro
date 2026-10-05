# Upstream refresh, October 4, 2026

## Result

All 22 configured remotes in the 11 existing checkouts fetched successfully with
`git fetch <remote> --no-prune`. The count rose from 21 because `~/git/LMCache` now
has the user fork as `origin` (`sjug/LMCache`, forked from `LMCache/LMCache`, which
is `upstream`). Local `rtx6kpro/master` fast-forwarded one commit (`72d3f5f`, daily
summary publication) from `7288ac04678466168c57360392702dd2b0439eee` to
`72d3f5f07f2f2c985a842036f52ba7413ee3d32d`, matching `upstream/master`. The active
branch remains `spark` and the working tree was not touched. No checkout, worktree,
remote addition, push, build, or serving action was performed.

Eighteen Karmic betas were published after the October 3 report's `d4499365`, all on
October 4 (UTC). **None changes the B12X autotune selection rule behind our Qwen C1
regression**, and [b12x#463](https://github.com/local-inference-lab/b12x/issues/463)
is open with no comments. Most of the work is FP4-CSF checkpoint performance, and
the recipe presets now default to CSF checkpoints.

## Publication and source identities

Latest observed publication: [`bdd7e579`](https://github.com/local-inference-lab/blackwell-llm-docker/releases/tag/karmic-kraken-beta-bdd7e57984392a48d581b725b389681830b86d0e13a530597af889506dd35e50), published 2026-10-04 20:25:46 UTC.

Image: `ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20261004-bdd7e57984392a48`.

| Component | October 3 beta `d4499365` | October 4 beta `bdd7e579` |
| --- | --- | --- |
| b12x | `78ee52c302abcef019d7ce17634c80b76f0975c9` | `52640cb15d4ad1c7230f747e72c45d45dc2da681` |
| vllm | `1286ae9c9a3fd9376b82da39f7bc87f6a99037d9` | `00a33e23142097f99fef95b1ecab6da4ea271e2e` |
| recipe | `353efc679f631206e0b001e67047dea80ee6d76e` | `c411dbf93f296de0aa6ad04c932df9103dfbee46` |
| flashinfer | `dbd6238c6655b98195fdf77f04bba6facf5a38a4` | unchanged |
| instanttensor | `95d4729b6d6a991bb8de61877147a9d9d9100b23` | unchanged |
| lmcache | `820af25ff630f4c00f7faefbcc31bc5ccd7bab71` | unchanged |
| nccl | `93fe05d9f9b6963ef841166a69cd0b30e4efe97b` | unchanged |

The fetched vLLM and B12X beta branch tips match these pins, and both
`canonical-candidate` branches (`b12x` `karmic-20260930`, `vllm` `karmic-20260924`)
now point at the same tips. The image's LMCache source is
`local-inference-lab/LMCache` `integration/local-inference-lab`, which was not a
configured remote at fetch time; `git ls-remote` showed it still at `820af25f`. At
the user's request it was then added to `~/git/LMCache` as `lil` and fetched;
`lil/integration/local-inference-lab` is `820af25f`, matching the image pin.

## What landed since the October 3 check

B12X `78ee52c3..52640cb1`:

- [#469](https://github.com/local-inference-lab/b12x/pull/469), [#471](https://github.com/local-inference-lab/b12x/pull/471): W4A16 skips empty M blocks and stages NVFP4-CSF scales per pipeline stage.
- [#472](https://github.com/local-inference-lab/b12x/pull/472), [#473](https://github.com/local-inference-lab/b12x/pull/473), [#476](https://github.com/local-inference-lab/b12x/pull/476): opt-in A4 (NVFP4 activation) prefill for W4A16 MoE layers, restricted to A16-only layers and selected per row type.
- [#477](https://github.com/local-inference-lab/b12x/pull/477): NVFP4-CSF scales can be expanded ahead of the MoE call.
- [#479](https://github.com/local-inference-lab/b12x/pull/479): compact W4A8 kernels read MXFP4-CSF scales inline (DS4.1 CSF decode 3-4% faster).

vLLM `1286ae9c..00a33e23`:

- [#970](https://github.com/local-inference-lab/vllm/pull/970), [#977](https://github.com/local-inference-lab/vllm/pull/977): W4A16 A4 prefill wiring (`B12X_W4A16_A4_PREFILL_MIN_TOKENS`, opt-in).
- [#971](https://github.com/local-inference-lab/vllm/pull/971): MXFP4-CSF for DS4 Flash.
- [#974](https://github.com/local-inference-lab/vllm/pull/974): PLE storage dtype resolved from an FP4-CSF checkpoint root (`qwen4_exp`, CSF checkpoints only).
- [#976](https://github.com/local-inference-lab/vllm/pull/976): NVFP4 MoE input scales are zero-initialized, so W4A16 layers without calibration never enable A4 prefill from uninitialized memory. The failure needed the opt-in A4 prefill setting.
- [#978](https://github.com/local-inference-lab/vllm/pull/978): GLM CSF prefill expands the next layer's scales during attention (2-3% with A4, 1-2% with A16).
- [#980](https://github.com/local-inference-lab/vllm/pull/980): with `expandable_segments` enabled, weight loading no longer applies the `max_split_size_mb:20` limit that stranded pages (0.5 GiB more free per GPU on glm53-tp2).

Recipe `353efc67..c411dbf9`:

- [#122](https://github.com/local-inference-lab/blackwell-llm-docker/pull/122), [#123](https://github.com/local-inference-lab/blackwell-llm-docker/pull/123): presets default to FP4-CSF checkpoints (QAD GLM and Qwen, DS4.1, DS4, DS4 Vision), with the original checkpoints selectable.
- [#124](https://github.com/local-inference-lab/blackwell-llm-docker/pull/124), [#129](https://github.com/local-inference-lab/blackwell-llm-docker/pull/129): GLM TP2 DMA allreduce and the final glm53-tp2 preset. [#125](https://github.com/local-inference-lab/blackwell-llm-docker/pull/125), [#126](https://github.com/local-inference-lab/blackwell-llm-docker/pull/126): A4 prefill limited to the QAD GLM checkpoint.
- [#127](https://github.com/local-inference-lab/blackwell-llm-docker/pull/127): documents that the Qwen original checkpoint's C16 deficit against CSF (878 vs 1,079 tok/s on RTX PRO 6000) is KV capacity, not kernel speed. Our profile caps `max_num_seqs` at 4.
- [#128](https://github.com/local-inference-lab/blackwell-llm-docker/pull/128): bundled llm-inference-bench 0.7.7 with `--tool-eval`.

## Applicability to our fleet

- **Qwen C1 regression.** Between `78ee52c3` and `52640cb1`, the only change under `preparation/`, `gemm/`, `_lib/dense_gemm.py`, `norm/` and `sequence/` moves `ld_shared_u16_zx` from `gemm/_shared/mxfp8_bmm.py` into the shared intrinsics module, with no functional change. Winner selection is unchanged. Nothing here is a fix; any new image still needs a matched Spark comparison.
- **Allocator.** Our launchers set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, so #980 applies to us. It changes post-load free memory and therefore the KV sizing a profile gets; that is a measured qualification item, not a free win.
- **Checkpoints.** We serve original (non-CSF) checkpoints. Upstream presets and their measurements now default to CSF, so upstream performance claims increasingly describe a checkpoint we do not run. A CSF move would be a separate checkpoint qualification per model.
- **Opt-in features.** A4 prefill and the GLM TP2 preset do not affect our TP4 GLM, TP2 Qwen or TP2 DS4 Vision profiles unless enabled.

## Fetch audit

| Checkout | Remotes fetched | Principal ref moves |
| --- | --- | --- |
| `rtx6kpro` | `origin`, `upstream` | `upstream/master` `7288ac04` → `72d3f5f0` |
| `vllm` | `lil`, `origin`, `upstream` | `lil/integration/karmic-kraken-beta` `1286ae9c` → `00a33e23`; `upstream/main` `84bcbc62` → `0c16eee3` |
| `b12x` | `origin`, `upstream`, `voip` | `upstream/integration/karmic-kraken-beta` `78ee52c3` → `52640cb1` |
| `blackwell-llm-docker` | `origin`, `upstream` | `upstream/main` `353efc67` → `c411dbf9` |
| `flashinfer` | `origin`, `upstream`, `voip` | `upstream/main` `c3c33367` → `372bbbaa` |
| `LMCache` | `origin` (new), `upstream` | `upstream/dev` `1ca53a8a` → `3efdd261` |
| `cutlass` | `upstream` | none |
| `sglang` | `upstream` | `upstream/main` `b016ca40` → `f385390b` |
| `spark-vllm-docker` | `origin`, `upstream` | `upstream/main` `cb51860d` → `bb6ee761` |
| `dgx-spark-infra` | `origin` | none |
| `llm-inference-bench` | `origin`, `upstream` | `upstream/main` `7c142e83` → `c71ec1f2` |

118 remote refs moved or appeared, most of them new feature branches.
