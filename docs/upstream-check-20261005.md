# Upstream refresh, October 5, 2026

## Result

All 23 configured remotes in the 11 existing checkouts fetched successfully with
`git fetch <remote> --no-prune` (the count includes the `lil` remote added to
`~/git/LMCache` on October 4). Local `rtx6kpro/master` fast-forwarded one commit
(`58291d6`, daily summary publication) from `72d3f5f07f2f2c985a842036f52ba7413ee3d32d`
to `58291d646631a8e58da7dfb4855d030383eaa28c`, matching `upstream/master`. The active
branch remains `spark` and the working tree was not touched. No checkout, worktree,
remote addition, push, build, or serving action was performed.

**Karmic main** has no new release: the latest is still
`ghcr.io/local-inference-lab/vllm:karmic-kraken-20260929-b174ad0302c4236c` (published
2026-09-30 00:02 UTC; vLLM `ab86b707`, B12X `a489f972`), and `lil/dev/karmic-kraken`
is unchanged at `ab86b707`.

**Karmic beta** published two images after the October 4 report's `bdd7e579`. Neither
touches the B12X autotune rule behind the Qwen C1 regression; B12X is unchanged and
[b12x#463](https://github.com/local-inference-lab/b12x/issues/463) is open with no
comments.

## Publication and source identities

Latest observed publication: [`b3523c2c`](https://github.com/local-inference-lab/blackwell-llm-docker/releases/tag/karmic-kraken-beta-b3523c2cdfe47054df319), published 2026-10-05 07:58:03 UTC.

Image: `ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20261005-b3523c2cdfe47054`.

| Component | October 4 beta `bdd7e579` | October 5 beta `e1ccbaf0` | October 5 beta `b3523c2c` |
| --- | --- | --- | --- |
| b12x | `52640cb15d4ad1c7230f747e72c45d45dc2da681` | unchanged | unchanged |
| vllm | `00a33e23142097f99fef95b1ecab6da4ea271e2e` | unchanged | `f1c2508f1018f47ee8a78478a819d202f359887b` |
| recipe | `c411dbf93f296de0aa6ad04c932df9103dfbee46` | unchanged | unchanged |

FlashInfer `dbd6238c`, InstantTensor `95d4729b`, LMCache `820af25f` and NCCL `93fe05d9`
are unchanged. The fetched vLLM beta tip matches `f1c2508f`.

## What landed since the October 4 check

- `e1ccbaf0` republishes the same sources with the final glm53-tp2 recipe text
  ([docker #129](https://github.com/local-inference-lab/blackwell-llm-docker/pull/129)): NVFP4 prefill activations, LMCache RAM tier on by default, a 4096-token budget and a 7,296 MiB KV cache; `glm53-spark-tp2` is removed.
- `b3523c2c` adds vLLM [#982](https://github.com/local-inference-lab/vllm/pull/982) (DeepSeek-V4.1-Flash MXFP4-CSF at TP3) and [#983](https://github.com/local-inference-lab/vllm/pull/983) (startup logs name the GLM MoE layers that prefill with A4; the MTP draft layer no longer warns as if A4 were off). 6 files, +105/-13.
- After `b3523c2c`, recipe `main` moved to `e881183c` with [docker #130](https://github.com/local-inference-lab/blackwell-llm-docker/pull/130) (merged 13:15 UTC, not yet in a published image): the TP4 `glm53-flash` profile's CSF variant now serves `GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD` @ `a1559e26` (MXFP8 attention and shared experts), prefill activations default to A4, `gpu-memory-utilization` 0.93 to 0.96, and DFlash2 with an FP4-CSF target loads again. `CHECKPOINT=original` is unchanged.

## Applicability to our fleet

- **Qwen:** no change on our path. B12X is unchanged and vLLM #983 only changes logging.
- **GLM:** #130 is a recipe preset for a new MXFP8 CSF checkpoint; it does not change our
  TP4 runner or the original QAD checkpoint we serve. It is a candidate for the CSF
  qualification sequence, not an image change.
- **DS4 Vision:** nothing new.

## Fetch audit

| Checkout | Principal ref moves |
| --- | --- |
| `rtx6kpro` | `upstream/master` `72d3f5f0` → `58291d64` |
| `vllm` | `lil/integration/karmic-kraken-beta` `00a33e23` → `f1c2508f`; `upstream/main` `0c16eee3` → `ff53f324` |
| `b12x` | none on Karmic branches |
| `blackwell-llm-docker` | `upstream/main` `c411dbf9` → `e881183c` |
| `flashinfer` | `upstream/main` `372bbbaa` → `188bdd76` |
| `LMCache` | `upstream/dev` `3efdd261` → `bffb77a7`; `lil/integration/local-inference-lab` unchanged at `820af25f` |
| `sglang` | `upstream/main` `f385390b` → `91f9bf8d` |
| `cutlass`, `spark-vllm-docker`, `dgx-spark-infra`, `llm-inference-bench` | none on principal branches |

65 remote refs moved or appeared in total, most of them feature branches.
