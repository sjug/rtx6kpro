# Upstream check 2026-10-08: Karmic beta against the three serving clusters

Fetch pass and fork sync ran 2026-10-08 (all AGENTS.md checkouts, all remotes, no failures; sjug forks fast-forwarded: rtx6kpro e350130 to efcd534, vllm 9367d8b969 to ab905a885d, blackwell-llm-docker f52d610 to 05eaafa, LMCache dev e64b43ef to 8bf2cf23, flashinfer e77db996c to fa7c741e0, spark-vllm-docker 73f01ce to a4c1729; b12x equal; llm-inference-bench left diverged 2 ahead / 8 behind on user instruction).

## Lineage

Karmic Kraken is still the newest lineage. The channel map at blackwell-llm-docker `05eaafa` (`tools/jovian_wheel_runtime/community-channel.json`) now lists only `karmic-kraken` (vLLM `dev/karmic-kraken`) and `karmic-kraken-beta` (vLLM `integration/karmic-kraken-beta`, FlashInfer `integration/karmic-kraken-beta`). The Jovian channels are gone from the map, and no newer `dev/*` branch exists on local-inference-lab/vllm or b12x. B12X as a component is gone too: since docker #144 the images take B12X from FlashInfer 0.7.1 (`flashinfer/experimental/b12x`, same `b12x` import package); the standalone b12x repository is archived.

Beta is rebuilt on canonical (docker-144): vLLM beta = `dev/karmic-kraken` plus the GLM-5.3 744B work and the hybrid A4 prefill; FlashInfer beta = LIL main plus the hybrid A4 prefill. Consequence: the beta tip no longer descends from the vLLM commits our October images were built from (`f1c2508f` Oct 5 kit, `abdf960a` nous cdcdf9ee; merge-base `ab86b70734` of 2026-09-29). Compare by change fragments, not ancestry.

## Newest publication

| Item | Value |
| --- | --- |
| Release | `karmic-kraken-beta-20261008.2` (tag `karmic-kraken-beta-38f14a00…`), 2026-10-08T20:44Z |
| Image | `ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20261008.2`, digest `sha256:cf8e87217901a88a258beeb545a52e893323671735b926cd25bd3879e48e965e` |
| Recipe | `05eaafa9473068844c0342f93286ed9759443251` (docker #148) |
| vLLM | `19f2c20ed4d64ae22bd009ceacadd85e2a81e354` (#1016 merge) |
| FlashInfer (carries B12X) | `b4c35e6ec1d712f6b832679c08ce5719fae57cd6` |
| LMCache | `3171cdb537de7af7d66d8e6ec7bf26bdcf3ba3bd` (unchanged since cdcdf9ee) |
| InstantTensor / NCCL | `747390dc…` / `300f3a73…` |
| Scope | native GPU smoke and LMCache contract tests only; serving performance unqualified |
| Platform | linux/amd64 only |

Tags are now channel plus date (docker #147); the canonical `karmic-kraken` channel publishes again on FlashInfer 0.7.1 (docker #145).

## DGX Spark arm64 twins (new since 2026-10-06)

Upstream builds a linux/arm64 (GB10, SM121) twin of beta releases through `.github/workflows/spark-container-release.yml` and `tools/jovian_wheel_runtime/spark_release.py`: every component rebuilt from the release's source commits under QEMU, image pushed as `karmic-kraken-beta-spark-<date>-<id>` with `container-release-linux-arm64.json` on the same release, alias `karmic-kraken-beta-spark`. Hardware profile `runtime/hardware/dgx-spark.yaml` is `status: research-only` (PCIe all-reduce off, NCCL IB over RoCE, B12X autotune off, one compile worker, GLM MTP draft head bf16, Qwen PLE CPU offload off). Presets `glm53-dgx-spark-x2` (GLM-5.3-Flash-NVFP4-Spark, TP2 over two boxes, MTP3, 8 slots, 351,730 KV tokens, about 30 tok/s C1) and `qwen38-dgx-spark-x2` (original checkpoint, TP2 two boxes, 16 slots, 3.39M KV, about 50 tok/s C1). No DS4 Vision Spark preset.

Published arm64 twins: `karmic-kraken-beta-spark-20261006-ff997f2f…`, `…-20261007-cdcdf9ee…` (vLLM abdf960a, the nous image), `…-20261007-d071fb3d…` (vLLM 5009fa56, recipe 475fcd2, docker #141). Nothing on 2026-10-08: the Spark workflow has not run since 2026-10-07T23:30Z (that run resolved nothing and skipped the build) although the publisher ran eight times on 2026-10-08. The FlashInfer 0.7.1 components do carry arm64 locks (`ci/lil_wheels/linux-arm64/` in FlashInfer `b4c35e6e`, `tools/jovian_wheel_release/linux-arm64/` in vLLM `19f2c20e`), so a twin is buildable by manual dispatch on upstream's side.

These twins are two-box TP2 presets at 8 to 16 slots with research-only status, built under QEMU with B12X autotune off. They are not a drop-in for our four-box GLM TP4, nor for the four-slot, max-native-context contracts our clusters run. They are a reference, not a replacement for the spark kit line.

## Serving pins compared

| Cluster | Model / layout | Image | vLLM | B12X | LMCache |
| --- | --- | --- | --- | --- | --- |
| sparky/buddy/rocky/lucky | GLM-5.3-Flash NVFP4 QAD 175ae8ce, TP4/DCP1/MTP3, 4 slots | karmic-beta-20260929-spark-sm121 `500ae05b` | `e77be225` | `10a553ef` | `413ac987` |
| dusty/kirby | Qwen3.8-Flash-Next QAD 7c4f1bc1, TP2, HC off, MTP3 | same | same | same | same |
| rusty/toby | DS4-Flash Vision 6821d6ad, TP2, b12x-a8 | same | same | same | same |
| nous | GLM-5.3-Flash CSF fd660d51, TP4, 4 slots | karmic-kraken-beta cdcdf9ee (amd64) | `abdf960a` | `236ddff0` | `3171cdb5` |

Oct 5 kit `4102660a` (vLLM f1c2508f): gates passed, Qwen not promoted (decode step-rate regression, b12x#463 open); production stays on `500ae05b`.

Fragments added on the vLLM beta since `e77be225`: 145. Since `abdf960a`: 36. B12X beta since `10a553ef`: 287 commits, 56 fragments. LMCache since `413ac987`: 78 commits, 22 fragments.

## What matters per cluster

### GLM-5.3-Flash TP4 on four Sparks (500ae05b, vLLM e77be225)

Fixes with a direct bearing on our serving contract:

- vllm-950: TP hybrid serving with boundary checkpoints could stop with `wait_for_boundary_checkpoint_copies timed out` after a burst of parallel long prompts (engine took the wrong rank's reply under async scheduling). Affected beta images from 20260928-d7935130 on. Our 20260929 kit is in that range. We have not seen it, but it is the one correctness fix to carry.
- vllm-931: grammar-constrained requests with MTP could emit tokens outside the schema and 500 after a skipped step.
- vllm-932: GLM parser keeps generated whitespace so the next turn reuses about 99.7% of the prompt instead of 79% (with the docker-110 chat template).
- vllm-988: serving thread count set before warmup; cold first prefill on 4x Spark GLM TP4 +1.03 s to +0.22 s (measured upstream on #681).
- vllm-restore-admission-stall: prefill compute-share servers (GLM profile) could step forever with idle GPUs when every slot was taken and restores waited. We serve 4 slots with vram cache; lower exposure, still a correct fix.
- vllm-904 / 908 and lmcache-98: image conversations get boundary checkpoints (only with an external cache; our GLM runs CACHE_MODE=vram).
- Spark-specific: vllm-995 (NVFP4 MTP draft head accepted on GB10, 12.1), vllm-989 (DFlash split-KV draft attention on GB10), vllm-996 / b12x-spark-arm64-wheel / lmcache-spark-arm64-wheel (arm64 bundles). The dgx-spark hardware profile pins the draft head to bf16 anyway.

Performance items are mostly CSF or A4 prefill and do not apply to the BF16 QAD checkpoint we serve: vllm-nvfp4-csf-scale-prefetch, vllm-1016 (CSF load 214 s to 144 s on glm53-tp2), b12x-487 (W4A16 CSF decode close to plain NVFP4), hybrid A4 prefill (docker-146, opt-in, refused for the Spark checkpoint by docker-125). B12X master syncs since 09-23 bring the grouped MoE decode and A16 token cutoff changes, which were already in the Oct 1 and Oct 5 kits and did not beat production on Qwen; GLM was not re-run on those kits.

Verdict: the fixes are worth a kit (vllm-950 above all), the speedups are not for this checkpoint. Build only in a user window, one job per node.

### Qwen3.8-Flash-Next TP2 on dusty/kirby (500ae05b)

- vllm-883: CACHE_MODE=native start regression (introduced by #864 after our pin, fixed again); not in our image, nothing to do.
- vllm-865: QSA KV corruption when CUDA graphs exceed the decode-row limit. Verified 2026-10-02 as already in our 99cbe782 line; confirm presence in e77be225 before claiming exposure (see memory rule).
- vllm-950 and vllm-931 apply to Qwen as to GLM.
- vllm-886: TP4/DCP4 for official checkpoints (not our layout).
- vllm-961 + docker-117: REPLICAS mode and shared PLE table (amd64 two-GPU hosts; the dgx-spark preset turns PLE CPU offload off).
- b12x#463 decode step-rate regression that blocked the Oct 1 and Oct 5 kits: no fragment on the beta names it fixed. Expect the same result on a new kit until that issue closes.

Verdict: nothing that justifies a new Qwen kit before b12x#463 moves.

### DS4-Flash Vision TP2 on rusty/toby (500ae05b)

- docker-123 + vllm-mxfp4-csf-ds4-flash: the ds4-vision profile now defaults to CHECKPOINT=csf (`DeepSeek-V4-Flash-Vision-Exp-lossless-CSF@b04bad8e`, mxfp4_csf). Upstream on two RTX PRO 6000: KV 1,850,842 vs 1,306,708 tokens (+42%), decode within 3% per step, prefill -1.4% / -2.9%. Routed experts run MXFP8 activations like the native B12X MXFP4 path. This is the one meaningful gain for this cluster, and it is a checkpoint plus loader change, not a drop-in; it is unmeasured on GB10 and no Spark preset exists for it.
- vllm-904 / 908, lmcache-98: image boundary checkpoints (external cache only).
- vllm-upstream-56271: DeepSeek V4 tool calls keep parameters that omit `string=`.
- No DS4-Vision-specific fix among the 145 fragments otherwise.

Verdict: candidate for a later CSF trial on the staging pair, not a reason to move production.

### nous GLM TP4 (cdcdf9ee, vLLM abdf960a)

Already planned for the user's window: #1002 HF-layout CSF reader, #1004 / #1006 DCP gathers (glm53-tp2 trial), plus now #1016 (faster CSF load, 2 of 85 shards for the draft), vllm-load-split-expandable-segments (+0.5 GiB free per GPU on glm53-tp2), vllm-nvfp4-csf-scale-prefetch (prefill +2-3%), b12x-487 and the stage-scale decode path, hybrid A4 prefill (docker-146 switches a4 to `B12X_W4A16_A4_PREFILL`; refuses the old MIN_TOKENS name on new images), vllm-tuning-schema22 (first start retunes; new JIT cache dir as planned). The FlashInfer 0.7.1 move changes the b12x packaging, not the launch contract. The newest amd64 image for that window at this check is `karmic-kraken-beta-20261008.2`.

## Checkout gaps

- `~/git/flashinfer` has no remote for local-inference-lab/flashinfer (`voip` is voipmonitor/flashinfer and has no `integration/karmic-kraken-beta`). The beta FlashInfer tip `b4c35e6e` was read with `git ls-remote` and `gh api`; adding a remote needs the user's go.
- `~/git/b12x` upstream still fetches the archived repository; its `integration/karmic-kraken-beta` tip `236ddff0` is the last standalone state and equals the B12X in cdcdf9ee.

## Not done

No builds, no container changes, no node access. Qwen Oct 5 candidate containers remain retained on dusty/kirby. No arm64 twin was pulled.
