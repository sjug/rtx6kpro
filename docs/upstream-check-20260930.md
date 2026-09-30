# Upstream check, September 30, 2026

Refresh of the [September 29 check](upstream-check-20260929.md). Every configured remote in the
eleven existing checkouts was fetched with `git fetch <remote> --no-prune` (22 remotes, all
succeeded); all tips are fast-forwards. Separately, and at the user's request, the fork's
`master` was fast-forwarded to `upstream/master` (`816274a` to `0812ee4`, no fork-only commits;
pushed to `origin` only). No build or serving change was made for this check.

Production at the time of the check: all eight Spark nodes serve Karmic beta image `500ae05b`
(vLLM `99cbe782`, B12X `1b6cd278`): GLM TP4, DS4 Vision TP2, Qwen TP2.

## Model checkpoints: the largest finding

The Hugging Face `main` refs of two production models moved on 2026-09-16. Earlier checks did not
compare served revisions against `main`.

| Model | Served | `main` now | Change |
| --- | --- | --- | --- |
| `local-inference-lab/GLM-5.3-Flash-NVFP4` | `46aaae8a` (Sep 4) | `175ae8ce` (Sep 16), 199.4 GB | "Publish GLM-5.3-Flash quantization-aware distilled checkpoint" (`2f259cfb`); 40 of 47 weight shards differ; index, `generation_config.json` and `quatrain-materialization.json` changed |
| `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` (the served `-4p89` repo now redirects here) | `c374e7e2` (Aug 28) | `7c4f1bc1` (Sep 16), 105.9 GB | "Publish Qwen3.8-Flash-Next quantization-aware distilled checkpoint" (`b13380df`); all shards replaced (37 to 36), plus `config.json`, index and export manifest |
| `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` | `6821d6ad` | `6821d6ad` | unchanged |

- GLM `generation_config.json` (`4b10bddc`, Sep 9) now carries temperature 1.0 and top_p 0.95,
  the same values our launcher sets with `--override-generation-config`.
- Qwen `main` keeps the served quantization layout (MXFP8 467, NVFP4 48 and W4A16 NVFP4 29
  layers; MTP layers W4A16 NVFP4). Its config renames the model type to `qwen4_exp` and lists
  `Qwen4ExpForConditionalGeneration` first. Production vLLM `99cbe782` registers both
  (`registry.py:608-612`, `transformers_utils/config.py:871`).
- Branches also exist for intermediate QAD steps: GLM `qad_tvn_step_{1500,2500,3500}`, `qad-step2500`
  and `qad-step-1750`; Qwen `qad-step-4000` and `qad-step5500-ple1000`. The Qwen
  `qad-step5500-ple1000` branch exports its MTP experts as MXFP8, which B12X does not run. The
  recipe now works around that (below). `main` is not affected.

## Publication

Six releases since the September 29 report (five beta, one main). Newest:

| | Main | Beta |
| --- | --- | --- |
| Image | `karmic-kraken-20260929-b174ad0302c4236c` | `karmic-kraken-beta-20260930-e47b71ea03eff6b1` |
| Digest | | `sha256:f7efe94fb203eb73...` |
| Recipe | `f28aea04babb` | `05aea88bc2e1` |
| vLLM | `ab86b7073400` | `5df66adcee68` |
| B12X | `a489f972e0dd` | `1b6cd278626a` (unchanged, same as production) |
| LMCache | `820af25ff630` | `820af25ff630` |

FlashInfer (`2206a14e`), InstantTensor (`95d4729b`) and NCCL (`93fe05d9`) are unchanged. There is still
no ARM64/SM121 publication.

The beta publication's vLLM `5df66adc` is production `99cbe782` plus five fixes: vllm #946 (scheduler),
the Qwen GDN state pool, the MiMo loader, #926 (DS4.1 ring mapping) and #950 (executor). It does not
include the CUTLASS DSL 4.7.1 wheel-build change (`da9fc45f69`), and its B12X is not the 4.7.1 update.

## vLLM

Main `0296a817` to `ab86b707` (+3) and beta `99cbe782` to `a2b44f52` (+13, seven non-merge):

- [0b1bccb858](https://github.com/local-inference-lab/vllm/commit/0b1bccb858) (#946): with prefill
  compute sharing, a decode that needed a KV block from a full pool could leave every step empty.
  Generation stopped with idle GPUs until clients aborted. Reported on GLM-5.3-Flash TP2 with
  LMCache restore admission, four slots and KV at 98 percent. It applies to any profile under KV
  pressure. Our pools are large (GLM 6.4M tokens), so it is unlikely but possible.
- [9ce5adbd56](https://github.com/local-inference-lab/vllm/commit/9ce5adbd56) (#950): with the batch
  queue, a collective RPC (`wait_for_boundary_checkpoint_copies`) answered before an earlier
  single-rank RPC was dropped as stale, and the engine stopped with `TimeoutError`. It was reproduced
  on GLM TP4/DCP4/MTP3 with 8 parallel 30K prompts. That collective belongs to boundary checkpoints.
  Our GLM runs DCP1 with `aligned` checkpoints, which lowers but does not rule out exposure.
- [ab86b70734](https://github.com/local-inference-lab/vllm/commit/ab86b70734): B12X GDN prefill trial
  callbacks keep the state pool they were bound to. This is Qwen GDN and a correctness guard.
- [b0b49023cc](https://github.com/local-inference-lab/vllm/commit/b0b49023cc) (#926): DS4.1 ring
  mapping, covered in the September 29 report. [c32d700276](https://github.com/local-inference-lab/vllm/commit/c32d700276):
  MiMo loader, not served.
- `16c52365e1` and `da9fc45f69`: wheels require CUTLASS DSL 4.7.1 with B12X. Production has 4.6.2.

## B12X

Main `a489f972` to `0d6600e6` (+3) and beta `1b6cd278` to `b6fada83` (+6):

- [4bacd509](https://github.com/local-inference-lab/sparkinfer/commit/4bacd509): CUTLASS DSL 4.6.2 to
  4.7.1 for all five compiler packages. The same GPU corpus passes on both; compiler timing is not
  established.
- [0d6600e6](https://github.com/local-inference-lab/sparkinfer/commit/0d6600e6): block-quantized launch
  heuristics derived from reuse, occupancy and SM count. Heuristic over autotuned kernel latency:
  geometric mean 0.954 (dense) and 0.9999 (MoE), with residual cases 1.65 to 5.07 percent slower.
  Serving throughput was not measured. It applies where no autotuning metadata exists; our boots
  autotune.
- [ba090286](https://github.com/local-inference-lab/sparkinfer/commit/ba090286): restores the legacy
  tensor-based entry points over the typed preparation APIs.

Adopting this B12X means a native rebuild on CUTLASS DSL 4.7.1, not a Python-only refresh.

## Recipe and other repositories

- Recipe `f28aea04` to `20e61711` (+9): CUTLASS DSL 4.7.1 and QuACK 0.6.5; lil-bench 0.7.5
  (hybrid-model KV capacity). An MXFP8 MTP drafter now gets `moe_backend` auto (vLLM picks Marlin),
  checked against the drafter's own `hf_quant_config.json` in both export layouts. These are
  launcher-side fixes for the Qwen `qad-step5500-ple1000` revision.
- FlashInfer +41: SM12x items only touch FlashInfer's own B12X fused MoE (#5451, skip unrouted
  expert ids) and W4A16 MoE expert parallelism (#4302). We run B12X directly. The rest is SM100/SM103.
- LMCache dev +15 (hugepages, io_uring, MP fixes), SGLang +68 (Qwen QSA fusions, DSpark coordination,
  AMD CI): out of scope.
- llm-inference-bench +2: lil-bench 1.3.2 reports KV capacity from `kv_cache_size_tokens`.
  rtx6kpro upstream +2: a daily summary and a Kimi X4T DSpark recipe document. CUTLASS,
  spark-vllm-docker and dgx-spark-infra are unchanged.

## Fetched tips

| Checkout | Ref | Tip |
| --- | --- | --- |
| rtx6kpro | upstream/master | 0812ee4f1c0b |
| vllm | lil/dev/karmic-kraken | ab86b7073400 |
| vllm | lil/integration/karmic-kraken-beta | a2b44f520bd3 |
| b12x | lil/master | 0d6600e64955 |
| b12x | lil/integration/karmic-kraken-beta | b6fada83d6f2 |
| blackwell-llm-docker | origin/main | 20e617110c50 |
| flashinfer | upstream/main | 9e9ec38922a2 |
| LMCache | origin/dev | c89bbe9daa5f |
| cutlass | origin/main | 0b55a2f691d6 |
| sglang | origin/main | 28c5e7f5cbac |
| spark-vllm-docker | upstream/main | 42f62e349c61 |
| dgx-spark-infra | origin/master | f52bf95a7b3d |
| llm-inference-bench | upstream/main | a50025a30f56 |

## Recommendation

1. **Checkpoints first.** GLM and Qwen `main` are QAD-distilled replacements of the weights we serve
   (the standing rule is to serve the latest revision). They need downloads (199.4 GB GLM on four
   nodes, 105.9 GB Qwen on two), the model manifests and runner revisions updated, and a full
   qualification on the current image. Quality and performance can move with new weights, so each
   needs its own grid against today's baseline with matched request defaults.
2. **Beta vLLM refresh.** A Python-only refresh of production to `5df66adc` (#946, #950, GDN) is low
   cost with the existing tracked-source kit. It fits the same windows as the checkpoint updates.
3. **Hold B12X 4.7.1** until it is published in a beta release. It needs a native rebuild and brings
   no measured serving gain yet.
4. DS4 Vision: no change needed.
