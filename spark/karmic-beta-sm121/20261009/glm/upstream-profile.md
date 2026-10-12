# GLM-5.3-Flash on DGX Spark: upstream profile catch-up (recipe 05eaafa)

Purpose: bring the Spark GLM TP4 deployment to the October 8 recipe's `glm53-flash` profile (publication karmic-kraken-beta-20261008.2, image d9ba012f on Spark) except where we keep a Spark-specific choice. Sources: `runtime/profiles/glm53-flash.yaml`, `runtime/hardware/dgx-spark.yaml`, `runtime/presets.yaml` (glm53-dgx-spark-x2) and `runtime/launcher.py` at blackwell-llm-docker 05eaafa; the resolved launch of the same profile on nous (`print-config.log`, engine log `quantization=modelopt_mixed load_format=instanttensor`, "Serving FP4-CSF checkpoint <snapshot>"); vLLM 19f2c20e and FlashInfer b4c35e6e (B12X) trees for what the image actually reads.

Runner: `run-glm-tp4-node-upstream.sh` (kit runner `run-glm-tp4-node.sh` plus the rows marked adopt). Container `glm53-flash-csf-qad-upstream-profile-karmic-beta-20261009-tp4`.

## Adopted from upstream

| Option | Upstream | Ours until now | How the runner applies it | Evidence it works on the Spark image |
| --- | --- | --- | --- | --- |
| Checkpoint | `local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD` (HF-layout FP4-CSF, profile pin dec48abd) | `GLM-5.3-Flash-NVFP4` 175ae8ce | MODEL = local snapshot of refs/main; expected 177,680,678,236 bytes, 36 shards; config must carry `weight_scale_encoding: csf` recipes and no manifest.json | vLLM 19f2c20e has no dedicated `nvfp4_csf` loader; it reads the recipes through `modelopt_mixed` + `instanttensor` (modelopt.py), which is what upstream's launcher resolves to (`prepare_hf_layout_csf`) and what nous ran on the same beta. The launcher's flags are unchanged. |
| expert-activations bf16 | W4A16 routed experts | fp4 (W4A4) | `VLLM_B12X_MOE_FP4_FORCE_A16=1` | envs.py + nvfp4 oracle at 19f2c20e; B12X w4a16 kernel has `_W4A16_REGS_SM121`. GB10 is compute-poor, so W4A16 may cost decode: measured in the window. |
| router-weights fp32 | FP32 top-k combine | bf16 | `B12X_W4A16_FP32_TOPK_WEIGHTS=1` | B12X w4a16/kernel.py reads it; applies only with W4A16. |
| prefill-activations a4 | NVFP4 prefill rows | not set | `B12X_W4A16_A4_PREFILL=1` (semantic switch; the MIN_TOKENS variable is gone from both trees) | vllm/utils/b12x.py and B12X prefill_a4.py read it; QAD experts are the qualified case (pre-QAD checkpoints are refused upstream). Upstream measured +12-14% prefill at TP4 on RTX; no accuracy loss claimed. Unmeasured on GB10. |
| recurrent-checkpoint-policy request_boundaries | profile default | aligned (R27 qualification) | runner passes `request_boundaries`; the launcher drops its own aligned default when the flag is given | `RecurrentCheckpointPolicy` includes it at 19f2c20e. The prefix-pair/triple gates cover reuse behaviour. |
| prefill scheduling | `prefill-schedule-interval 1`, `prefill-compute-share 0.4`, `max-parallel-prefills 1` | interval 8, no compute share | `PREFILL_SCHEDULE_INTERVAL=1`, runner passes `--prefill-compute-share 0.4` and `--max-parallel-prefills 1`; the launcher's `FAIRNESS_ENGINE=compute_share` branch is NOT used because it emits `--fairness-engine`, which vLLM 19f2c20e does not accept (attempt 1 failed on it) | `prefill_compute_share` and `max_parallel_prefills` are scheduler-config fields at 19f2c20e (`--help=all` in the image lists both); a compute share requires interval 1. Head-of-line and concurrency gates measure the effect. |
| MTP head | `VLLM_MTP_NVFP4_LM_HEAD=0`, `VLLM_LM_HEAD_A16=1` | tree defaults (runtime NVFP4 own head) | both set explicitly | `mtp.py`: with 0 the drafter owns no runtime-quantized NVFP4 head; `prepare_draft_lm_head` then builds the GLM draft-only copy selected by `VLLM_GLM53_MTP_DRAFT_HEAD` (kept nvfp4). `VLLM_LM_HEAD_A16` is consulted only inside runtime NVFP4/MXFP8 head quantization (`vocab_parallel_embedding.py`), both off here, so it records the profile and changes nothing. Acceptance and steps/s in the window decide; see "watch". |
| `B12X_MHC_PDL=1` | on | unset (B12X default 0) | set | FlashInfer `b12x/norm/mhc/_kernels.py` reads it; `norm/mhc/api.py` supports SM120/SM121 and vLLM `glm5next/nvidia/model.py` selects the B12X MHC path on the SM12x family (Codex review 2026-10-11). |
| kernel environment | `VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel`, `VLLM_USE_FLASHINFER_SAMPLER=1`, `VLLM_CAUSAL_CONV1D_UPDATE_HOIST=1`, `VLLM_GLM53_KDA_GATE_SIDE_STREAM=1`, `VLLM_MXFP8_LM_HEAD=0`, `VLLM_DISABLE_SHARED_EXPERTS_STREAM=0`, `VLLM_USE_V2_MODEL_RUNNER=1` | unset (defaults equal except DISABLED_KERNELS and V2, which defaults to auto) | set explicitly | all read by 19f2c20e; the MXFP8 attention and shared experts of the CSF checkpoint are the layers Marlin FP8 would otherwise serve. |
| `--enable-prompt-tokens-details` | on | off | added | API-only (cached-token usage details). |

## Kept on purpose (Spark-specific choices)

| Option | Upstream | Ours | Why we keep it |
| --- | --- | --- | --- |
| max-num-seqs | 32 (RTX profile), 8 (Spark x2 preset) | 4 | user rule 2026-09-29 (max 4 sequences). |
| max-model-len | 1,048,576 | 1,048,576 | same (user rule: native context). |
| CUDA graph capture | 1..256 | 1 2 4 8 12 16 (= seqs x (MTP+1)) | derived from the 4-slot rule; larger captures only cost memory. |
| gpu-memory-utilization | 0.96 (RTX), 0.83 (Spark x2 preset) | 0.85 | TP4 UMA headroom; tunable later per user (serving hosts need no host headroom), kept fixed for this comparison. |
| VLLM_GLM53_MTP_DRAFT_HEAD | nvfp4 (RTX profile), bf16 (dgx-spark hardware file) | nvfp4 | Sept 24 GB10 bisect: bf16 draft head cost 4.2-4.9% engine steps. |
| KDA prefill backend | `kda_prefill_backend b12x` | flashkda | SM121 FlashKDA build is part of the Spark kit foundation gate; the pinned tree's resolver offers auto/triton/flashkda. |
| L2 prefetch, work source | `VLLM_GLM53_L2_PREFETCH=1`, `B12X_DYNAMIC_WORK_SOURCE=persistent_grid` | off (SM121 default), materialized_queue | r26 launcher drift: forcing both on GB10 cost 3-4%. |
| reasoning_effort default | high | max | user choice (Max effort + top_p 0.95 defaults, 2026-09-30). |
| served-model-name | GLM-5.3-Flash-NVFP4 | GLM-5.3-Flash | our API contract. |
| autotune | `B12X_AUTOTUNE=0` (dgx-spark hardware file) | natural autotune | user rule 2026-10-05 (no manual autotune; upstream disables it only to protect a 4% earlyoom floor we do not run). |
| transport | two-box RoCE preset, NCCL env for a switchless ring | four-node switched 200G, RoCEnante + PyNCCL, NCCL_PROTO LL,Simple both rails, TC 106 | qualified contract; the ABI 5 proxy was qualified on it 2026-10-11. |
| InstantTensor BUFFERED, expandable segments, kv fp8, block 256, mamba align, target page 2048, recurrent page auto | same | same | no change. |

## Inapplicable on this image or hardware (not set)

- `B12X_DYNAMIC_SPLIT_ROUTE_COMPUTE`, `_DIRECT_EXPERT_SCALES`, `_SPLIT_LOW_SMEM`, `_SKIP_SPLIT_BARRIER_RESET`, `_SPLIT_FAST_PREPARE`, `_SPLIT_COMPUTE_MAC`, `VLLM_GLM53_ONLINE_DENSE_MXFP8`: no reader in vLLM 19f2c20e or B12X b4c35e6e (dead environment in the profile).
- `B12X_PCIE_ONESHOT_*`, `VLLM_ENABLE_PCIE_ALLREDUCE=1`, `VLLM_PCIE_*`: PCIe all-reduce paths; the Spark has none (hardware file sets PCIE allreduce off, as we do).
- `VLLM_GLM53_DFLASH_ATTN=1`: DFlash drafter attention fast path; we serve MTP.
- `VLLM_USE_FASTOKENS=1`: the `fastokens` package is not installed in the Spark image (import check 2026-10-11).
- `VLLM_GLM53_L2_PREFETCH_PERSIST_MB`: only with L2 prefetch.
- LMCache L1/L2 cache options: LMCache is packaged but not qualified on Spark; the launcher refuses it (unchanged).
- NCCL ring settings of the two-box preset (`NCCL_CUMEM_ENABLE`, `NCCL_NVLS_ENABLE`, `NCCL_IGNORE_CPU_AFFINITY`, `NCCL_IB_ADDR_FAMILY`, compile-worker limits): research-only preset for a different topology.

## Watch in the window

1. W4A16 experts on GB10: decode steps/s may drop against W4A4; prefill a4 may rise. Both measured against the same image's current-config grid (14:51 EDT 2026-10-11) and, if the delta is inside 3%, a same-window current-config boot.
2. `VLLM_MTP_NVFP4_LM_HEAD=0` with `VLLM_GLM53_MTP_DRAFT_HEAD=nvfp4`: confirm the boot log's draft-head line and compare greedy acceptance (`probe-acceptance-ab.py`, ARM_BODY greedy) with the 2.664 measured on the current config.
3. request_boundaries: prefix-pair and triple gates plus the short-pool gate must pass; watch KV (checkpoint retention costs memory).
4. compute_share 0.4: head-of-line and concurrency probes; a slower single-stream prefill is the expected trade.
5. KV tokens vs 7,191,950 (current config on this image): the CSF checkpoint should leave more.
6. Drafter MoE backend: CSF expert recipes dispatch to `Nvfp4CsfMoEMethod` before ordinary backend selection, so `moe_backend: marlin` in the speculative config may not describe the drafter's experts on this checkpoint; read the resolved backend from the boot log before claiming it (Codex review).

## Review

Codex (read-only, 2026-10-11) against the pinned trees: one mapping gap (`B12X_MHC_PDL`, now adopted), the executor now passes the checked refs/main revision to every rank and the qualify script asserts it in every container, the runner resolves the checkpoint below the stop branch, the qualify script rejects a non-production policy or revision early unless the production comparison is skipped, and the idle check distinguishes an ssh failure from an idle node. Confirmed correct: CSF loader mapping (modelopt_mixed + instanttensor on an HF-layout snapshot), implicit MTP snapshot inheritance, template and launcher digests, the W4A16/FP32 top-k/A4 variable names, request-boundary forwarding, scheduling flags and V2 selection; FlashKDA is the only SM12x KDA prefill choice the tree accepts, L2 prefetch and materialized_queue are the SM121 defaults. Codex also notes that none of the kept choices is proven optimal for the new checkpoint and precision: capture sizes, 0.85 utilization and the nvfp4 draft head each want a same-checkpoint measurement.

## Result (2026-10-11 window)

All gates passed. Against the same image with the current config: engine steps/s +27.5/+21.2/+15.4% and output tok/s +31.6/+20.1/+16.9% at C1/C2/C4, acceptance unchanged, grid prefill +4 to +9%, fresh prefill +4 to +7%, KV -1.1% (weights -4.5 GiB per rank, request_boundaries retention in the pool). Watch items: W4A16 on GB10 did not cost decode, it gained; the draft head resolved as the draft-only NVFP4 copy; prefix gates passed under request_boundaries; head-of-line and concurrency probes passed with the 0.4 compute share; the drafter MoE backend line was not separately read from the log. Full record in `runs/glm-5.3-flash/nvfp4/2026-10-karmic-beta-20261009-sm121-qualification/upstream-profile-vs-current-r01.md`.
