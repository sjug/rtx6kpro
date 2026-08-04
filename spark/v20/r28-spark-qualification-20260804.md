# r28-spark qualification record (2026-08-04)

Image: `gilded-gnosis-v20-r28-spark-sm121-vllm47d1950-si200c1db-fi7ad08da-cu132-20260804`
(local ID `5ac378214692`, 25.7 GB, built on dusty, staged on all 8 nodes).

Composition: upstream gilded-gnosis-v20 r28 (canonical bases GG `3003860`,
SparkInfer `272a84bd`, LMCache `9cebd405`; exact r28 PR manifests incl. vLLM
#235 reasoning contract, updated #217 tiered offload, #228/SI#117 shared-H
mixed Trellis) + Spark overlay (CMake arch 12.1, MTP-3D, PR#234-refined
frozen-q_len guard, fail-closed EXL3 loader) + FlashInfer `7ad08da`
(PR#3932) + DeepGEMM SM121 patch. Locks:
`blackwell-llm-docker/patches/releases/gilded-gnosis-v20-r28-spark/`.
Canonical r28 trees were reconstructed byte-exact (e1e9426/200c1db/9a05c88)
before overlay application; the composed vLLM tree round-trips to
`47d19501`.

EXL3 spike verdict: DEFER, fail closed (see upstream report item 7): all 68
units compile for sm_121a behind pure x86 guards; GEMM family self-consistent
(0.09% vs reconstruct+hgemm); mul1 codebook bit-exact; 3INST/MCG codebooks
show cross-unit encoder-vs-decode codeword divergence invariant to fast-math.
Image rejects glm52-exl3 at startup with the recorded reason.

## Build gates (all PASS, dusty)

Labels/provenance; launcher byte-provenance (DS4 helpers from composed tree,
GLM suite = launcher `6a61804` incl. r26 DCP policy, grep-verified);
GPU smoke incl. #235 low/high/max distinct prompts, #217 TieringOffloadingSpec,
SI#117 ABI 6; PR#3932 arithmetic discriminator; frozen-q_len guard matrix
(22 cells); DS4 helper dry-runs (dspark/mtp2/lucifer-cutlass/a16); EXL3
fail-closed gate.

## Runtime qualification (all PASS)

### Laguna-S-2.1-FP8 + DFlash, dusty/kirby TP2 (FULL-96/K5/0.85)

- Reproducer A (max_tokens=8, spec truncation): token-exact vs PIECEWISE refs.
- Reproducer B (8,200-token tail-fusion prompt): token-exact, prompt_tokens 8200.
- 61 "Replaying FULL CUDA graph" lines; engine alive; 0 errors.

### GLM-5.2 NF3 hybrid, sparky/buddy/rocky/lucky TP4/DCP2/MTP3 (262k, 0.86)

- PYNCCL/RoCE ag_rs init, correct rank placement, nvfp4_ds_mla KV.
- Decode 24.9 tok/s cc1 (512 tokens); coherent output.
- Uncached prefill: 8k 766 tok/s, 64k 779 tok/s, 128k (154,631 tokens)
  771 tok/s: flat, no long-context cliff.
- 0 errors across all four nodes. Upstream PCIe calibrator never runs
  (runner bypasses the helper suite; DCP mechanisms explicit).
- Acceptance-normalized decode A/B vs v20p1 rides the queued CKV workstream.

### DeepSeek-V4-Flash-0731 DSpark, rusty/toby TP2 (b12x-a8, K5, 0.87)

- Served names: `DeepSeek-V4-Flash-0731` + new alias `DeepSeek-V4-Flash`.
- #235 reasoning contract LIVE: prompt_tokens low/high/max = 11/90/103
  (pre-#235 low==high). Helper default reasoning_effort=high is now real.
- `tool_choice=required`: correct single call `get_weather({"city":"Berlin"})`.
- Truncation probe (max_tokens=8, K=5): clean, engine alive (q_len guard).
- Concurrent correctness 8/8 (2.9 s); decode 41.1 tok/s cc1; KV 568,506
  tokens; 0 errors both nodes.

## Rollback

Previous production images remain loaded on their nodes: Laguna v20p3p1
(`vllm41ae549`), GLM v20p1 (`vllm2167295`), DS4 v20p3 (`vllm92b27a4`).
