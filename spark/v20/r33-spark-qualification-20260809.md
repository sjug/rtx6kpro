# r33-spark qualification record (2026-08-08/09)

Image: `gilded-gnosis-v20-r33-spark-sm121-vllm28e8eaf-b12x06db0f4-fi1ac6942-cu132-20260808`
(respin; supersedes first-pass `vllm44700e3`, retained on nodes as history).
Staged on all 8 nodes. DATE_TAG is our build date; canonical r33 is dated
2026-08-09.

Composition: upstream gilded-gnosis-v20 r33 manifests (canonical bases GG
`e2666d9a`, b12x `9bbae678`, LMCache `9cebd405`; canonical trees
`fa13d334`/`06db0f4b`/`9a05c881` reconstructed byte-exact first). Our vLLM
PR #234 q_len guard rides UPSTREAM in the r33 manifest at its refined head
`eaf24cc15`. FlashInfer moved to the upstream 0.6.18 integration pin
(`voipmonitor @ 1ac6942`, #3932 quantfix verified present); the sjug
`7ad08da` mirror is retired. Spark overlay, 4 files (composed tree
`28e8eaf1`, round-tripped): CMake arch 12.1, MTP-3D residual fallback,
fail-closed online EXL3 quantization at MODE SELECTION (before cache
identity/lookup; only the literal "1" in `VLLM_EXL3_ONLINE_UNSAFE_OVERRIDE`
bypasses; belt-and-braces loader guard retained), and the env registration
for that override.

EXL3 ships ENABLED for sm_121a behind the pure x86 arch-guard patch
(`exllamav3-sm121-aarch64-guards-20260804.patch`, 6 files, no kernel
changes).

## EXL3 correctness posture (exhaustive codebook oracle, 2026-08-08)

- decode/reconstruct/GEMM family is oracle-BIT-EXACT on SM121 for all
  65,536 indices on all three codebooks (3INST, MCG, MUL1).
- The ENCODER fast path is inconsistent with the decode codebook for
  3INST/MCG on BOTH SM120 and SM121 (identical mismatch counts; fork-wide
  a1-retile bug, NOT an SM121 miscompile). MUL1 encoder is bit-exact.
- Consequence: serving pre-quantized checkpoints (TR3 = MCG-coded) is fully
  supported; ON-DEVICE quantization (online K6, hard-coded mcg=True) is
  fail-closed. Checkpoint-only mode is also the higher-quality
  configuration per upstream's KLD table.

## Build gates (all PASS on dusty, respin)

Everything from the r28 battery (adjusted for the b12x rename and the r33
`serve-ds4-flash.sh` hash) plus: exhaustive decode-vs-oracle
(bit-pattern + finiteness), MUL1-encoder regression guard, EXL3 GEMM
execution parity (fused vs reconstruct+hgemm, K3/K4, M 1/32/128), b12x r33
mixed K3/K4 shared-H suite (vendored, 15 passed), online-quant override
matrix (unset/"0"/invalid closed; literal "1" opens), EXL3 artifact
assertions independent of GPU checks, `SKIP_EXLLAMAV3!=0` refused at entry.

## Adopted KLD figures (upstream r28 table; our measurement attempt is
## documented in the upstream report)

Transfer justified by decode bit-exactness + byte-identical model-math
trees. EXL3 3.42bpw checkpoint-only: 0.074 @fp8 KV; 0.108 @nvfp4 KV
(FP4 cache costs ~46% relative). NF3 hybrid has NO KLD figure anywhere
(open item; needs the direct-logits capture feature upstream dropped).

## GLM-5.2 EXL3 production decision

`willfalco/GLM-5.2-EXL3-TR3-3.42bpw @ ae68c65` (328 GB, staged all 4
cluster nodes), TP4 cross-node, **FP8 KV + DCP2** (quality-first: capacity
recovered via DCP instead of cache quantization), MTP3 greedy/triton,
ONLINE_QUANT=none, runner `run-glm52-exl3-tp4-node.sh`. Config matrix
measured on the cluster:

| Config | Decode cc1 | Prefill 8k/64k/128k | KV pool @262k |
|---|---:|---|---:|
| NVFP4 KV, DCP1 (first boot) | 25.8 | 776/801/737 | 284,800 |
| FP8 KV, DCP2 (production) | 24.3-23.7 | 685-798/762-793/767-769 | 309-325k |
| NF3 hybrid (old prod, FP4 KV, DCP2) | 24.9 | 766/779/771 | 739,200 |

EXL3 pool is smaller than NF3-hybrid's at equal settings because
checkpoint-only mode keeps dense layers BF16 (86.95 vs 84.12 GiB/rank) plus
a ~1 GiB prefill trellis arena. Optional pool lift to ~470k via explicit
`--kv-cache-memory-bytes` ~12.5 GiB (validated headroom: engine reports
15.49 GiB available at full utilization; keep >=4-5 GiB UMA host margin).
Upstream is deprecating NF3-era formats (QSRT), reinforcing the migration.

## Per-profile qualification on vllm28e8eaf

- **GLM-EXL3** (sparky/buddy/rocky/lucky): READY, coherent, decode 23.7,
  prefill 798/793/767, pool 309,120, fp8 confirmed in engine banner, MTP3
  acceptance 3.20, 0 errors x4.
- **Laguna** (dusty/kirby): both frozen-wrapper reproducers token-exact vs
  the PIECEWISE refs, engine alive, 0 errors. First verification of the
  guard as composed UPSTREAM (#234) rather than via our overlay. Pair
  re-parked after qualification (user-directed).
- **DS4** (rusty/toby): flipped to r33; both served names, reasoning
  contract stable (11/90/103), required tool call OK, truncation probe
  clean, 8/8 concurrent, decode 43.7 tok/s cc1 (41.1 on r28-spark), KV
  568,139, 0 errors x2.

## Fleet state after cutover

DS4 + GLM-EXL3 serving on r33-spark; Laguna parked (qualified); r28-spark
retained on all nodes as rollback; NF3 hybrid checkpoint + runner retained
as the GLM rollback path. All 8 nodes now share one 200G switched fabric;
distribution is a direct dusty fan (2-wide scp batching; 7-wide overloads
the source). SSH trust flattened to match (host keys verified out-of-band
via the workstation channel).

## Open items

- GLM-EXL3 bench/soak pass on the production config.
- NF3-hybrid KLD figure (blocked on upstream direct-logits capture).
- KV pool lift boot (~12.5 GiB explicit budget) if capacity is wanted.
- Upstream report delivery (encoder oracle findings, dropped
  `return_prompt_logits`, UMA capture OOM, r31 L2-offload cache poisoning
  awareness).
