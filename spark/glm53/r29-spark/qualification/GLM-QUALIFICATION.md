# Corrected R29 GLM qualification

Completed 2026-09-09 14:39:33 EDT, driver exit 0, QUALIFICATION-PASS.
Verdict: qualified for the existing aligned TP4/DCP1/MTP3 profile and left
serving on sparky, buddy, rocky and lucky. This is not qualification of auto
checkpoint policy, LMCache, a new draft head or a clear_thinking override.

## Identity and contract

- Image: `0b15723cb87646bb4628c5cd67aa2a7c879ac5a27105109ea5db06c30113dc0b`.
- Tag: `localhost/voipmonitor/vllm:jj-r29-spark-sm121`.
- Source-lock SHA: `3fb73885dd8f5b5e40de1342c63aeac37c075e6cee3a86548f9abff889b88754`.
- Checkpoint: local-inference-lab/GLM-5.3-Flash-NVFP4,
  `46aaae8a82032f77100f2f03e9cc11b391df3b4d`.
- TP4, DCP1, MTP3, BF16 draft head, aligned recurrent checkpoints, FP8 KV,
  max length 1,048,576, max sequences 8, batched tokens 4096, utilization 0.85.
- FlashKDA, RoCEnante plus PyNCCL, NCCL_PROTO=LL,Simple,
  InstantTensor BUFFERED, LMCache disabled. No clear_thinking override.
- Boot KV capacity: 6,196,490 tokens, this boot only.

All four receiver image IDs and launcher identities passed before shutdown.
Workers stopped before head and started before head. First correct completion
arrived at 14:13:05. Final inspections show all four containers running on the
corrected image, none OOMKilled. R28 stopped containers and images are retained.

## Correctness and serving behavior

- Semantic battery: 15/15 across three repetitions, covering reasoning levels,
  vision and tool round trips.
- Exact needle retrieval at 2,048, 2,049, 262,000 and 1,048,000 tokens,
  all finish_reason stop. These probes were cache-warm, not cold-prefill timings.
- Identical burst: 121.2 aggregate tok/s. Later fresh bursts 110.4 to 124.5;
  repeat burst 118.0. No serialization or head-of-line stall: fresh TTFT
  0.3 to 0.4 seconds while long requests continued decoding.
- Frozen prefix pairs and triples completed. Long triple reused 258,048 tokens
  on the 1,048,000-token extension, 389.45 seconds versus R28's 389.32 seconds
  at the same reused-token count. This is partial-hit latency parity.
- Final completion returned exactly 333.

## Standard benchmark

Unchanged run_bench.sh: contexts 0/16K/32K/64K/128K, concurrency 1/2/4,
30-second duration and 3-second warmup. All 15 cells valid. Geometric means:

| Concurrency | R28 steps/s | R29 steps/s | Change | R28 output tok/s | R29 output tok/s |
|---|---:|---:|---:|---:|---:|
| 1 | 19.938 | 19.921 | -0.09% | 51.691 | 49.823 |
| 2 | 31.175 | 31.078 | -0.31% | 81.519 | 81.485 |
| 4 | 45.118 | 45.630 | +1.14% | 119.860 | 121.362 |

Output is aggregate generated throughput, not engine steps. At c1 output is
3.62% lower and acceptance is 2.501 versus 2.593 (-3.53%); engine speed is flat.
This does not establish why acceptance differs or prove broad quality parity.

| Prefill context | R28 tok/s | R29 tok/s | Change |
|---|---:|---:|---:|
| 8K | 2776 | 2745 | -1.12% |
| 16K | 2896 | 2938 | +1.45% |
| 32K | 2950 | 2934 | -0.54% |
| 64K | 2941 | 2927 | -0.48% |
| 128K | 2839 | 2853 | +0.49% |

Historical single-boot comparison, not a repeatability study. No material
engine or prefill regression is demonstrated, and no GLM speedup is claimed.

Measured window 14:26:58 to 14:39:33: no captured jit_monitor warnings,
tracebacks, CUDA errors or engine-failure messages. Workers emitted no new
container log lines during this window. One-Hz GPU receipts contain 755 to 756
samples per node; rocky has six nonzero clock-event samples, others zero.
Mean SM clocks sparky/buddy/rocky/lucky: 2459/2479/2516/2435 MHz;
maximum temperatures 81/85/84/81 C. Do not claim an entirely unthrottled run.

## Receipts and final state

Receipts: `glm-metadata-rebuild-20260909/`, including comparison JSON,
driver status, semantic JSONL, prefix metrics, node inspections and GPU logs.
Build receipts: `../build-receipts/20260909T173029Z-765802/`.

Benchmark results repository, campaign `2026-09-jj-r29-vs-r28`:
`20260909T142658-0400__jj-r29-aligned-sm121-tp4-dcp1-mtp3-native1m__r01.json`,
SHA `8fe3d82c0a0864a621e9ba5669e82b410eee11f89d1fa81a9178f2023dc665e8`.
Historical R28 baseline:
`20260908T190749-0400__jj-r28-aligned-sm121-tp4-dcp1-mtp3-native1m__r01.json`,
SHA `3a502970e06e0005c138f1e0fc0af1fecf584aaacee0c670e48d0ba7cf971f36`.

Qwen was paused for the rebuild and restored on dusty/kirby to its original
qualified R29 image `ee996ef8...`, not the corrected GLM image. DS4/nous was
untouched. No commit or push performed.

## Subsequent source preservation, 2026-09-10

Implementation commit 43b34e2 preserves the corrected image's source lock
byte-for-byte (SHA 3fb73885dd8f5b5e40de1342c63aeac37c075e6cee3a86548f9abff889b88754).
The image was built from the then-uncommitted recipe, so its recipe-commit label
predates this source commit. Source-lock and recipe input digests, not that older
commit label, establish the built-content identity. No rebuild is implied.
