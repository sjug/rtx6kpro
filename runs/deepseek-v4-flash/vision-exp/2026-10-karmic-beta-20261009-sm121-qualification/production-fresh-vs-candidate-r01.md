# DS4 Vision: candidate d9ba012f vs fresh-boot production 500ae05b, 2026-10-09 (same window, frozen harness cc9bb06a)

Candidate grid: throughput/20261009T195837-0400__karmic-beta-20261009-tp2-dspark-k3-b12x-a8-native1m__r01.json. Production grid: throughput/20261009T201617-0400__production-20260929-qad-ds4-vision-fresh-boot__r01.json (production container restarted 20:14 EDT after the candidate; its (c4,128K) cell is flagged capacity_limited by the harness, so ds4-vision/compare.py refuses the pair; numbers below are the same geometric means computed directly, with and without that cell).

| C | tok/s prod -> cand | steps/s | acceptance |
| --- | --- | --- | --- |
| 1 | 37.82 -> 38.19 (+1.0%) | 17.53 -> 17.54 (+0.1%) | 2.16 -> 2.18 (+0.9%) |
| 2 | 55.82 -> 57.87 (+3.7%) | 26.48 -> 26.78 (+1.1%) | 2.11 -> 2.16 (+2.5%) |
| 4 | 80.68 -> 82.60 (+2.4%) | 37.64 -> 37.77 (+0.4%) | 2.14 -> 2.19 (+2.0%) |

Excluding the flagged cell: C4 tok/s +2.7%, steps/s +0.9%, acceptance +1.8%. Prefill prod -> cand: 8K -2.1%, 16K -1.5%, 32K -0.9%, 64K -1.0%, 128K -0.9%.

KV: candidate 1,983,708 tokens vs production 2,196,661 (-9.7%) at the same 0.85 utilization. Memory profile per rank (rusty): consumed memory (weights + non-torch) 88.35 vs 87.28 GiB (+1.07 GiB), CUDA graph capture 0.60 vs 0.27 GiB, available KV memory 13.23 vs 14.62 GiB. The 1 GiB non-torch growth comes with FlashInfer 0.7.1 and the FlashInfer-carried B12X; not attributed further. 1.98M tokens is 1.89x the 1M context (production 2.10x); recoverable with a higher utilization if wanted (user: memory settings may be tuned for serving).

Health: candidate no Xid, one kernel NV_ERR_NO_MEMORY retry on rusty during the semantic battery, none on toby, no container-side error; vision battery (long cases), 32 structured cases, grid validation and post-grid semantic check passed.
