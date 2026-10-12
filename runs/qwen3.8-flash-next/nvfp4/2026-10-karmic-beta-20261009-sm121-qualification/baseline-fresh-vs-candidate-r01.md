# Candidate d9ba012f vs fresh-boot production 500ae05b, 2026-10-09 (same window, harness cc9bb06a)

compare-production.py refuses the pair on KV budget (5,189,473 vs 78,619,040); no cell in either grid was capacity-limited, so the budget did not bind. Geometric means over contexts 0/16K/32K/64K/128K:

| C | tok/s prod -> cand | steps/s | acceptance |
| --- | --- | --- | --- |
| 1 | 49.41 -> 48.07 (-2.7%) | 23.86 -> 23.73 (-0.5%) | 2.07 -> 2.03 (-2.2%) |
| 2 | 76.42 -> 76.90 (+0.6%) | 37.93 -> 38.19 (+0.7%) | 2.01 -> 2.01 (-0.1%) |
| 4 | 116.40 -> 116.91 (+0.4%) | 56.42 -> 57.22 (+1.4%) | 2.06 -> 2.04 (-1.0%) |

Prefill tok/s prod -> cand: 8K 3061 -> 3100 (+1.3%), 16K 2952 -> 3010 (+2.0%), 32K 2862 -> 2910 (+1.7%), 64K 2720 -> 2746 (+1.0%), 128K 2468 -> 2495 (+1.1%).

Baseline drift, fresh production vs the Oct 4 production grid: tok/s -0.4/-4.3/-3.4%, steps/s -0.6/-3.9/-4.0% at C1/C2/C4.
