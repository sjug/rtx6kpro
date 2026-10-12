# GLM-5.3-Flash on DGX Spark: upstream profile (CSF QAD + recipe 05eaafa options) vs current config, same image (r01)

Window 2026-10-11 EDT on sparky/buddy/rocky/lucky, image d9ba012f (Karmic beta 20261008.2 Spark build). Three grids on the same frozen harness (cc9bb06a / 5c79b976), 30 s cells, C1/C2/C4 at contexts 0 to 128K, 6,412,288-token budget, one grid each:

- Production: image 500ae05b, current config, fresh boot 14:00, grid 14:15 (`throughput/20261011T141551-0400__production-20260929-qad-glm-fresh-boot__r01.json`).
- Current config on the candidate image: original checkpoint 175ae8ce, aligned policy, W4A4 experts, interval 8; fresh boot 14:30, grid 14:51 (`throughput/20261011T145055-0400__karmic-beta-20261009-aligned-sm121-tp4-dcp1-mtp3-nvfp4head-seq4-native1m__r01.json`).
- Upstream profile on the candidate image (`glm/run-glm-tp4-node-upstream.sh`, `glm/upstream-profile.md`): CSF QAD checkpoint dec48abd, W4A16 experts with FP32 top-k and A4 prefill, request_boundaries, prefill compute share 0.4 at interval 1, one parallel prefill, upstream head and kernel environment, B12X MHC PDL; everything we keep on purpose unchanged (4 slots, 1M, nvfp4 draft head, FlashKDA, no L2 prefetch, materialized queue, 0.85, Max effort). Fresh boot 17:13 (ready 17:20), gates 17:20 to 17:33, grid 17:33 to 17:45 (`throughput/20261011T173327-0400__karmic-beta-20261009-upstream-profile-csf-qad-sm121-tp4-dcp1-mtp3-seq4-native1m__r01.json`).

## Gates (upstream profile)

First completion, short-prompt pool 7/7, semantic x3, concurrency (identical burst vs distinct, fresh/extension/verbatim-repeat, head-of-line), frozen prefix pairs, prefix triples short and long, native-context needles to 1,048,000 tokens, grid validation, post-grid completion: all passed. No RoCE, NCCL, Xid, CUDA or allocation errors in any of the four container logs. RoCEnante all-reduce and all-gather live on both rails. Attempt 1 of the window (17:09) failed at argument parsing on the launcher's stale `--fairness-engine` flag and was replaced within six minutes; see the kit README.

## Grid geomeans

| Concurrency | Output tok/s prod | current | upstream | Engine steps/s prod | current | upstream | Accept len prod | current | upstream |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | 55.91 | 53.51 | 70.41 (+31.6% vs current, +25.9% vs prod) | 20.56 | 20.58 | 26.25 (+27.5% vs current, +27.7% vs prod) | 2.72 | 2.60 | 2.68 (+3.2% vs current, -1.3% vs prod) |
| C2 | 84.23 | 84.47 | 101.49 (+20.1% vs current, +20.5% vs prod) | 31.72 | 31.67 | 38.40 (+21.2% vs current, +21.1% vs prod) | 2.66 | 2.67 | 2.64 (-0.9% vs current, -0.5% vs prod) |
| C4 | 128.71 | 125.78 | 147.05 (+16.9% vs current, +14.2% vs prod) | 47.01 | 47.35 | 54.64 (+15.4% vs current, +16.2% vs prod) | 2.74 | 2.66 | 2.69 (+1.3% vs current, -1.7% vs prod) |

| Prefill size | prod tok/s | current | upstream | upstream vs current |
| --- | --- | --- | --- | --- |
| 8K | 2527 | 2719 | 2958 | +8.8% |
| 16K | 2904 | 2922 | 3090 | +5.7% |
| 32K | 2946 | 2957 | 3130 | +5.9% |
| 64K | 2932 | 2954 | 3093 | +4.7% |
| 128K | 2892 | 2914 | 3032 | +4.0% |

| Cell | Steps/s current | upstream | Accept current | upstream | TTFT avg s current | upstream | ITL p50 ms current | upstream |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 0K | 20.70 | 26.67 | 2.52 | 2.60 | 0.24 | 0.23 | 18.8 | 14.0 |
| C1 16K | 20.76 | 26.53 | 2.62 | 2.72 | 1.51 | 0.07 | 18.2 | 13.7 |
| C1 32K | 20.61 | 26.43 | 2.61 | 2.89 | 1.45 | 0.09 | 18.3 | 13.1 |
| C1 64K | 20.36 | 26.02 | 2.55 | 2.65 | 1.37 | 0.17 | 19.1 | 14.5 |
| C1 128K | 20.47 | 25.60 | 2.70 | 2.57 | 1.82 | 0.28 | 17.9 | 15.0 |
| C2 0K | 31.91 | 39.26 | 2.62 | 2.70 | 0.28 | 0.06 | 23.8 | 18.9 |
| C2 16K | 32.35 | 39.01 | 2.78 | 2.73 | 2.50 | 0.11 | 22.6 | 18.7 |
| C2 32K | 31.41 | 38.24 | 2.66 | 2.53 | 2.44 | 0.15 | 24.2 | 20.4 |
| C2 64K | 31.45 | 38.22 | 2.59 | 2.69 | 2.31 | 0.29 | 24.6 | 19.3 |
| C2 128K | 31.25 | 37.28 | 2.69 | 2.58 | 2.83 | 0.45 | 23.8 | 20.4 |
| C4 0K | 47.34 | 56.16 | 2.65 | 2.71 | 0.70 | 0.11 | 31.3 | 26.0 |
| C4 16K | 47.54 | 55.45 | 2.66 | 2.64 | 4.10 | 0.17 | 32.5 | 27.2 |
| C4 32K | 48.20 | 53.38 | 2.62 | 2.68 | 4.14 | 0.22 | 33.0 | 27.7 |
| C4 64K | 48.25 | 55.47 | 2.63 | 2.75 | 3.71 | 0.45 | 32.5 | 26.3 |
| C4 128K | 45.46 | 52.79 | 2.73 | 2.67 | 4.59 | 0.76 | 33.2 | 28.0 |

Per-cell engine steps/s rose in all 15 cells (smallest +10.7% at C4 32K, largest +29.1% at C1 16K); inter-token latency p50 at C1 0K fell from 18.8 to 14.0 ms; the C1 0K cell ran 800 engine steps in 30 s against 622. Acceptance length is unchanged within noise, so the gain is engine step time, not drafting. TTFT in the context cells fell from seconds to tenths of a second: the request_boundaries policy retains the recurrent state at request boundaries, so the harness's repeated context prefixes are served from cache end to end, where the aligned policy re-ran part of the prefill. No loops, no errors, no capacity-limited cells, token accounting consistent between client and server.

## Same-window idle probes

Fresh prefill, zero prefix-cache hits (`glm/probe-fresh-prefill.py`):

| Prompt | Production | Current config | Upstream profile |
| --- | --- | --- | --- |
| 8K | 2,945 | 3,048 / 3,052 | 3,238 / 3,260 |
| 128K | 3,041 / 3,045 | 3,080 / 3,071 | 3,212 / 3,210 |

Greedy identical-prompt acceptance, 4 rounds x 3 prompts x 1,024 tokens, alternating arms (`glm/probe-acceptance-ab.py`):

| | Production | Current config | Upstream profile |
| --- | --- | --- | --- |
| accept len greedy | 2.671 | 2.664 | 2.695 |
| accept len server default | 2.665 | 2.660 | 2.698 |
| per-user tok/s greedy | 53.8 | 54.5 | 69.2 |
| per-user tok/s default | 54.0 | 54.2 | 69.4 |

## Memory and boot (per rank, candidate image)

| | Current config | Upstream profile |
| --- | --- | --- |
| Model weights in memory | 46.93 GiB | 42.46 GiB |
| Consumed memory (weights + non-torch) | 56.34 GiB | 54.05 GiB |
| Available KV memory | 46.25 GiB | 48.52 GiB |
| GPU KV cache tokens | 7,191,950 | 7,113,158 |
| B12X compilations on first boot | 707 (0:54) | 1,301 (1:34) |
| Launch to first correct completion | 6 min 15 s | 6 min 41 s |

The CSF checkpoint saves 4.5 GiB of weights per rank; the KV pool is 2.3 GiB larger but holds 1.1% fewer tokens because request_boundaries keeps recurrent checkpoints in the same pool. Draft head resolved as a draft-only NVFP4 copy (85 MiB per rank) with the verifier head BF16, as intended. V2 model runner was already in use on the current config, so it is not the source of the gain.

## Attribution

Not separated: the profile was adopted as a block (checkpoint, expert precision, prefill activations, policy, scheduler, environment). The decode step-rate gain is consistent with the W4A16 expert path on the compressed CSF scales (upstream's stated reason for the checkpoint) and the prefill gain with A4 prefill rows; a per-knob A/B on this image would attribute them if wanted. Correctness on the gates is unchanged; the CSF checkpoint's accuracy claims are upstream's (five-run parity with the BF16 QAD checkpoint on their suites) and were not re-measured here beyond the semantic, prefix and needle gates.

## Verdict

Every gate passed. Decode +15 to +28% engine steps/s, +17 to +32% output tok/s, fresh prefill +4 to +7%, acceptance unchanged, KV within 1%, no errors. Candidate containers retained on all four nodes as `glm53-flash-csf-qad-upstream-profile-karmic-beta-20261009-tp4-r1`; production 500ae05b (current config) restored. Promotion is the user's decision: the image alone (parity) or the image with the upstream profile (the gain above).
