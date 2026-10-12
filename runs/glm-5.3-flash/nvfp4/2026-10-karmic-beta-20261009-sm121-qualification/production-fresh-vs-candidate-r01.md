# GLM-5.3-Flash NVFP4 TP4: Karmic beta 20261009 image vs same-window fresh-boot production (r01)

Window 2026-10-11 EDT on sparky/buddy/rocky/lucky. Same checkpoint (175ae8ce), same runner arguments, same frozen harness (llm_decode_bench.py cc9bb06a, run_bench.sh 5c79b976), 30 s cells, C1/C2/C4 at contexts 0/16K/32K/64K/128K, 6,412,288-token budget. One grid each; boot variance not separated.

- Production: image 500ae05b, restarted fresh from its retained containers 14:00, grid 14:15 to 14:28 (`throughput/20261011T141551-0400__production-20260929-qad-glm-fresh-boot__r01.json`).
- Candidate: image d9ba012f, fresh boot 14:30 (ready 14:37), gate battery 14:37 to 14:51, grid 14:51 to 15:03 (`throughput/20261011T145055-0400__karmic-beta-20261009-aligned-sm121-tp4-dcp1-mtp3-nvfp4head-seq4-native1m__r01.json`).

## Grid geomeans

| Concurrency | Output tok/s prod | cand | delta | Engine steps/s prod | cand | delta | Accept len prod | cand | delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | 55.91 | 53.51 | -4.3% | 20.56 | 20.58 | +0.1% | 2.72 | 2.60 | -4.4% |
| C2 | 84.23 | 84.47 | +0.3% | 31.72 | 31.67 | -0.1% | 2.66 | 2.67 | +0.4% |
| C4 | 128.71 | 125.78 | -2.3% | 47.01 | 47.35 | +0.7% | 2.74 | 2.66 | -3.0% |

| Prefill size | prod tok/s | cand tok/s | delta |
| --- | --- | --- | --- |
| 8K | 2527 | 2719 | +7.6% |
| 16K | 2904 | 2922 | +0.6% |
| 32K | 2946 | 2957 | +0.4% |
| 64K | 2932 | 2954 | +0.8% |
| 128K | 2892 | 2914 | +0.8% |

| Cell | Steps/s prod | cand | Accept prod | cand | tok/s prod | cand |
| --- | --- | --- | --- | --- | --- | --- |
| C1 0K | 20.79 | 20.70 | 2.76 | 2.52 | 57.4 | 52.3 |
| C1 16K | 20.54 | 20.76 | 2.53 | 2.62 | 52.1 | 54.5 |
| C1 32K | 20.62 | 20.61 | 2.76 | 2.61 | 56.9 | 53.7 |
| C1 64K | 20.49 | 20.36 | 2.64 | 2.55 | 54.2 | 51.9 |
| C1 128K | 20.37 | 20.47 | 2.91 | 2.70 | 59.3 | 55.3 |
| C2 0K | 32.43 | 31.91 | 2.63 | 2.62 | 85.4 | 83.6 |
| C2 16K | 31.92 | 32.35 | 2.73 | 2.78 | 87.3 | 89.9 |
| C2 32K | 32.00 | 31.41 | 2.68 | 2.66 | 85.8 | 83.6 |
| C2 64K | 31.35 | 31.45 | 2.58 | 2.59 | 80.9 | 81.3 |
| C2 128K | 30.91 | 31.25 | 2.65 | 2.69 | 81.8 | 84.2 |
| C4 0K | 48.31 | 47.34 | 2.67 | 2.65 | 129.1 | 125.5 |
| C4 16K | 46.50 | 47.54 | 2.77 | 2.66 | 128.9 | 126.4 |
| C4 32K | 47.31 | 48.20 | 2.69 | 2.62 | 127.5 | 126.3 |
| C4 64K | 47.05 | 48.25 | 2.73 | 2.63 | 128.6 | 126.7 |
| C4 128K | 45.93 | 45.46 | 2.82 | 2.73 | 129.6 | 124.1 |

Engine step rate is at parity in every concurrency and within 2% in every cell. The output tok/s differences at C1 and C4 are acceptance length, which at the serving defaults (temperature 1.0, top_p 0.95) varies with the sampled text. Grid prefill (first-at-size scouts) is +0.4 to +0.8% from 16K up and +7.6% at 8K.

## Same-window probes (both images, endpoint idle)

Fresh prefill, zero prefix-cache hits (`glm/probe-fresh-prefill.py`, receipts `qualification/glm-transport/fresh-prefill*.jsonl`):

| Prompt | Production tok/s | Candidate tok/s |
| --- | --- | --- |
| 8K first probe | 1,643 (first use after boot) | 3,048 |
| 8K second probe | 2,945 | 3,052 |
| 128K | 3,041 and 3,045 | 3,080 and 3,071 |

Acceptance, identical prompts, 1,024 tokens with ignore_eos, 4 rounds x 3 prompts, alternating a greedy arm (temperature 0, top_p 1.0) with the server default (`glm/probe-acceptance-ab.py`, receipts `qualification/glm/aligned-mtp3/acceptance-greedy-vs-default-*.jsonl`):

| Arm | Production accept len | Candidate accept len | Production tok/s | Candidate tok/s |
| --- | --- | --- | --- | --- |
| greedy | 2.671 | 2.664 | 53.8 | 54.5 |
| server default | 2.665 | 2.660 | 54.0 | 54.2 |

Per prompt the greedy arms differ by up to 6% in both directions (the two images do not emit identical greedy text), and the aggregates agree within 0.3%. The grid's acceptance dip is sampling variation, not a draft-head or kernel change.

## Memory and boot

| Per rank | Production | Candidate |
| --- | --- | --- |
| Consumed memory (weights + non-torch) | 58.01 GiB | 56.34 GiB |
| Available KV memory | 44.45 GiB | 46.25 GiB |
| GPU KV cache tokens | 6,927,805 | 7,191,950 (+3.8%) |
| Launch to first correct completion | 4 min 0 s | 6 min 15 s |

The candidate's slower boot is the B12X RoCE kernel compile (707 compilations, 54 s, nothing cached) plus first-boot warmup; production read 22 cached kernels in 3 s. A second candidate boot reuses the cache.

## Transport

RoCEnante (B12X RoCE proxy ABI 5, per-peer rail routes) on the switched 10.11.11.x fabric: world 4 on rocep1s0f0 and roceP2p1s0f0, all-reduce and all-gather both live, backend list exactly ['B12X_ROCENANTE', 'PYNCCL']. No RoCE, NCCL, Xid, CUDA or allocation errors in any container log during the window; GPU throttle flags were the usual short SW power and thermal samples on all four nodes.

## Verdict

Parity on engine step rate and acceptance, +1 to +3.6% fresh prefill, +3.8% KV, all serving gates passed. The ABI 5 transport is qualified on the four-node fabric. Promotion is the user's decision. Production 500ae05b restored 15:15 (ready 15:18); candidate containers retained on all four nodes as `glm53-flash-nvfp4-karmic-beta-20261009-tp4-r1`.
