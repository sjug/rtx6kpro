# JJ r22 Spark qualification

Date: 2026-09-04

## Deployment

- Model: `local-inference-lab/GLM-5.3-Flash-NVFP4`
- Model revision: `2e8b6daeb2be8716c06b54d746d12e581e551cca`
- Served name: `GLM-5.3-Flash` (one advertised model)
- Nodes: sparky, buddy, rocky, and lucky at TP4/DCP1 over the switched 200G
  f0 rails
- Image ID: `907c1265f308ea87c77d98e0b787455af228a5dbdfa228b1c7653a48d3433e0a`
- Release tag: `localhost/voipmonitor/vllm:glm53-jj-r22-spark-sm121-vllm4fbb1c2-b12xe1edb6d-lmcache976a97f-cu133-torch213-20260904-r1`
- Qualification build tag: `localhost/voipmonitor/vllm:glm53-jj-r22-spark-sm121-vllm4fbb1c2-b12xe1edb6d-lmcache976a97f-cu133-torch213-20260904-r1-scratch`
- Runtime: MTP3, FP8 KV, native 1,048,576-token model length, percentage-sized
  KV at 0.85 utilization, LMCache disabled

The Docker-v2 archive was transferred only over the switched
`10.11.11.0/24` fabric. Its SHA-256 was verified as
`fc15e668b7c899df6a14aa0e6637da5f567e6fc69d94ba1cefaf78596f7aaa65`
on every receiver before import. Every running container resolves to the exact
image ID above.

The image was built and qualified before the release commit, following the
program's build, validate, then sign ordering. Its baked recipe SHA-256 is
`359bc15cb8a55f4907ee3fe52f3bcdc88f24ca8b552623f19ce396d7fd65bdd1`,
and its source-lock SHA-256 is
`fe3558327415c379496ae1cd2b6e319b97f711d18e37f97bf24739196eace744`.
The embedded source lock, both in-image launchers, and runtime verifier were
checked byte-for-byte against this release tree. The post-build changes are
limited to host-runner release defaults and qualification records; they do not
change the image payload. The release tag therefore promotes the already
qualified image ID without rebuilding or converting it.

## Admission

The four ranks formed the distributed group on `10.11.11.1`. The TP
communicator selected exactly `['B12X_ROCENANTE', 'PYNCCL']`, while the EP
communicator remained `['PYNCCL']`. Both the first routed 65,536-byte
RoCEnante all-reduce and the first `(8, 38720)` BF16 shard all-gather executed
on live warmup traffic. The ABI-3 proxy loaded from the immutable in-image
cache without a runtime compile.

R22 selected split GLM cache pages with a 2,048-token target and recurrent
block size. It reported 41.04 GiB available for KV and 6,283,542 KV tokens,
or 5.99 maximum-concurrency requests at the native context length. The API was
ready in approximately 3 minutes 42 seconds after the head started.

## Correctness

The first completion returned exactly `333` for `17 * 23 - 58`.

The repeated semantic suite passed every case across three repetitions:

- reasoning effort `low`, `high`, and `max`;
- deterministic synthetic-image understanding; and
- strict tool-call round trip with exact arguments and result.

Receipt:
`results/jj-r22-semantic-admission-3x-20260904.jsonl`.

Exact retrieval passed on both sides of the R22 split-cache boundary and at
the legacy comparison points: 2,048, 2,049, 2,784, and 2,785 prompt tokens.
Receipt:
`results/jj-r22-boundary-2048-2049-2784-2785-20260904.jsonl`.

Native-context retrieval passed at:

- 262,000 prompt tokens in 91.920 seconds; and
- 1,048,000 prompt tokens in 391.478 seconds.

The first 262K attempt used the historical 16-token output ceiling. It emitted
the correct retrieval preamble but ended before emitting the code and is
retained as a harness-budget false negative. The 64-token retry emitted the
exact code and stopped normally. Receipts:

- `results/jj-r22-native-context-needle-20260904.jsonl` (excluded truncated
  attempt);
- `results/jj-r22-native-context-262k-retry64-20260904.jsonl`; and
- `results/jj-r22-native-context-1048k-20260904.jsonl`.

The post-benchmark completion returned exactly `READY` with the established
GLM reasoning budget.

## Standard benchmark

Every performance measurement used `llm-inference-bench/run_bench.sh` version
0.4.29 under campaign `2026-09-jj-r22-sm121-qualification`. The standard pass
used 30-second windows at contexts 0, 16K, 32K, 64K, and 128K for concurrency
1, 2, and 4. The c8 production-envelope cell used the matched R17p1
`--skip-prefill` methodology. The logged 6,283,542-token KV capacity was passed
as the benchmark budget.

Raw R22 MTP-normalized engine steps per second:

| Concurrency | 0 | 16K | 32K | 64K | 128K |
|---:|---:|---:|---:|---:|---:|
| 1 | 19.173 | 18.789 | 18.915 | 18.795 | 18.512 |
| 2 | 29.755 | 30.050 | 29.511 | 29.445 | 29.342 |
| 4 | 44.493 | 43.687 | 43.045 | 45.337 | 41.533 |

Geometric means against the two qualified R17p1 RoCEnante boots:

| Concurrency | R17p1 mean | R22 | Delta |
|---:|---:|---:|---:|
| 1 | 17.0969 | 18.8357 | +10.17% |
| 2 | 26.7480 | 29.6195 | +10.74% |
| 4 | 40.8797 | 43.5996 | +6.65% |
| 8 (context 0) | 59.8426 | 64.3605 | +7.55% |

The c8 result was 172.184 output tok/s at an effective acceptance length of
2.6753. All 16 measured cells completed with zero request errors, zero warmup
timeouts, zero queueing, and no capacity limits.

R22 prefill versus the geometric mean of the two R17p1 boots:

| Context | R17p1 tok/s | R22 tok/s | Delta |
|---:|---:|---:|---:|
| 8K | 2,088 | 2,870 | +37.43% |
| 16K | 2,283 | 2,945 | +29.00% |
| 32K | 2,344 | 2,942 | +25.54% |
| 64K | 2,366 | 2,918 | +23.33% |
| 128K | 2,345 | 2,850 | +21.54% |

The 16K-through-128K geometric-mean prefill gain is +24.82%. The independent
near-native 1,048K retrieval latency improved from 524.922 seconds on R17p1 to
391.478 seconds on R22, a 25.4% reduction.

Primary receipts:

- `llm-inference-bench/results/runs/glm-5.3-flash/nvfp4/2026-09-jj-r22-sm121-qualification/throughput/20260904T130007-0400__jj-r22-sm121-tp4-dcp1-mtp3-native1m__r01.json`
- `llm-inference-bench/results/runs/glm-5.3-flash/nvfp4/2026-09-jj-r22-sm121-qualification/throughput/20260904T131656-0400__jj-r22-sm121-tp4-dcp1-mtp3-native1m-c8__r01.json`

The interrupted c8 setup artifact at timestamp `20260904T131605-0400` is
excluded. It ran only an unwanted 8K prefill scout before the methodology
mismatch was detected; no c8 measurement occurred.

## Health and verdict

After the complete correctness and benchmark battery, every node reported:

- zero container `ERROR`, traceback, OOM, engine-failure, or killed-process
  records;
- zero kernel Xids and zero `NV_ERR_NO_MEMORY` records;
- zero system OOM or memory-pressure records; and
- untouched swap.

`MemAvailable` after qualification was approximately 4.96 GiB on sparky,
7.34 GiB on buddy, 7.35 GiB on rocky, and 9.95 GiB on lucky.

JJ R22 therefore passes GLM-5.3-Flash NVFP4 qualification for this exact
four-node SM121 TP4/DCP1/MTP3/RoCEnante profile. It preserves semantic,
vision, tools, cache-boundary, and native-context correctness while materially
improving both prefill and acceptance-normalized decode over qualified R17p1.
The R17p1 image remains locally available as rollback.
