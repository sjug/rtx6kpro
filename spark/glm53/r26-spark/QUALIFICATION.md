# JJ r26 Spark qualification

Status: Qwen phase qualified; GLM phase completed but R26 is not promoted

Build date: 2026-09-05

## Candidate identity

The source composition and build gate ladder are complete. Qwen serving
correctness and performance qualification passed on dusty and kirby. The same
image archive is staged, but not loaded, on the four GLM nodes.

- Image ID:
  `bb9cb676e464425acb945118bc541fd917fa7eb7f3024418f1733b70de31e8b8`;
- image tag:
  `localhost/voipmonitor/vllm:glm53-jj-r26-spark-sm121-vllm59c9787-b12x00248b0-lmcachefe5442f-cu133-torch213-20260905-r1`;
- recipe SHA-256:
  `b3a9275c7d7b9fe799e14eb6ccb72f71d2ec764c9fa5fff4e48ac9ec3bccda1c`;
- source-lock SHA-256:
  `2f735f777c53ab2ab35631247fb3aa0516a93e0d192b31d008f9e22b45e319ee`.

Locked source identities:

- vLLM Spark tree: `d4571e5ba189fcc05534ecae88dee763ffef4e35`;
- vLLM package subtree: `59c9787400c85d5d4547ef898c5dee41804e7088`;
- B12X tree: `c20b6aab67ed791cc226ee91de23f7c45509f436`;
- B12X package subtree: `00248b09689830e55d8fbce8ee630600b63ef663`;
- LMCache tree: `008ac3e09ae5917aa0849147480d7bd5b9f8b37a`;
- LMCache package subtree: `fe5442fbf258accaa7f26d2bbb00d8b7b5c349ca`.

Build gates passed on a physical GB10, including the source/native runtime
contract, all six draft-head mode/capability tests, the real SM121 NVFP4 draft
head at rows 1/4/32, FlashKDA near-collinear-key finiteness, and the B12X
persistent-epilogue repeated-output regression.

The image uses the normal release tag above. Failed builds are replaced under
that tag rather than creating special development or scratch suffixes.

## Phase 1: Qwen TP2

Model: `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-4p89`

Revision: `c374e7e24b54f6cb0017d0c2e6d26823d2f2fb5d`

Nodes: dusty and kirby over their direct 200G links

Qualified profile: TP2/MTP3, native 262,144-token model length, 0.85 GPU memory
utilization, LMCache disabled, `NCCL_PROTO=LL,Simple`, and no channel pin.

The service is live on dusty and kirby with the exact image ID above. The final
MTP3 boot advertised exactly one model and returned the expected meaningful
completion. It reported 43.06 GiB of KV cache, 5,300,423 KV tokens, and 20.22x
maximum concurrency at the native model length. The first MTP3 boot reported
42.71 GiB and 5,257,359 tokens; the difference is normal boot-time memory
variation.

### Correctness results

- The three-run semantic battery passed all 18 reasoning, non-thinking,
  vision, and tool-call checks.
- MTP3 exact retrieval passed at 2,848, 2,849, 131,072, and 262,000 input
  tokens. Every case returned the exact token IDs for `739184`.
- MTP0 exact retrieval passed at 2,784, 2,785, and 131,072 input tokens.
- Four sequential greedy technical responses were meaningful and passed the
  collapse checks. Their hashes were not identical across runs. Concurrent
  fixed-corpus outputs were also not bit-repeatable, as was already true for
  the r12, r15p, and r22 controls. This candidate is semantically qualified,
  not bitwise deterministic.
- The fixed-token MTP3 run completed 12 requests and 6,144 output tokens. Its
  mean acceptance length was 2.1423, versus 2.2588 for r22.

Receipts:

- `spark/qwen38-flash-next/jj-r26-mtp3-semantic-admission-3x-20260905.jsonl`;
- `spark/qwen38-flash-next/jj-r26-mtp3-boundary-native-context-20260905.jsonl`;
- `spark/qwen38-flash-next/jj-r26-mtp0-boundary-native-context-20260905.jsonl`;
- `spark/qwen38-flash-next/jj-r26-mtp3-chat-greedy-20260905.jsonl`;
- `spark/qwen38-flash-next/jj-r26-fixed-token-acceptance-20260905.json`.

### Standard benchmark result

The standard `llm-inference-bench/run_bench.sh` 15-cell pass completed under
campaign `2026-09-jj-r26-vs-r22`. Engine steps/s is the primary decode metric.

| Concurrency | R22 steps/s geo | R26 steps/s geo | Change | R26 raw tok/s geo change |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 16.267 | 19.166 | +17.82% | +9.48% |
| 2 | 27.771 | 31.911 | +14.91% | +13.96% |
| 4 | 43.161 | 48.978 | +13.48% | +13.91% |
| All | 26.915 | 31.057 | +15.39% | +12.43% |

All 15 decode cells improved in engine steps/s, with changes from +10.90% to
+19.96%. R26 raw throughput and engine steps/s by context were:

| Concurrency | Contexts 0/16K/32K/64K/128K raw tok/s | Engine steps/s |
| ---: | --- | --- |
| 1 | 35.4 / 35.8 / 38.9 / 41.1 / 35.2 | 19.5 / 19.4 / 19.1 / 19.1 / 18.7 |
| 2 | 68.8 / 60.3 / 60.5 / 65.1 / 63.7 | 32.1 / 32.2 / 31.8 / 31.6 / 31.8 |
| 4 | 107.1 / 108.3 / 107.1 / 105.0 / 104.1 | 49.3 / 48.6 / 48.7 / 48.8 / 49.5 |

Prefill improved substantially in the same harness:

| Prompt | R22 tok/s | R26 tok/s | Change |
| ---: | ---: | ---: | ---: |
| 8K | 1,277 | 2,670 | +109.08% |
| 16K | 1,158 | 2,042 | +76.34% |
| 32K | 1,445 | 2,741 | +89.69% |
| 64K | 1,373 | 2,383 | +73.56% |
| 128K | 1,437 | 2,238 | +55.74% |

This is one boot per release, so it establishes a large directional gain and
successful qualification, not a multi-boot estimate of variance.

The result receipt is
`llm-inference-bench/results/runs/qwen3.8-flash-next/nvfp4-4p89/2026-09-jj-r26-vs-r22/throughput/20260905T205045-0400__jj-r26-sm121-tp2-mtp3__r01.json`.

## Phase 2: GLM TP4

Model: `local-inference-lab/GLM-5.3-Flash-NVFP4`

Revision: `46aaae8a82032f77100f2f03e9cc11b391df3b4d`

Nodes: sparky, buddy, rocky, and lucky over the switched 200G f0 fabric

Measured first-admission profile: TP4/DCP1/MTP3, BF16 proposal head, native
1,048,576-token model length, percentage-sized KV at 0.85 utilization, FP8 KV,
LMCache disabled, and RoCEnante enabled.

The Docker archive was transferred over the switched 200G fabric to all four
nodes and verified at 32,442,739,712 bytes with SHA-256
`c46799b317f73d35d5321b3c6173d15424f91a0f95a461d46e9efd3365cc4eca`.
It remains staged at
`/home/jugs/transfers/jj-r26/glm53-jj-r26-spark-r1.docker.tar`. The archive
loaded to the exact image ID on every node with 149 layers. The deployed runner
was byte-identical on all four nodes.

The checkpoint revision advanced from `2e8b6dae` to `46aaae8a`. The only model
tree changes were `chat_template.jinja` and removal of `lil.yaml`; the 44 weight
shards and model configuration were unchanged. The exact new snapshot was
materialized on every node by reusing the existing weight blobs and installing
only the new template blob. No Hugging Face cache content was cleared.

### Runtime admission and correctness

The engine reached READY and reported the exact TP backend list
`['B12X_ROCENANTE', 'PYNCCL']`; the EP group used `['PYNCCL']`. NCCL 2.31.2 was
loaded coherently, and RoCEnante all-reduce and all-gather both ran. The server
advertised exactly `GLM-5.3-Flash` at the pinned revision and native 1,048,576
model length. Boot reported 40.28 GiB of KV cache and 6,194,512 KV tokens; the
later metrics view reported 6,412,288 tokens.

Correctness passed:

- the first meaningful completion returned `FINAL: 333`;
- the three-run semantic battery passed low, high, and max reasoning, vision,
  and strict tool round trips, three of three in every arm;
- exact needle retrieval passed at 2,048, 2,049, 262,000, and 1,048,000 input
  tokens;
- the post-benchmark completion returned `READY`;
- all four containers remained running with `OOMKilled=false`, and the service
  reported no container-level CUDA, NCCL, traceback, or runtime error.

Receipts:

- `spark/glm53/jj-r26-semantic-admission-3x-20260905.jsonl`;
- `spark/glm53/jj-r26-native-context-needle-20260905.jsonl`.

The full-native-context request took 502.199 seconds, versus 391.478 seconds on
R22, a 28.3% latency regression. This is a correctness pass but not a
performance pass.

### Standard benchmark comparison

The standard 15-cell `llm-inference-bench/run_bench.sh` pass completed with no
request errors, queueing, warmup timeout, or capacity limit. Engine steps/s is
the primary decode metric because speculative acceptance moved between images.

| Concurrency | R22 steps/s geo | R26 steps/s geo | Change | R26 raw tok/s geo change |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 18.836 | 18.113 | -3.84% | -4.98% |
| 2 | 29.620 | 28.477 | -3.86% | -4.96% |
| 4 | 43.600 | 42.169 | -3.28% | -1.31% |

R26 raw throughput and engine steps/s by context were:

| Concurrency | Contexts 0/16K/32K/64K/128K raw tok/s | Engine steps/s |
| ---: | --- | --- |
| 1 | 44.0 / 46.0 / 46.3 / 48.4 / 47.5 | 18.261 / 18.146 / 18.081 / 18.107 / 17.970 |
| 2 | 73.2 / 75.8 / 73.9 / 71.1 / 74.8 | 28.780 / 28.699 / 28.412 / 28.458 / 28.040 |
| 4 | 111.7 / 117.3 / 117.8 / 107.9 / 109.4 | 42.271 / 42.975 / 43.167 / 41.510 / 40.965 |

Prefill regressed consistently:

| Prompt | R22 tok/s | R26 tok/s | Change |
| ---: | ---: | ---: | ---: |
| 8K | 2,870 | 2,787 | -2.89% |
| 16K | 2,945 | 2,845 | -3.40% |
| 32K | 2,942 | 2,823 | -4.04% |
| 64K | 2,918 | 2,809 | -3.74% |
| 128K | 2,850 | 2,735 | -4.04% |

At c8 and zero context, R26 measured 163.6 raw tok/s and 62.0 engine steps/s,
versus 172.18 and 64.36 on R22, changes of approximately -5.0% and -3.7%.

A short repeat at zero context confirmed that the normalized gap was not a
single-pass artifact. R26 measured 18.272 steps/s at c1 and 42.730 at c4,
respectively 4.70% and 3.96% below matched R22. Its higher c1 acceptance length
made raw throughput 1.02% higher, which is why raw tok/s alone is not a valid
promotion metric here.

Benchmark receipts:

- `llm-inference-bench/results/runs/glm-5.3-flash/nvfp4/2026-09-jj-r26-sm121-qualification/throughput/20260905T222033-0400__jj-r26-sm121-tp4-dcp1-mtp3-native1m__r01.json`;
- `llm-inference-bench/results/runs/glm-5.3-flash/nvfp4/2026-09-jj-r26-sm121-qualification/throughput/20260905T223623-0400__jj-r26-sm121-tp4-dcp1-mtp3-native1m-c8__r01.json`;
- `llm-inference-bench/results/runs/glm-5.3-flash/nvfp4/2026-09-jj-r26-sm121-qualification/throughput/20260905T223809-0400__jj-r26-sm121-tp4-dcp1-mtp3-native1m-c1-c4-repeat__r01.json`.

### Health verdict and disposition

The post-run service is responsive and every container is still running on the
exact image ID. However, rocky recorded 29
`NVRM ... Out of memory [NV_ERR_NO_MEMORY]` kernel messages during startup and
CUDA graph capture between 22:07:20 and 22:07:31. The other three nodes
recorded none. Post-run `MemAvailable` was approximately 5.45 GiB on sparky,
7.66 GiB on buddy, 7.67 GiB on rocky, and 10.19 GiB on lucky.

R26 therefore does not pass the GLM promotion gate. Correctness is green, but
the repeated 3% to 4% engine-step and prefill regressions, the 28.3% native
context latency regression, and the startup NVRM allocation errors prevent it
from replacing R22 as the qualified GLM release. R26 remains live temporarily
for investigation. It is neither promoted nor automatically rolled back. If a
subsequent source/config correction passes the same battery, that corrected
R26 line should remain deployed rather than reverting to R22.

The NVFP4 proposal head, DCP4, LMCache, compute-share fairness, and DFlash2
remain outside this first-admission result.

### Attribution correction, 2026-09-06

The measurements above compare different launcher environments. They do not
establish a regression caused by the R26 engine source. The R26 launcher adds
`VLLM_GLM53_L2_PREFETCH=1`, `B12X_DYNAMIC_WORK_SOURCE=persistent_grid`, direct
expert scales, and split-route/compute settings including a compute MAC of 224.
The source L2-prefetch default enables it only on capability (12, 0), so the
explicit export changes GB10 behavior. These settings require a controlled
comparison before attributing the observed loss to the release.

The earlier explanation naming new GLM KDA side-stream overlap, dynamic MoE
M1/M8/FC1 changes, and GB10 mHC grouping as R26 deltas is withdrawn. The
composition patch does not change the GLM model/KDA/L2-prefetch files, the vLLM
B12X MoE integration, or the dynamic MoE kernel. The paged selected-forward
addition belongs to QSA and is not evidence of a changed GLM DSA top-k path.

Next control: retain the same R26 image and BF16 proposal head, restore the
R22 launcher environment by removing the added L2-prefetch and dynamic-policy
overrides from the effective process environment, verify that environment on
every rank, then repeat the standard 15-cell benchmark and prefill measurements.
Use run_bench.sh, retain its warmup, and record any compilation during timed
cells. If recovery is partial, measure prefetch and dynamic overrides separately.
The NVFP4 proposal head is a subsequent optimization experiment.

The single native-context request establishes a measured latency difference,
not a repeatable 28.3 percent regression. Repeat it warm with one-second GPU
clock and thermal telemetry on all four ranks. No service restart or launcher
deployment was performed when recording this correction.

### Completed launcher control, 2026-09-06

The same image `bb9cb676...` was restarted worker-first with a read-only
launcher overlay removing the added R26 policy block. Every rank used the
same checkpoint revision, BF16 MTP3 head, TP4/DCP1, FP8 KV, native context,
utilization 0.85, and RoCEnante. Process-environment receipts confirm that
neither L2-prefetch overrides nor B12X_DYNAMIC_* overrides survived.

The GLM semantic battery passed all three repetitions. The standard 15-cell
run_bench.sh grid completed without request errors. A controller readiness
request and JIT events overlapped c1/zero-context; that cell is excluded and
replaced in the comparison by an isolated same-boot run_bench.sh repeat with
no JIT. Both raw receipts remain intact. This is a composite comparison,
not a second independent full grid.

| Concurrency | R22 steps/s | Original R26 steps/s | Corrected R26 steps/s | Corrected vs R22 |
|---|---:|---:|---:|---:|
| 1 | 18.836 | 18.113 | 18.833 | -0.02% |
| 2 | 29.619 | 28.476 | 29.573 | -0.16% |
| 4 | 43.600 | 42.169 | 43.512 | -0.20% |

Prefill at 8K/16K/32K/64K/128K measured 2868/2963/2977/2945/2880 tokens/s,
between -0.07% and +1.19% of R22. The broad 3% to 4% loss is resolved by
removing the launcher overrides. The experiment identifies the override set
as the supported explanation; it does not isolate each knob's contribution.
No remaining ordinary-grid gap justifies splitting the overrides further.

Warm retrieval passed at 262000 tokens in 91.531 seconds and 1048000 tokens
in 477.142 seconds, with the same needle 739526 and 128-token output budget.
The native-context result is 5.0% faster than original R26's 502.199 seconds,
but remains 21.9% longer than R22's 391.478-second receipt. Across a conservative
468-second interior of the 1M request, software thermal slowdown (NVML 0x20)
was sampled 29 times on sparky and 24 on buddy, zero on rocky and lucky.
Minimum/maximum SM clocks were 2281/2444, 2333/2463, 2457/2502, and 2398/2418
MHz respectively. The remaining long-context difference is unresolved;
neither source causality nor thermal causality is established by these runs.

Startup allocation warnings also remain separate: sparky/buddy/rocky/lucky
recorded 47/20/0/57 NV_ERR_NO_MEMORY entries during loading or capture, with
none during the benchmark or native-context run. This boot reported 6172749
KV tokens. The service remained responsive and no container was OOM-killed.

The policy block is removed from the source launcher, and the canonical host
runner now mounts its reviewed companion launcher on every future start.
Both files are deployed on all four nodes. The running control retains the
same launcher bytes and image identity. Qwen serving was untouched. R26 is
left serving with the corrected environment; ordinary-grid parity is recovered,
while full native-context performance and startup allocation health remain open.

Receipts, commands, excluded harness attempts, telemetry, and full per-cell
comparisons are in the local archive:
[control-r22-env-20260906](/home/jugs/git/rtx6kpro-artifacts/20260906-pre-r27/spark/glm53/r26-spark/control-r22-env-20260906/README.md)
and [RESULTS.md](/home/jugs/git/rtx6kpro-artifacts/20260906-pre-r27/spark/glm53/r26-spark/control-r22-env-20260906/RESULTS.md).

### Cache-accounting correction and direct extension test

The 391.478-second R22 native-context receipt is a partial-prefix-hit
measurement and must not be used as a cache-equivalent latency baseline.
The archived R22 journal shows the prefix hit rate rising from zero to 16.3%
and a completion-window prompt rate consistent with about 790K fresh tokens.
The R26 requests show essentially full re-prefill. Thus the previously quoted
21.9% residual latency difference does not establish a prefill kernel regression.

The logger accumulates computed prompt tokens, excluding cached/transferred
tokens, and divides by actual monotonic elapsed time. Multiplying its rounded
rate by exactly ten seconds is only an estimate. In particular, 1008 blocks of
256 equal 258048 tokens, not 258127, and the control's ten-second estimate
1030813 differs from 1048000. Approximate fresh-token rates are informative,
but neither exact cache counts nor a controlled 9% kernel speedup follow from
those estimates and different cached-prefix lengths.

A direct live R26 test on 2026-09-06 used a fresh needle and verified the
tokenized common prefix. It recorded both full API usage and before/after
Prometheus counters:

| Request | Shared token-ID prefix | Freshly computed | Cache hits | Seconds |
|---|---:|---:|---:|---:|
| 131072 tokens | 0 | 131072 | 0 | 44.286 |
| 262144-token extension | 131049 | 262144 | 0 | 92.501 |

Both retrievals passed. API prompt_tokens_details was null, so cached_tokens
is recorded as unavailable, not zero; exact zero hits are established by
the prefix-cache and prompt-source counter deltas. Total prompt deltas equal
the two requests exactly, with no evidence of interfering inference traffic.
The general native-context harness now preserves the complete usage object
and optional cached_tokens field for future runs.

This confirms a long-prefix miss in the live R26 profile. It is not yet a
matched R22/R26 regression test. The shorter prompt's closing instructions
mean the shared prefix ends 23 tokens before its full length; Mamba checkpoint
retention and block boundaries must be considered when explaining the miss.
The cache-manager patch is relevant investigation material, but much of its
new pinning/offload logic concerns external connectors, disabled in this run.
No causal source hunk has been identified and no cache workaround is deployed.

Startup allocation warnings also occurred in the R22-era kernel journal,
so they are a separate loading/capture resource investigation, not established
R26 regressions. Brief thermal flags remain factual observations, not an
explanation for unequal cached work. Claims based on the mislabeled
/tmp/r22-*-reconstruct trees are withdrawn; use immutable image contents,
verified composition identities, patches, and changed-path manifests instead.

Direct receipts are in control-r22-env-20260906/prefix-extension.jsonl,
with complete counter snapshots and the historical journal excerpts beside it.

### Completed R22 twin, 2026-09-06

The requested fixed-token three-pair matrix is complete. Both R26 and R22
missed all extensions: a 131049-token unaligned shared prefix, a 131072-token
shared prefix aligned to the 2048-token recurrent boundary, and a 4073-token
short shared prefix. Exact cache and fresh-token counters agree. All twelve
answers were correct, with matching input IDs and the current checkpoint
revision pinned on both releases.

This does not establish an R26-specific regression or a long-only threshold.
It does not invalidate the older R22 partial-hit receipt, whose precise
conditions still need reproduction. See
[PREFIX-MATRIX.md](/home/jugs/git/rtx6kpro-artifacts/20260906-pre-r27/spark/glm53/r26-spark/control-r22-env-20260906/PREFIX-MATRIX.md) in the local archive for the controls,
limits, and complete receipts. No cache-policy or kernel changes were made.

## Repeated-predecessor follow-up, September 6

Both 4K/4K/8K triples missed completely, whether the first request stopped
normally or exhausted its 16-token budget. The subsequent R26
262000/262000/1048000 sequence reproduced the historical hit magnitude:
zero hits on both predecessors, then 258048 cached and 789952 freshly computed
tokens on the extension. The extension returned the correct needle and stopped
normally in 386.590 seconds. Exact cumulative metrics, not logger-rate
estimates, provide those counts.

This is not an R26-specific inability to reuse long prefixes. The tested
single-predecessor misses remain known behavior on both images, not a claim
about every conversational turn. Repetition, termination, exact length, and
retained-boundary effects are not yet isolated. No source or policy changes
were made and R26 remained serving. See
[PREFIX-TRIPLES.md](/home/jugs/git/rtx6kpro-artifacts/20260906-pre-r27/spark/glm53/r26-spark/control-r22-env-20260906/PREFIX-TRIPLES.md) in the local archive for all receipts
and the historical-replay limitations.

## Provenance note

The built image's recipe hash was captured before the post-build node runners
were pinned to the exact image ID and before this qualification record was
filled in. The executable image payload remains identified by its image ID,
source lock, tree hashes, patch hashes, native-artifact hashes, and build-gate
receipts. No rebuild is needed for documentation-only and runner-pin changes.
