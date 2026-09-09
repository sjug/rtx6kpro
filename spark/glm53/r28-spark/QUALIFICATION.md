# Qwen R28 Spark qualification

Status: **Qwen aligned MTP3 qualification passed**, 2026-09-08. The standard
grid, repeated semantic tests, retrieval, acceptance, concurrency and MTP0
controls are complete. Aligned MTP3 was restored and its final smoke passed;
both dusty/kirby containers were verified running at 16:32 EDT. GLM and nous
were untouched. No claim of R28 GLM qualification is made here.

The subsequent authorized GLM rollout is recorded separately in
[GLM qualification](qualification/glm-20260908/RESULTS.md), completed at
19:20 EDT on 2026-09-08. This document retains the Qwen qualification scope.

## Candidate and comparison

- R28 image `cd93d80b3f9547f70e1e4608cd42af7c4b0f2ff33d92d5e520913cc9bab2d8d1`.
- Runner SHA256 `d00f73937267f4d095d39aa065b7b66b9ba09714244c453ea139cee4a2b31dda`.
- Qwen3.8-Flash-Next-NVFP4-4p89 revision
  `c374e7e24b54f6cb0017d0c2e6d26823d2f2fb5d`.
- TP2, MTP3, aligned checkpoint policy, full/piecewise CUDA graphs, context
  262144, max sequences 4, batch budget 4096, utilization 0.85, FP8 KV.
- InstantTensor BUFFERED, LMCache off, PyNCCL on the existing private f1 rails,
  NCCL_PROTO=LL,Simple. No launcher tuning changes.
- Initial boot's reported KV capacity: 5194550 tokens. R27 aligned's historical
  boot reported 5245700; do not interpret the difference as a throughput delta.
- R27 baseline: `20260906T183107-0400__jj-r27-aligned-sm121-tp2-mtp3__r01.json`,
  SHA256 `93280d53cde465f3fc0fd5f4e3edd76c3a538981a6e7c3d630a41c425da58538`.
- Benchmark v0.4.29, checkout HEAD `b6ec758fdd93f0500a08b3d60b27a15fcba6d61a`,
  dirty metadata/timestamp additions preserved. Script SHA256
  `2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3`;
  run_bench.sh SHA256 `5c79b9760a2381b4b5233f5bbc8f1f279841b46f596dc718a127eaa8eea3e4f2`.

## Protocol

1. Existing aligned MTP3 boot: semantics x3, exact retrieval at 2848/2849/131072/
   262000, frozen-token acceptance x3, predicted padded transitions c1/c3/c1/c2/
   c4/c1, and prefix reuse.
2. Identical/distinct bursts, fresh/extension/repeat phases, and a head-of-line
   probe. The R28 streaming probe records only nonempty content as TTFT and
   propagates request failures, correcting the old R27 substring-based timer.
3. Standard unchanged 15-cell run_bench.sh grid and integrated prefill scouts,
   c1/c2/c4, compared with the historical R27 aligned grid. Single-boot evidence,
   not proof of repeatable gains. JIT warnings are reviewed against measured
   windows, not treated as failures merely for appearing during warmup.
4. Separate R28 auto MTP3 control: semantic smoke, both concurrency reproducers,
   and prefix reuse. No presumption that either R27 scheduling defect persists.
5. Separate aligned MTP0 control: exact retrieval at 2784/2785/131072 and prefix
   reuse. Restore aligned MTP3 and verify meaningful output, leaving it serving.

Readiness requires an actual correct completion within a bounded 20-minute
window, not /health or /v1/models alone. Worker-first graceful teardown and
worker-first startup apply to the pair. No model/JIT caches are cleared. Never
run podman system reset.

## Receipts and limitations

### Aligned MTP3 correctness completed, 15:59 EDT

- Semantic admission: 18/18.
- Exact needle retrieval: 2848, 2849, 131072, 262000 tokens, all returned
  `739184` with finish_reason=stop. Elapsed 1.648, 1.153, 50.211, 72.201 s;
  these are cache-state-dependent correctness timings.
- Frozen corpus SHA256 `cc9230856e9dab760645436993ec44669c3168e2edc3cfbc8a897c72de0ffff4`:
  acceptance 2.197425 versus historical R27 aligned 2.186477. All 12 requests
  produced 512 tokens. Output hashes vary across waves on both releases; this
  is not a bit-reproducibility pass or a newly demonstrated R28 difference.
- Predicted padded transition c1/c3/c1/c2/c4/c1: acceptance lengths 2.489,
  2.595, 2.460, 2.607, 2.628, 2.469. No collapsed phase; client overlaps matched
  each phase's concurrency. Runtime graph dispatch remains unobserved.
- 32K prefix probe: fresh 0 hits, extension/repeat/extension-repeat each 28480
  hits by isolated metric deltas. Correct answers in all four requests. TTFT
  10.351, 2.556, 1.291, 1.310 s; first extension includes any lazy-path overhead.
- Concurrent bursts: c4 distinct 130.7 tok/s, identical 127.0, distinct again
  127.6. Fresh/extension/repeat phases all batch normally.
- Corrected streaming HOL probe: repeat TTFT 0.318 s; fresh TTFT 0.296 and
  0.280 s while the two 600-token requests remained active. All streams
  completed their token budgets, with meaningful nonempty output.
- Standard grid ran 16:01:43 to 16:14:49 EDT: all 15 cells complete, zero
  errors, warmup timeouts, underfilled cells, or capacity-limited cells.

### Standard grid versus historical R27 aligned

Geometric means across the five contexts. Steps/s is the harness's aggregate
acceptance-normalized metric, not per-user output speed.

| Concurrency | R27 steps/s | R28 steps/s | Change | R27 output tok/s | R28 output tok/s |
|---|---:|---:|---:|---:|---:|
| 1 | 19.394 | 21.848 | +12.65% | 38.125 | 44.989 |
| 2 | 32.821 | 36.470 | +11.12% | 63.745 | 76.299 |
| 4 | 50.398 | 56.339 | +11.79% | 101.861 | 115.788 |

| Prefill context | R27 tok/s | R28 tok/s | Change |
|---|---:|---:|---:|
| 8K | 2913 | 3100 | +6.42% |
| 16K | 1972 | 2185 | +10.80% |
| 32K | 2712 | 2919 | +7.63% |
| 64K | 2529 | 2743 | +8.46% |
| 128K | 2251 | 2486 | +10.44% |

All 15 matched step-rate cells improved in this run. This is a historical,
single-boot comparison, not proof of repeatable gains or kernel attribution.
The R27 reference predates the user's dusty storage reset; no contemporary
R27 reboot was performed for this task.
Fresh randomized grid prompts differ between runs; fixed-token acceptance is
reported separately above. Prefill scouts are single samples.

The only observed inference JIT warning in the grid was `_topk_topp_kernel`
on both ranks at 20:07:47 UTC (16:07:47 EDT). C2/0K began at 16:07:45;
measurement began at the ready event at 16:07:51. The compile was in warmup,
not the measured window. Rank logs are preserved by the subsequent profile
switch. No runtime errors were observed.

Raw result: `20260908T160143-0400__jj-r28-aligned-sm121-tp2-mtp3__r01.json`,
SHA256 `6fafd72a251df984811c73724878feebac887f5a9fec75caef69442149e9194c`.
The derived, cell-by-cell comparison is `qualification/qwen-20260908/aligned-mtp3/grid-comparison.json`.
The harness's hardware summary is client-local and is not evidence of Spark
thermals. Its block-count-derived max_total_tokens is not the hybrid model's
effective KV capacity; use the engine-reported 5194550 above.

### Auto MTP3 scheduling control completed, 16:22 EDT

Worker-first restart began at 16:15:49 EDT; correct first completion and both
profile identity checks passed at 16:19:14. The first semantic attempt failed
the literal final-string requirement: the low-effort answer correctly computed
333 but ended in `**FINAL: 333**`. This is a recorded instruction-format failure,
not evidence of arithmetic corruption. The unmodified gate is repeated once;
both attempts are preserved, and a retry does not erase the first failure.
The first failure is in `auto-mtp3/control.log`; that attempt's JSONL is empty
because the existing semantic tool writes a response only after its checks pass.

The unchanged semantic retry passed 6/6. This is not a clean first-attempt
semantic pass for auto. The scheduling and cache probes completed:

- C4 distinct 126.0 tok/s, identical 124.6, distinct again 120.9. No whole-burst
  serialization in this probe. Fresh/extension/repeat phases: 119.5, 125.3,
  132.8, 129.4, 128.6 tok/s.
- Matched streaming HOL probe: repeat TTFT 0.198 s, fresh 0.447 and 0.423 s.
  Both 600-token requests remained active; all streams completed their budgets.
  Neither R27 scheduling symptom reproduced on this R28 auto boot. These are
  focused probes, not a soak or a complete auto-policy qualification.
- Same 32K prefix probe: fresh 0 cached tokens; extension 32044; exact repeat
  32038; extension repeat 32069, by isolated counter deltas. Answers correct
  in all four. TTFT 10.084, 0.274, 0.096, 0.096 s. Aligned MTP3 retained
  28480 in each warm request, so exact-boundary reuse is demonstrably better
  under auto for this sequence. No divergent-prefix or native-context auto
  qualification is claimed.
- Auto boot KV capacity: 5392945 tokens, logged 20:18:35 UTC. Different
  checkpoint policy and boot; not a controlled KV-capacity comparison.

The R27 report must not describe its two scheduling symptoms as demonstrated
on R28. No report is filed in this task. Aligned remains the serving contract;
auto would need its own full policy qualification before changing that default.

### Aligned MTP0 control completed, 16:27:51 EDT

- Correct first completion and both rank identity checks passed at 16:26:48.
- Exact needle retrieval at 2784, 2785 and 131072 tokens, all `739184`, all
  finish_reason=stop. Elapsed 1.885, 1.033 and 45.789 s, correctness timings
  rather than cold-prefill comparisons.
- Same 32K prefix probe: fresh 0 hits; extension, repeat and extension repeat
  each 30624 hits. TTFT 9.311, 0.577, 0.541, 0.565 s. All answers correct.
- KV 6448224 tokens, logged 20:26:05 UTC. This is the MTP0 control's capacity,
  not the profile being left serving.
- No runtime failure in the control.

### Final serving state, 16:32 EDT

- Aligned MTP3 restored on the same R28 image and unchanged runner hash.
  Correct first completion and both identity gates passed at 16:31:55 EDT.
- Final semantic smoke: 6/6, including all three reasoning levels,
  non-thinking arithmetic, image understanding, and tool round trip.
- Dusty container `8c3d2b4ca272b5e832c37248a5c264dd4dd66640b3ff644ed71ad38750c69743`;
  kirby `5cda3000faf0a0b855d3e21a361c85d6d58f216792abf5605fd4b6c0f58c4305`.
- Final boot KV capacity **5282370 tokens**, logged 20:31:21 UTC. The grid
  belongs to the initial 5194550-token boot; do not assign final capacity to
  that benchmark. No fixed KV-byte override was introduced.
- Model list contains exactly `Qwen3.8-Flash-Next-NVFP4-4p89`. Context 262144,
  max sequences 4, MTP3, aligned, NCCL_PROTO=LL,Simple verified from runtime.
- MemAvailable at 16:32:55/56 EDT: dusty 5990256640 bytes, kirby 8979841024.
  Swap used 4001792 and 344064 bytes respectively. No serving OOM or runtime
  exception was observed in the qualification runs.
- Final logs, inspect JSON, model list, runner hashes and host snapshots are
  under `qualification/qwen-20260908/aligned-mtp3-restored/`.
- Benchmark catalog rebuilt and checked: 209 results, zero warnings.

Verdict: leave R28 aligned MTP3 serving on dusty/kirby. No correctness or
performance regression was observed under the tested aligned profile. Gains
are single-boot observations. Auto's two scheduling controls now pass, but
that does not qualify its entire policy or change the serving default. Its
one stochastic strict-format failure is retained above. No image builds,
cache clearing, commits or upstream filings were performed. Normal serving
JIT cache writes occurred; only the named Qwen serving containers were
replaced for the controls.

Raw qualification receipts: `qualification/qwen-20260908/`, split by profile.
The private benchmark campaign is `2026-09-jj-r28-vs-r27` under
`llm-inference-bench/results/runs/qwen3.8-flash-next/nvfp4-4p89/`.
Retrieval elapsed times may include prefix hits and are not cold-prefill
benchmarks. Graph padding coverage is predicted from shapes, not observed
engine dispatch. Auto controls are not a serving-default promotion.
