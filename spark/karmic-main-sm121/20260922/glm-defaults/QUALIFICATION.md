# GLM Karmic-main upstream-default profile

September 23-24, 2026. Correctness and standard benchmark complete.
Not recommended for promotion: decode is slower than the saved R38 profile.
Claude independently confirmed the arithmetic and clean timing. At the user's
request, R38 was restored and a real completion verified at 00:57 UTC.
The candidate is stopped and retained. See RESTORATION.md.

Image: `1a7a8acff71044e3b8bf97dacb7617cd1bdb50d0ab2fc2e35364a4d4dc165dcc`.
Model: local-inference-lab/GLM-5.3-Flash-NVFP4 at
`46aaae8a82032f77100f2f03e9cc11b391df3b4d`.
Profile: `78a82d4ec6e674df078e1564e3be73c0cfe5194bb361ea8be11c285863398703`.
Runner: `86c778bc8c64588b68c060bc959d27b5b5ec782c598eb26520c439ae003e2cb4`.
Launcher: `cc8c25cc5f40c13f75dbdd273ee6f166f210f0846e1396141ba27921b208bad8`.

TP4/DCP1, MTP3, request_boundaries, NVFP4 draft head, B12X KDA prefill,
32 sequences, capture256, FULL_AND_PIECEWISE, native context, utilization 0.85,
reasoning high and clear_thinking=false server defaults. LMCache disabled.
This is a whole-profile comparison with R38, not an engine-only A/B.

## Startup fix

The first launch failed because direct Python entry bypassed NGC shell
initialization and selected host libcuda 580.173.02 instead of the bundled
615.65.02 compatibility library. The same real Marlin repack operation failed
with CUDA launch status 222 under direct Python, passed under Bash, and passed
under direct Python with only the compat library search path added. All arms
disabled the CUDA disk cache. See EXECUTION.md and the diagnostic receipts.

The corrected runner enters through Bash and checks the exact loaded libcuda
and driver API before exec. No model option, image or host driver changed.
All four nodes passed the small repack probe and the actual boot passed the
formerly failing MTP conversion. Claude cleared the correction before retry.

## Correctness and behavior

- First completion: exact 333 with stop; four actual command/environment and
  driver-library receipts match the reviewed contract.
- Short pool boundaries: 7/7, lengths 128/129 and 2047-2051.
- Semantic/vision/tool battery: 15/15 over three repetitions. Exact tool
  response format passed; no weakening of the earlier beta gate.
- Concurrent identical, distinct, repeat and multi-turn requests completed.
  Identical c4 took 8.5 s versus distinct 8.3 s in the first pair. Fresh requests
  beside two long decodes had TTFT 0.4/0.5 s; no old head-of-line serialization.
- Frozen prefix matrix: 6/6 valid and isolated. Both long extensions reused
  126,976 tokens; the short extension with 4,073 shared tokens missed.
- Both short triples completed. Exact 4K repeats hit all 4,096 tokens; their
  different-suffix 8K extensions missed. Both initial requests stopped naturally,
  so the max_tokens=16 arm did not exercise actual truncation on this boot.
- Exact native retrieval: 2048, 2049, 262000 and 1048000 all passed with stop.
  262K elapsed 92.35 s; 1M elapsed 391.47 s. These are correctness receipts,
  not a fresh-token prefill comparison, since preceding prompts can share cache.

## Health and capacity

This boot admitted 6,457,321 KV tokens. Per-rank KV budgets: sparky 43.82 GiB,
buddy 43.50, rocky 43.96, lucky 43.97. Per-boot capacity is not throughput.

Before the grid: 92 NV_ERR_NO_MEMORY allocation warnings during startup
(sparky 22, buddy 13, rocky 15, lucky 42), plus two on sparky at first 131K
admission. No Xid or OOM kill. Sparky's sampled available memory reached about
2.6 GiB during the 1M request, with stable worker headroom. These warnings are
not resolved by the entrypoint fix and remain a qualification caveat.

Observers recorded clocks, memory, reclaim counters and buddyinfo through the
grid and were stopped after it. Final container and kernel logs are retained.
The timed window (00:26:51-00:38:42 UTC) has no JIT/engine warning, Xid, new
NVRM warning or OOM kill, and no foreign inference client. All POSTs came from
the benchmark workstation, 192.168.2.2. A post-grid completion returned 333
with stop. No clock-throttle flags were sampled during timing. Mean sampled
SM clocks were 2461/2483/2516/2431 MHz on sparky/buddy/rocky/lucky; minimum
MemAvailable was 3.12/5.16/4.02/5.94 GiB in that interval.

Thermal caveat outside the grid: the long-prefix/1M qualification interval had
software thermal slowdown on sparky, buddy and lucky, and five hardware
slowdown samples on sparky. The last event ended at 00:26:27 UTC, before the
benchmark. Those correctness timings are not thermally clean performance
measurements. The grid itself has zero sampled throttle flags.

## Performance

Unchanged run_bench.sh and llm_decode_bench.py hashes are checked before timing.
15 cells, c1/c2/c4, 30 seconds per cell, contexts 0/16K/32K/64K/128K, five
prefill scouts. Baseline is the saved September 15 R38 r04 grid; no new baseline
run. Benchmark-repository files are untouched. All 15 cells validated without
flags. Candidate raw JSON SHA-256:
`70e46729dfb7bc32ae0208d10685e57169d88c384db56e5ec3be2698b48db027`.

Geometric means across the five context cells:

| Concurrency | R38 tok/s | Candidate tok/s | Change | R38 steps/s | Candidate steps/s | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 53.63 | 48.86 | -8.90% | 20.44 | 19.17 | -6.18% |
| 2 | 84.37 | 73.05 | -13.42% | 31.92 | 29.06 | -8.96% |
| 4 | 123.05 | 110.73 | -10.01% | 46.00 | 43.26 | -5.95% |

Effective acceptance length also decreased by 2.90%, 4.90%, and 4.31%,
respectively. The output loss is therefore not just an acceptance change.
Every cell has lower engine steps than its saved R38 counterpart.

| Prefill context | R38 tok/s | Candidate tok/s | Change |
| --- | ---: | ---: | ---: |
| 8K | 2775 | 2759 | -0.58% |
| 16K | 2955 | 2903 | -1.76% |
| 32K | 2916 | 2913 | -0.10% |
| 64K | 2947 | 2917 | -1.02% |
| 128K | 2872 | 2843 | -1.01% |

This is one boot against a historical baseline, not a repeatability claim.
The compared profiles differ in retention, draft head, graph envelope and
other upstream defaults. No individual code change is attributed. The
32-sequence envelope is admitted, but throughput qualification stops at c4;
this is not a c32 load qualification. R38 containers remain retained.

Claude's final independent review completed after a long Herdr wait. It
recomputed all three geometric means exactly, verified 15 fully occupied cells,
29.9-30.0 seconds of measurement, no warmup timeout or capacity flag, and
client/server token agreement within a few tokens. No timing contamination was
found. Its recommendation is not to promote this slower profile. Possible
effects from prefetch/work-source and draft-head defaults remain hypotheses;
no further settings changes or experiment were authorized or made.

Receipts: `qualification/entrypoint-20260923/`. The earlier failed boot and R38
restoration remain under `qualification/window-20260923/`.
