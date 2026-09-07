# GLM R27 qualification window

User authorized staging and GLM cutover on 2026-09-06. Qwen and DS4 remain
untouched. R26 is retained as rollback. This is qualification, not promotion.

Initial live preflight found R26 serving on all four GLM nodes, with neither
the R27 image nor its node runner present. The existing 32749006848-byte
Docker archive on dusty is being copied unchanged over 10.11.11.0/24 using
aes128-gcm SSH and no compression. Source route is enp1s0f0np0, 10.11.11.7.
Receivers: sparky .1, buddy .2, rocky .4, lucky .3. Each checks the archive
checksum and exact image ID ef669fa1cde3e99936c02575eca8f610990bb6c64dd1bbfcb04fd42dde87afae.

Runner SHA256 ea3173e5b24e9a03bdd3e735893bea8d3154373c7b14bca71e6bd1ad2bf09382
and launcher SHA256 613a257b4648aa35f074a2d695eb7a1d3cc66fba7115207a76df2a352980cfaa
match on all four nodes. Model revision remains 46aaae8a, BF16 MTP3 head,
TP4/DCP1, FP8 KV, 1M context, 0.85 utilization, RoCEnante, LMCache disabled,
and no explicit recurrent-checkpoint policy (auto).

Pre-execution driver corrections: resolve the frozen-prefix helper from the
actual repository root; read container cmdline rather than host /proc/1;
create missing benchmark campaign metadata pointing to corrected R26.
Runner render tests and shell syntax checks passed. No serving policy changed.

All four archive checksums and image IDs verified. R26 stopped workers first,
then head, with approximately 122.5 million kB MemAvailable per node afterwards.
The stopped R26 containers and image remain available as rollback.

R27 started workers buddy/rocky/lucky, then sparky. Qualification started at
2026-09-06T21:12:27-04:00, with receipts under qualification/glm-20260906/.
The streaming probe also now ignores empty delta headers and rejects incomplete
thread results, so its TTFT means first nonempty generated text.

First correct completion: 2026-09-06T21:16:13-04:00. All-rank identity and
unchanged-policy checks passed. TP dispatch is B12X_ROCENANTE followed by PYNCCL.
KV admission: 6,245,692 tokens, logged 21:15:07 for this boot.
Semantic admission: 15/15 checks across three runs (reasoning, vision, tool round
trip).

The unchanged auto policy reproduces the Qwen scheduling hazard on GLM:
four identical requests complete serially at 50.4 aggregate output tok/s;
four fresh distinct requests reach 113.9, 118.3, and 113.4 tok/s in successive
probes. Repeating four previously completed distinct prompts also serializes
(56.8 tok/s). Multi-turn extensions batch at 94.9 tok/s in this probe.
These are workload probes, not the standard benchmark grid.

Head-of-line probe: two long requests start at 0.3 and 0.9 s TTFT. A repeated
prompt queued behind them gets 13.2 s TTFT; fresh requests queued after it
get 10.5 and 7.5 s. Attribution remains provisional, but the auto profile
has a demonstrated serving regression and is not suitable for promotion.
No policy change has been made during this window.

Frozen cache matrix: all six responses valid and counter-isolated. All three
extensions miss (shared prefixes 131049, 131072, and 4073 tokens), as on R22/R26.
The boundary-aligned pair aligns the shared prefix, not the whole base prompt
(131095 tokens), so it is not an exact request-endpoint continuation.
Both short triples reuse all 4096 tokens on the exact repeat, after either
budget exhaustion or normal stop. Both 8192-token divergent extensions miss.
This demonstrates repeat reuse, not a general divergent-prefix fix.

Long triple uses the exact corrected-R26 corpus digest 61afc9c8...:

| Stage | R26 seconds / hits | R27 seconds / hits |
| --- | --- | --- |
| 262000-token first, budget 16 | 90.97 / 0 | 91.52 / 0 |
| Identical repeat, budget 64 | 91.96 / 0 | 1.48 / 262000 |
| 1048000-token divergent extension | 386.59 / 258048 | 476.05 / 0 |

All R27 responses contained the needle; the repeat and extension stopped normally.
Each request passed counter isolation. The extension time difference includes
different processed-token counts and is not evidence of a kernel slowdown.
R27 does not retain the same divergent-prefix hit seen after R26's recomputed
second predecessor. Exact repeats improve substantially, with the scheduling
hazard described above.

Native retrieval passed at 2048, 2049, 262000, and 1048000 tokens, all stopping
normally. The native gate repeats
the same 262K/1M prompts from this triple, so its times must not be called fresh
prefill measurements.

Standard run_bench.sh campaign started 21:33:15. C1 completed with no errors or
warmup timeouts: steps/s by 0/16K/32K/64K/128K are 19.8403, 19.8548, 19.8600,
19.5055, 19.2960 (geomean 19.6700). Corrected-R26 C1 geomean is 18.833, so
this is a single-boot increase, not a demonstrated repeatable gain.
The first C2 cell reports effective_concurrency=1 and warmup_timed_out=true.
Do not interpret the underfilled auto-policy cells as isolated kernel speed.
The full grid finished at 22:00:20 EDT. Every C2/C4 cell is underfilled at
effective concurrency one and reports a warmup timeout. All 15 cells have zero
request errors. Keep these receipts as evidence
of the unchanged-policy failure; do not replace them with salted-prompt results.

Next decision after this window: a matched GLM aligned-policy control would test
whether Qwen's workaround transfers, using the same image/checkpoint/head and
the same concurrency probes and grid. This is not yet run or applied to GLM.

Telemetry caveat: the benchmark's automatic hardware_summary describes the
two-GPU workstation, not the four Sparks. Use the four node-specific GPU CSVs
saved alongside the driver for this deployment's clock/thermal evidence.

Final engine steps/s geometric means: C1 19.66997, C2 19.72920, C4 19.83203.
Output tok/s geometric means: 52.20021, 47.72531, 49.89382. C2/C4 are
underfilled observations, not qualified concurrency results. The harness's
capacity_limited flag and printed KV-deficit legend are generic classifications:
the raw timeout reasons are running_reqs=1/2 and running_reqs=1/4, not a
measured KV deficit. The fresh/distinct probes demonstrate concurrent admission.

Result: `20260906T213315-0400__jj-r27-sm121-tp4-dcp1-mtp3-native1m__r01.json`.
The raw JSON is copied into `glm-20260906/llm-inference-bench/`; the run log
is `glm-20260906/benchmark.log`.
No new JIT warning was logged during the timed grid; last head warning was
21:18:20, before the 21:33:15 benchmark start. Kernel extracts show five
NV_ERR_NO_MEMORY lines on rocky and one on buddy during startup profiling,
with no serving-time matches, Xid, or OOM-kill in the captured window.

Final post-grid answer contains 333 and finish_reason=stop. Both first and last
readiness replies also contain a literal </think> in content, so the weak
readiness check must not be treated as a non-thinking formatting gate.
The separate semantic battery passed all of its reasoning/vision/tool tests.
All four containers report running=true and oom=false. Driver exit zero means
the test sequence completed, not that the demonstrated scheduler hazard passed.
R27 remains running for testing under auto; R26 is retained, not restarted.
