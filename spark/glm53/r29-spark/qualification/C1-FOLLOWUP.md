# R29 c1 repeat and explicit clear_thinking sweep

2026-09-09. User authorized one unchanged c1 repeat followed by the full
standard grid with request chat_template_kwargs={"clear_thinking":true}.
No restart, image change or server-default change. Image verified running:
0b15723cb87646bb4628c5cd67aa2a7c879ac5a27105109ea5db06c30113dc0b.

## Unchanged c1 repeat

run_bench.sh v0.4.29, contexts 0/16K/32K/64K/128K, duration 30 seconds,
warmup 3 seconds, same client token limit 6412288. All five cells valid with
zero errors, capacity limitation, underfill or warmup timeout.

| Run | Output tok/s geo | Engine steps/s geo | Acceptance geo |
|---|---:|---:|---:|
| Historical R28 | 51.6912 | 19.9382 | 2.5926 |
| R29 initial | 49.8226 | 19.9209 | 2.5010 |
| R29 unchanged repeat | 51.4556 | 19.9010 | 2.5856 |

The initial 3.6% output deficit did not repeat at that magnitude: repeat output
is 0.46% below historical R28. Engine speed remains flat and acceptance recovers.
This is a same-boot repeat with fresh randomized benchmark prompts, not a
matched contemporary R28 control or proof of equality across boots.

Receipt in benchmark results campaign 2026-09-jj-r29-vs-r28:
20260909T164052-0400__jj-r29-aligned-sm121-tp4-dcp1-mtp3-native1m__r02.json.

## Explicit true arm, complete

Standard full 15-cell sweep started 16:47:04 EDT. Variant:
jj-r29-aligned-sm121-tp4-dcp1-mtp3-native1m-clear-thinking-true.

The live checkpoint template defaults clear_thinking to false. Its relevant
branch only clears previous assistant reasoning. build_messages in the standard
harness has no previous assistant reasoning, only an acknowledgement. A live
/tokenize control with that message structure yielded identical 40-token lists
for omitted kwargs and explicit true. Thus this sweep is an explicit-parameter
control and performance repeat, not a test of clearing reasoning-heavy histories.
No speed difference should be attributed to the setting without a rendered-input
difference. Hardware data automatically sampled on the benchmark client is not
four-node Spark telemetry and must not be cited as cluster power or temperature.

A positive tokenization control with prior inline <think> reasoning gives 57
tokens with false and 36 with true, confirming the parameter is honored. A
separate reasoning_content field in the /tokenize probe did not preserve that
reasoning (36 tokens under both settings), so that probe is not evidence about
chat-completions handling of that field.

All 15 cells passed with explicit true present in saved metadata and no errors,
underfill, capacity limits or warmup timeouts. Geometric means:

| Concurrency | Output tok/s | Engine steps/s | Acceptance |
|---|---:|---:|---:|
| 1 | 50.9404 | 19.9294 | 2.5560 |
| 2 | 80.9258 | 31.0573 | 2.6057 |
| 4 | 119.3576 | 45.0744 | 2.6480 |

Prefill 8K/16K/32K/64K/128K: 2747/2939/2953/2937/2860 tok/s.
Receipt: 20260909T164704-0400__jj-r29-aligned-sm121-tp4-dcp1-mtp3-native1m-clear-thinking-true__r01.json.

Conclusion: no demonstrated performance gain from explicit true on this workload.
The c1 deficit is not consistently reproduced at its original magnitude; engine
throughput is stable while output and acceptance vary. No server default changed.
No image restart, promotion, rollback or commit was performed.

Post-sweep exploratory smoke with enable_thinking=false in addition to true
returned the correct arithmetic but exposed a closing think tag in content.
That request is outside the measured profile and is not a clean formatting gate;
it must not be described as proof of non-thinking-mode correctness.
The subsequent exact-arm smoke (clear_thinking=true only) returned content 333,
reasoning in its separate field and finish_reason stop.
