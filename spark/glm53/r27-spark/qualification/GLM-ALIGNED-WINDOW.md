# GLM R27 matched aligned control

User authorized execution on 2026-09-06 after preparing and reviewing the kit.
Scope: sparky/buddy/rocky/lucky only. Qwen and DS4 unchanged. R26 remains
available as rollback; this is qualification, not automatic promotion.

Only intended serving change from the completed auto window:
`--recurrent-checkpoint-policy aligned` on all four ranks. Same image
`ef669fa1cde3e99936c02575eca8f610990bb6c64dd1bbfcb04fd42dde87afae`,
checkpoint revision `46aaae8a`, BF16 MTP3 head, TP4/DCP1, native 1M context,
0.85 utilization, RoCEnante, and LMCache disabled.

Runner SHA256 `8fae752b581eb414fac17469eec8cd1788b373b0320ec7bdcdd96446e9fe3698`
matches on all four nodes. Workstation driver SHA256
`15f3a4853e7fecfb7cdac1568a33340d13680e3db8ff3f31546cf25e5585f0c8`.
Runner tests, driver shell syntax, and campaign-copy comparison pass.
Same corpus, request order, probes, and run_bench.sh arguments as auto.

Worker-first teardown and launch completed. Prelaunch MemAvailable was about
122 million kB per node. Qualification started 2026-09-06T22:10:59-04:00;
raw receipts are under `glm-aligned-20260906/`.

First completion passed at 22:14:00 EDT. All-rank image/policy gates passed.
KV admission: 6,281,563 tokens for this boot, versus auto's 6,245,692.
Same TP dispatch list: B12X_ROCENANTE followed by PYNCCL.
Semantic admission: 15/15 over three runs, including reasoning, vision, tools.

Initial concurrency probes, aggregate output tok/s (not the standard grid):

| Probe | Auto | Aligned |
| --- | ---: | ---: |
| C4 identical | 50.4 | 105.5 |
| Repeat four previously completed distinct prompts | 56.8 | 113.2 |
| Four multi-turn extensions | 94.9 | 103.4 |

Aligned's fresh distinct probes reach 115.1 to 123.4 tok/s. The serialization
seen with auto is absent in these aligned probes.

Head-of-line TTFT: repeated request 0.3 s, fresh requests behind it 0.7 and
0.6 s (auto: 13.2, 10.5, 7.5 s respectively). The two long requests started
in 0.2 and 0.6 s. Thus the same-image policy control removes the measured
blocking as well as serialization.

Frozen pairs all pass correctness and counter isolation. Long-unaligned
extension hits 126976 tokens (49.68 s); long-aligned extension hits 129024
(49.10 s). Auto hits zero in both (92.22 and 92.04 s). The short divergent
extension still hits zero. Both 4K triples also hit zero for repeats and
extensions, versus auto's 4096-token exact-repeat hits. This is a measured
policy-dependent tradeoff, not a general all-prefix caching claim.

Long triple: first 262000-token request 91.26 s, zero hits, budget exhausted
as intended with the needle present. Exact repeat: 2.93 s, 258048 hits,
correct needle and normal stop. Auto's repeat was 1.48 s with 262000 hits;
corrected R26's was 91.96 s with no hits. Same input digest in all three.
The 1048000-token extension passed with normal stop in 386.73 s and 258048
hits. Auto took 476.05 s with zero hits; corrected R26 took 386.59 s with
258048 hits. This matched R27 policy A/B attributes the missing partial reuse
in this sequence to auto, without claiming a kernel-speed difference.
Native retrieval passed at 2048, 2049, 262000, and 1048000 tokens, all stopping
normally. These latter two are cache-warm repeats of the triple, not fresh
prefill performance measurements.

The standard run_bench.sh campaign ran 22:27:47 to 22:40:17 EDT, aligned
variant in the same campaign as auto. All 15 cells have zero request errors,
zero warmup timeouts, and effective concurrency equal to requested concurrency.
Post-benchmark completion passed; all four containers report running=true,
oom=false. The requested qualification battery passed. Aligned remains running
for testing; no production promotion or default-policy edit was performed.

## Performance

Geometric means across the five contexts. Corrected R26 uses the documented
clean C1/zero-context replacement, not its contaminated first cell. Raw files
are unchanged. Differences between boots remain possible; this is one boot
per arm, not a repeatability study.

| Concurrency | R26 steps/s | Aligned steps/s | Change | R26 output tok/s | Aligned output tok/s | Aligned acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 18.833 | 19.742 | +4.83% | 48.30 | 50.13 | 2.539 |
| 2 | 29.573 | 31.198 | +5.50% | 76.47 | 80.82 | 2.591 |
| 4 | 43.512 | 45.427 | +4.40% | 116.99 | 122.26 | 2.691 |

Per cell: **engine steps/s / aggregate output tok/s / acceptance length**.

| Context | C1 | C2 | C4 |
| --- | --- | --- | --- |
| 0 | 19.784 / 50.44 / 2.550 | 31.664 / 79.86 / 2.522 | 47.746 / 125.57 / 2.630 |
| 16K | 19.872 / 49.61 / 2.497 | 31.650 / 83.73 / 2.646 | 45.937 / 123.12 / 2.680 |
| 32K | 19.802 / 49.39 / 2.494 | 31.521 / 83.28 / 2.642 | 45.026 / 122.08 / 2.711 |
| 64K | 19.673 / 50.95 / 2.590 | 30.560 / 75.53 / 2.472 | 44.474 / 120.57 / 2.711 |
| 128K | 19.583 / 50.27 / 2.567 | 30.617 / 81.98 / 2.678 | 44.046 / 120.03 / 2.725 |

Fresh-prefill client tok/s at 8K/16K/32K/64K/128K:
2786/2974/2966/2958/2882. Corresponding server fresh-token rates at 16K
through 128K: 2994/2985/2975/2898. The 8K scout is about 2.9% below the
corrected-R26 single scout; do not infer a systematic prefill regression or
gain from these one-sample cells. Longer-context rates are near baseline.

## Evidence and caveats

- Actual command-line receipts match auto's token-for-token after removing
  the aligned flag/value from each of the two recorded surfaces (PID1 argv
  and Podman Args). Verified on all four ranks. Same exact image ID.
- Last logged JIT compilation on every rank was 22:17:56, before the grid.
- Kernel extracts contain startup NV_ERR_NO_MEMORY warnings only: sparky 24,
  buddy 23, rocky 18, lucky 33, all between 22:12:36 and 22:13:11. No matched
  serving-time warning, Xid, or OOM-kill in the captured window. These warnings
  are residual UMA-admission behavior, not evidence of a clean allocator path.
- Benchmark automatic hardware summaries describe the workstation; use the
  four remote GPU CSVs for this deployment's clocks and thermals.
- First/post arithmetic replies both contain the correct answer and stop
  normally, but also include a literal </think> in content with the probe's
  enable_thinking=false setting, as in auto. Non-thinking formatting remains
  unqualified; the separate reasoning/vision/tool semantic checks pass.
- Current runner default is still auto. Reproduce this passing profile with
  RECURRENT_CHECKPOINT_POLICY=aligned on all four nodes. Do not plain-restart
  and assume this control became the default. R26 remains available.

Raw result: `20260906T222747-0400__jj-r27-aligned-sm121-tp4-dcp1-mtp3-native1m__r01.json`
in the existing campaign, with a byte-identical copy under
`glm-aligned-20260906/llm-inference-bench/`. Driver, responses, cache metrics,
kernel extracts, and node telemetry are under `glm-aligned-20260906/`.
