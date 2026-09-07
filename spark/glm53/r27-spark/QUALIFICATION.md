# JJ r27 Spark qualification

Status: Qwen TP2 and GLM TP4 passed their requested batteries under `aligned` and remain running for testing; both auto profiles rejected for concurrent-serving behavior. GLM aligned control completed 2026-09-06 22:40 EDT; its runner default remains auto, so explicit aligned selection is required.

Composition date: 2026-09-06. Build date: 2026-09-06.

## Candidate identity

- Image ID:
  `ef669fa1cde3e99936c02575eca8f610990bb6c64dd1bbfcb04fd42dde87afae`
  (third build; launchers on `--load-format instanttensor` with no extra
  loader config. Superseded builds `371fbb60…` (instanttensor plus the removed
  `instanttensor_copy` option) and `3c3a2d39…` (fastsafetensors) differ only
  in the two launcher files and the recipe hash; their gate logs are in
  `build-receipts/`);
- image tag:
  `localhost/voipmonitor/vllm:glm53-jj-r27-spark-sm121-vllmf3c3fef-b12x95fdcb1-lmcachefe5442f-cu133-torch213-20260906-r1`;
- the qualified Qwen candidate is this image paired with the deployed host
  runner `run-qwen38-flash-next-jj-r27-spark-tp2-node.sh` (sha256
  `dd39ab5ffb7207ab10e05683d9d63ef8f12224852194e4617674140c15cc630c`), which
  passes `--recurrent-checkpoint-policy aligned` explicitly; the image alone
  boots the rejected `auto` profile;
- size 32,748,538,850 bytes; built on dusty from `~/git/bld-jj-r27-spark`
  with `ALLOW_DIRTY_BUILD=1`, `DOCKER_COMMIT=3e37145f`, and the explicit
  release tag (the directory is not yet committed).

Build gates passed on a physical GB10 (receipt
`build-receipts/build-dusty-20260906-r1-instanttensor.log`): source and package tree
identities before and after both refreshes, patch and manifest digests,
byte-identical preserved native products (ten vLLM `.abi3.so` objects, the SM121
FlashKDA object at `484f88de…`, vllm-rs, Triton kernels, the LMCache wheel
objects and cumem interposer), the PR 674 `tl.cast` markers, the
`recurrent_instruction_boundary` renderer marker, the SM121 draft-head overlay,
the B12X persistent-store and fill-elision invariants, RoCEnante API 1 and proxy
ABI 3 from the immutable cache, launcher dry runs with no policy exports, the
hollow-package preflights, and these GPU regressions: RoCE health 5 passed, PR
674 restore scalars 18 passed, r27 hybrid-cache regressions (partial prefix
hits, Mamba align chunk split, Mamba prefix state index) 130 passed, GLM draft
head mode and capability 6 passed, the real SM121 NVFP4 draft head at rows
1/4/32, FlashKDA near-collinear finiteness, and the B12X persistent-CTA epilogue
store test.

One build correction was needed and is part of the recipe: in r27, importing
`vllm.v1.worker.gpu_worker` reaches `boundary_checkpoint.py`, which evaluates
`tl.constexpr` at import time. Without a GPU the container disables Triton and
that import fails, so the RoCE health test now runs with the GPU device
(`build-receipts/build-dusty-20260906-attempt1-cpu-gate-failure.log`).

Locked source identities:

- vLLM r27 integration commit `63a82f8d323e8538cbe6f88ae1812a1c01577a0f`,
  tree `f54cd9ca2b9434727715197d32150b75e82a9ebf`;
- vLLM PR 674 head `04813092b20f76b71159b75672a4f0d8f97c0f9e`, pinned as an
  overlay;
- vLLM Spark tree `8881b7772e1644d63162ad3b7494888ef7ac54fd`, package subtree
  `f3c3fef548738807f7d1ef8157df63815c498607`;
- B12X r27 integration commit `e8ad299b174f16e2e8fb5879bea272f4efbb53f2`,
  tree `f3cd8a9eb00d3226a1acbbed1efedf10cc1c3e71`, package subtree
  `95fdcb1cfea380480b8882fa44055cfef358ddbb`;
- LMCache package subtree `fe5442fbf258accaa7f26d2bbb00d8b7b5c349ca`
  (unchanged, inherited);
- FlashKDA commit `3b225bf26bb8e218928a1fe14751cb48cf31d11b`, SM121 object
  digest `484f88deea08f7e07fec6d1c275560cccbeeeb9928fcb047de0b25b4135f3116`
  (unchanged, inherited).

Offline checks completed on the workstation:

- the r26 Spark overlay and PR 674 apply cleanly to the r27 integration tree;
- the generated refresh patch reproduces the r26 Spark tree from the r26
  integration tree plus overlay, then the r27 Spark tree from the r26 Spark
  tree (round trip verified by `git write-tree`);
- the vLLM refresh touches 157 paths, none native or build-system; the B12X
  refresh touches 63 paths within the r26 extension allow-list, including the
  new loader C sources;
- both r26 B12X build-gate invariants still hold in the r27 tree.

## Phase 1: Qwen TP2

Boot attempt 1 (image `371fbb60…`, 14:41 local, kirby worker then dusty head):
the head failed during weight loading with `ValueError: Unexpected extra
config keys for load format instanttensor: {'instanttensor_copy'}`. r27
removed that option (vLLM `6575b5ac8`). Both containers were stopped, worker
first.

Boot attempt 2 (image `3c3a2d39…` with `--load-format fastsafetensors` on
both launchers, 14:51 local): the pair initialized (PYNCCL tp:0, r27 version
string) and entered weight loading; within twenty minutes dusty stopped
answering ssh while still answering ping, kirby's worker exited with a
`ProcessGroupNCCL` watchdog timeout on a broadcast, and the driver's
completion probe never succeeded. The journals confirm memory exhaustion on
both ranks during fastsafetensors loading: dusty logged eight
`NVRM NV_ERR_NO_MEMORY` failures at 14:54:42 to 14:54:53 (two minutes into
loading), then `nv-monitor` hung-task reports and journald memory-pressure
flushes until the host was reset at about 16:57; kirby logged five
`NV_ERR_NO_MEMORY` failures at 14:52:49 while inside
`default_loader.get_all_weights`. The asymmetry is only in who died: kirby's
rank was terminated by the NCCL watchdog abort at 15:04, which freed its
memory, while dusty's rank 0 never completed its flight-recorder dump, could
not acquire the GIL to exit, and thrashed against the 16 GiB swap file for two
hours. Upstream saw the same class of failure on GB10 (`b7e3d0336`). dusty was
hard-reset by the operator.

Both launchers now keep `--load-format instanttensor` without the removed
option; r27's iterator opens InstantTensor with `copy=True` (owned tensors),
the same mode the r26 GLM launcher used on Spark, and this checkpoint's largest
tensors (1.271 GB `embed_tokens` and `lm_head`) fit the 1.28 GB buffer, so the
removed oversized-tensor fallback is not reached. This is supporting evidence,
not a loading qualification; lazy `--load-format safetensors` (upstream
`b7e3d0336`) is the fallback candidate if loading fails again.

Boot attempt 3 (image `ef669fa1…`, InstantTensor `copy=True`, 17:09 local):
loaded cleanly (98.5 GB target in 52 s at 2.0 GB/s, 6.16 GB MTP drafter,
no memory warnings), KV 5,143,024 tokens at 0.85 utilization (19.62x at
262,144; r26 reported 5,300,423), first correct completion 17:14:43. The pair
is serving on dusty/kirby with the r26 Qwen launcher environment.

### Correctness results (receipts in `qualification/qwen-20260906/`)

- semantic admission 18/18 across three repetitions (reasoning at low,
  medium, xhigh; non-thinking; vision; tool round trip);
- MTP3 retrieval exact at 2,848, 2,849, 131,072 (54.5 s), and 262,000
  (136.2 s) prompt tokens, 64-token budget, `finish_reason: stop`;
- four greedy 256-token responses, zero repeated 6-grams, all valid;
- fixed-token MTP3 acceptance 2.2547 on the frozen corpus (r26 2.1423,
  r22 2.2588);
- PR 667 transition probe: c1 2.436, c3 2.564 (predicted padded replay, 12
  tokens into the 16-token graph), c1 2.508, c2 2.554, c4 2.600, c1 2.429; no
  collapsed phase; coverage labeled as predicted, not observed;
- runtime JIT compiles during the run: 14, all first-shape (including
  `_restore_auxiliary_state_kernel`, which shows the endpoint-checkpoint
  restore path executing on a repeated prompt), none inside the timed cells
  except one `W4A16FusedMoeKernel` shape at 21:25 UTC during the first
  prefill scout.

### Standard benchmark (campaign `2026-09-jj-r27-vs-r26`)

| Concurrency | R26 steps/s geo | R27 steps/s geo | Change |
| ---: | ---: | ---: | ---: |
| 1 | 19.166 | 19.312 | +0.76% |
| 2 | 31.911 | 19.324 | -39.44% |
| 4 | 48.978 | 19.291 | -60.61% |

Prefill: 8K 2,850 (+6.7%), 16K 2,952 (+44.6%), 32K 2,860 (+4.3%), 64K 2,659
(+11.6%), 128K 2,347 (+4.9%) tok/s versus r26.

The c2 and c4 cells are not a decode regression. The server logged
`Running: 1, Waiting: 1|3` throughout them at 1 to 2% KV usage: r27's
`auto` recurrent checkpoint policy publishes an endpoint checkpoint for every
completed prompt, and a request whose entire prompt matches one takes the
`boundary_logits_only` path, which the scheduler runs as the only work in its
step (`scheduler.py` 1245 to 1250 and 1646 to 1651). The harness sends the
identical padded prompt to every concurrent client and reuses it across
cells, so every request after the first was an exact repeat and they were
admitted one at a time. Live probes (`probe-concurrency-*.py`,
`probe-concurrency-results.txt`) confirm the scope: four fresh distinct
prompts batch at 110.7 tok/s, four multi-turn extensions of completed prompts
batch at 118.4 tok/s, and only four verbatim repeats serialize (49.2 tok/s).

The same path causes head-of-line blocking (`probe-repeat-head-of-line.py`):
with two 600-token decodes running, a verbatim-repeat request submitted two
seconds later waited 14.7 s for the engine to drain, and two fresh requests
submitted one second after it waited 9.1 s and 12.1 s to first token instead
of under one second, because the repeat at the head of the waiting queue
breaks the admission loop. This is an r27 serving hazard for bursts of
identical prompts, independent of the harness, and is not reported upstream
as of 2026-09-06 (open PR #682 extends the same mechanism to GLM DCP/DFlash).

Pending decisions: which retention policy to qualify (`auto` as shipped, or
`--recurrent-checkpoint-policy aligned`, the r26-equivalent behavior, which
the runner can pass through `EXTRA_VLLM_ARGS` for a control), the c2/c4 grid
under that policy or under salted prompts, and the MTP0 boundary relaunch.

### Aligned-policy control (operator decision 2026-09-06 evening)

R27 stays on dusty/kirby as staging only; not promoted; GLM untouched. The
control keeps everything fixed except `--recurrent-checkpoint-policy aligned`,
passed through the Qwen runner's `EXTRA_VLLM_ARGS` (rendered in the dry run).

Step 1, aligned + MTP0 (boot 18:09 local, KV 6,414,538 tokens, 24.47x at
262,144): boundary retrieval exact at 2,784, 2,785, and 131,072 (51.3 s)
prompt tokens with a 64-token budget and `finish_reason: stop`
(`jj-r27-aligned-mtp0-boundary-native-context-20260906.jsonl`).

Prefix reuse for one 32,038-token prompt (`tests/probe-prefix-reuse.py`,
server prefix-cache counter deltas in tokens):

| Policy | Fresh TTFT | Extension reuse | Extension TTFT | Verbatim repeat reuse | Repeat TTFT |
| --- | ---: | ---: | ---: | ---: | ---: |
| auto, MTP3 | 10.88 s | 32,044 / 32,069 | 0.262 s | 32,038 / 32,038 | 0.102 s |
| aligned, MTP0 | 10.12 s | 30,624 / 32,069 | 0.600 s | 30,624 / 32,038 | 0.571 s |

`aligned` re-prefills the unaligned tail (about 1,400 tokens here) instead of
restoring the exact endpoint state; the capability cost is roughly 0.3 to
0.5 s per turn at this length against a 10 s cold prefill.

Step 2, aligned + MTP3 (boot 18:19 local, KV 5,245,700 tokens, 20.01x at
262,144; driver `tests/qualify-qwen38-r27-aligned-mtp3.sh`, exit 0, receipts
`jj-r27-aligned-mtp3-*` and `driver-aligned-mtp3.log`). The blocking behavior
is gone and correctness holds:

- identical-prompt burst: c4 identical 110.3 tok/s against c4 distinct 114.3
  and 114.0 tok/s (auto: identical serialized at 49.2);
- fresh / extension / verbatim-repeat phases: A 115.4, B 90.1, C 121.1, D
  (verbatim repeat of A) 119.1, E 116.8 tok/s; the repeat phase batches
  normally. Phase B (four multi-turn extensions) is slower than under auto
  (118.4) because each extension re-prefills the unaligned tail;
- head-of-line reproducer: every request reached first token in 0.3 s
  (repeat, fresh0, fresh1; auto: 14.7, 9.1, 12.1 s);
- semantic admission 18/18; MTP3 retrieval exact at 2,848, 2,849, 131,072
  (55.5 s), and 262,000 (84.2 s) prompt tokens;
- fixed-token MTP3 acceptance 2.1865 (auto 2.2547, r26 2.1423);
- PR 667 transition probe: c1 2.572, c3 2.581 (predicted padded replay), c1
  2.482, c2 2.506, c4 2.635, c1 2.572; no collapsed phase.

Prefix reuse under aligned + MTP3 (same 32,038-token probe): fresh 11.11 s,
extension 28,480 / 32,069 reused (TTFT 1.46 s), verbatim repeat 28,480 /
32,038 (1.36 s), extension repeat 28,480 (1.38 s). Reuse is coarser than
under aligned + MTP0 (30,624) and under auto (32,044); the retained
checkpoint granularity evidently depends on the speculative window, which
is not yet source-traced.

Weaker cache observations, kept separate because they are not a matched
A/B: the native-context probes build each prompt as prefix + filler +
question, so the 262,000-token prompt shares roughly 131K tokens with the
131,072-token prompt but diverges before that prompt's endpoint. In the
server's 10 s log windows the 262K request processed about 262K prompt tokens
under auto (cumulative hit rate 0.6%, 136.2 s) and about 134K under aligned
(cumulative hit rate 31.6%, 84.2 s). Cumulative hit-rate percentages cannot
establish per-request cached tokens, and the two runs were separate boots,
so this is suggestive only. A direct probe under aligned
(`probe-divergent-prefix.py`,
`jj-r27-divergent-prefix-aligned-mtp3-20260906.jsonl`) sent three requests
sharing a 77,5xx-token prefix with different questions: the first was a cold
prefill (0 reused, TTFT 31.5 s) and the second and third each reused 74,048
tokens (TTFT 1.9 s, about 3,500 re-prefilled tokens plus overhead). The same
probe has not been run under auto, so the divergent-prefix comparison across
policies stays open. Exact-endpoint reuse is the demonstrated auto benefit.

Standard grid, unchanged harness (`20260906T183107-0400__jj-r27-aligned-sm121-tp2-mtp3__r01.json`):

| Concurrency | R26 steps/s geo | R27 auto | R27 aligned | aligned vs R26 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 19.166 | 19.312 | 19.394 | +1.19% |
| 2 | 31.911 | 19.324 | 32.821 | +2.85% |
| 4 | 48.978 | 19.291 | 50.398 | +2.90% |

Geo tok/s: cc1 38.12 (r26 37.21), cc2 63.74 (63.60), cc4 101.86 (106.31;
acceptance variance, steps/s is the engine metric). Prefill scouts: 8K
2,913, 16K 1,972, 32K 2,712, 64K 2,529, 128K 2,251 tok/s (r26 2,670 /
2,042 / 2,741 / 2,383 / 2,238; the 16K scout is a single sample and swung
2,042 to 2,952 to 1,972 across the three runs).

Decision (operator, 2026-09-06): `aligned` is the Qwen default for this
image. It removes the measured blocking behavior, passes correctness, and
restores scaling with modest gains over r26 (c1 +1.1%, c2 +2.9%, c4 +2.9%,
single boot; not yet shown repeatable across boots). The Qwen launcher now
defaults to it, the build gate requires it, the runner test requires it in
every role, and the host runner passes the flag explicitly for image
`ef669fa1` (built before the launcher default) so a plain restart preserves
the profile; the `EXTRA_VLLM_ARGS` control hook is removed. GLM's policy is
untouched pending its own qualification. The aligned + MTP3 pair stays
serving on dusty/kirby as staging; not promoted. `auto` returns to
consideration once the exclusive boundary-step scheduling is fixed upstream
and retested.

## Phase 2: GLM TP4

Executed in the user-approved window, 2026-09-06 21:12 to 22:00 EDT, on
sparky/buddy/rocky/lucky. Exact image `ef669fa1...` is running on all four;
R26 stopped containers and image are retained as rollback. Qwen and DS4
were not changed. GLM's checkpoint policy stayed unchanged (`auto`).

**Verdict: correctness gates passed; concurrent-serving qualification failed.
Do not promote this GLM auto profile.** The driver exited zero because its
concurrency probe records behavior rather than asserting a latency threshold;
that exit status is not an overall qualification pass.

- Semantic admission: 15/15 across three runs, including reasoning, vision,
  and tool round trips. Native retrieval passed at 2048, 2049, 262000, and
  1048000 tokens. Frozen pairs and triples passed answer and counter-isolation
  checks. KV admission was 6,245,692 tokens for this boot.
- Identical C4 prompts serialize at 50.4 aggregate tok/s; fresh distinct
  probes reach 113.4 to 118.3. Repeating previously completed distinct
  prompts also serializes. The repeated head-of-line request has 13.2 s TTFT,
  and fresh requests behind it have 10.5 and 7.5 s TTFT.
- The unchanged 15-cell `run_bench.sh` pass completed with zero request errors.
  All ten C2/C4 cells timed out during readiness with effective concurrency
  one. Their observed rates are underfilled scheduler measurements, not
  valid full-concurrency kernel-throughput measurements.

| Requested concurrency | Corrected R26 steps/s geo | R27 auto observed steps/s geo | R27 output tok/s geo |
| --- | ---: | ---: | ---: |
| 1 | 18.833 | 19.670 | 52.20 |
| 2 | 29.573 | 19.729 (underfilled) | 47.73 |
| 4 | 43.512 | 19.832 (underfilled) | 49.89 |

The C1 increase is single-boot evidence only. Fresh-prefill client tok/s at
8K/16K/32K/64K/128K: 2827/2949/2961/2950/2842. Server fresh-token rates,
where available at 16K through 128K: 2968/2978/2966/2856.

Exact repeat reuse improves: the 262000-token repeat reuses all tokens and
completes in 1.48 s, versus 91.96 s with zero hits on corrected R26 using
the same corpus. Divergent-prefix pairs still miss. The long triple's 1M
extension takes 476.05 s with zero hits; R26 took 386.59 s with 258048 hits.
Those times do not compare identical amounts of prefill work.

Post-grid arithmetic completion and all-rank running/OOM checks passed.
The first/post arithmetic responses both contain a literal `</think>` in
content despite the probe's `enable_thinking=false`; they establish liveness
and the correct answer, not a clean non-thinking formatting contract. The
separate reasoning/vision/tool semantic battery passes its own checks.

Receipts and full caveats: [GLM window report](qualification/GLM-WINDOW.md),
raw logs and responses in `qualification/glm-20260906/`, and benchmark
campaign `2026-09-jj-r27-sm121-qualification` under `llm-inference-bench/results`.
At the end of this auto window, the aligned control was not yet authorized.
The subsequent approved control is recorded below; no automatic promotion
or rollback occurred.

## Phase 2b: matched GLM aligned control

Completed 2026-09-06 22:40:17 EDT on the same four nodes and exact image.
Actual argv receipts differ from auto only by the aligned flag/value.
Semantic 15/15, frozen cache pairs/triples, retrieval through 1048000 tokens,
all 15 benchmark cells, post-completion, and all-rank running/OOM checks passed.
Every benchmark cell achieved requested concurrency without warmup timeout or
request error. The driver exit status is supported by these individual gates.

| Concurrency | Corrected R26 steps/s | R27 aligned steps/s | Change | R27 aligned output tok/s |
| --- | ---: | ---: | ---: | ---: |
| 1 | 18.833 | 19.742 | +4.83% | 50.13 |
| 2 | 29.573 | 31.198 | +5.50% | 80.82 |
| 4 | 43.512 | 45.427 | +4.40% | 122.26 |

Single-boot comparisons, not demonstrated repeatable gains. Fresh-prefill
scouts remain near baseline overall; the 8K scout is about 2.9% lower.

Identical C4 requests now batch at 105.5 tok/s versus auto's 50.4. The repeated
head-of-line request reaches first token in 0.3 s and fresh arrivals in 0.6
to 0.7 s, versus auto's 13.2 and 7.5 to 10.5 s. Long divergent pairs recover
126976/129024 hits. The 1M triple extension recovers 258048 hits and completes
in 386.73 s, matching R26's same-hit 386.59 s, versus auto's 476.05 s and
zero hits. This matched A/B establishes the policy-dependent reuse difference
for these cases. Aligned's 262K exact repeat takes 2.93 s with 258048 hits
versus auto's 1.48 s with 262000. Short 4K repeats still miss under aligned.

At the end of this control R27 aligned remained serving for testing, without
changing defaults. The subsequently approved promotion makes aligned the
runner default without restarting the already-qualified serving processes.
R26 remains the rollback. Qwen and DS4 were untouched. See the
[promotion identity record](qualification/GLM-PROMOTION.md).
Startup allocation warnings and the non-thinking formatting caveat remain;
see [full aligned report and raw numbers](qualification/GLM-ALIGNED-WINDOW.md).
