# R38 Spark rollout, 2026-09-15

Status: in progress. The user authorized the usual complete rollout:
Qwen on dusty/kirby first, then GLM on sparky/buddy/rocky/lucky if Qwen passes.
Leave successful candidates serving. Subsequent explicit direction keeps
Qwen on R32, qualifies GLM independently, and adds DSv4 Vision Exp on
rusty/toby as a separate qualification. Nous remains out of scope.

## Frozen candidate

- Image: localhost/voipmonitor/vllm:jj-r38-spark-sm121
- Image ID: ea031e1d3d051033f077fc986bf6f8fce04cf9ab52483d5a719ba13114567fc5
- Source lock SHA256: 9a0dc0bb8eac7d3ccf5778765146651d35982810d8fb1ed9e39832f855247be0
- Runtime receipt: build-receipts/runtime-20260915T163156Z-64815
- Build gates: 540 pytest cases passed, plus the recorded native and launcher checks.
  Seven selected groups lack exact-count enforcement, but their observed 210
  passing cases are present in this receipt. Tightening that future-build
  safeguard does not change the outcome of this build.
- Image size: 35,870,037,422 bytes; 214 layers.
- The original build kit is preserved as build-kit.tar and its SHA256 beside
  the runtime receipt. Rollout scripts added after BUILD-OK did not build this
  image and must not be retroactively described as its build inputs.

R32 rollback ID on all six nodes:
74e53e710bef141f6f68e722582569f9c6aa388bce405ad6f1423566a2300c9c.
Stop worker containers first, head last, with podman stop -t 60. Preserve
the R32 containers, images and all model/JIT caches.

## Distribution

Docker archive saved on rusty, then copied on switched 200G addresses only.
No compression, OCI conversion, public-NIC bulk transfer or cache cleanup.
Rusty's SSH identity did not authenticate to kirby; the verified established
dusty-to-kirby path is used for that second leg.

Docker-archive load preserved the full image ID, config, history, rootfs,
layer digests and layer sizes. Podman changed layer descriptor media types
from diff.tar to diff.tar.gzip without changing layer bytes, which changed
the manifest digest. Both remain Docker v2 manifests. Raw manifest receipts
are retained, and comparison permits only this measured media-type alias.
No image conversion or content rebuild was performed to address it.

## Serving and evidence

The qualified R32 settings are carried over without tuning: aligned recurrent
checkpoints, MTP3, utilization 0.85, InstantTensor BUFFERED, LMCache disabled.
Qwen is TP2 with its short public alias Qwen3.8-Flash-Next. GLM is TP4/DCP1,
BF16 draft head, FlashKDA prefill, RoCEnante, and no clear_thinking override.

Use llm-inference-bench/run_bench.sh, unchanged harness SHA
2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3.
No benchmark-source edits or commits are part of this rollout.
Campaigns: 2026-09-jj-r38-vs-r32, separately under each model family.

Qwen requires first meaningful completion, semantic 18/18, native retrieval
through 262K, fixed-token acceptance, padded MTP transitions, concurrency and
head-of-line probes, the standard 15-cell grid, MTP0 boundaries, then a second
MTP3 correctness battery and grid. GLM requires short-pool and semantic tests,
concurrency, the frozen prefix pairs/triples, native retrieval through 1M
and the standard grid. Record boot KV, acceptance, JIT and clocks separately.

Historical clean R32 baselines:

- Qwen: 20260910T120516-0400, repetition 02 in 2026-09-jj-r32-vs-r29.
- GLM: 20260910T160142-0400, repetition 02 in 2026-09-jj-r32-vs-r29.
  Do not use the earlier GLM grid contaminated by unrelated client traffic.

The comparator accepts only the documented Qwen alias rename while enforcing
the same checkpoint, harness and checkpoint policy. Raw results stay immutable.
Engine steps, output throughput and acceptance are separate metrics.
Historical or single-boot differences alone are not repeatable speedups.

## Timeline

- 12:40:36 EDT: runtime BUILD-OK.
- 13:15 EDT: Docker archive save completed on rusty.
- 13:29:17 EDT: kirby's copy, load and image checks completed.
- 13:29:30 EDT: authorized Qwen cutover began.
- 13:29:53 EDT: R38 head and worker started; first-boot qualification active.

Results will be appended as the live gates complete.

### Qwen first R38 boot

Correct first completion at 13:34:57 EDT, about five minutes after launch.
The actual container arguments match the previous R32 container after only
the release-path substitution. Environment differences are cache/provenance
identities and assertion/logging settings, not serving-profile tuning.

- Semantic: 18/18, including thinking levels, non-thinking, image and tool cases.
- Retrieval: exact 739184 at 2,848, 2,849, 131,072 and 262,000 input tokens;
  every response stopped normally. These are correctness probes, not clean
  prefill-speed comparisons.
- Fixed-token acceptance: 2.273871 on the unchanged corpus, 12 requests and
  6,144 output tokens. Outputs are not bit-repeatable across waves; this is
  explicitly recorded, as in prior qualifications.
- Padded transition phases C1/C3/C1/C2/C4/C1: acceptance 2.500, 2.686, 2.508,
  2.627, 2.618, 2.523. Coverage remains predicted from batch shape rather
  than asserted from an instrumented graph-dispatch trace.
- Identical/distinct C4 probe: 130.0 versus 126.0 output tok/s. No observed
  identical-request serialization in this aligned profile.
- Head-of-line probe: fresh maximum TTFT 0.292 s; repeat TTFT 0.259 s.
- Engine KV capacity at this boot: 5,135,502 tokens. Per-rank available KV:
  dusty 41.72 GiB, kirby 43.28 GiB. Utilization remains 0.85.
- Standard grid began at 13:41:59 EDT. Performance acceptance is pending
  completion and the matched second boot; no speedup is claimed here.

### Two R38 grids and the fresh-control decision

Both complete 15-cell grids passed structural checks: no request errors,
warmup timeouts, capacity-limited cells or underfilled cells.
Against historical R32, aggregate acceptance-normalized steps/s changed:

| R38 boot | C1 | C2 | C4 |
| --- | ---: | ---: | ---: |
| Initial | +0.11% | +0.21% | -0.09% |
| Second | -0.87% | -0.46% | +0.54% |

Output throughput changed +1.16/-5.40/+0.89% initially and
-1.15/-3.69/+1.50% on the second boot. Acceptance explains much of the
variable C2 output difference; fixed-token acceptance was 2.273871 and
2.266322, respectively.

The initial 16K prefill outlier (2101 tok/s) did not repeat (2904 tok/s).
No JIT warning was logged during that first timed grid; its cause is not
established. Second-boot prefill was 2989/2904/2823/2661/2403 tok/s from
8K through 128K, 3.1-4.0% below the five-day-old R32 baseline.

MTP0 retrieval passed at 2784, 2785 and 131072 with normal stops.
The second MTP3 boot passed the complete semantic/retrieval/padded battery;
KV capacity was 5,275,977 tokens (dusty 42.86 GiB, kirby 43.98 GiB).

At 14:23:34 EDT the initial chain completed successfully. A fresh R32
control followed immediately, with an R38 return arm queued after it.
Both use the same correctness warmup and standard benchmark sequence.
This comparison is needed before attributing the smaller prefill difference
to the release. GLM cutover is still pending that verdict.

### Fresh R32 control and R38 return

The fresh R32 control completed at 14:46:55 EDT with all 15 cells and five
prefill scouts present, no request errors, capacity flags, underfill or warmup
timeouts. Its receipt SHA256 is
f034cc9732795af8bb4225cb1e9d96cbd006a3b37ee8f1e8fd22c17b9c841889.
The preserved R38 containers restarted at 14:47 EDT for the return arm.

| Metric | Fresh R32 | R38 initial | R38 second |
| --- | ---: | ---: | ---: |
| C1 steps/s, geometric mean | 21.910 | 21.867 | 21.653 |
| C2 steps/s, geometric mean | 36.618 | 36.593 | 36.348 |
| C4 steps/s, geometric mean | 57.324 | 55.737 | 56.087 |
| 16K prefill tok/s | 2995 | 2101 | 2904 |
| 32K prefill tok/s | 2908 | 2849 | 2823 |
| 64K prefill tok/s | 2748 | 2678 | 2661 |
| 128K prefill tok/s | 2483 | 2424 | 2403 |

Fresh R32 fixed-corpus acceptance was 2.234995 and its engine KV capacity was
5,327,879 tokens. Semantic, retrieval and padded-transition checks passed.
Do not equate randomized-grid output differences with kernel throughput.

Whole-grid mean clocks were within 0.15% across the first two R38 arms and
R32 on each node. R32 had no nonzero clock-event samples. Those aggregate
samples do not exclude a brief per-cell effect, but show no sustained
R32 clock advantage. The return arm remains necessary.

One R32 JIT warning occurred at 14:39:52.982 for _topk_topp_kernel. The
corresponding C2 zero-context requests started at 14:39:52.679 and the
receipt records 5.558 seconds of cell warmup, placing this warning before
measurement. Retain the warning and warmup evidence rather than claiming
there was no JIT anywhere in the benchmark invocation. No source tuning
was applied between arms.

GLM remains on R32. A confirmed Qwen regression will not silently count as
a promotion pass. The user has been asked whether to hold R38 altogether
or qualify GLM independently while retaining Qwen on R32. The user chose
the latter. The R38 return arm still completes for evidence, then Qwen is
restored to R32. Rusty's direct switched-200G access to all four GLM nodes
was verified, allowing GLM distribution without Qwen-node I/O.

### GLM distribution and launcher permission correction

All four GLM nodes verified the R38 image by 15:02:39 EDT. Cutover began
at 15:02:57, preserving R32 containers. The first launch failed before vLLM:
the host-mounted GLM launcher was mode0644, shadowing the executable image
copy. The digest gate checked bytes but not the execute permission.
Every rank exited1 with `catatonit: failed to exec pid1: Permission denied`.
This is a rollout artifact error, not a model or kernel regression.

The waiting driver was stopped and its failed-start receipts preserved.
Mode0755 was restored to the same launcher bytes locally and on all four
nodes, and the existing failed containers were restarted worker-first.
The retry uses qualification/glm-20260915-retry, leaving the first receipts
unchanged. Both the pre-cutover checker and node runner now reject a
non-executable launcher. A two-case test failed on the old checker and
passes after the fix. No image rebuild or serving-setting change occurred.

### Qwen completed comparison and independent review

The return arm completed at 15:10:19 EDT, all 15 cells valid. Raw SHA256:
e76c6a1e62082a38f80f156b68cc862845fab5ab4bd04baf8dc3c841f74e0c9c.
Its semantic18/18 and all four retrieval cases passed; fixed-token
acceptance was 2.194286. Engine KV capacity was 5,333,520 tokens.

| Metric | Fresh R32 | R38 return | Change |
| --- | ---: | ---: | ---: |
| C1 steps/s, geometric mean | 21.910 | 21.730 | -0.82% |
| C2 steps/s, geometric mean | 36.618 | 36.242 | -1.03% |
| C4 steps/s, geometric mean | 57.324 | 56.461 | -1.51% |
| 16K prefill tok/s | 2995 | 2904 | -3.04% |
| 32K prefill tok/s | 2908 | 2839 | -2.37% |
| 64K prefill tok/s | 2748 | 2677 | -2.58% |
| 128K prefill tok/s | 2483 | 2422 | -2.46% |

Claude independently reviewed the first two R38 grids, historical and fresh
R32 controls, their launch profiles, timestamps and correctness receipts.
Arithmetic and comparability hold. The persistent 32K-128K prefill loss is
supported by both client and server timing, without a clock advantage for
R32. The return arm supports that finding. This does not identify a kernel
cause or prove a population-level confidence interval.

C4 regression is not established: fresh R32 is 2.75% faster than its
historical boot, while both earlier R38 boots are within -0.09/+0.54% of
historical R32. Return R38 is about1.21% above historical R32. Do not label
the fresh-control C4 percentage alone as a repeatable release regression.

The pinned 2c447f16 digest names llm_decode_bench.py, not its run_bench.sh
wrapper. The wrapper digest is
5c79b9760a2381b4b5233f5bbc8f1f279841b46f596dc718a127eaa8eea3e4f2.
Neither benchmark source file was changed or upgraded during this rollout;
the existing external-repository edits remain the user's work.

Claude also found eight Kirby SW-power-cap samples in R38 r02, none inside
a measured decode window, and two foreign GET /v1/models requests during
r01's 128K scout, but no foreign inference traffic. R38's host memory margin
was about1-2GiB lower in these boots; KV and MemAvailable are different
measurements and neither proves the source of that difference.

Per the user's explicit decision, the preserved R32 Qwen containers are
restored after the completed return arm. Correct first completion and both
image/state checks passed at 15:13:39 EDT. GLM qualification proceeds
independently. No Qwen R38 promotion is claimed.

Claude also reviewed the completed r03: all15cells, no structural flags,
JIT warnings, foreign clients or clock events. It confirms the long-prefill
cost under this profile with no kernel cause asserted. R03's C4 falls
inside the observed range of the two R32 boots; a material decode regression
is not established. Its smaller sub-1% cross-boot decode differences are
not promoted into a claim of equivalence or regression.

The corrected GLM launch reached first meaningful completion at15:12:34,
then passed seven short-pool cases and the three-repetition semantic battery.

### GLM correctness and cache probes

The identical C4 burst reached 121.8 output tok/s versus 115.3 and 112.8
for the surrounding distinct bursts. In the head-of-line test, repeat and
fresh arrivals reached first token within 0.4-0.7 seconds while the long
decodes remained active. These are liveness checks, not the throughput grid.

The frozen long-prefix extensions reused 126,976 tokens (unaligned) and
129,024 tokens (boundary-aligned). The short 4K-to-8K pair and both short
triples still missed, preserving the known aligned-policy behavior.
The 262K/262K/1M triple reused 258,048 tokens for the repeat and extension;
the extension completed with the correct needle in 387.222 seconds.
This is a partial-hit measurement, not cold full-context prefill timing.

Exact retrieval at 2,048, 2,049, 262,000 and 1,048,000 tokens passed with
normal stops. The latter two reuse prefixes seeded by the preceding probes,
so their 2.632/5.151-second durations demonstrate cached correctness, not
cold prefill speed. The standard fresh-prompt grid started at 15:26:28 EDT.

### GLM grid interrupted by independent Pi traffic

The first grid is not accepted. At C1/64K its receipt reports average and
maximum running requests of 2, with readiness `running_reqs=2/1`, rather
than the requested single request. The resulting 16.155 steps/s cannot be
used as an R38 performance result. The standard underfill/error flags do
not reject this opposite, overfilled condition.

The access log shows an additional POST from 192.168.2.2:58272 at
15:30:13.134, before the benchmark stream from :58284 at 15:30:14.185.
A read-only socket inventory identified :58272 as workstation Pi PID364980,
not the benchmark. No Pi process or request was stopped.
At 15:33:29 the qualification controller was stopped deliberately, leaving
all four R38 serving containers running. Its partial grid and all completed
correctness/cache receipts remain intact. No QUALIFICATION-PASS was emitted.
The user was asked to pause GLM inference before a separate quiet grid.
DSv4 qualification continues independently on rusty/toby.

The user confirmed GLM traffic paused. A separate quiet run started at
15:34:11 under jj-r38-glm-qualification-quiet.service, with repetition2
and receipts in qualification/glm-20260915-quiet. It rechecks native-context
retrieval and then runs the complete unchanged grid on the same serving boot.
The shared post-run validator now rejects average or maximum running-request
counts above the requested concurrency. Its new two-case test failed before
the check and passes after it; completed Qwen r03 and quiet R32 GLM controls
still validate. No llm-inference-bench source was changed.

That attempted quiet run was also interrupted, not accepted. The same Pi
process issued POSTs at 15:34:45.578 (:60686) and 15:35:07.883 (:50892);
the latter socket was independently mapped to Pi PID364980. Its 8K scout
was therefore contaminated (1506.772 tok/s), and no complete timing cell
had been saved when the controller was stopped at 15:36:59. The native
retrieval rechecks passed. A second request to pause/cancel the active Pi
turn was sent; no foreign process or serving container was stopped.

The user chose to let the active Pi turn finish. GLM timing remains paused
while DSv4 continues. Queue checks at 15:37:47 showed zero running/waiting
requests, with the last logged generation draining by 15:37:22; a quiet
interval is required before the next timing attempt.

At 15:39:03 the endpoint was still idle with no new inference after the
drained turn. A final fail-closed zero-running/zero-waiting check passed,
then jj-r38-glm-qualification-quiet-r03.service started at15:39:26 using
repetition3 and qualification/glm-20260915-quiet-r03. No model restart.
The first C1 timing cell records exactly one active request.

The quiet interval did not prove that Pi's agent turn was complete. The
same PID364980 resumed calls on :46306 at 15:44:14, 15:44:50 and15:45:25.
C1/128K then recorded average/max running requests of2 and15.041 steps/s,
so that cell is invalid. Its clean C1/64K predecessor recorded20.363 steps/s,
versus16.155 in the first contaminated attempt; neither contaminated value
is evidence of a release regression. The third timing controller was stopped
at15:46 after a socket check mapped :46306 to the same Pi process.

No further GLM timing attempt starts on an inferred idle interval. Explicit
confirmation that the active Pi turn has finished is required. All three
partial/contaminated timing receipts are excluded from the promotion
decision. Correctness/cache checks passed; a complete isolated performance
grid remains outstanding. R38 remains live for the user's ongoing turn,
not declared fully qualified. Qwen remains on R32. DSv4 Vision completed
qualification and remains on R38; see spark/ds4-vision/r38/QUALIFICATION.md.

The user explicitly confirmed the Pi turn fully finished. At 15:52:50 EDT,
the preflight verified zero running/waiting GLM requests and all four ranks
still on the expected healthy R38 image. The new isolated run uses
jj-r38-glm-qualification-quiet-r04.service, repetition4 and receipts under
qualification/glm-20260915-quiet-r04. No image or serving setting changed.

### GLM isolated qualification completed

The r04 controller emitted QUALIFICATION-PASS at 16:05:41 EDT on
2026-09-15. All 15 grid cells matched requested concurrency; no JIT warnings,
server errors or clock-event samples occurred in the benchmark window.
The post-grid completion returned exactly 333 with a normal stop, and all
four ranks remained running on the expected R38 image without OOM flags.

R32 to R38 engine-step geometric means are 20.102 to 20.438 at C1,
31.243 to 31.916 at C2, and 45.321 to 46.003 at C4. These are measured
improvements of 1.67%, 2.15% and 1.50%, not yet proven repeatable gains.
Fresh-prompt prefill scouts are within 1% of R32. The three interrupted
timing attempts remain excluded, with their raw receipts preserved.

R38 is qualified for this GLM serving profile and remains serving on
sparky/buddy/rocky/lucky. R32 remains the stopped rollback. Qwen remains
on R32 on dusty/kirby; DSv4 Vision remains qualified on R38 on rusty/toby.
See qualification/GLM-QUALIFICATION.md for the final result and provenance.

### Qwen investigation closed for now, 2026-09-16

The preserved original R38 pair subsequently passed a startup/correctness
replay. The explicit FlashInfer-prefill screen then booted but selected CUDA
decode and returned garbled output on three greedy arithmetic requests.
The correctness gate prevented all timing; no additional R38 control grid
was taken. This two-backend change does not isolate FlashInfer prefill as
the cause. The original R32 pair was restored and returned exactly 333 at
21:08:58 EDT on September 15. The user explicitly chose to stay on R32 and
pause further Qwen experiments and restarts. GLM and DSv4 were not touched.

See `qualification/stock-r38-replay-20260915/RESULTS.md` and
`qualification/flashinfer-direct-20260915/RESULTS.md` for the recorded
identities, failures, and recovery. Experimental scripts are retained as
historical reproducers, not as production launch recommendations.
