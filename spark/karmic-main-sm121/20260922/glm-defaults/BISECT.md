# GLM default-profile performance investigation

User authorized investigation on September 24. Only the GLM cluster is in
scope. R38 remains the rollback, Qwen and DS4 are not changed. No new image,
driver, source patch, MTP0 arm or benchmark-repository edit is planned.

The completed default grid is the red signal: all 15 decode cells have lower
engine steps than saved R38; geometric means are down 6.2/9.0/6.0%. Acceptance
is also down 2.9/4.9/4.3%. The existing run_bench.sh full grid is the measurement
loop; a kernel microbenchmark cannot establish an end-to-end recovery here.
The minute-scale loop is intentional because serving initialization and MTP
state are part of the workload. Neither repeatability across boots nor a
single underlying cause has been established.

Ranked hypotheses and predictions:

1. NVFP4 draft-head logits reduce acceptance. BF16 should improve acceptance
   on comparable workloads; it can also change head latency, which is measured
   separately through steps. One stochastic grid is not a numerical proof.
2. L2 prefetch costs more than it saves on GB10. Disabling it should improve
   decode steps with much smaller effect on large-chunk prefill. Pinned source
   otherwise defaults it off for capability 12.1.
3. Persistent-grid work distribution loses load balancing. Materialized queues
   should improve MoE-dependent steps; the source calls that the production
   default. It can affect prefill as well as decode.
4. Scheduler cadence, graph envelope, or another source/foundation delta costs
   time per step. If the first three arms fail to recover steps, inspect these
   before adding unrelated kernel patches. Current evidence does not exclude
   scheduler or graph effects merely because all context cells are affected.

Claude's source review adds an explicit higher-priority engine-feature arm:
RecoverSSM is auto-enabled for this GLM MTP profile, with its engagement
confirmed in the candidate boot. Disable it alone to test the cost of the
recovery backend and eager state commits. Record KV changes because retaining
full speculative states can trade memory for latency. This is not MTP0.

Sampling is another confound: candidate explicitly forces top_p=0.95 while the
R38 checkpoint generation_config has top_p=null. A future neutral top-p arm
must remain separate from template changes. Neither different draft logits
nor sampling can be declared acceptance-only: both change executed kernels
and could affect step latency too. No causal attribution from static source.

Each initial arm is independent over the frozen default profile, not cumulative:

| Arm | Only changed engine setting |
| --- | --- |
| bf16-head | VLLM_GLM53_MTP_DRAFT_HEAD=bf16 |
| no-prefetch | VLLM_GLM53_L2_PREFETCH=0 |
| queue | B12X_DYNAMIC_WORK_SOURCE=materialized_queue |
| no-replayssm | --no-use-replayssm |

Prioritize no-replayssm, no-prefetch, then bf16-head and queue after review.
Do not bundle a large R38-shaped profile first: a recovery or residual loss
there would not prove that the remaining cause is purely profile or B12X.
The known SM121 overlay explains the draft-head architecture check difference;
it is not an unknown image/source mismatch.

`experiment.py` verifies the original profile digest and derives an effective
profile digest. The runner labels it, the launcher verifies it, and all four
boot receipts must match it. Explicit finite choices only; no arbitrary
environment passthrough. CLI/model/transport/compat library remain unchanged.
The shared JIT mount remains; B12X environment changes can change its cache
namespace and warmup must finish before measurements. No cache deletion.

After independent review, stop R38 workers then head gracefully, retaining
containers. Launch the first arm workers then head. Each arm must pass actual
completion, four-rank identity, short-pool, semantic 3x, and concurrent/multi-turn
checks before the unchanged 15-cell grid. Diagnostic screens are labeled
`screen`, not full qualification; they do not repeat 1M admission for every
single-knob arm. Any promising combined profile needs the full correctness,
native-context and grid battery before promotion. Keep per-rank logs, journals,
telemetry and KV capacity; reject timing contaminated by foreign traffic or JIT.

The screen also runs the frozen short and 131K/262K prefix matrix to exercise
recurrent caching and warm long-prefill shapes before timing. Start one bounded
`observe.sh` unit per node before each boot, record its exact arm-specific name,
and collect final container logs, kernel journals since boot, and observer logs
before stopping those exact units. No grid is accepted without that evidence.
Record actual KV capacity as a consequence of the arm, not a sufficient proof
of engagement by itself; the explicit flag and source path provide that proof.

The historical defaults receipts predate experiment records and cannot pass the
new `verify_boot.py`. They passed the then-current verifier and their actual
argv/environment remains retained. `compare_arms.py` explicitly accepts their
missing arm metadata only as the frozen default reference, not as a new arm.

Inspect outcomes before selecting further arms. Confirm a winning configuration
with a repeat and, when necessary for attribution, a return to the Karmic
default control. Do not infer a general engine regression from whole-profile
comparisons. Never time a failed correctness arm or silently relax a gate.
During an authorized investigation window, a cleanly stopped
failed arm can transition directly to the next reviewed independent arm without
an intervening R38 boot. Restore R38 if host safety is uncertain, no reviewed
next arm is ready, or the window ends. The first interim restoration was an
unnecessary extra model load; the user corrected that sequencing.
Stop and remove only the task's transient observer units.

## Execution

Claude cleared the arm kit after independent review. Added its requested
cross-host launcher/helper digest checks against workstation bytes. User
confirmed active GLM clients were paused. R38 stopped cleanly on all ranks,
workers before head, with exit 0 and no OOM kill; containers remain retained.

At 01:19:45 UTC September 24, started bounded telemetry units
`glm-bisect-no-replayssm-20260924.service` on all four hosts, then launched
the no-replayssm workers and head. Over 116 GiB MemAvailable per node before
launch. Screen receipts: `qualification/bisect-no-replayssm-20260924/`.
No other cluster was changed.

The no-replayssm arm passed boot identity and 7/7 short-pool checks, then failed
the first semantic tool-round-trip exact-output gate. It returned correct
arithmetic with extra text: `17 × 23 = 391\n\nFINAL: 391`. The existing gate
requires exactly `FINAL: 391`. No timing or long-prefix tests ran. This is a
failed screen, not proof of numerical corruption or a general RecoverSSM
correctness defect. Its failed receipt is retained and not superseded by a retry.

KV capacity was 6,507,086 tokens, versus 6,457,321 on the measured default boot.
The direction disproves the proposed use of KV decrease as an engagement test:
boot allocations vary. Engagement rests on the verified CLI and source path.
Kernel allocation warnings: sparky 5, buddy 8, rocky 10, lucky 39; no Xid/OOM
kill. Restoring R38 per the failure rule while Claude reviews the evidence.

R38 restoration completed with an exact real completion. Claude recommended
retaining the no-replayssm failure and moving to no-prefetch. The user corrected
the unnecessary interim restoration: proceed directly between reviewed arms
in this authorized window when safe.

The no-prefetch arm started at 01:30:38 UTC with arm-specific observer units.
The first driver invocation began before the head was created and failed its
container-exists check, with no inference. That preflight receipt is retained
at `qualification/bisect-no-prefetch-20260924/`; after all four launches finished,
the real screen began in `qualification/bisect-no-prefetch-20260924-run1/`.

No-prefetch passed the screen and all 15 benchmark cells. Relative to the
frozen Karmic defaults, geometric-mean steps improved 5.51/5.29/3.60 percent
at C1/C2/C4; output throughput improved 1.83/5.41/3.06 percent. Prefill was
within -0.69 to +1.45 percent. This is one boot, not a repeatability claim.
The raw receipt digest is
`85ef9324527f1b781aa6ea55bf4e2be67629c1a8bcfa096cc57ce2d836594113`.
Final container logs, kernel journals and observer output were retained before
stopping the exact arm-specific observers. No Xid, OOM kill or inference JIT
warning was found in those logs; startup allocation warnings remain recorded.

The next independent BF16-head arm was started directly after gracefully
stopping the no-prefetch containers, all exit 0 without OOM kill. Prefetch is
back on for this independent comparison. Its receipts are
`qualification/bisect-bf16-head-20260924-run1/`.

Local pinned-source follow-up while BF16 loads: the B12X tree at `10a553ef`
reads `B12X_DYNAMIC_WORK_SOURCE`, but a full `b12x/` search finds no reads of
`B12X_DYNAMIC_DIRECT_EXPERT_SCALES`, `B12X_DYNAMIC_SPLIT_ROUTE_COMPUTE` or
`B12X_DYNAMIC_SPLIT_COMPUTE_MAC`. Do not spend standalone boots attributing a
performance effect to those inherited launcher strings. Also, the vLLM
scheduler rejects `prefill_schedule_interval > 1` while compute-share is set;
an interval-only future arm would be invalid under this profile.

BF16-head passed the screen. An additional fixed-input acceptance probe uses
the existing acceptance runner with prompts freshly rendered for GLM once,
then frozen in `qualification/glm-fixed-token-corpus-20260924.json` (canonical
digest `39e15bb9ad6b2893692896e85f4e97b5924d5fa315bc51dd47f0a802cc02502c`).
At temperature zero, 3 waves of 4 requests and 512 output tokens measured
acceptance 2.3522. All four output hashes varied across waves. Thus this is a
fixed-input probe, not a fixed-output or deterministic acceptance oracle;
do not use it to eliminate token-stream differences. It is supplementary to
the standard grid, not a new correctness gate. No head comparison exists yet.

The BF16-head standard grid passed all 15 cells but is rejected for performance:
steps 18.243/27.741/41.438 at C1/C2/C4, down 4.86/4.53/4.22 percent from
Karmic defaults; output 46.427/71.955/106.410 tok/s. Aggregate acceptance
changed -0.13/+3.18/+0.33 percent and did not offset the step cost. Preserve
NVFP4 in subsequent arms. JIT warnings in this boot occurred during correctness
warmup before the grid; final logs and telemetry are retained alongside the
comparison. Next independent arm is materialized queue, with prefetch on and
NVFP4 head, not a cumulative combination yet.

The queue arm passed the screen with 6,583,150 KV tokens. No post-boot Xid,
NV_ERR_NO_MEMORY or OOM-kill appeared before timing. The same fixed-input probe
measured acceptance 2.3823 with non-repeatable output hashes, so its small
difference from BF16's 2.3522 is not evidence of a head-quality difference.
The standard grid is running in `qualification/bisect-queue-20260924-run1/`.

Additional hypothesis if a residual remains: Karmic explicitly sets
`B12X_MHC_PDL=1`. The retained R38 container does not set this variable; the
pinned B12X mHC module defaults it to zero. It adds dependent-launch signaling
and synchronization in decode partial kernels and changes prefill launch
attributes. Prediction: disabling this alone on the best measured profile
could improve steps if its launch/synchronization cost exceeds the overlap
benefit on GB10. No measurement yet and no assertion that it is harmful.

Queue grid completed: steps changed -0.19/-0.54/+0.30 percent against the
default at C1/C2/C4. This is not a demonstrated win; keep persistent_grid.
BF16 and queue have therefore not been combined with no-prefetch.

Next optimization is `no-prefetch-k5`: retain the measured no-prefetch winner,
NVFP4 head, RecoverSSM and request boundaries, change draft count from 3 to 5,
and add exact target graph sizes 6/12/18/24 for C1/C2/C3/C4. Keep the 32-request
and capture256 envelope, 1M context and 0.85 utilization. Pinned vLLM explicitly
supports recovery with 1 through 7 speculative tokens. Profile digest:
`e4bf32f3177b723066d6d2f2ccb25909bdc281b5d4e9170553e4621741143c9f`.
Seven contract tests pass normally and under -O; historical arm digests remain
unchanged. Benchmark metadata and variant now derive K from the effective argv.
Prediction: added accepted tokens may outweigh additional drafting/verification
cost. This is window tuning, not attribution of the R38-to-Karmic regression.
Across K3/K5, output throughput and latency are primary; raw step rates are not
like-for-like because the work per step changes. The mHC PDL hypothesis is
deferred behind this potentially larger optimization, not considered settled.

K5 passed the diagnostic screen: 7/7 short-pool and 15/15 semantic checks,
concurrency without head-of-line serialization, and the frozen prefix matrix.
The boot admitted 6,453,508 KV tokens (43.83 GiB on rank zero). Startup
NV_ERR_NO_MEMORY warnings are retained in the per-node pre-grid journals;
there was no Xid or OOM kill. The standard grid started at 04:29 UTC.
The exact experiment helper used for this boot is retained in its receipt
before preparing any further local arms. No updated helper is synced while
this arm is serving.

Two additional finite profiles are prepared locally, not yet executed:
`no-prefetch-k2` changes only K3 to K2 and adds exact graph sizes 3/6/9/12;
`no-prefetch-no-mhc-pdl` changes only B12X_MHC_PDL from 1 to 0 over the
no-prefetch K3 profile. Both keep the existing maximum graph and request
capacity. Nine contract tests pass normally and under -O. The K5 effective
profile remains byte-identical to its running receipt. K2 tests whether fewer
draft passes are a better trade on GB10; the PDL arm tests synchronization
cost without changing speculative acceptance policy. Neither is a measured
improvement or a source-level fix.

K5 grid completed at 04:41 UTC, all 15 cells valid. Geometric-mean output
47.739/68.295/100.130 tok/s at C1/C2/C4 is down 4.05/11.30/12.26 percent
against no-prefetch K3. Reject K5. Its larger acceptance does not pay for
the extra verification/draft work. No JIT warning fell inside the grid,
no Xid or OOM kill was found, and no clock event was sampled. Sparky had
eight movable-allocation/compaction stalls at 04:30:02/12/22 during the
overall benchmark window; do not label the entire window stall-free.
Minimum MemAvailable was 2.31/2.69/3.45/5.77 GiB on the four nodes.
Full logs, journals, telemetry and comparison are retained; old observer
units were stopped. All four containers exited 0 without OOM kill.

The next K3 arm, `no-prefetch-no-mhc-pdl`, started directly around 04:43 UTC.
Effective profile `b9a765710364d7ee49a5de264f8da03d626620db2d5da323c70ad785cf6b6fbb`
differs from no-prefetch K3 only in `B12X_MHC_PDL=0`. Nine runtime contract
tests passed on each node; the workstation's twelve tests include the new
offline telemetry summarizer. Bounded observers use the exact unit
`glm-bisect-no-prefetch-no-mhc-pdl-20260924.service`. No other cluster changed.

Additional source-backed hypothesis, not yet deployed: block verification
(`rejection_sample_method=block`) instead of standard token-by-token rejection.
Pinned vLLM `e77be225` routes it in the generic V2 rejection sampler without
a DSpark-only condition. It adds cumulative joint-ratio and residual-mass
kernels, so any acceptance gain must exceed their latency. Unlike synthetic
acceptance, the algorithm is intended to preserve the target distribution;
that is a claim to gate, not an excuse to accept arbitrary draft tokens.
The existing upstream tests cover distribution preservation (10 cases),
acceptance expectation (2 cases), placeholder truncation (6 cases, including
one standard-method control), and GLM-vocabulary int64 indexing (2 cases).
`run_sampler_gate.py` pins their bytes and the sampler implementation, rejects
skips/count drift, requires SM121 and the actual source-tree runtime, and runs
one pytest session per process. Run these only after stopping the GLM model,
never alongside serving; some cases allocate several GiB.
Two finite block arms are prepared over no-prefetch K3, with and without the
mHC PDL change. Select the parent only after its measurements. They change
only the rejection algorithm, not temperature, top-p, draft count, model,
context or capacity. Thirteen local tests pass normally and under -O.

mHC-PDL-off completed all 15 cells: 51.408/78.890/115.851 tok/s, versus
49.752/76.994/114.117 for no-prefetch K3. Engine-step changes are only
+0.21/+0.38/-0.33 percent. The token-rate difference is predominantly
acceptance variation; no demonstrated mHC performance win, so do not carry
the override into the next arm. No Xid, OOM kill or clock event; rocky had
one movable-allocation and two compaction stalls in the overall benchmark
window. Full telemetry and journals are retained. All containers stopped
cleanly, workers first, and their exact observers were stopped.

At approximately 05:06 UTC, began the sampler gate in a separate retained
container `glm-block-sampler-gate-20260924` on idle sparky. Preflight required
no running containers and at least 80 GiB MemAvailable. Exact same image,
network disabled, source-file digests verified, no host cache mount or model
weights. Log: `qualification/block-sampler-gate-20260924.log`. No serving
block-verification arm is authorized by a collection-only pass; all execution
counts must pass first. Chosen next parent is plain no-prefetch K3 with
the existing mHC PDL setting, not the neutral mHC-off variant.

Sampler GPU gate completed: 12 distribution/acceptance, 6 placeholder and
2 GLM-indexing cases passed, with exact collection counts and no skips.
Container exited 0 without OOM kill; its post-test kernel journal has no
NVRM/Xid/OOM lines. At approximately 05:08 UTC the four ranks launched
`no-prefetch-block`, effective profile
`897bcc4882f6cb3b894c063c2fa6ff9c2fe54ad18f5949968c5dd67d67068a6d`.
Ten runtime contract tests passed on every node before launch. The current
screen is in `qualification/bisect-no-prefetch-block-20260924-run1/`, including
snapshots of the runner, launcher, helper, profile, verifier and benchmark
driver. Benchmark metadata now explicitly records the rejection method;
the comparison rejects a mismatch rather than inferring it from the name.
Active bounded observers: `glm-bisect-no-prefetch-block-20260924.service`.

Block verification passed the original temperature-zero screen in full.
An additional temperature-one semantic run passed 14 checks, then its third
tool response violated exact formatting: `The result of 17 × 23 is 391.\n\nFINAL: 391`.
The tool call, arguments, result and finish reason were correct, but the
existing exact-format assertion failed. No benchmark was started. This is
not yet attributed to block verification, and the failed check is retained.

`probe_tool_format.py` repeats the unchanged tool gate under seeds 0 through
19 at temperature one, retaining every full synthetic request and response.
Its output is explicitly diagnostic, never a qualification PASS. Same initial
prompts/seeds are used across arms; generated tool messages can differ.
On block verification, 16/20 passed exact formatting; four added explanatory
text before the correct final answer. A standard-verification control is
now being booted to test whether this is shared sampled formatting behavior.
The original no-prefetch container identities were checked against their
receipts before renaming them with `-boot1-20260924`; no container was deleted.
Block containers are stopped and retained, with all observers stopped and
final logs/journals/telemetry collected. Current standard-control observers
are `glm-standard-format-control-20260924.service`.

The standard sampler returned 19/20 exact-format tool answers in that first
diagnostic, versus block's 16/20. Generated intermediate tool messages differ,
so these are not identical final-input comparisons. The standard sampler's
seed-10 final tool request is frozen in
`qualification/tool-response-corpus-20260924.json` (SHA-256
`6e2254a083ff446d215d169daffa5414a9519829afcb75561b173c39e17294ec`).
`probe_fixed_tool_response.py` replays that same final input with seeds 0-99
and a fresh cache salt per trial. Standard passed exact formatting 85/100;
all 15 misses retained the correct numeric result with additional text.
This is diagnostic distribution evidence, not a qualification pass.

The standard control's logs, journals and telemetry were saved under
`qualification/standard-format-control-20260924/`. All four containers exited
zero without OOM kill after worker-first shutdown; their observers stopped.
At about 05:35 UTC the retained block containers restarted after verifying
the image and both mounted helper digests. The new bounded observer is
`glm-block-format-return-20260924.service`; readiness receipts are under
`qualification/block-format-return-20260924/`. This return repeats the exact
100-request final-input diagnostic before deciding whether block verification
can proceed to performance screening. Qwen and DS4 remain untouched.

The first return verifier failed because `podman logs` includes both starts
of a retained container, while the identity gate requires exactly one startup
record. The driver now scopes logs to each inspected `State.StartedAt`; no
identity assertion was relaxed. The failed receipt remains, and a new
`block-format-return-20260924-verified` receipt passed all four boot identities.

Matched fixed-final-input result: standard 85/100 exact-format responses,
block 81/100. Every response retained the correct numeric result; misses
added text. This small sample does not establish a block-specific formatting
regression or prove distribution equivalence. The extra sampled-semantic
failure remains recorded, not converted to PASS. Together with the 20 pinned
GPU tests and original temperature-zero screen, this permits diagnostic timing
after a fresh unchanged screen, not promotion. The return screen is in
`qualification/bisect-no-prefetch-block-20260924-run2/`.

The returned block arm completed 15 valid benchmark cells at
`20260924T014524-0400__karmic-main-no-prefetch-block-sm121-tp4-dcp1-mtp3__r01.json`.
Geometric mean throughput is 51.606/78.348/115.831 tok/s (C1/C2/C4), versus
49.752/76.994/114.117 for the standard no-prefetch parent. Step changes are
-0.47/+0.35/+0.73 percent, and acceptance changes +4.22/+1.40/+0.77 percent.
The token-rate increase is not established repeatable and remains below R38
in all three groups. Do not promote or carry block verification as a proven win.

Health: no Xid or OOM kill; startup NV_ERR_NO_MEMORY counts are
40/39/7/59 on sparky/buddy/rocky/lucky. All observed JIT warnings precede the
05:45:24 UTC benchmark start. Buddy has 14 nonzero clock-event samples
(`0x4`) in the overall run: four in warmup prefill and ten in the 128K scout,
all before timed decode readiness. Thus the lower 128K prefill number is
not a clean sampler attribution. No direct reclaim or compaction stalls
occurred in the overall benchmark window. Minimum MemAvailable was
2.39/3.25/3.22/5.56 GiB. Boot KV was 6,550,740 tokens (rank-zero 44.21 GiB).
Logs, journals, comparison and telemetry summary are retained with the grid.

After saving evidence and stopping its exact observers, all block containers
stopped cleanly, workers first, with exit zero and no OOM kill. The finite
`no-prefetch-k2` arm is now booting with standard verification and matching
3/6/9/12 target graph sizes added. Ten runtime tests passed on every host.
Receipts: `qualification/bisect-no-prefetch-k2-20260924-run1/`; bounded unit
`glm-bisect-no-prefetch-k2-20260924.service`. This tests draft-depth cost without
bundling the unproven block change. No other cluster is touched.

K2 passed the unchanged complete screen, including all seven short cases,
15 semantic cases, concurrent multi-turn/repeat checks and six prefix requests.
The grid began at 06:08 UTC, but foreign client `192.168.2.33` resumed POSTs
at 06:14:45 UTC and continued through concurrent cells. Timestamped head logs
show three running requests during C2 and five during C4. The corresponding
drop in benchmark-client throughput cannot be attributed to K2.

At approximately 06:18 UTC, interrupted only the verified local benchmark
PID with SIGINT. Its partial JSON and resume file remain under
`20260924T020814-0400__karmic-main-no-prefetch-k2-sm121-tp4-dcp1-mtp2__r01`.
Thirteen cells were saved; the grid validator correctly rejected incompleteness.
This run is contaminated and NOT an accepted performance result. Do not resume
its saved cells into a clean run. The pre-interference C1 group was 50.37 tok/s;
there is no accepted full-grid K2 verdict.

Requested confirmation that clients are paused and the active turn is fully
finished before repeating measurements. Do not infer this from a quiet interval.
K2 containers remain serving and were not restarted or stopped; no foreign
request was cancelled. All four final container inspections, timestamped logs,
kernel journals and observer logs were retained in the K2 receipt directory.
Its exact observer units were stopped after collection. No Xid or OOM kill
appears in the retained kernel journals. Qwen and DS4 remain untouched.
Live experiments are blocked on that quiet-window confirmation. Next: a fresh
K2 grid, then bounded kernel profiling if the residual performance gap remains.

At 12:03 UTC September 24, resumed after the user explicitly confirmed clients
were paused and the active turn was finished. Metrics were idle. The original
K2 containers remain on the same boot and a fresh real completion plus four-rank
identity gate passed. `benchmark.sh` now allows an explicit prior correctness
receipt only when each current container ID and StartedAt match that receipt;
it always passes `--no-resume`. No correctness marker was manufactured and no
contaminated cell is reused. The new receipt is
`qualification/bisect-no-prefetch-k2-20260924-clean-repeat/`, with the original
screen linked in benchmark metadata. Observer unit on each GLM node:
`glm-k2-clean-repeat-20260924.service`. Thirteen local tests and shell syntax
checks pass. No engine/launcher, image, model or benchmark-repository change.

### Clean K2 result, September 24 12:16 UTC

The fresh grid completed all 15 cells. Aggregate geometric means C1/C2/C4:
50.618846 / 80.407504 / 117.998600 tok/s, versus the no-prefetch K3 parent's
49.751945 / 76.993999 / 114.117391. Gains are 1.74 / 4.43 / 3.40 percent in
this single complete run. K differs, so the higher step rates are not a
like-for-like engine speedup. This remains below the saved R38 record
53.634140 / 84.370850 / 123.047670 and is not promoted.

Prefill 8K/16K/32K/64K/128K: 2850 / 2939 / 2949 / 2922 / 2854 tok/s.
Health throughout 12:04:44 to 12:16:37 UTC: no foreign inference, Xid,
NV_ERR_NO_MEMORY, OOM kill, logged JIT warning or exception; no clock-event
samples, swap growth, direct reclaim or compaction stalls on any rank.
Minimum MemAvailable: 4.66 / 4.82 / 4.71 / 7.57 GiB in rank order.
Raw grid, comparison, validation and four-rank telemetry are retained in the
clean-repeat receipt. The contaminated r01 is not included in this verdict.

Next diagnostic is the bounded no-prefetch K3 profile in PROFILE-PLAN.md.
It adds only profiler configuration and does not change model, image,
sampling, graphs, autotuning, transport or retention. Instrumented throughput
will not be scored. The finite profile and its invariants pass 14 local tests.

### Bounded profiler result

Receipt: `qualification/profile-no-prefetch-k3-20260924/`. The unchanged
screen passed. All eight traces contain five worker forwards, zero prefill
tokens, and the intended C1 or C4 K3 batch. `profile-summary.json` excludes
synthetic GPU annotations, which otherwise double-count overlapping streams.
No profiler error, foreign inference, Xid or OOM kill was found. Allocation
warnings before capture: sparky 4, buddy 36, rocky 0, lucky 41. This is a
diagnostic boot, not qualification for promotion.

At C4, kernel-time shares agree across ranks: target dynamic MoE 46-47%,
BF16 WMMA GEMMs 32%, RoCEnante 6-7%, Marlin draft MoE 2.7%, GDN recovery
family 2.6-2.7%, FlashInfer draft head 1.2-1.3%. At C1, BF16 GEMMs are
39-46%, target MoE 27-31%, and the draft head 1.8-2.1%. C1 collective time
differs substantially between rank zero and the workers. Instrumentation
and rank synchronization are confounders; these are not uninstrumented
latencies or a causal explanation of the historical R38 difference.

The draft-head arm is still a useful cheap discriminator of both kernel and
acceptance differences, but direct head-kernel savings alone cannot plausibly
close the whole gap at C4. Larger opportunities are target MoE and dense
projections. The pinned unquantized dispatcher returns torch F.linear for
linear-backend b12x; its optional FlashInfer BF16 alternative is SM100-gated,
so it is not an available flag-only GB10 optimization.

Next finite arm: `no-prefetch-shared-head`, effective SHA
`ee59f222e8c05666ae38563b1b5e71735bac901d9064327b4b48322c809bfe5c`.
Relative to no-prefetch K3 it changes only VLLM_MTP_NVFP4_LM_HEAD to 1.
The verifier now distinguishes the runtime-quantized shared head from the
separate GLM draft copy. Fifteen local tests pass, including under -O.

### Shared-head result, September 24 13:00 UTC

The screen passed, followed by a complete 15-cell harness result at
`20260924T084841-0400__karmic-main-no-prefetch-shared-head-sm121-tp4-dcp1-mtp3__r01.json`.
C1/C2/C4 geometric-mean throughput: 49.957454 / 77.073682 / 115.811145 tok/s.
Engine steps: 20.140686 / 30.716223 / 44.912493, versus the no-prefetch parent
-0.45 / +0.41 / +0.20 percent. This does not show a meaningful engine gain or
close the historical R38 gap. Prefill: 2785 / 2911 / 2907 / 2904 / 2821 tok/s.
It is a single diagnostic grid, not promotion qualification.

Execution bookkeeping error: the local benchmark wrapper was edited to add the
future selective-quantization gate while its shell was still running. After the
unmodified harness saved the complete result, the shell resumed at a shifted
offset and exited 127 (`--metadata: command not found`). The standalone pinned
raw-grid validator and identity/protocol comparison were then run successfully.
No benchmark cell was rerun or altered and no completion marker was fabricated.
Do not edit an executing wrapper in place; wait for it to exit first. Exact
remote experiment/launch helpers are retained with the receipt.

Health caveat: no foreign inference, timed JIT warning, Xid, OOM kill or clock
event was found, but this was not a pressure-free run. During the timed window,
sparky lost 423.5 MiB SwapFree, with 34 movable allocation stalls and 202
compaction stalls; buddy/rocky recorded 2/1 allocation and compaction stalls.
Minimum MemAvailable in rank order was 3.03 / 2.62 / 2.93 / 5.65 GiB. The boot
also logged allocation warnings before timing: sparky 1, lucky 65, others zero.
These prevent attributing sub-percent differences to the head kernel alone.
The full telemetry and kernel journals are retained, not only container logs.

Next: the separately authorized 56-projection MXFP8/A16 arm described in
SELECTIVE-MXFP8.md, against no-prefetch K3, with full correctness and native-1M
qualification before timing. The shared-head result does not change that parent.

### Selective MXFP8/A16 first result, September 24 13:36 UTC

Full correctness including native 1M passed. The 15-cell grid completed normally
and validates. C1/C2/C4 output: 54.056 / 80.652 / 118.575 tok/s; steps:
21.323 / 31.771 / 46.068. Relative to the no-prefetch K3 parent, steps improve
5.39 / 3.85 / 2.78%, while prefill regresses 7.23 to 7.90%. No timed driver,
JIT, foreign-client, throttle, swap-growth or reclaim event was found. Startup
and early correctness allocation warnings remain in the record.

Claude reviewed the completed evidence and recommended repeating before changing
numerics again. A same-boot repeat is running from 13:40:50 UTC. The A16 large-M
path is a source-supported explanation for the prefill tradeoff, not a measured
kernel attribution. A potential explicit activation-quantized arm needs separate
authorization and full qualification; it has been proposed, not launched.

Review limits: auto precision is not inherently unqualifiable, but its selected
per-shape modes must be recorded. The earlier K2 gains are single-run evidence,
not established noise. A single shared-head grid does not prove acceptance is
unaffected by the head. Long retrieval timings are not cold-prefill comparisons
unless processed tokens match. No expert-target expansion is authorized.
