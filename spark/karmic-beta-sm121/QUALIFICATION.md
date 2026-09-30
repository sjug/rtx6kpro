# Karmic beta Spark qualification

Status: in progress, September 20/21, 2026. No model promoted yet.

Candidate `f30dc6d9a2a6f6fc0ac9ff8cddb04d9f631f4a87b48ca7cac69254802fe83233`
passed the build/native gates and was transferred to kirby using the unchanged
Docker archive over the switched 200G fabric. Both nodes verified the same ID.

## Qwen, initial MTP3 boot

Dusty/kirby began the candidate boot around 2026-09-21 00:49 UTC. R32 containers
and image remain retained. The existing serving contract is unchanged: TP2,
MTP3, aligned retention, 262,144 maximum length, four sequences, 4,096 batch
budget and 0.85 memory utilization. Both ranks select B12X GDN prefill and decode.

Passed before timing:

- First completion returned exactly `333` with a stop finish.
- Semantic battery: 18/18, including reasoning, non-thinking, vision and tools.
- Exact retrieval at 2,848, 2,849, 131,072 and 262,000 tokens, all stop finishes.
- Fixed-token acceptance battery and c1/c3/c1/c2/c4/c1 transition probe. Effective
  acceptance was 2.618 to 2.696 in the transition probe, with no collapsed phase.
  Padded-graph coverage remains predicted, not observed internal graph coverage.
- Prefix extension and repeat each reused 28,480 tokens and returned the expected
  answer. The cold 32K seed took 10.21 seconds, extensions/repeats 1.35 to 1.39.
- Identical and distinct c4 bursts both batched. These small functional probes
  are not the standard performance comparison.
- Head-of-line probe: fresh-request maximum TTFT 0.325 seconds, repeat 0.303.

The standard unchanged `run_bench.sh` sweep began at 2026-09-21 01:06:32 UTC.
Its receipts go to this repository's campaign directory. Neither benchmark
source nor its separate results repository is edited. The update prompt was
explicitly declined. Clock/memory sampling is recorded alongside the run.

### Capacity and health caveats

At 00:54:59 UTC the engine reported **5,266,950 effective KV tokens**, 20.09x
concurrency at 262,144 tokens. Rank-local available KV memory was 42.79 GiB on
dusty and 44.2 GiB on kirby. The harness prints 79,766,784 from raw blocks times
block size; this is not the hybrid model's effective capacity and must not be
used as such. Actual graph pools were 0.22/0.20 GiB, versus estimates 0.46/0.43.

Both nodes logged startup allocation-pressure NV_ERR_NO_MEMORY messages. They
remained responsive and completed admission, but these messages are not a clean
health verdict. Preserve and inspect the journals through sustained inference.
A layer-normalization JIT warning occurred during semantic warmup before timing.

The first grid completed at 01:19:45 UTC with 15/15 cells and no request errors,
underfilled cells or warmup timeouts. Geometric mean steps/s were 22.859, 36.623
and 55.839 at C1/C2/C4; output tok/s were 47.230, 72.971 and 116.820. Relative to
the September 15 R32 control, steps changed +4.33%, +0.01% and -2.59%. This is a
historical comparison, not a demonstrated repeatable release delta.

Dusty logged two identical `_memdescAllocInternal` NV_ERR_NO_MEMORY warnings at
01:16:05 UTC. The timestamp falls in C4/32K admission: the four streaming requests
began around 01:16:02, and the cell warmup lasted 12.152 seconds before measured
decode. It does not overlap that cell's steady-state measurement. The failed
allocation's owner and size are not available in these logs, so the cause is
unresolved and no runtime fix is claimed. No Xid or host OOM kill was observed.

The admission warnings coincide with a five-second host-memory change on dusty:
between 21:16:03 and 21:16:08 EDT, MemAvailable rose from 5,091,576 to 6,100,852
KiB (4.86 to 5.82 GiB, an increase of 0.96 GiB), while SwapFree fell from
13,318,396 to 13,157,732 KiB (157 MiB more swap occupied). This supports host
memory pressure/reclaim as a hypothesis. These gauges do not measure swap-out
traffic or identify the allocation, and do not prove the driver retried or
changed page size. The retained dusty journal contains 94 matching lines total:
92 before readiness and the two admission warnings, not 96 total.

Across the first timed sweep, 794 clock samples per node reported no active
throttle reason. Mean SM clocks were 2469 MHz (dusty) and 2479 MHz (kirby), with
maximum temperatures 79/85 C. Minimum MemAvailable was 4.73/6.54 GiB. Dusty's
SwapFree declined about 0.19 GiB; this alone does not identify the allocating
process or demonstrate swap traffic inside a particular measured cell.

Read-only journal comparison confirms the identical driver warning occurred in
the retained R32 boot on kirby: 11 lines at September 16, 21:33:46-48 EDT, after
the recorded 21:31:22 container start. Therefore the signature itself is not new
to Karmic. That historical startup observation does not explain the two Karmic
request-admission warnings or establish that all instances are harmless.

The user declined another R32 run and removed MTP0 from qualification. A second
unchanged Karmic MTP3 sweep began at 01:31:53 UTC, supervised by
`watch-qwen-repeat.py`, with per-cell completion timestamps, fresh telemetry and
post-run kernel journals. This tests same-boot repeatability and warning
recurrence, not boot-to-boot variance. There was no restart or configuration fix.

### Completed repeat

The repeat completed at 01:45:06 UTC. The watcher observed exit 0 and preserved
the kernel summaries immediately. All 15 cells passed the comparison validator
(no request errors, warmup timeouts, capacity limits or underfilled cells).
There were zero NVRM/Xid entries on either node during the repeat and no new
inference JIT warnings. The post-benchmark semantic check passed all six cases.
Node telemetry retained 797 samples each, no active throttle reasons, mean SM
clocks 2466/2484 MHz and maximum temperatures 83/85 C.

Geometric means across the five contexts:

| Metric | C1 | C2 | C4 |
| --- | ---: | ---: | ---: |
| First Karmic output tok/s | 47.23 | 72.97 | 116.82 |
| Repeat Karmic output tok/s | 46.44 | 75.84 | 120.46 |
| First Karmic steps/s | 22.859 | 36.623 | 55.839 |
| Repeat Karmic steps/s | 22.803 | 36.760 | 55.530 |
| Repeat steps vs first | -0.25% | +0.37% | -0.55% |
| Repeat steps vs September 15 R32 | +4.08% | +0.39% | -3.13% |

Repeat prefill tok/s at 8K/16K/32K/64K/128K: 3089, 2994, 2896, 2674, 2441.
Changes versus the same R32 control: +1.88%, -0.03%, -0.41%, -2.69%, -1.69%.
Same-boot engine performance is consistent within 0.6% across the two sweeps.
Output differences are larger because acceptance changes. The C4 deficit versus
that historical R32 control persists, but these are two samples from one Karmic
boot, not independent boot distributions. No broad performance upgrade is claimed.

The allocation warning did not recur, including at the previous C4/32K trigger.
It is retained as an unresolved, nonfatal admission observation, not described
as fixed. No configuration or kernel change was made to suppress it.

### Review hardening

Boot journals, historical R32 evidence and small build gate logs are now retained
in the kit. `build.lock.json` centralizes as-built IDs and wheel hashes with
consistency tests. Exact pass-count matching, complete future receipt input
hashing, NCCL host-flag checking, startup connection-error retry, restart recovery
guidance and cluster-window documentation were tightened. Distribution now
requires an idle receiver in one selected cluster, verifies disk headroom and
image identity, and only then deletes its transferred archive. The DS4-only
distribution subsequently completed on rusty and toby with exact image IDs.
All 20 local tests and shellcheck pass. Extra identity labels are future-build
changes only; the running image is unchanged. Apt pinning and independent B12X
component packaging remain documented candidate limitations, not silently closed.

## DS4 Vision on rusty/toby, 2026-09-20

Qualification execution completed at 22:41:06 EDT; controller exit 0 at
22:41:07. No commit or promotion was performed. Candidate image
`f30dc6d9a2a6f6fc0ac9ff8cddb04d9f631f4a87b48ca7cac69254802fe83233`
is still serving as `ds4-vision-karmic-tp2` on both nodes. The unchanged R38
containers remain stopped, intact, as rollback. Qwen and GLM were not interrupted.

The exact Docker archive was transferred over the switched 200G fabric, loaded
and ID-verified on each receiver before its local archive was removed. The
source archive remains on dusty. The CPU head/worker launcher/parser gate passed
before worker-first startup. Serving settings match the qualified R38 profile:
TP2/DCP1, DSpark K3, 524288 context, four sequences, 4096 batched tokens,
utilization 0.85, FP8 KV, PYNCCL, LL/Simple, max reasoning by default.

### Correctness and admission

- Real first completion: exact `333`, normal stop.
- Pre-grid text/reasoning, vision and vision continuation, tool round trip,
  mixed text/image C4, and post-grid repeat battery: all passed.
- Exact 16384-token retrieval: `739184`, normal stop, 9.156 seconds.
- Exact 524000-token retrieval: `739184`, normal stop, 406.655 seconds;
  16128 cached tokens. Historical R38 took 336.510 seconds with the same prefix
  reuse. Neither is a completely cold prefill measurement. This candidate took
  20.8% longer in this single historical comparison.
- Boot effective KV capacity: 1,175,814 tokens at the 524288 envelope, reported
  concurrency 2.24x and rank-zero available KV memory 11.31 GiB. Capacity is
  boot-specific, not derived from the benchmark's fixed admission budget.

### Standard unchanged benchmark

All 15 cells passed the completeness/error/capacity/warmup checks. Harness
`llm_decode_bench.py` SHA remains `2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3`.
`run_bench.sh` was used without modifying its repository or separate results
repository. Results are in this repository's Karmic DS4 campaign. Candidate raw
JSON SHA is `bba6fcecab9a07843063751bc8b19a699418b00dad6640be26309c4898733618`.
Protocol and image/checkpoint/harness identities are checked by
`ds4-vision/compare.py`; its full output is
`qualification/ds4-vision/comparison-r38.json`.

| Concurrency | R38 output tok/s | Karmic output tok/s | R38 steps/s | Karmic steps/s | Steps change |
|---|---:|---:|---:|---:|---:|
| 1 | 36.20 | 37.05 | 17.35 | 17.37 | +0.13% |
| 2 | 56.01 | 56.94 | 26.33 | 26.28 | -0.19% |
| 4 | 79.03 | 77.16 | 37.25 | 35.94 | -3.51% |

Geometric means across five contexts. Output changes are +2.35%, +1.66% and
-2.38%; effective acceptance changes are +2.22%, +1.86% and +1.18%. Output
throughput is not interchangeable with engine-step performance.

| Prefill context | R38 tok/s | Karmic tok/s | Change |
|---|---:|---:|---:|
| 8K | 2104 | 2082 | -1.05% |
| 16K | 2253 | 1955 | -13.23% |
| 32K | 2219 | 2099 | -5.41% |
| 64K | 2142 | 1896 | -11.48% |
| 128K | 2003 | 1645 | -17.87% |

This is one candidate boot versus a historical R38 receipt, not a matched
multi-boot attribution study. Correctness passed, but performance qualification
is not an unconditional upgrade: prefill and C4 deserve investigation before
promotion. No cause is asserted and no additional R38 run was requested.

### Retained health observations

Full boot-to-finish kernel journals, container logs, one-second GPU telemetry,
five-second memory/buddyinfo/vmstat telemetry and container inspect receipts are
under `qualification/ds4-vision/`. Both containers finished running, not
OOM-killed. No Xid or OOM-kill appeared. Toby logged 65 NV_ERR_NO_MEMORY lines
during weight loading; rusty logged none. These are retained warnings, not fixed.
Minimum observed MemAvailable was 5.47 GiB on rusty and 6.41 GiB on toby.

Each rank logged five admission JIT warnings, the last at 22:20:33, all before
the benchmark started at 22:27:21. None occurred during the grid. Transformers
also logged missing video-processor argument docstrings, not a serving exception.
No traceback occurred. GPU temperatures peaked at 88 C on rusty and 86 C on
toby. There were brief `0x20` clock-event samples during the 524K prefill on both
nodes; rusty also had four during the benchmark window (22:28:27, 22:28:40,
22:28:45 and 22:33:43), while toby had none during that window. Startup `0x4`
samples are retained separately in raw telemetry. Clock events prevent claiming
an entirely unthrottled run or attributing the prefill gap solely to software.

### Prepared-tail follow-up

The no-restart follow-up completed 15 requests at the five exact scout sizes,
three fresh prefixes each, with zero cached tokens and no preparation or JIT
warnings on either rank. Warm median TTFT at 16K/32K/64K/128K was
7.46/15.09/35.91/90.04 seconds, versus the original candidate's
8.26/15.29/33.73/77.59 seconds. Thus the long-context gap survives warm plans;
the first-use warning does not explain it away. Short-context recovery is not
uniform, and the preparation cost was not directly measured. Server counters
corroborate the timings. No new NVRM or Xid occurred. The full method, limitations
and receipts are in `qualification/ds4-vision/warm-prefill/RESULTS.md`.
Candidate configuration and containers remain unchanged; promotion remains held.

## Remaining work

September 21 fresh R32 return benchmark: all 15 cells passed. R32 prefill was
0.3% to 3.4% faster than the two Karmic receipts, while C1 engine steps were
3.2% to 3.5% lower and C2/C4 steps were within 0.5%. The historical R32 C4
advantage did not repeat and must not be treated as an established Karmic
regression. See `qualification/qwen-r32-return/RESULTS.md` for raw means,
comparisons and JIT caveats. R32 remains serving; this result made no deployment
change.

September 21 operational update: Qwen was rolled back to the retained qualified
R32 pair on dusty/kirby at the user's request. Real completion and canonical
alias checks passed; Karmic containers remain stopped and retained. See
`qualification/qwen-r32-restored-20260921/RESTORE.md`. Historical Karmic results
above describe the completed experiment, not the current Qwen serving image.

September 21 DS4 restoration: after the user rebooted rusty/toby, the retained
`ds4-vision-jj-r38-tp2` containers were started worker first, unchanged, using
image `ea031e1d3d051033f077fc986bf6f8fce04cf9ab52483d5a719ba13114567fc5`.
The canonical Vision alias and a real arithmetic completion (333, stop) passed.
KV capacity is 1,516,572 tokens. Rusty logged two NV_ERR_NO_MEMORY warnings
during startup at 10:36:44 EDT; no Xid was observed, and toby's kernel check was
clean. MemAvailable after restoration was 6.9 GiB on rusty and 7.8 GiB on toby.
Receipts are under `qualification/ds4-r38-restored-20260921/`. This restores
R38 serving, not Karmic promotion; Qwen and GLM were not touched.

Qwen correctness and two MTP3 sweeps are complete; promotion remains a decision
given the mixed historical comparison and retained warning caveat. MTP0 is
explicitly out of scope. Do not rerun R32. DS4 Vision correctness and its grid
are complete, with performance promotion held as described above. Qualify GLM
independently on sparky/buddy/rocky/lucky in its own authorized window.
Retain each model's existing qualified rollback. Run only one cluster window at
a time, keeping the other models serving. DS4.1 is not part of this work.

September 21 GLM attempt: Claude reviewed and identified stale GLM preflight
paths, repaired in serving-only image 9b23ca237881 (native bytes unchanged).
That image passed live preflight and native smoke but stalled during model
loading with sustained memory pressure and swap use. The attempt was stopped;
no model qualification or benchmark passed. R38 was restored on all four GLM
nodes with a correct completion at 11:07 EDT. See
`qualification/glm-20260921/EXECUTION.md`. Missing inherited expandable-segments
configuration is verified drift and a diagnostic lead, not an established cause.

September 21 allocator-only GLM retry: the same image with only
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` added loaded the main model
in 68.25 seconds and admitted 6,867,183 KV tokens. All seven short-pool checks
passed. Two full semantic batteries each passed 14 checks and failed the third
tool round trip on extra explanatory text before the exact required answer.
The arithmetic and tool arguments were correct; five isolated repetitions
passed. This is a retained exact-format qualification failure, not a waived
gate or a demonstrated numerical corruption. No concurrency, long-context or
benchmark tests ran. R38 restoration followed. Evidence is under
`qualification/glm-allocator-20260921/` and
`qualification/glm-allocator-confirmation-20260921/`; restoration evidence is
under `qualification/glm-r38-restored-after-allocator-20260921/`.

The successful boot supports the allocator setting as a sufficient change on
this trial, not a proven memory mechanism. DS4 Vision already exported that
setting inside its launcher, so this finding does not explain its prefill gap.
