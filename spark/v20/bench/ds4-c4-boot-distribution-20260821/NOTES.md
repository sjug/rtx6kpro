# DS4 r18p performance investigation

Dates: 2026-08-21 through 2026-08-22

Hosts: `dusty` (rank 0) and `kirby` (rank 1), TP2 over the two qualified
200 Gbit/s RoCE rails. Production on `rusty` and `toby` is out of scope and
must remain untouched.

## Question

Determine whether the apparent II r18p concurrency-four regression against
corrected GG r34 is repeatable across independent engine starts. If it is,
identify and optimize its source. If it is not, stop treating the single-boot
delta as a kernel regression and continue only with independently repeatable
performance opportunities.

## Prior evidence and reason for this study

The original contemporary single-boot comparison measured r18p at -4.09% at
concurrency four. The subsequent LL128 A/C/A produced 28.25, 31.00, and 31.76
steps/s geometric means across nominally equivalent r18p starts. The fresh
plain `LL,Simple` control was 12.4% above the earlier r18p baseline and 7.8%
above the single corrected-r34 result. LL128 was admitted but not selected at
the relevant collective sizes and is rejected as an optimization arm.

Therefore, the -4.09% target is not established. Engine start is the
experimental unit in this study.

## Frozen study contract

- Engine-start order: r18p/r34/r34/r18p/r18p/r34.
- Engine restarts only; no host reboot and no cache clearing.
- Identical images, model revision, A8 backend, K5 probabilistic DSpark,
  capture ladder, and RoCE configuration used by the qualified deployments.
- r18p retains its qualified exact-four NCCL contract. Corrected-r34 retains
  its qualified unpinned-channel contract. Both use `NCCL_PROTO=LL,Simple`.
- Two fixed-input c4 sweeps per start at contexts 0, 16K, 32K, 64K, and 128K.
  Each sweep has its own stable prompt identifier, repeated across every
  engine start, so the second sweep cannot reuse the first sweep's prefix.
- The benchmark's per-cell warmup runs before every measured window. On-disk
  JIT caches remain mounted and are never removed.
- Boot medians are compared for acceptance-normalized steps/s, output tok/s,
  acceptance length, and 64K prefill. The boot, not each cell or sweep, is the
  independent sample.
- Workstation `nvidia-smi` data are disabled in the benchmark. Spark telemetry
  is collected explicitly from both remote nodes before and after each sweep.

## Initial state

At 22:10 EDT, both staging nodes were healthy on r18p image ID
`445f9ac3196dd10a47d0d90441ba43a8417903ba029f59a1a9c7fbef8ecfa4a1`.
The active containers matched `NCCL_PROTO=LL,Simple`, exactly four minimum and
maximum channels, the qualified model revision, and the persistent Hugging
Face and `/cache` mounts. This existing engine start becomes r18p boot 1.

Both nodes also have corrected-r34 image ID
`276f00868134ed3116ffaf44db975f1b4d8803c7f528c8f772bdaae43506fbd6`
available locally.

## Results

The original c4 regression was not repeatable. Across three independent engine
starts per image, r18p A8 was 3.27% faster than corrected-r34 in
acceptance-normalized c4 steps/s, within 1% at c1, and at 64K prefill parity.
The only qualified alternative profile is static packed W4A16 for a deployment
known to remain near c4: it gains about 6% c4 steps/s but loses about 6.5% at
c1, 7.5% at prefill, and 5-6% of KV capacity. It is not a better general
default.

Ten follow-on policy and integration discriminators found no broad replacement
for stock A8. M32 tiling, vLLM-managed OMP lifecycle, smaller resident grids,
persistent-grid work scheduling, disabling materialized intermediates,
disabling shared-input reuse, source-native W4A16, and a source-format hybrid
were neutral, inapplicable, or slower. The only unresolved positive lead is a
narrow M=4-only W4A8 dynamic selector, worth about 3% for that individual MoE
shape in the single-GPU screen; its end-to-end relevance has not yet been
established.

### Execution log

#### r18p boot 1

The two fixed-input sweeps completed without request errors, warmup timeouts,
or capacity limits. Their geometric means and the resulting boot median were:

| Sample | c4 steps/s | c4 output tok/s | acceptance length | 64K prefill tok/s |
|---|---:|---:|---:|---:|
| Sweep 1 | 31.049 | 95.574 | 3.078 | 2,280 |
| Sweep 2 | 30.874 | 94.436 | 3.059 | 2,276 |
| Boot median | 30.961 | 95.005 | 3.068 | 2,278 |

The 0-context engine rates were 32.96 and 32.71 steps/s. Across the full
context ladder, the two sweep-level geometric means differed by only 0.57%.
Output tok/s varied more than engine steps/s because probabilistic K5 produced
different acceptance lengths even with fixed inputs; this is expected and is
kept separate from engine-rate attribution.

#### corrected-r34 boot 1

Both sweeps completed without errors, warmup timeouts, capacity limits, or
decode queueing.

| Sample | c4 steps/s | c4 output tok/s | acceptance length | 64K prefill tok/s |
|---|---:|---:|---:|---:|
| Sweep 1 | 30.004 | 87.539 | 2.918 | 2,245 |
| Sweep 2 | 30.000 | 90.074 | 3.002 | 2,253 |
| Boot median | 30.002 | 88.807 | 2.960 | 2,249 |

Within this first interleaved pair, r18p is +3.20% in acceptance-normalized
steps/s and +1.29% at 64K prefill. It is +6.98% in output tok/s, but about half
of that raw difference is explained by its +3.66% acceptance-length advantage.
This is one engine start per arm and is not yet an image-level conclusion.

#### corrected-r34 boot 2

The second independent r34 engine start also completed both sweeps without an
invalid cell.

| Sample | c4 steps/s | c4 output tok/s | acceptance length | 64K prefill tok/s |
|---|---:|---:|---:|---:|
| Sweep 1 | 30.050 | 90.255 | 3.003 | 2,278 |
| Sweep 2 | 29.812 | 88.880 | 2.981 | 2,277 |
| Boot median | 29.931 | 89.568 | 2.992 | 2,277.5 |

The individual zero-context cells moved down by roughly 4% versus r34 boot 1,
but the five-context boot geometric mean changed by only -0.24%. This is the
first direct evidence that the large single-cell boot spread is substantially
attenuated by the planned context-ladder geometric mean.

#### r18p boot 2

Both sweeps completed cleanly.

| Sample | c4 steps/s | c4 output tok/s | acceptance length | 64K prefill tok/s |
|---|---:|---:|---:|---:|
| Sweep 1 | 30.885 | 92.462 | 2.994 | 2,291 |
| Sweep 2 | 30.935 | 95.167 | 3.076 | 2,276 |
| Boot median | 30.910 | 93.815 | 3.035 | 2,283.5 |

r18p boot 2 is -0.17% from r18p boot 1 in steps/s. The two r18p boot
geometric means and the two corrected-r34 boot geometric means are therefore
already much tighter than the earlier single-cell boot spread. A third boot
per arm remains required by the frozen protocol.

#### r18p boot 3

Both sweeps completed cleanly.

| Sample | c4 steps/s | c4 output tok/s | acceptance length | 64K prefill tok/s |
|---|---:|---:|---:|---:|
| Sweep 1 | 30.982 | 94.739 | 3.058 | 2,237 |
| Sweep 2 | 30.615 | 93.022 | 3.038 | 2,234 |
| Boot median | 30.799 | 93.881 | 3.048 | 2,235.5 |

The third r18p steps/s median is -0.53% from boot 1 and -0.36% from boot 2.
Its prefill is about 2% below the two earlier r18p starts; because this sample
is late in the sequence, the final late-sequence r34 boot is needed to separate
time or thermal drift from an image effect.

#### corrected-r34 boot 3

Both final sweeps completed cleanly.

| Sample | c4 steps/s | c4 output tok/s | acceptance length | 64K prefill tok/s |
|---|---:|---:|---:|---:|
| Sweep 1 | 29.926 | 90.123 | 3.011 | 2,264 |
| Sweep 2 | 29.752 | 88.514 | 2.975 | 2,279 |
| Boot median | 29.839 | 89.318 | 2.993 | 2,271.5 |

### Distribution verdict

All 12 sweeps and all 60 decode cells completed with zero request errors,
warmup timeouts, capacity limits, or nonzero queue fraction. Across independent
engine starts:

| Arm | Boot 1 steps/s | Boot 2 steps/s | Boot 3 steps/s | Median | Range |
|---|---:|---:|---:|---:|---:|
| II r18p | 30.961 | 30.910 | 30.799 | 30.910 | 30.799–30.961 |
| corrected GG r34 | 30.002 | 29.931 | 29.839 | 29.931 | 29.839–30.002 |

r18p beat the time-matched r34 boot by +3.20%, +3.27%, and +3.22%. The median
result is therefore +3.27% acceptance-normalized steps/s, +5.11% output tok/s,
and +1.86% acceptance length. The original single-boot c4 regression is
falsified; r18p is repeatably faster at c4 under this production-faithful
contract.

The steps/s advantage is present at every context: +2.92% at zero context,
+4.04% at 16K, +5.00% at 32K, +3.40% at 64K, and +3.50% at 128K. Median 64K
prefill is +0.29%, which is parity at the resolution of this experiment. The
late r18p prefill dip is not a repeatable image-level regression.

## Optimization work after the distribution study

The next experiment is deliberately conditional:

- Repeatable steps/s gap: run the A16 discriminator, then use short Nsight
  Systems traces and VeloQ to separate B12X kernel time, CUDA-graph/host gaps,
  and NCCL overlap. Hand only isolated hotspot kernels to Nsight Compute.
- Output-only gap with steps parity: measure acceptance using the pinned input
  corpus; do not call it a CUDA regression.
- No actionable gap: record c4 parity and spend the remaining window on
  repeatable opportunities, beginning with the A8/A16 crossover and the 64K
  prefill result already embedded in this study.

## Phase 2: current-r18p A8/A16 crossover

The distribution study falsified the c4 regression and instead found r18p
faster than corrected-r34. The next candidate comes from an older but concrete
II r15 result: forced A16 was +6–7% over forced A8 at c2/c4, while it was about
-6% at c1. r18p's paired-FC2 restoration may change that crossover, so the old
number is only a lead.

Source inspection of the installed r18p image confirms there is no unforced
dynamic A8/A16 policy. `B12X_MOE_FORCE_A8=0` and
`B12X_MOE_FORCE_A16=0` selects the native NVFP4 mode; forced A8 and forced A16
are load-time weight/execution choices. Phase 2 therefore uses an A8/A16/A8
engine-start sequence on the r18p image, two fixed-input sweeps per arm, and
measures c1 and c4 together across the same five contexts. A16 is useful as a
general replacement only if its c4 gain is material and its c1 loss is no
longer material; otherwise it remains a workload-specific profile.

### Phase 2 results

All six sweeps and all 60 cells completed with zero request errors, warmup
timeouts, capacity limits, or decode queueing. The two A8 controls bracketed
the A16 boot and remained close: control 2 was -0.77% at c1, -0.57% at c4,
and -0.44% at 64K prefill relative to control 1. The A16 deltas against the
median of those two controls were:

| Metric | A8 control | A16 | Delta |
|---|---:|---:|---:|
| c1 steps/s geometric mean | 14.704 | 13.747 | -6.51% |
| c4 steps/s geometric mean | 30.948 | 32.763 | +5.86% |
| c1 output tok/s geometric mean | 41.836 | 39.405 | -5.81% |
| c4 output tok/s geometric mean | 93.895 | 101.334 | +7.92% |
| 64K prefill tok/s | 2,263.5 | 2,094.5 | -7.47% |

The c4 engine-rate gain was positive at every context: +2.62% at zero
context, +6.99% at 16K, +5.39% at 32K, +7.38% at 64K, and +7.04% at 128K.
The c1 loss was likewise present at every context (-5.60% to -7.56%). A16
also reduced the reported total KV pool from 1,062,569-1,077,222 tokens on
the two A8 boots to 1,008,939 tokens, a 5.0-6.3% capacity cost depending on
which A8 start is used.

This is a real crossover, not a replacement verdict. A16 is an optional
high-concurrency profile for a deployment known to stay near c4, where it
buys about 6% more engine steps and 8% more observed output. It is inferior
as the general default because it loses about 6.5% at c1, 7.5% in prefill,
and roughly 5-6% of KV capacity. Staging was returned to r18p A8 after the
experiment.

The A16 boot logged ten post-engine-start compilation misses per rank while
both A8 boots logged none. The per-cell benchmark warmup and repeated sweep
kept the measurement valid, and the A16 advantage survived both sweeps. This
is recorded as a cold-path limitation rather than promoted into a separate
optimization project.

## Phase 3: contemporary c1 image A/B/A

The original c4 deficit is gone and Phase 2 found a deliberate A8/A16
crossover. The remaining possible image-level target is c1: current r18p A8
measures roughly 14.7 steps/s, close to but not directly comparable with older
r34 results. Profiling that difference before proving it would repeat the
single-boot error that started this study.

The opening r18p control is the immediately preceding `a8-control-2` boot and
its two completed sweeps. Phase 3 then cold-starts corrected-r34 for two c1
sweeps and closes with another r18p A8 start and two sweeps. All arms use the
same two fixed prompt seeds, five-context ladder, 30-second cell duration,
K5 probabilistic DSpark contract, image-specific qualified NCCL settings, and
persistent caches. If r18p is not at least 3% slower in acceptance-normalized
c1 steps/s with stable opening/closing controls, there is no justified c1
profiling target.

### Phase 3 results

All four new sweeps and all 20 new cells completed with zero errors, warmup
timeouts, capacity limits, or decode queueing. The opening and closing r18p
controls were 14.647 and 14.640 c1 steps/s geometric mean (-0.05%), while r34
was 14.753. Relative to the median r18p control:

| Metric | r18p A8 | corrected-r34 | r18p delta |
|---|---:|---:|---:|
| c1 steps/s geometric mean | 14.644 | 14.753 | -0.74% |
| c1 output tok/s geometric mean | 41.525 | 41.679 | -0.37% |
| acceptance length | 2.836 | 2.826 | +0.36% |
| 64K prefill tok/s | 2,266 | 2,242 | +1.07% |

The c1 steps deltas by context were -0.84%, +0.13%, -1.90%, -1.24%, and
+0.16% from zero through 128K. This is parity and is far below the 3%
predeclared profiling threshold. There is no repeatable r18p c1 regression to
optimize. Together with Phase 1, r18p A8 is about 3.3% faster than r34 at c4,
within 1% at c1, and at prefill parity or slightly ahead.

## Phase 4: default W4A16 execution probe

With the image-level gaps closed, the next candidate was an execution mode
already present in r18p's B12X integration but not exposed by the DS4 launcher.
The arm cleared both A8 and A16 force flags. It was initially labeled "native
NVFP4," but that interpretation was wrong: the integration defaults to
`nvfp4` only when `quant_config.quant_dtype == "nvfp4"`; DS4's expert source is
E8M0/MXFP4, so this arm selected `w4a16`.

The saved receipts prove the actual path: `_w4a16_route_count_kernel`,
`_w4a16_route_prefix_from_counts_kernel`, and `W4A16FusedMoeKernel` with BF16
activations and `scale_format=e8m0_k32`. The raw experiment remains valid, but
its interpretation is a second W4A16 observation, not an FP4-activation or
native-NVFP4 result. The directory and result filenames retain their historical
names so receipt paths do not move.

### Phase 4 results

Default W4A16 booted, passed the arithmetic serving smoke, and completed both
fixed-input sweeps (20/20 valid cells). Relative to the A8 controls:

| Metric | A8 | Default W4A16 | Delta |
|---|---:|---:|---:|
| c1 steps/s geometric mean | 14.704 | 13.731 | -6.62% |
| c4 steps/s geometric mean | 30.948 | 33.020 | +6.70% |
| c1 output tok/s geometric mean | 41.836 | 38.277 | -8.51% |
| c4 output tok/s geometric mean | 93.895 | 98.936 | +5.37% |
| c1 acceptance length | 2.845 | 2.788 | -2.02% |
| c4 acceptance length | 3.034 | 2.996 | -1.25% |
| 64K prefill tok/s | 2,263.5 | 2,088.5 | -7.73% |

The cc4 steps gain was repeatable at every context (+5.68% to +7.70%), as
was the c1 loss (-5.92% to -7.48%). Native's reported local KV pool was
71,304 tokens, inside the A8 boot band.

This arm is rejected as a distinct mode because it is W4A16 under a different
launcher spelling. Its small differences from forced A16 are run variation and
cold-path effects, not a precision-mode distinction. The forced-A16 result is
the clearer specialized high-concurrency receipt, and A8 remains the general
default. The launcher patch remains a corrected historical artifact only; it
is not a release proposal.

## Phase 5: batch-selective A8/native experiment

The static native result exposes a useful but narrower opportunity. A8 wins
prefill and c1, while native wins every measured c4 context. The exact r18p
B12X planner accepts `("w4a8_nvfp4", "nvfp4")` as one source-native plan:
both recipes require no materialized repack, retain the source allocation, and
share the prepared runtime alphas. A weight-free in-image planner check
confirmed this contract on the 128-expert DS4 TP2 shard.

Phase 5 will test an opt-in vLLM integration variant that prepares both modes
once and chooses native only for full K5 speculative-verify batch sizes 12, 18,
and 24. M=6 c1 decode, reduced-depth boundary steps, and large prefill batches
remain on A8. The experiment is off by default, bind-mounted only on staging,
and must preserve the existing A8 source allocation and KV capacity.

Acceptance criteria are stricter than the static probe:

- boot, arithmetic smoke, graph capture, and all fixed-input cells must pass;
- c1 and 64K prefill must remain within 2% of the bracketed A8 control;
- c4 must improve by at least 3% in acceptance-normalized steps;
- c4 output throughput must improve, not merely raw kernel rate;
- reported KV capacity must remain in the A8 band; and
- staging must return to the exact r18p A8 contract after the experiment.

This is an integration experiment, not yet a release patch. A positive result
would still need output-equivalence and broader serving correctness gates
because the selected c2-c4 MoE calls use FP4 activations.

### Phase 5 attempt 1: rejected before measurement

The first hybrid source failed closed during model initialization. The retained
worker journal showed that DS4 has both ModelOpt NVFP4 expert layers and E8M0
MXFP4 expert layers. The initial policy globally required the dual
`w4a8_nvfp4`/`nvfp4` plan, so an E8M0 layer correctly resolved to `w4a8_mx`
and then hit the experiment's incompatible-mode guard. No graph was captured
and no hybrid benchmark request ran.

The corrected policy is layer-scoped: only a layer whose forced-A8 mode is
`w4a8_nvfp4` receives the dual source-native plan or can select `nvfp4` at the
named token counts. `w4a8_mx` layers retain their original single-mode plan and
execution unchanged. The harness now archives both container logs and inspect
records before automatic restoration on any later failure. The original A8
opening controls were complete before the failed boot and are reused after
validation rather than rerun.

The corrected boot then exposed the more fundamental scope result: its only
forced-A8 source warning is `w4a8_mx for E8M0 FP4 weights`; no DS4 expert layer
selects `w4a8_nvfp4`. Therefore the corrected experiment is intentionally a
null selector on this checkpoint. Its serving sweep is retained as a parity
control, but an A8/native-NVFP4 hybrid cannot optimize this DS4 model because
the required NVFP4 source format is absent.

### Phase 5 results

The corrected hybrid and both bracketed A8 controls completed all 60 cells
without request errors, warmup timeouts, capacity limits, or decode queueing.
The A8 controls were within +0.13% at c1, -0.30% at c4, and -0.35% at 64K
prefill. Relative to their median, the hybrid measured -0.69% c1 steps/s,
+0.07% c4 steps/s, and +0.09% 64K prefill. Its c4 context deltas ranged from
-1.00% to +1.01%.

The raw output differences (+6.45% at c1 and -3.55% at c4) track equally
large probabilistic-acceptance movements and are not engine-performance
effects. This is the expected parity result for a selector that source and
boot-log evidence prove never changes a DS4 layer. The experiment is rejected
as inapplicable, and both staging nodes were verified back on the exact r18p
A8 image, original integration-module hash, and exact-four NCCL contract.

## Phase 6: source-native W4A16 probe

The actual crossover is `w4a8_mx` versus `w4a16` on E8M0 source weights. A
weight-free matrix against the exact r18p B12X planner established:

- A8 alone requires the `qmma_repacked` representation;
- default W4A16 alone requires the incompatible `mma_packed` representation;
- W4A16 alone can instead use `SOURCE_NATIVE` with `transfer_source` storage;
- A8 plus default W4A16 is rejected as two incompatible model-sized repacks;
- A8 plus source-native W4A16 is rejected because it would retain both the
  source-native model weights and the A8 model-sized repack.

This rules out a cheap per-batch A8/W4A16 hybrid. Before considering planner
and memory-policy development, Phase 6 will measure the already-supported
source-native W4A16 mode as a static arm. The useful outcomes are either a
better high-concurrency profile than default W4A16, or evidence that the
source-native kernel narrows the c1/prefill penalty enough to revisit the
default. Its KV pool reports the memory side of that decision directly.

### Phase 6 results

Source-native W4A16 loaded and captured successfully, passed the deterministic
arithmetic serving smoke, and completed all 20 cells without request errors,
warmup timeouts, capacity limits, or decode queueing. Its model-memory result
was effectively identical to A8: 1,084,729 KV tokens versus 1,084,789 on the
nearest A8 boot. Avoiding the model-sized repack therefore does not buy a
material capacity advantage in this deployment.

Its execution performance is decisively worse:

| Metric | A8 | Packed W4A16 | Source-native W4A16 | Source-native vs A8 |
|---|---:|---:|---:|---:|
| c1 steps/s geometric mean | 14.720 | 13.747 | 8.444 | -42.64% |
| c4 steps/s geometric mean | 30.961 | 32.763 | 30.393 | -1.83% |
| 64K prefill tok/s | 2,263.5 | 2,094.5 | 1,987.5 | -12.19% |

The c1 loss is 42.1-43.8% at every context. Source-native W4A16 is also 7.23%
slower at c4 and 5.11% slower at prefill than packed W4A16. The supported
source-native layout is rejected both as a default and as a specialized
profile. Any future A8/W4A16 dynamic policy would require deliberate support
for two incompatible model-sized representations; this experiment supplies no
performance or capacity reason to pursue that complexity.

## Phase 7: W4A8 low-M tile-policy discriminator

With the cross-precision paths exhausted, the next target stays inside the A8
kernel already used in production. The exact installed planner selects M16 for
routed-row counts 36 through 383 and moves to M32 at 384. Full K5 verify is
approximately 36 routed rows at c1 and 144 at c4, so both current serving cells
are below the existing DS4-specific M32 band.

The supported `B12X_DYNAMIC_TILE_MN=32x128` override provides a clean
discriminator. Phase 7 uses a default/M32/default engine-start sequence with
two fixed-input sweeps per arm. If M32 improves c4 but hurts c1, the next
iteration will lower the DS4-specific planner threshold only for the c4-sized
band. If it does not improve c4, no selector patch is justified.

### Phase 7 results

Forced M32 passed the arithmetic smoke and all 20 serving cells, but it lost
performance against the four bracketed default-M16 controls. MTP-normalized
steps/s fell 1.69% at c1 and 1.15% at c4, with nine of ten context/concurrency
cells negative. The only positive cell was c4/16K at +0.49%, well inside the
run noise. The candidate also reduced 64K prefill by 0.75%.

The candidate boot exposed 1,114,937 KV tokens versus 1,082,807 and 1,087,612
on the opening and closing defaults, respectively. That approximately 2.7%
capacity movement is real at the allocator output but is not a usable trade:
M32 loses both decode and prefill, and its available-memory diagnostic does
not move monotonically with token capacity across these independent boots.
No lower M32 crossover or selector patch is justified. Stock M16 remains the
production policy for all observed c1 and c4 verify shapes.

### Related batch-budget audit

Every boot warns that speculative drafting reduces
`max_num_scheduled_tokens` from 8,192 to 8,176. Source inspection shows the
reduction is exactly four extra draft slots times four maximum sequences. A
batch budget of 8,208 would remove the warning and restore an 8,192-token
scheduler ceiling, but the current reduction is only 16/8,192 (0.20%). It
cannot explain a multi-percent decode or prefill result, and contemporary A8
prefill is already at r34 parity. No separate multi-boot experiment is
justified unless a later workload is demonstrably limited at that exact
boundary.

## Phase 8 candidate: restore vLLM-managed OMP lifecycle

Live boot logs exposed a separate launcher-level candidate. The DS4 launcher
unconditionally defaults `OMP_NUM_THREADS` to 16. r18's vLLM runtime is
explicitly designed to choose a bounded multi-thread count during weight load,
mark that choice with `VLLM_OMP_NUM_THREADS_SET_BY_VLLM=1`, and then reduce
Torch intra-op threads to one for steady-state serving. An externally supplied
OMP value suppresses the transition and produces the live warning that
multi-threaded CPU ops can degrade serving through OpenMP spin-wait contention
and cgroup CPU-quota throttling.

The correct release candidate is therefore removal of the launcher export,
not globally setting OMP to one and slowing weight loading. A serving
discriminator can preserve the current 16-thread startup while setting the
vLLM ownership marker, then assert that the boot log changes from the warning
to `Reducing Torch threads from 16 to 1 for serving`. It remains isolated from
the M32 experiment and will run only after that A/B/A restores stock A8.

### Phase 8 results

The candidate produced the exact intended log transition on both nodes and
passed the arithmetic smoke, proving the discriminator changed the steady-state
Torch intra-op pool from 16 threads to one. It nevertheless produced no engine
throughput benefit: versus the four bracketed stock controls, normalized
steps/s moved +0.13% at c1 and -0.13% at c4. The per-context deltas straddled
zero, and 64K prefill fell 0.97%.

The apparent +6.73% c1 output-token gain was paired with +6.59% speculative
acceptance and therefore does not represent faster execution. This experiment
rejects removal of the launcher OMP override as a performance optimization for
the measured DS4 service. The source-level warning may still motivate a
resource-isolation cleanup, but it is not part of the r18p performance path and
stock 16-thread serving behavior was restored on both staging nodes.

## Phase 9 candidate: SM121 prepared-W4A8 resident grid

The live pinned checkpoint config records hidden size 4096, 256 routed
experts, and top-k 6. A full K5 verify is therefore approximately 36 routed
rows at c1 and 144 at c4. The c1 shape enters the prepared-W4A8 decode regime;
the c4 shape is outside its 64-row bound and uses the generic dynamic regime.
Both resolve to a logical cap of 48 on this 48-SM device, which the
two-CTA-per-SM prepared-W4A8 launcher expands to 96 physical CTAs.

The source does contain an SM121 guard that caps 24 through 48 routed rows at
24 physical CTAs, but it is explicitly restricted to K=6144. It does not
apply to this K=4096 checkpoint. Treating it as active was a stale-shape
assumption caught by reading the live checkpoint config before the first
candidate arm.

The c4 path asks for the generic `dynamic` backend cap, not the more specific
`dynamic_w4a8_decode` cap used below 64 rows. Therefore
`B12X_DYNAMIC_MAX_ACTIVE_CLUSTERS` is the authoritative override: it changes
the generic logical cap before the two-CTA expansion and also serves as the
fallback for c1. A 24/32/40/default screen compares 48/64/80/96 physical CTAs
at both c1 and c4. This is a more direct test than changing tile M: it asks
whether two full resident waves add useful work or merely increase barrier
participation for the sparse 36- and 144-row domains. The first pass is a
fixed-input screen bracketed by default boots; any mover must then pass a full
A/B/A before becoming a candidate.

The first launch was stopped before any result or candidate arm after a
pre-measurement source recheck caught the wrong, decode-regime-specific env
name. The interrupted boot was restored to the exact stock image and env, and
the corrected screen uses the generic selector above. Its log retains the
aborted pre-candidate attempt so the restart is explicit rather than hidden.

A second preflight produced one complete default control before the live
script was edited while Bash was still reading it, causing a parse failure
before any candidate launch. That control and its receipts are retained under
the `preflight-` prefix but excluded from the final screen. The final script
was syntax-checked, frozen for its full lifetime, and reran the complete
default/24/32/40/default sequence.

### Phase 9 results

All five final arms passed the fixed-input result validator: ten cells per
arm, zero request errors, no warmup timeouts, no capacity limits, no decode
queueing, and positive 64K prefill. The opening and closing stock controls
form the baseline below. Candidate-specific B12X kernels compiled in each
candidate arm's benchmark warmup path; the receipts retain those events.

| Logical cap | Physical CTAs | c1 steps/s | vs stock | c4 steps/s | vs stock | 64K prefill | vs stock |
|---:|---:|---:|---:|---:|---:|---:|---:|
| stock 48 | 96 | 14.694 | baseline | 30.934 | baseline | 2,262.5 | baseline |
| 24 | 48 | 14.694 | +0.00% | 30.882 | -0.17% | 2,146.0 | -5.15% |
| 32 | 64 | 14.767 | +0.50% | 31.176 | +0.78% | 2,174.0 | -3.91% |
| 40 | 80 | 14.807 | +0.77% | 30.633 | -0.97% | 2,227.0 | -1.57% |

No lower cap produces a broad decode improvement: the per-context deltas
straddle zero, and the largest aggregate gain is only +0.78%. In contrast,
prefill falls monotonically as the grid shrinks. The behavior is consistent
with the stock second resident wave being useful for the larger-M work that
the global override also controls. No source selector or lower-cap follow-up
is justified. Stock 96-CTA prepared-W4A8 execution remains the balanced
production policy, and the final arm restored it on both staging nodes.

## Phase 10 candidate: prepared-work scheduler

The production default uses `materialized_queue`: host/device preparation
publishes the complete work domain and resident CTAs atomically claim tasks.
The same r18p kernel supports `persistent_grid`, which consumes that identical
published domain by arithmetic grid striding. It removes per-task atomic claim
traffic but can lose load balance, so source explicitly describes it as a
controlled A/B option rather than the production default.

Phase 10 will run only after the MAC screen restores stock. It uses a full
queue/persistent/queue A/B/A, two fixed-input sweeps per arm, the same
arithmetic serving smoke, and thermal receipts. This is a legitimate lower-
level discriminator because it changes neither precision, tile geometry,
routing output, nor the resident-grid cap; it only changes how resident CTAs
obtain already materialized work items.

### Phase 10 results

All six sweeps passed the fixed-input validator with zero request errors,
warmup timeouts, capacity limits, or decode queueing. Temperatures and clocks
were in the same band across the arms. Relative to the four bracketed
`materialized_queue` controls, `persistent_grid` measured:

- -0.03% c1 normalized steps/s;
- -0.55% c4 normalized steps/s; and
- -0.66% 64K prefill throughput.

The per-context changes straddle zero and the apparent c1 output movement is
fully explained by a matching acceptance-length movement. Arithmetic grid
striding therefore does not recover useful atomic-claim overhead on this work
domain; its reduced load balancing is slightly worse at c4. The candidate is
rejected, and the closing arm restored `materialized_queue` on both nodes.

The candidate also exposed 1,062,208 total KV tokens versus 1,115,177 and
1,081,967 on the independent opening and closing stock boots. CUDA-graph
memory profiling varied across those boots, so this is not attributed as a
structural scheduler cost. It supplies no compensating reason to keep a
candidate that already lost the performance comparison.

## Phase 11: production-geometry W4A8 kernel screen

The remaining supported W4A8 policy switches can be screened more cheaply
than repeated two-node boots. `a8-kernel-screen/microbench.py` prepares one
deterministic synthetic expert package at the exact checkpoint shard geometry
(E=256, K=4096, N=1024, top-k=6), then times CUDA-graph replay at token counts
1, 3, 4, 5, 6, 24, 256, and 1,024. Those points cover tiny decode, reduced
speculative depth, full c1/c4 verification, and larger-M execution.

The stock arm is compared with `persistent_grid`, disabling the materialized
intermediate, disabling shared-input reuse, and disabling tiny decode. Every
arm reuses identical weights and inputs, checks finite output, cosine
similarity, norm ratio, and error statistics against stock, and writes partial
JSON after each completed arm. This is a discriminator only: a microbenchmark
win must still pass the two-node serving A/B before it can become a candidate.

### Phase 11 results

The broad screen completed stock plus three alternatives before the global
no-tiny arm stopped on the deliberately strict stock-equivalence gate. That
stop was safe: it occurred before accepting a timing and the wrapper restored
both nodes to the exact stock service. The completed arms showed no positive
candidate:

- `persistent_grid` was within -0.18% to +0.19% of stock for M=3 through
  M=1,024 and 3.74% slower at M=1, independently corroborating Phase 10;
- disabling the materialized intermediate was at parity through M=24 but
  1.52% and 2.21% slower at M=256 and M=1,024; and
- disabling shared-input reuse was never materially faster and was 1.58% and
  2.03% slower at M=256 and M=1,024.

The no-tiny difference was then resolved against the actual pinned B12X
W4A8-MX reference instead of weakening the stock-equivalence gate blindly.
Both stock tiny decode and the dynamic fallback were finite and passed the
B12X cosine threshold at M=1, 3, and 4. The dynamic path was closer to the
reference at all three sizes, but performance crossed over:

| M | Stock tiny | Dynamic fallback | Dynamic delta | Stock/reference cosine | Dynamic/reference cosine |
|---:|---:|---:|---:|---:|---:|
| 1 | 187.262 us | 221.700 us | +18.39% slower | 0.998932 | 0.999877 |
| 3 | 551.652 us | 575.148 us | +4.26% slower | 0.998988 | 0.999788 |
| 4 | 741.450 us | 719.391 us | 2.98% faster | 0.998881 | 0.999831 |

Therefore globally disabling tiny decode is rejected, and stock is retained
for M=1 and M=3. The only positive microbenchmark lead is an M=4-only selector:
dynamic saves about 22 us per exact-geometry MoE invocation and is numerically
closer to the reference. It is not yet a serving candidate. Its value depends
on how often the DSpark service actually executes M=4 MoE batches, and a
source-level selector must still pass a bracketed two-node serving A/B before
it can justify a release change.

The first focused-reference attempt used a newer B12X call signature and
failed before timing. The exact r18p signature was then read from the pinned
tree, the harness corrected, linted, and rerun. Both failures and restorations
are retained in `kernel-screen.log`; `tiny-results.json` contains the accepted
focused result. The final staging state is the exact r18p image on both nodes,
with stock A8 policy, exact-four NCCL, and a healthy endpoint.

## Model-identity correction

The production II r10 runner historically passed both
`DeepSeek-V4-Flash-0731` and `DeepSeek-V4-Flash` to vLLM's
`--served-model-name`. This was intentional aliasing, but it is an invalid
identity contract: those names designate different checkpoints. The study
launchers copied that production convention, so their retained `/v1/models`
receipts advertise two entries even though every arm loaded only the pinned
0731 checkpoint and every benchmark request addressed
`DeepSeek-V4-Flash-0731`. The performance results are therefore unaffected,
but the advertised identity was wrong.

Both DS4 production-capable runners and all eleven study launchers are
corrected in the worktree. `SERVED_MODEL_NAME=DeepSeek-V4-Flash-0731` is now
the sole owner of DS4 model identity; topology-only `EXTRA_VLLM_ARGS` no
longer contain a model name; both runners reject attempts to reintroduce the
flag through `EXTRA_VLLM_ARGS_APPEND`; and every study readiness path fails
unless `/v1/models` contains exactly the single expected ID. Historical
receipts are preserved verbatim.

The production pair was restarted worker-first on 2026-08-22 using the exact
corrected-r34 image. Before teardown, both container specifications were
captured and the replacement render was checked to preserve the unpinned r34
NCCL contract. After readiness, both nodes matched the previous image ID,
environment, and cache mounts with `EXTRA_VLLM_ARGS` as the only intended
change. Neither node carries an NCCL channel cap. `/v1/models` advertises only
`DeepSeek-V4-Flash-0731`; a canonical chat request returned HTTP 200 and the
correct arithmetic response; and a request naming the distinct
`DeepSeek-V4-Flash` model returned HTTP 404. Both live commands contain exactly
one `--served-model-name DeepSeek-V4-Flash-0731` argument.

The dusty/kirby staging pair was subsequently restarted worker-first with the
same correction. Both nodes retained r18p image ID `445f9ac3...`, exact-four
NCCL channels, all prior environment values, and the same persistent cache
mounts; only `EXTRA_VLLM_ARGS` changed. Staging now advertises only
`DeepSeek-V4-Flash-0731`, completes canonical requests successfully, and
returns HTTP 404 for `DeepSeek-V4-Flash`. The new boot allocated 1,063,589 KV
tokens, or 2.03 full 524,288-token requests, with 17.05 GiB reported on dusty
and 16.53 GiB on kirby.
