# DS4 Vision dense-backend evaluation, September 28

Status: both evaluation sessions completed. Earlier R38p containers were
stopped by request. The subsequent user-started eugr recipe was benchmarked
and remains running unchanged. Other fleets unchanged.

## Question and fixed identities

Does explicitly selecting B12X dense linears improve the existing R38p Vision
profile on rusty/toby, compared with its current automatic dense selection?

- Image on both nodes: `ab3ed5285a81c9fd4df3c9001021c420368e7667177470aa2e055420f89ecdcf`.
- Model revision: `6821d6ad3681a4b137b066b76094fa82ebd0a380`.
- Existing container: `ds4-vision-jj-r38p-tp2` on each node.
- Rusty container ID: `bff596a7c6057c9cecf6cd4363033304cda40b57fc2351c0ce18b7c9b71597ba`.
- Toby container ID: `af14372bc06c5fad4353063f083f2aae72ae78016810aa2f8b0d8bb5ae00aa93`.
- Source launcher SHA256: `3542a3f6663503dc8697c85e27e5a0a67b80c68c9ffc76143a39403f2bf0d18c`.
- `run_bench.sh` SHA256: `5c79b9760a2381b4b5233f5bbc8f1f279841b46f596dc718a127eaa8eea3e4f2`.
- `llm_decode_bench.py` SHA256: `2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3`.

The harness hashes match September 22. Its unrelated local changes remain
untouched. Current startup logs report `linear_backend='auto'`,
`DeepGemmFp8BlockScaledMMKernel`, and PYNCCL. A selection log alone does not
establish every projection's final implementation after weight processing.

## Arms and sequence

1. A: current R38p container, `BACKEND=b12x-a8-dglin`, DSpark K3.
2. B: same image and launch arguments, `BACKEND=b12x-a8`, DSpark K3.
3. C: B12X dense at K6, graph cap 48, conditional on correctness. The user
   rejected the extra isolated K3/cap-48 grid; test the combined profile next.
4. Originally planned A-return was canceled by the user during the K6 grid.
   Finish that grid, then stop all containers on rusty/toby. Keep Qwen and GLM
   on the other six nodes untouched, as explicitly clarified.

The pinned launcher adds `--linear-backend b12x` for B. Preserve the old
`VLLM_USE_B12X_FP8_GEMM=0` environment value in both arms: this image reports
it as unknown, and changing it would add an unnecessary configuration delta.
Require the actual B12X kernel selection marker and reject silent fallback.
Use a distinct candidate container name and the immutable image ID. Stop and
start workers before the head. Preserve original containers and all caches.
On candidate failure, restore the originals and verify a real completion.

Hold TP2/DCP1, probabilistic drafting, maximum reasoning, sampling defaults,
524288 context, four sequences, 4096 scheduled tokens, utilization 0.85,
FP8 KV, block256, InstantTensor BUFFERED, NCCL
LL/Simple and the existing direct-pair fabric fixed. Do not change other fleets.

## Gates and measurements

Before each timed arm, run the existing Vision semantic/tool/image/concurrency
checks and 32-case structured-output stress. Check long-context correctness
before considering the alternative eligible for promotion. Observe actual
commands, backend markers, image identities, KV capacity, startup warnings and
host memory on both ranks. A failed correctness gate stops timing for that arm.

Use the unchanged standard `run_bench.sh` protocol: C1/C2/C4, short/16K/32K/
64K/128K contexts, 30-second sustained-decode cells and integrated prefill
scouts. Preserve calibration and record token-targeting method. Record output
tokens/s, aggregate engine steps/s, effective acceptance, per-cell flags and
remote clock/thermal/memory telemetry. Do not use client GPU telemetry as Spark
telemetry. Reject cells with foreign traffic, warmup timeout or capacity flags.

Keep all arms in one benchmark campaign inside this task's results tree;
temporary observer and preparation files belong in session scratch, not new
receipt folders or sibling checkouts. Preserve raw results and report measured
conclusions here. No commits or publication are part of this evaluation.

## Interpretation and limits

One B boot bracketed by two A observations can identify a promising backend
effect, not establish a confidence interval or universal performance parity.
Compare B against both A observations and retain per-cell regressions, not just
geometric means. Kernel preparation and normal KV profiling can vary across
boots; capture those differences rather than fixing KV or clearing caches.

The pinned DSpark speculator explicitly drafts N tokens in one parallel pass;
its token-block width is distinct from the draft network's three layers. The
speculative-config validator exempts DSpark from the MTP divisibility rule.
The historical prose claiming three layers imply a K3 limit is therefore not
sufficient evidence of a runtime restriction. Actual K6 startup, configured
depth, proposal metrics and correctness remain execution gates.

The initial graph ladder tops out at 16 rows. K6 C1/C2 verifier widths (7/14)
fit that envelope, but C4 can require 28 rows. Following the user's question,
test K6 with cap 48 as a combined profile, without an extra K3/cap-48 grid. The expanded
ladder is 1/2/4/8/12/16/24/32/48. Verify actual runtime capture, not just the
configured maximum. Kambiz's effective ladder is unknown; cap 48 alone does not
prove identical graph dispatch.

The 60+ tokens/s external claim has no verified image, prompt or measurement
identity yet. The K3 comparison isolates dense-backend choice. The next arm combines K6
and expanded graph capture on the same image. Neither comparison establishes
parity with an unidentified external image and workload.

## Execution observations

The user reported possibly sending additional requests during evaluation. The
first baseline short-context C1 cell recorded two running requests and is
excluded. Its 32K C1 cell also failed warmup stability and is excluded. A same-boot
two-cell recheck passed with one running request. The provisional baseline below
combines those two rechecks with the other 13 original cells; it is not one clean
sweep. The final baseline return is needed for a quiet comparison. Expected
concurrency does not prove that all possible external overlap was absent.

| Arm | C1 geometric mean tok/s | C2 | C4 | Grid status |
| --- | ---: | ---: | ---: | --- |
| A, with two replacement cells | 37.731 | 58.346 | 81.742 | Composite, provisional |
| B12X K3, cap16 | 36.910 | 57.206 | 78.617 | 15/15 validation passed |

Both arms passed 18 semantic/image/tool/concurrency checks and 32 structured
checks. B12X K3 alone has not shown a speedup. Rusty had at least one nonzero
clock-event bitmask sample (`0x20`) during B12X K3. Sampled SM clocks ranged
2385-2437 MHz and temperature reached 85 C. The quiet baseline return and
per-cell telemetry matter when interpreting small differences. No kernel
journal entries occurred during that grid; direct-reclaim counters did not
increase during its sampled window, although startup had increased them.

K6/cap48 started successfully, selecting B12X dense kernels and the explicit
1/2/4/8/12/16/24/32/48 graph ladder. It reported 1,462,124 KV tokens, compared
with 1,480,274 for B12X K3. All 18 semantic and 32 structured checks passed.
The initial observer incorrectly required verified draft tokens divided by
drafts to equal six. That counter counts verified tokens after filtering,
including grammar filtering, rather than a constant proposal width. Source
inspection at the pinned vLLM revision confirmed this distinction. Actual
sixth-position counters showed 228 verified and 94 accepted tokens, establishing
execution beyond K3. The corrected observer requires actual sixth-position
verification and acceptance before timing. No serving code was changed.

## Source comparison leads, not a speedup claim

The local `spark-vllm-docker` checkout at `535f9fc` selects
`eugr/spark-vllm-b12x:latest` for the prebuilt `--exp-b12x` path. Its source-build
defaults select vLLM `dev/karmic-kraken` and B12X `master`, not our JJ R38
composition. These are mutable references, not the identity of Kambiz's running
image. The supplied process listing does not establish its digest or source pins.
No image was pulled or built during this review.

R38's B12X pin `ce419b52681b` predates the preparation system and subsequent
compact-W4A8 routing/tuning changes. Inspected candidate leads include B12X
`1dc77276e9d0` (parallel grouped expert prefixes and compact W4A8 route plans),
`6debca95` (repacked W4A8 grid tuning), and vLLM `cff8aebfeb73` (block-FP8 and
DeepSeek WO capacity-plan preparation). Their presence does not establish an
end-to-end Vision speedup. Some nearby prominently labeled GB10 speedups,
including `d11d4f35a8`, target DS4.1-specific paths and must not be credited to
Vision without tracing applicability.

The September 22 JJ-main investigation already found a preparation-related
Vision KV capacity loss, so switching to newer source has a known memory
qualification cost. This is a reason to measure a coherent newer candidate,
not evidence that the external 60+ C1 rate is impossible or reproduced. Exact
external image identity, prompt/context, sampling, acceptance and throughput
definition remain missing.

## Final results and stopped state

The K6/cap48 grid completed with 15/15 valid cells. At the user's explicit
request, all running containers on rusty and toby were then stopped. Both
nodes returned an empty `podman ps -q`; original containers, candidate
containers and caches were retained. Other six serving nodes were untouched.
The planned baseline return was canceled. No alternative was promoted, and
K6 was not subjected to the separate 524K correctness qualification.

Geometric means of aggregate output tok/s across five context cells:

| Configuration | C1 | C2 | C4 |
| --- | ---: | ---: | ---: |
| Auto K3 cap16, provisional composite | 37.73 | 58.35 | 81.74 |
| B12X K3 cap16 | 36.91 | 57.21 | 78.62 |
| B12X K6 cap48 | 33.52 | 46.47 | 60.26 |

K6/cap48 is slower than B12X K3 in every measured cell: geometric-mean
changes are -9.2% at C1, -18.8% at C2 and -23.3% at C4. C1 effective
acceptance rises from 2.079 to 2.260 tokens per engine step, but engine
steps/s falls from 17.757 to 14.829. This identifies the measured tradeoff,
not the underlying kernel bottleneck. Depth and capture changes are bundled.

| Concurrency | Target context | Auto K3 composite | B12X K3 | B12X K6 cap48 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 39.05 | 34.80 | 33.93 |
| 1 | 16384 | 36.77 | 38.74 | 32.91 |
| 1 | 32768 | 36.01 | 37.52 | 33.94 |
| 1 | 65536 | 39.20 | 35.83 | 32.99 |
| 1 | 131072 | 37.73 | 37.80 | 33.82 |
| 2 | 0 | 56.85 | 55.57 | 46.06 |
| 2 | 16384 | 61.07 | 58.52 | 45.87 |
| 2 | 32768 | 57.85 | 57.64 | 47.36 |
| 2 | 65536 | 59.89 | 55.88 | 48.68 |
| 2 | 131072 | 56.23 | 58.50 | 44.50 |
| 4 | 0 | 82.34 | 79.17 | 62.99 |
| 4 | 16384 | 81.71 | 79.81 | 62.21 |
| 4 | 32768 | 82.38 | 80.99 | 60.29 |
| 4 | 65536 | 83.50 | 78.73 | 59.69 |
| 4 | 131072 | 78.85 | 74.53 | 56.37 |

K6 remote telemetry: 49 samples per node; all sampled clock-event bitmasks
were zero. Rusty SM clocks were 2366-2437 MHz, toby 2405-2515 MHz; both
peaked at 83 C. Direct-reclaim counters stayed unchanged within the sampled
window. Kernel journal entries were AppArmor perfmon denials from Ubuntu Pro
helper processes, not GPU faults. These sampled observations do not exclude
short unsampled events.

K6 integrated prefill scouts at 16K/32K/64K/128K were
2304/2260/2178/2024 tok/s, close to K3. Its initial 8K scout was 1346 tok/s
and overlapped inference-time JIT warnings during initial warmup; do not use
that one cold scout as a steady-state prefill comparison. The last logged
JIT warning was at 15:18:45 EDT, before the decode cells.

The harness used the shared 6.18 chars/token calibration cache, not identical
tokenized prompts across boots. Context values are targets, not exact prompt
lengths. Rates are sustained aggregate output tokens per measurement second;
C2/C4 values are aggregate across requests. Harness local GPU telemetry is
not the authoritative Spark telemetry.

Raw immutable measurements:

- [baseline](../runs/deepseek-v4-flash/vision-exp/2026-09-28-vision-dense-depth-evaluation/throughput/20260928T143609-0400__baseline__r01.json), SHA256 `0aacf2b68e8d3ce5088c489f60285ae3dee704c14fb48406ef488733065f4536`.
- [baseline-recheck](../runs/deepseek-v4-flash/vision-exp/2026-09-28-vision-dense-depth-evaluation/throughput/20260928T145057-0400__baseline-recheck__r01.json), SHA256 `e6a096eb5e435538a98606e6c386533b0f92e50caf9b409560a83b65750a7346`.
- [b12x-k3](../runs/deepseek-v4-flash/vision-exp/2026-09-28-vision-dense-depth-evaluation/throughput/20260928T145856-0400__b12x-k3__r01.json), SHA256 `32da520d36b9a564a88fefb9a6a69da8536471ded5a021c941b5e1d6503ab84b`.
- [b12x-k6-cap48](../runs/deepseek-v4-flash/vision-exp/2026-09-28-vision-dense-depth-evaluation/throughput/20260928T151823-0400__b12x-k6-cap48__r01.json), SHA256 `3804fe69d9730ed7cd23f9f976e48602277740d42a51a7c828c84f5d03980c4d`.

## User-started eugr recipe, second session

After stopping the earlier experiment, the user started `vllm_node` on both
nodes and explicitly requested benchmarking it. Logs were inspected before
inference; the benchmark then began at 16:15:56 EDT. No container configuration
was changed. The earlier stop instruction was fulfilled for the earlier test;
the user-started recipe is left running.

Both nodes report image ID
`bdd4cd9474e88adcd2fe87bf08813b9ddff7a573a357e9a0eb573c3220c3225a`. Rusty
records registry digest
`docker.io/eugr/spark-vllm-b12x@sha256:1890879a1edda692a2660ca3209f9ce1b267b732e811d8fafd82433ba2f7ff52`.
Toby records a different local manifest digest after transfer, with the same
image ID. Installed packages: vLLM `0.1.dev21546+g502d6cb5a.d20260928`,
B12X `1.3.0`, Torch `2.13.0+cu130`, FlashInfer `0.7.0`, InstantTensor `0.2.0`.
No retained git metadata was present in `/workspace/vllm`, so the full vLLM
source hash and exact B12X source pin remain unresolved.

Startup selects B12X dense kernels and MXFP4/MXFP8 MoE, with separate B12X
weights/state/bind preparation and eight compiler workers. DSpark loads only
3/48 checkpoint shards and finishes draft weight loading in 1.27 seconds.
Engine initialization takes 1094.59 seconds. The model does not support
the requested piecewise compilation, so runtime explicitly switches to
FULL_DECODE_ONLY and captures full decode graphs successfully. API health
returns 200 and startup completes. Legacy VLLM_USE_B12X_* environment
variables are reported unknown; explicit backend flags are effective.

The recipe uses B12X loader, K6 probabilistic drafting, capture cap48, eight
sequences, 8192 batched tokens, high reasoning, and auto context resolved to
1048576. The effective initial graph ladder is 1/2/4/8/16/24/32/40/48.
It admits 1,065,212 aggregate KV tokens; rank0 reports 10.51 GiB and rank1
10.22 GiB. Both soft/hard nofile limits remain 500000. Model revision is
configured as mutable `main`, not the earlier explicitly pinned snapshot.

All 18 semantic and 32 structured checks passed before timing. The existing
correctness scripts were copied to scratch with only the served model name
changed to `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`. The benchmark harness
remains unchanged. This is an end-to-end recipe comparison, not an isolated
image or source-revision comparison.

### Completed eugr recipe measurement

All 15 decode cells passed validation, with zero reported request errors,
concurrency overruns, warmup timeouts or capacity flags. Geometric means
over five target contexts:

| Recipe | C1 tok/s | C2 aggregate tok/s | C4 aggregate tok/s |
| --- | ---: | ---: | ---: |
| R38p B12X K3 cap16 | 36.91 | 57.21 | 78.62 |
| R38p B12X K6 cap48 | 33.52 | 46.47 | 60.26 |
| eugr B12X K6 cap48 | 31.82 | 45.86 | 68.22 |

Relative to R38p B12X K3, this recipe is about 13.8% slower at C1, 19.8%
slower at C2 and 13.2% slower at C4. Relative to R38p K6, C4 improves about
13.2%, while C1 is 5.1% slower and C2 is 1.3% slower. These are observed
single-run differences across complete configurations, not isolated engine
performance effects. The original auto K3 baseline remains provisional due
to the two replaced cells and canceled return run.

For C1, eugr engine steps/s is 14.802 versus R38p K6 14.829, while effective
acceptance is 2.150 versus 2.260. The new recipe did not increase C1 engine
speed on this workload. This weakens an image-age-only explanation for the
external 60+ claim without establishing its actual cause.

| Concurrency | Target context | eugr aggregate tok/s |
| ---: | ---: | ---: |
| 1 | 0 | 32.27 |
| 1 | 16384 | 32.84 |
| 1 | 32768 | 31.39 |
| 1 | 65536 | 33.84 |
| 1 | 131072 | 28.97 |
| 2 | 0 | 45.32 |
| 2 | 16384 | 48.07 |
| 2 | 32768 | 45.59 |
| 2 | 65536 | 45.00 |
| 2 | 131072 | 45.38 |
| 4 | 0 | 69.73 |
| 4 | 16384 | 65.97 |
| 4 | 32768 | 67.44 |
| 4 | 65536 | 68.61 |
| 4 | 131072 | 69.44 |

Prefill scouts at 8K/16K/32K/64K/128K: 2277/2378/2239/2198/2004 tok/s.
The harness used the same calibration-cache path but a different served-model
identity, so do not assume the same cached calibration entry or identical
prompts. No benchmark-source files or raw JSON measurements were edited.

Remote telemetry recorded 49 samples per node with zero clock-event bitmasks.
Rusty clocks ranged 2379-2450 MHz, maximum temperature 83 C; toby
2411-2535 MHz, maximum 79 C. Both kernel journals were empty during the run.
Final captured container logs contained no ERROR or Traceback lines.
The containers remain running with the user's configuration unchanged.

Raw result: [20260928T161556-0400__b12x-eugr-k6-cap48__r01.json](../runs/deepseek-v4-flash/vision-exp/2026-09-28-vision-dense-depth-evaluation/throughput/20260928T161556-0400__b12x-eugr-k6-cap48__r01.json); SHA256 `0b157b0e714146a7c8e1efd5f5ea8d1691f1198949f002a63caf39e6c6610678`.
