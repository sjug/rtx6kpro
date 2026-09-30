# Qwen Karmic main qualification, September 22, 2026

Status: correctness and full grid passed; Claude's independent final review
confirmed comparability, arithmetic and the health/performance caveats.
Performance is mixed, not a clean R32 replacement.
This is one cold candidate boot, not repeatability evidence or a promotion.
Only dusty/kirby were interrupted. GLM and DS4 Vision remain outside this window.

## Identity and contract

Image `88867036403cb3c9bc3026eb8c907ad64bc1af9c28d316d22b670dde4f561152`,
tag `localhost/voipmonitor/vllm:karmic-main-spark-sm121`. Both ranks verified the
same Docker archive and image ID. Source identities are in `source.lock.json`;
the retained distribution version string names the old base and is not the
source identity. Small build/transfer gate evidence is in `evidence/build-gates/`.

Qwen checkpoint `c374e7e24b54f6cb0017d0c2e6d26823d2f2fb5d`, served alias
`Qwen3.8-Flash-Next`; TP2, MTP3, aligned recurrent checkpoints, 262144 context,
four sequences, 4096 batched tokens, utilization 0.85, InstantTensor BUFFERED,
NCCL `LL,Simple`. No MTP0 or fresh R32 control was run.

Boot began 2026-09-23 01:45 UTC. First completion was available around 01:53 UTC.
The runtime's reported KV capacity was 5,259,616 tokens, with 42.73 GiB rank 0
and 43.92 GiB rank 1 available to KV. Compare the saved R32 return's 5,272,780
and JJ-main's 4,857,562 tokens as boot-specific capacity, not performance.

## Correctness and interaction

Receipts: `qualification/qwen-mtp3/`.

- First completion: exact `333`, finish reason `stop`.
- Semantic battery: 18/18, covering three reasoning levels, non-thinking,
  synthetic image input and tool round trips.
- Retrieval: exact `739184` at 2848, 2849, 131072 and 262000 prompt tokens;
  all finished `stop`. These are correctness requests, not fresh-prefill timings.
- Fixed-token acceptance: 2.294249, corpus
  `cc9230856e9dab760645436993ec44669c3168e2edc3cfbc8a897c72de0ffff4`.
  Outputs were not identical across waves, also true of the saved JJ-main arm;
  no bitwise parity claim is made.
- Padded transition c1/c3/c1/c2/c4/c1: acceptance 2.40/2.67/2.41/2.62/2.56/2.42;
  overlap checks passed and no collapsed phase. Graph coverage remains predicted,
  not directly observed dispatch coverage.
- Prefix repeat and extension: each reused 28,480 tokens, with correct answers.
- Head-of-line probe: maximum fresh-request first token 0.266 seconds.
- Short concurrency check: identical c4 129.6 tok/s; distinct c4 132.4 and
  123.7 tok/s. No serialization signature. These are not the standard grid.

## Health interpretation

Retain exact kernel journals separately from container logs. The startup snapshot
contains 103 NV_ERR_NO_MEMORY lines on dusty and 72 on kirby. Dusty's warnings
span loading and later admission/correctness phases, not just one second.
They are allocation-pressure evidence, not a resolved driver mechanism.
Final totals: dusty 108, kirby 72. Five additional dusty warnings occurred at
22:09:43 EDT during c4/32K admission. The harness started that cell at 22:09:39,
declared it ready at 22:09:50, then began the 30-second measurement. No warning
fell inside that measured interval. No Xid or OOM-kill was recorded.

During the grid (22:00:08 to 22:13:23 EDT), minimum MemAvailable was 3.380 GiB
on dusty and 6.757 GiB on kirby. Dusty lost 0.044 GiB of SwapFree, with 138
additional allocstall_normal and 3776 pgscan_direct counts; kirby had no increase.
Mean sampled SM clocks were 2470.3/2487.8 MHz, with zero active throttle flags.
Whole-boot minima were 3.142/6.757 GiB. Completion does not mean abundant headroom.
No container warning/error or inference JIT warning appeared during the grid.
The two first-use Triton warnings preceded it during correctness testing.
All 54 grid-window POSTs came from workstation address 192.168.2.2; this cannot
independently distinguish another client on the same workstation.

The reported final graph memory (-1.80/-0.24 GiB) is a free-memory delta across
capture, not an isolated graph allocator measurement. The pinned source computes
free-before minus free-after; the enormous percentage clamps the denominator to
one byte for negative values. KV admission uses the earlier profile estimate,
not this final delta. Do not copy the resulting KV recommendation into a launcher.
Attribution of the free-memory rise to a particular allocation owner is unproven.

The benchmark's hardware summaries describe the workstation, not the remote pair;
use the per-node observer telemetry for Spark clocks, power and memory. Its
reported 79,655,712-token KV budget is also not the engine's admitted capacity;
use the boot's 5,259,616-token figure above.

## Benchmark method

Unmodified `~/git/llm-inference-bench/run_bench.sh`, 15 cells: c1/c2/c4 at
0/16K/32K/64K/128K, 30 seconds each, plus integrated prefill scouts. The existing
benchmark working-tree diff was retained read-only; this task did not edit it.
Harness SHA256s and container inspections are retained with the receipts.

Compare engine steps, output throughput and acceptance separately against saved
R32 and JJ-main grids. Prefill scouts have one sample per context. The complete
grid has 15 unique cells and no errors, underfill, capacity or warmup flags.
The comparison helper checks harness, model revision, alias, policy and protocol.
Raw results are in the corresponding `runs/` campaign; comparisons are under
`qualification/qwen-mtp3/`.

Geometric means across the five contexts:

| Concurrency | R32 tok/s | Main tok/s | Output change | R32 steps/s | Main steps/s | Steps change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 44.19 | 48.31 | +9.32% | 22.07 | 23.15 | +4.92% |
| 2 | 74.92 | 78.49 | +4.77% | 36.58 | 37.43 | +2.32% |
| 4 | 115.79 | 120.03 | +3.67% | 55.79 | 56.37 | +1.03% |

Acceptance accounts for the rest of the output-rate gain. Grid acceptance is
not the fixed-corpus acceptance test above.

| Context | R32 prefill tok/s | Main prefill tok/s | Change |
| --- | ---: | ---: | ---: |
| 8K | 3124 | 2932 | -6.15% |
| 16K | 3004 | 2870 | -4.46% |
| 32K | 2918 | 2783 | -4.63% |
| 64K | 2764 | 2646 | -4.27% |
| 128K | 2501 | 2412 | -3.56% |

Server-side prefill at 16K through 128K also shows the deficit. This is one
candidate boot and one scout per size; no new R32 control was requested or run.
There is no causal attribution or repeatability claim.
The deficit also appears at the previously exercised 128K shape, with no JIT
warnings or B12X compilation lines during the grid. First-use compilation does
not explain these measured scouts.

Against the prior JJ-main evaluation grid, output is -2.98/-1.48/-1.22% at
c1/c2/c4, while engine steps are +1.66/+0.18/-2.91%. Main is not a demonstrated
decode improvement over that immediate predecessor either.

Recommendation: do not promote this as an across-the-board Qwen improvement.
Both nodes remain on the tested candidate for evaluation, with previous JJ-main
containers retained and stopped. No implicit rollback or additional cluster
window was performed. GLM and DS4 Vision remain unchanged.

Final raw-grid SHA256:
`b5f1be8b606537a4d66d3d050d0528afc906100749f58b4ca557c4c23a29b7b7`.
The driver exited zero. Both candidate containers were verified running on the
pinned image after the tests, with OOMKilled false. No commit or push was made.

## HC sharding diagnostic follow-up, September 23 UTC

Claude reviewed the isolated diagnostic runner and amendments before execution.
Both HC-off and warm HC-on-return arms subsequently completed correctness and
the unchanged 15-cell standard grid with exit status zero. Full numbers, source
attribution limits and receipt hashes are in
[the investigation report](../../docs/qwen-karmic-main-prefill-investigation-20260923.md#executed-onoffon-result).

HC-off improves 32K/64K/128K prefill by 4.16/4.43/4.15% against the warm on
return, at effectively identical mean GPU clocks. It remains 1.41/1.09/0.76%
below saved R32 at those lengths; 16K remains 7.19% below R32 and is unresolved.
The on/off/on sequence supports an HC-feature contribution, not an explanation
of the entire R32 gap. Only one off boot was run, with single-scout prefill.

HC-off versus warm on engine steps improve 1.95/4.59/5.41% at C1/C2/C4.
Output tok/s changes are -2.99/-0.58/+6.75% because acceptance also changes.
Both arms pass semantic 18/18, native retrieval through 262K and the other MTP3
gates. Neither arm has a benchmark-window allocation warning, Xid or OOM kill;
startup allocation warnings persist on dusty.

Current pair is the HC-on return (`HC_TP=1`), still the evaluation profile. Both
the initial on and off containers are retained stopped. Production defaults,
GLM and DS4 were not changed. New receipts are `qualification/window-hc-off/`,
`window-hc-on-return/`, `qwen-mtp3-hc-off/` and `qwen-mtp3-hc-on-return/`.
