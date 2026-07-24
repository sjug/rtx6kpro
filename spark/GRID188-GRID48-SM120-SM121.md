# Grid188 to Grid48: SM120 and SM121 Comparison

Status: Grid48 implemented, deployed, and measured (2026-07-18); see "Measured verdict" below

Recorded: 2026-07-18

## Purpose

This note records the device measurements and kernel reasoning behind an
SM121-shaped version of the B12X mapped Grid188 hybrid-decode kernel. The
specific SM121 target is the NVIDIA GB10 in the DGX Spark nodes, so the direct
equivalent is called **Grid48**: one persistent CTA per one of the 48 SMs.

Grid188 is used by the GLM-5.2 mixed NVFP4/NF3 TP4 decode path. It is not a
DeepSeek-V4-Flash/DSpark kernel.

## Conclusion

Grid48 is a geometry retarget of Grid188, not a new GEMM design.

The measured SM120 and SM121 devices have identical per-SM CUDA execution
limits relevant to the kernel:

- 65,536 registers per SM;
- 102,400 bytes of shared memory per SM;
- 101,376 bytes of opt-in shared memory per block;
- 1,536 resident threads per SM;
- 1,024 threads per block; and
- 32 threads per warp.

The baseline port should retain the Grid188 256-thread CTA, its 45,184-byte
shared-memory layout, its software whole-grid barriers, and one CTA per SM.
It should compile for `sm_121a`, require exactly capability `(12, 1)` and 48
SMs, and distribute the existing work across a 48-CTA grid.

## Measurement Provenance

### SM120 workstation

The local workstation contains two NVIDIA RTX PRO 6000 Blackwell Workstation
Edition GPUs. Board-level values were read through `nvidia-smi`; CUDA resource
values were read directly through `cudaGetDeviceProperties()` using the
local CUDA 13.3 toolkit (`nvcc` 13.3.73).

Both GPUs report the same architectural properties. CUDA-visible memory
differs by a few MiB between devices because of runtime/driver reservations:

```text
device 0: 102,011,043,840 bytes
device 1: 102,014,189,568 bytes
```

The local driver is `610.43.03`. Both cards currently have a configured
300 W power limit.

### SM121 DGX Spark

The GB10 properties were read from PyTorch's CUDA device properties in the
running `ds4-v18-tp2` container on `toby`. The queried device is the same
SM121 class deployed across the DGX Spark cluster.

```text
name: NVIDIA GB10
CUDA-visible memory: 130,663,165,952 bytes
integrated device: yes
```

## Exact Device Comparison

| Property | RTX PRO 6000 | GB10 | Grid consequence |
|---|---:|---:|---|
| Compute capability | 12.0 | 12.1 | Build separate `sm_120a` and `sm_121a` code |
| SM count | 188 | 48 | Grid188 becomes Grid48 |
| Registers per SM | 65,536 | 65,536 | Same per-SM register budget |
| Shared memory per SM | 102,400 B | 102,400 B | Same residency budget |
| Default shared memory per block | 49,152 B | 49,152 B | Same default limit |
| Opt-in shared memory per block | 101,376 B | 101,376 B | Existing 45,184 B CTA fits |
| Maximum threads per SM | 1,536 | 1,536 | Same thread residency limit |
| Maximum threads per block | 1,024 | 1,024 | Existing 256-thread CTA fits |
| Warp size | 32 | 32 | No warp-level redesign |
| Total L2 cache | 128 MiB | 24 MiB | Different cache pressure and reuse |
| L2 per SM, nominal | 697 KiB | 512 KiB | GB10 has about 73% as much per SM |
| Memory bus | 512-bit | 256-bit | Part of the bandwidth difference |
| Memory type | Discrete GDDR7 | Unified LPDDR5X | Different latency/coherency topology |
| Nominal memory rate | 28,002 MT/s | 8,533 MT/s | Used for theoretical bandwidth below |
| Theoretical bandwidth | 1,792.128 GB/s | 273.056 GB/s | RTX has 6.563x aggregate bandwidth |
| Bandwidth per SM | 9.533 GB/s | 5.689 GB/s | GB10 has 59.7% as much per SM |
| Reported/max SM clock | 3,090 MHz | 2,418 MHz | Per-SM throughput is not clock-identical |

The SM-count ratio is `188 / 48 = 3.917`. The bandwidth ratio is larger at
`1,792.128 / 273.056 = 6.563`, so the Grid48 port should be expected to face
more memory pressure per SM even though its execution-resource limits match.

The two products therefore differ in more than only SM count and aggregate
memory bandwidth: compute-capability minor version, clocks, L2 capacity,
memory capacity, and integrated-versus-discrete topology also differ. None of
those differences invalidates the one-resident-CTA-per-SM design.

## Existing Grid188 Workload

The Grid188 specialization in B12X release commit
`bc85ef36192cb6e444d42ba7be86e1e125cca98a` targets the exact GLM hybrid
decode shape:

| Parameter | Value |
|---|---:|
| Decode rows (`m`) | 4 |
| Hidden size | 6,144 |
| Intermediate size | 512 |
| Routed experts | 256 |
| NVFP4 experts | 64 |
| NF3 experts | 192 |
| Top-k | 8 |
| Input/output dtype | BF16 |
| Activation | gated SiLU |
| CTA threads | 256 |
| Dynamic shared memory | 45,184 B |
| FC1 tasks | 128 |
| FC2 tasks | 768 |

The current Grid188 admission logic requires capability `(12, 0)`, exactly
188 SMs, one resident block per SM, and no local-memory spill. It launches a
single 188-CTA kernel and uses software grid barriers between phases.

### Grid188 schedule

| Phase | Tasks | CTA allocation | Maximum waves |
|---|---:|---|---:|
| FC1 | 128 | 128 CTAs x 1; 60 CTAs idle | 1 |
| FC2 | 768 | 16 CTAs x 5; 172 CTAs x 4 | 5 |

The work iterator is conceptually:

```text
task = cta_id + wave * grid_x
```

That formulation is already independent of the literal value 188. The main
hard-coding is in device admission, constants, compile/cache metadata,
mapping proofs, custom-op integration, logging, and vLLM selection.

## Proposed Grid48 Schedule

Grid48 preserves the same task numbering and changes only `grid_x`:

| Phase | Tasks | CTA allocation | Maximum waves |
|---|---:|---|---:|
| FC1 | 128 | 32 CTAs x 3; 16 CTAs x 2 | 3 |
| FC2 | 768 | 48 CTAs x 16 | 16 |

The exact schedule vectors are:

```text
FC1 work counts: [3] * 32 + [2] * 16
FC2 work counts: [16] * 48
```

The FC2 phase has more serial work per CTA than Grid188. That is expected
from the 3.917x reduction in SM count. It does not change correctness, but it
makes route-dependent NVFP4/NF3 task balance an important benchmark target.

## Why Grid48 Should Be Resident-Safe

The existing CTA consumes 256 threads and 45,184 bytes of shared memory.
The GB10 allows 1,536 threads and 102,400 bytes of shared memory per SM, so
one such CTA per SM fits with substantial thread and shared-memory margin.

The port must still query the compiled SM121 function attributes. A correct
admission check must prove all 48 CTAs can be resident simultaneously before
allowing the software whole-grid barriers to execute. At minimum it should
reject the fast path if any of these are false:

- capability is exactly `(12, 1)`;
- the device exposes exactly 48 SMs;
- compiled shared memory is the expected 45,184 bytes;
- the compiled kernel has no local-memory spill;
- occupancy permits at least one CTA per SM; and
- the full 48-CTA grid can be resident at once.

These checks are deadlock-safety requirements, not merely performance
checks.

## Implementation Shape

The safest implementation is a separate Grid48 hardware profile sharing the
mapped-kernel core with Grid188. Do not relax the existing Grid188 SM120 gate.

Suggested profile constants:

```text
target_capability = (12, 1)
target_sms        = 48
grid_x            = 48
fc1_tasks         = 128
fc2_tasks         = 768
```

Required integration work:

1. Parameterize or duplicate the Grid188 device profile, mapping proof,
   resource admission, workspace sizing, compile metadata, and cache key.
2. Compile the new specialization for `sm_121a`.
3. Add a Grid48 custom-op entry point, or generalize the existing mapped-grid
   operation without changing its fail-closed behavior.
4. Make vLLM select Grid188 for `(12, 0) / 188 SMs` and Grid48 for
   `(12, 1) / 48 SMs` after eager preparation succeeds.
5. Preserve serial hybrid decode as the fallback if compilation or resource
   admission fails.
6. Emit distinct `armed` and `executing` messages for Grid48 so deployment
   logs prove the fast path actually ran.

The existing Spark TP4 launcher currently exports
`VLLM_NF3_GRID188_DECODE=1` in
[`v18/run-glm52-v18-hybrid-tp4-node.sh`](v18/run-glm52-v18-hybrid-tp4-node.sh).
The mapped-grid selector should eventually use a hardware-neutral name while
retaining the old environment variable as compatibility input.

## Validation Plan

### Mapping proof

- Prove that every FC1 task in `[0, 128)` appears exactly once.
- Prove that every FC2 task in `[0, 768)` appears exactly once.
- Prove that no CTA receives an out-of-range task.
- Check the exact Grid48 work-count vectors shown above.

### SM121 compile and admission

- Compile the actual kernel for `sm_121a` on GB10.
- Record registers per thread/block, static and dynamic shared memory, local
  memory, and CUDA occupancy.
- Prove all 48 CTAs are concurrently resident.
- Fail closed to serial decode for any metadata mismatch.

### Correctness

- Compare against serial hybrid decode for mixed, all-NVFP4, and all-NF3
  routing patterns.
- Exercise the exact `m=4`, BF16, top-k-8 shape.
- Test eager execution and CUDA graph capture/replay.
- Run multiple graph replays to expose stale barrier/workspace state.

### Deployment

- Run the four-node GLM TP4/DCP1 profile on the Spark cluster.
- Confirm every rank logs both Grid48 `armed` and `executing` messages.
- Confirm NCCL TP collectives still use the two switched 200G interfaces.
- Verify output parity before collecting performance numbers.

### Performance

- Benchmark serial hybrid decode versus Grid48 at the maintained decode
  contexts and concurrencies.
- Record kernel duration and end-to-end tokens/s, not only mapping proof
  success.
- Inspect CTA-tail imbalance caused by different NVFP4 and NF3 task costs.
- Treat Grid96/two-CTA-per-SM as a later experiment only if compiled register
  usage proves two-block residency. Grid48 is the direct Grid188 analogue.

## Current Spark Validation Caveat

The GPU-validation section in
[`v18/build-gilded-gnosis-v18-spark-sm121-cu132.sh`](v18/build-gilded-gnosis-v18-spark-sm121-cu132.sh)
calls `w4a16_hybrid_mapped_grid188_mapping_proof()` and validates the returned
task lists. That proof is pure mapping logic; it does not compile or execute
Grid188 on SM121. The current exact `(12, 0) / 188 SM` kernel admission gate
therefore rejects GB10 and vLLM falls back to serial hybrid decode.

A completed Grid48 port must add a real SM121 compile/resource check and an
execution test. A successful mapping proof alone is insufficient evidence.

## Source and Deployment Anchors

- Canonical v18 description: [`../models/glm5.2_v18.md`](../models/glm5.2_v18.md)
- B12X Grid188 release commit:
  `bc85ef36192cb6e444d42ba7be86e1e125cca98a`
- vLLM release source:
  `264bce1da81e27d638e7cf265b4cbd125d023c38`
- Spark v18 build:
  [`v18/build-gilded-gnosis-v18-spark-sm121-cu132.sh`](v18/build-gilded-gnosis-v18-spark-sm121-cu132.sh)
- Spark GLM TP4 launcher:
  [`v18/run-glm52-v18-hybrid-tp4-node.sh`](v18/run-glm52-v18-hybrid-tp4-node.sh)
- Spark v18.2 image tag:
  `voipmonitor/vllm:gilded-gnosis-v18p2-spark-sm121-vllm264bce1-b12xbc85ef3-fi801d57a-cu132-20260718`

The mapped MoE kernel executes locally on each TP rank. Retargeting Grid188
to Grid48 does not change the cross-node NCCL topology or the TP4 network
collectives; it changes only the per-rank hybrid decode kernel between those
collectives.

## Measured verdict (2026-07-18, GB10 SM121)

Layer-level microbench (`v18.4/microbench-hybrid-moe-layer.py`, m=4 GLM shape,
CUDA-graph replay, 200 iters; raw mean/median/batch in the script log):
serial two-tier 618.6 us median (kept 202.0 + nf3 417.8 + combine 6.7) vs
mapped Grid48 580.6 us — 1.066x median (1.053x mean/batch). Serving impact
~2-3% at cc1 MTP3 (38 us x 77 layers vs ~105 ms step), confirmed within noise
in the 21-cell sweep (0.993x geomean vs serial).

Root cause of the SM120/SM121 asymmetry, in effective weight-stream bandwidth
for this routing (8 kept + 24 NF3 routes, ~134.5 MB/layer floor): mapped
execution raises utilization from ~45% to ~78% of theoretical on 188-SM SM120
but only from ~80% to ~85% on 48-SM GB10. Grid188 primarily recovers bandwidth
lost when separate tier kernels underfill the large SM array; on GB10 each
tier already approaches the memory-system limit (kept FC1 still underfills 48
SMs but has enough per-CTA memory demand to drive ~77% of bandwidth alone), so
Grid48 is correct but yields only a 5-7% layer-level gain. The 38 us net
saving includes scheduling changes, removal of the separate output add and
route gathers, and bandwidth-utilization improvement — not launch overhead
alone. Byte floor at 273 GB/s is ~492.5 us vs Grid48's 580.6 us: material
further gains require reducing NF3 traffic or increasing reuse, not
scheduling. The crossover SM count moves with routing, tile geometry, and
bandwidth per SM — do not generalize "below N SMs" beyond this shape.

TODO when reusing the harness: add an output-equivalence check (nonzero
weights, mapped vs serial allclose).

Serving-level confirmation (2026-07-18 cc1 MTP3 on/off A/B, same v18p4 image,
3 sweeps/mode x 5 contexts): geomean ON/OFF = 1.0243, matching the 2-3%
microbench projection; accept length 2.879 vs 2.852 (mode-independent), so
~+1.5% survives acceptance normalization. Kernel is reached, is faster, and
the saving is real but small — closing the three-way discrimination.
