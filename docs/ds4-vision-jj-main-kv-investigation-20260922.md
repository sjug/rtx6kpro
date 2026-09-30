# DS4 Vision JJ-main KV-capacity investigation

Local source and retained-receipt investigation, 2026-09-22. No serving changes,
GPU experiments, builds, benchmark edits, or source-checkout mutations.
Claude investigated independently; Codex checked the source paths and memory
samples below after waiting for its completed report.

## Finding and evidence boundary

The loss is localized to the new B12X preparation/profiling lifecycle, not a
changed KV format or memory-budget formula. This identifies a source subsystem
and introducing commits, not a proved single faulty allocation or a safe revert.
The retained receipts do not split host allocations, GPU program/local-memory
reservations and plan state. Foundation changes remain a confound for their size.

Frozen vLLM comparison: R38 composed tree `077347fd` versus tested JJ-main tree
`976a9ac502a7` (source `8e1f1e587f` plus Spark overlay and cache-agreement fix
`0f60770e85`). B12X: `6abad734` versus `0f3a8cbf`.

On rusty, historical R38 admits 12.73 GiB KV; JJ main admits 6.85 GiB. Both load
81.11 GiB weights and report 1.08 GiB peak activation. Consumed memory changes
88.00 to 93.29 GiB, and graph reservation 1.63 to 2.22 GiB. Later R38 boots admit
more KV, so these are boot-specific differences, not invariant release costs.

## Code path

- `9c27ec090d` introduces model-owned preparation; `26c055577b` restructures its
  coordination. At the JJ pin, `vllm/v1/executor/abstract.py:357` runs the weights
  preparation stage before the worker's `determine_available_memory` RPC.
- The initial memory snapshot predates this stage. On integrated GPUs,
  `vllm/utils/mem_utils.py` uses host `psutil.virtual_memory().available` for free
  memory. This file is unchanged between the pins. Retained host and device
  memory therefore both reduce KV admission.
- B12X `preparation/session.py` retains winning plan state, program owners and
  explicitly declared owners. Compiler processes are drained and closed. Losing
  candidate cleanup exists, including `_lib/program_cache.py` collection and
  stack-limit reset bounded by retained kernels. Neither a leaked compiler pool
  nor a particular stack allocation has been demonstrated.
- `a6994a6b6a` reserves prepared-plan scratch and calls `reserve_all()`. DSpark
  uses two workspace lanes. This can reserve more than the old profile touched.
- `a663db14f8` sizes compressed-MLA scratch across shorter C128 metadata widths,
  not just maximum context. The split policy is non-monotonic; shorter widths
  can require more chunks. CPU evaluation of the pinned policy confirms an
  increased envelope for some geometries. The exact Vision-Exp configuration
  is needed to quantify it; a text-only example is not the production amount.
- `0493cdc071` already collects, synchronizes and releases unused allocator
  blocks before final admission. It is present, not a missing backport.

## Independently replayed retained samples

From `spark/jj-main-sm121/qualification/ds4-vision/{rusty,toby}-memory.log`:

| Point (EDT) | Rusty MemAvailable GiB | Toby MemAvailable GiB |
| --- | ---: | ---: |
| 13:23:52, weight loading complete | 25.1006 | 25.8074 |
| 13:25:02, weights preparation complete | 20.6693 | 21.3547 |
| 13:28:12, later profiling | 18.0222 | 18.7750 |

The first stage leaves approximately 4.43/4.45 GiB less available on both ranks.
These are five-second host samples, not per-allocation attribution. They localize
the increase but cannot establish ownership of every byte or prove a leak.

## Next discriminating test, not executed

An instrumented JJ-main boot should record per-lane workspace bytes, host/process
memory, Torch allocated/reserved bytes and CUDA stack limits around each stage.
A warm-selection-cache boot can distinguish cold tuning effects from persistent
winning-plan costs, but must verify whether profiling still performs compilation.
Any such boot requires a new rusty/toby window. Do not remove required workspace
reservation or move the initial snapshot merely to make the reported KV larger.
