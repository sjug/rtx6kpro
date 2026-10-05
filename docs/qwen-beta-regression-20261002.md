# Qwen beta regression investigation, October 2

Initial source inspection identified changed B12X kernel selection after the
CUTLASS DSL 4.6.2 to 4.7.1 upgrade and fresh autotuning. The subsequent authorized
October 2 experiment found selection variability and partial C1 recovery when
restoring old selections on the candidate. This supports a selection contribution
but does not establish a complete explanation or a qualified fix. The original
source investigation was read-only; the live experiment is recorded below.

Against the restored old-image repeat, candidate output throughput is lower by
1.77%, 2.47%, and 0.55% at C1/C2/C4; decode-step throughput is lower by
2.71%, 1.25%, and 1.63%. Old-image repeats also vary, and the restored run had
three dusty NV_ERR_NO_MEMORY messages during its grid. Acceptance and timing
variation limit attribution from these grids alone.

C1 nevertheless has a consistent context-matched signal: at each of the five
contexts, both candidate readings are below both old-image readings in steps/s.
Across the 20 context-matched pairwise comparisons the loss is 0.71–4.62%.
This is not a claim that every candidate reading is below every old reading
across different contexts. The raw C1 steps/s are:

| Context tokens | Old r01 | Old r02 | Candidate r01 | Candidate r02 |
| --- | --- | --- | --- | --- |
| 0 | 24.6652 | 24.7978 | 24.1844 | 24.0374 |
| 16384 | 24.2995 | 24.9219 | 23.8518 | 24.1267 |
| 32768 | 24.5231 | 24.7499 | 23.6061 | 23.8789 |
| 65536 | 24.0686 | 23.9904 | 23.6855 | 23.7502 |
| 131072 | 23.5358 | 23.5389 | 22.8739 | 22.8862 |

C2/C4 have weaker separation and do not establish a comparably consistent
regression with only two grids per image. C1 is the first experiment target.

## Actual comparison

| Component | Serving image | Candidate |
| --- | --- | --- |
| Image | `500ae05b98da` | `c4d4c7f56892` |
| vLLM | `99cbe782f3b67a7758a85bcc0e7938a1ec27012a` | `4a379ed42881ee022aaf5d9ada554f9096e5acf3` |
| B12X | `1b6cd278626aa6b44c519b5d9f4ad576c9af783b` | `b557d87850cc836268fd327ccd46eaa6a033cdbf` |
| FlashInfer | `2206a14e` | `dbd6238c6655b98195fdf77f04bba6facf5a38a4` |
| CUTLASS DSL | 4.6.2 | 4.7.1 |

The small afternoon re-pin was only a subset of this serving-to-candidate delta.
Checkpoint, TP2, MTP3, HC off, aligned checkpoints, and harness were held constant.
Comparisons are saved in
`runs/qwen3.8-flash-next/nvfp4/2026-09-qad-7c4f1bc1-qualification/`:
`original-vs-restored.json` and `restored-vs-candidate.json`.

## Saved tuning evidence

Read-only inspection on dusty found the same 93 selection keys in both caches;
44 configurations differ. Cache root: `~/.cache/vllm-jj-qwen38-4p89/jit/`.
Namespaces are `karmic-beta-20260929-d1bf7017c3744f9294f0` and
`karmic-beta-20261001-93d7583a39a5131a8dd7`. Selection file in each:
`b12x/preparation/d21ec6d108d51b356aaa0bf1d027ff9a06e5a9b6c690b836d5bfa3db459601ec.json`.

Changes include dense GEMM tiles/split-K, four activation-precision switches,
MoE routing, GDN prefill configurations, and normalization configurations.
These are saved choices, not timing traces proving each choice ran in each cell.

Exact dense-query shapes were recovered by reproducing the selection-key hash
from `b12x/preparation/session.py::_choice_key`, using the pinned
`gemm.blockscaled_precision` contract (query schema 7, config schema 4,
candidate contract 19). M is activation rows, N output features, K input features.

| M | N | K | Old selection | Candidate selection |
| --- | --- | --- | --- | --- |
| 1, 4 | 2560 | 3072 | BF16, tile 16x64x128, split 8 | BF16, tile 16x64x64, split 8 |
| 4, 8 | 8192 | 2560 | BF16, tile 16x64x64, split 8 | BF16, tile 16x128x128, split 8 |
| 16, 24 | 2560 | 3072 | BF16, tile 16x64x64, split 8 | Quantized activations |
| 32 | 2560 | 3072 | BF16, tile 16x64x64, split 1 | Quantized activations |
| 24 | 8192 | 2560 | BF16, tile 16x64x64, split 2 | Quantized activations |

Weights in these queries are MXFP8 in both images. Selected-program manifests
confirm the precision switches change activations from BF16 to E4M3 FP8 with
E8M0 scales and add a Triton quantizer. This changes arithmetic and cost; its
effect on speculative acceptance has not been independently measured.

For steady MTP3 decode, the logical verification pass processes four tokens per
sequence and each draft pass processes one row per sequence:

| Concurrency | Verify M | Draft M per pass |
| --- | --- | --- |
| C1 | 4 | 1 |
| C2 | 8 | 2 |
| C4 | 16 | 4 |

These are logical row counts; captured/padded execution shapes should also be
confirmed in the experiment. The precision switches at M=16/24/32 correspond
to C4 and above under this mapping. They cannot explain the steady C1 loss.
The identified C1 dense changes are BF16 tile changes with split-K remaining 8;
M=8 belongs to C2 verification, not C1 under this mapping. Focus C1 attribution
on those tile choices, compiler output for the selected kernels, and other
changed selections actually exercised during decode (norm/MoE, and any GDN
path shown active by a trace). Changed GDN prefill choices alone are not evidence
of a steady-decode bottleneck.

At the pinned revisions, vLLM
`vllm/model_executor/kernels/linear/b12x_blockscaled.py:106-151` declares
per-row-count plans with activation mode `auto`. B12X
`b12x/gemm/blockscaled/_tuning.py` allows both activation precisions; its MXFP8
candidate set was already present. B12X
`b12x/preparation/session.py:1348-1372` selects candidates by measured latency.
New compiler output or different measurement conditions can change winners;
these saved choices do not establish which caused the change. The upstream
`docs/cutlass-471-migration.md` documents correctness/resource evidence, not
matched SM121 serving performance qualification.

Autotuning variability is a concrete alternative to compiler slowdown. The
candidate populated caches during its initial cold boot, whose qualification
failed health for post-readiness NV_ERR_NO_MEMORY. The subsequent warm boot
reused populated caches; see
`spark/karmic-beta-sm121/20261001/qualification/qad-7c4f1bc1-warm/mtp3/REPLAY-HEALTH-REVIEW-OK`.
The old grids reused the old cache rather than independently retuning. Memory
pressure during tuning, or small isolated-timing differences that do not carry
over to steady serving, could select different winners. The health record does
not establish that the errors overlapped the tuning measurements or caused
those selections.

## Other hypotheses checked

FlashInfer `flashinfer/sampling.py`, `include/flashinfer/sampling.cuh`, and
`csrc/sampling.cu` are byte-identical between the old Git revision and the locked
candidate archive. This weakens a sampler-source regression hypothesis; rebuilt
binaries and all their dependencies have not been proven performance-identical.

Shared PLE storage is opt-in and unset in the runner. New MXFP8 MoE support is
format-guarded; this model uses NVFP4 experts. vLLM also changed scheduler
admission and multiprocess response bookkeeping, but source inspection does not
establish either as the measured decode bottleneck.

## Experiment design (subsequently executed below)

First, in an authorized GPU window, retune the same candidate once into an
isolated fresh cache under the experiment's tree on both ranks. Preserve the
existing caches. Keep image, profile and tuning policy fixed, record startup
health, and diff the resulting assignments/configurations against the current
candidate's 93 selections. Verify a complete retune, with no rank importing the
old selection cache through reconciliation. This startup-only check precedes
another benchmark grid, but still needs the serving pair and is not a free
workstation-only operation.

Different winners demonstrate selection variability under the two observed
startup conditions. They do not by themselves prove memory pressure caused it,
that the first winners were worse, or that pinning fixes performance. Unchanged
winners from one retune would not rule out variability either.

In a separately authorized window, compare the candidate's current selections
with the old selections held fixed, rebuilding kernels with 4.7.1. Keep compiler,
weights, workload, and sampling settings unchanged between those arms. Do not
copy old compiled objects into the new compiler namespace. Verify selected
plans, then measure steady decode and acceptance.

Copy only selection metadata, not compiled objects, for the pinned arm.
Matching keys support compatibility but are not the only validation: at the
pinned B12X revision, `_cache.py::_read` requires matching cache identity and
`session.py::_lookup` validates each assignment and its lowered configuration.
Reconciliation shares selections across ranks while executable artifacts stay
local. Confirm the selected configurations and freshly resolved 4.7.1 programs
on both ranks before attributing the result. Start with context-matched C1;
expand only if the result warrants it.

If old choices recover performance, tuning selection is implicated. Otherwise,
hold selections fixed while comparing compiler versions to distinguish compiler
code generation from remaining runtime changes. Repeating an unmodified full
grid alone will not isolate either cause.

## October 2 live experiment

The user authorized execution of the retune and pinned comparison. It ran in
existing workstation tmux pane `0:7.1`. Receipts, orchestration and isolated
node caches are under the task's
`spark/karmic-beta-sm121/20261001/qualification/autotune-ab-20261002/` tree.
Original caches and retained production containers are preserved. The script
restores the qualified September 29 pair after completion or failure.

A completely empty cache on both ranks produced 93 selections, of which **42
differ from the original candidate**. Both ranks produced the same file:
`e651c54b4b3c12250b0f04070c20262e83ef6e618f297dc0c948264456255cde`.
Seventeen changed configurations returned to the old image's choice; 41 choices
still differ from the old image overall. The M=1 and M=4, N=2560, K=3072 GEMMs
both returned to tile K=128 from the candidate's K=64. Variability therefore
includes identified C1 shapes. Tuning logs show fresh measurements with zero
initial cache hits; later preparation phases reuse choices made in that same
startup. There were startup NV_ERR_NO_MEMORY messages, but none in the collected
post-readiness smoke window; arithmetic smoke passed.

The measured arms use the same candidate image and 4.7.1 compiler. Each gets
an independent copy of the original candidate's compiled cache. The pinned arm
replaces only its selection JSON with the old image's selection JSON; no 4.6.2
compiled objects are copied. Logs confirm cached selection loading without
autotuning; any missing artifacts compile under the candidate. Both ranks retain
all 93 seeded configurations exactly. Selection-file SHA-256 values:

- Current: `0d93ea8aacbecfdbf68ab5b1ced087bc46c60ec85c5e73f44333a42cfa3334a8`
- Pinned old: `d6526615fa42aa6baa6750a90e218bb706ff18e01ffca9bc9475332e454667a6`

Each arm uses the unchanged pinned harness, C1, contexts 0/16K/32K/64K/128K,
30 seconds per decode cell, and token budget 78,619,040. Acceptance is recorded
alongside steps/s. This is a diagnostic comparison, not full qualification of
the pinned configuration.

### Measured result

| Context | Current selections, steps/s | Old selections on 4.7.1, steps/s | Change |
| --- | --- | --- | --- |
| 0 | 24.2724 | 24.3118 | +0.16% |
| 16K | 24.0744 | 24.4743 | +1.66% |
| 32K | 24.2692 | 24.4357 | +0.69% |
| 64K | 23.8537 | 24.2667 | +1.73% |
| 128K | 23.0132 | 23.4227 | +1.78% |
| Geometric mean | **23.8919** | **24.1791** | **+1.20%** |

Old-image C1 geometric means were 24.2152 and 24.3937. Pinning gets within
0.15% of the lower old-image mean, but remains below the observed range.
Recovery also differs by context: at context 0 the pinned arm still trails both
old-image readings by 1.43–1.96%; at 128K it trails by about 0.48–0.49%.
This is partial recovery, not uniform restoration to the old-image envelope.

Effective acceptance geometric mean was 2.0515 for current selections and
2.1170 for pinned selections (old-image repeats: 2.1127 and 2.0545).
Acceptance varied too; these runs do not isolate a precision-specific acceptance
effect. The benchmark kept the existing sampling settings, without adding a
fixed seed. One sequential A/B has no counterbalanced repeat, so temporal and
sampling variation remain limitations on the size of the selection effect.

Both measured arms had zero failed/capacity-limited cells, no post-readiness
Xid/NV_ERR_NO_MEMORY/kernel OOM messages, no container OOMKilled or restarts,
and successful arithmetic smoke. All selection-file hashes remained unchanged
after measurement on both nodes. The pinned arm logged zero tuning measurements,
84 cached choices in its first preparation phase and nine in subsequent phases;
15 then eight missing artifacts compiled with the candidate environment.

Raw runs under
`runs/qwen3.8-flash-next/nvfp4/2026-10-karmic-beta-sm121-qualification/throughput/`:

- `20261002T095115-0400__karmic-beta-20261001-tuning-current-c1__r01.json`
- `20261002T100114-0400__karmic-beta-20261001-tuning-pinned-c1__r01.json`

`comparison.json` in the experiment receipts contains per-context steps/s,
acceptance, and output throughput for these arms and the four historical grids.
No full C1/C2/C4 grid of the pinned configuration has been run, and it has not
been promoted. The result supports selection changes as a contributor while
leaving residual compiler/runtime effects unresolved. Further attribution needs
a repeat or counterbalanced comparison and, for the residual, a compiler arm
with selections and sources held fixed. An image-to-image comparison alone does
not isolate the compiler.

The experiment exited with status 0. The retained September 29 containers on
dusty and kirby were restored, worker first; API health passed at 14:10:26 UTC.
At 14:11:20 UTC both were verified running image `500ae05b98da`, and arithmetic
smoke returned `391`. See `status.txt` and `restoration.json` in the receipts.
