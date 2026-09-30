# Deterministic DSA top-k tie-break: patch, review and test plan

Artifacts in this directory (all `claude`-prefixed, no external checkout or
host was modified):

| File | Purpose |
| --- | --- |
| `claude-dsa-topk-tiebreak.patch` | Unified diff against pinned B12X `a7d7d29b`, path `b12x/attention/dsa_indexer/tiled_topk.py` |
| `claude_topk_patch_check.sh` | Applies the patch to a fresh `git show` copy in a `mktemp` directory, byte-compiles it, checks recorded digests and that the shared fallback is unchanged |
| `claude_test_topk_tiebreak.py` | GPU regression against a stable reference; `--stock-module` demonstrates the pre-patch failures |
| this note | Design, dispatch confirmation, review of the diagnostic repair, integration and qualification plan |

Recorded digests (the check script fails closed on drift):

| Object | SHA-256 |
| --- | --- |
| `claude-dsa-topk-tiebreak.patch` | `7d77a7fb6f2abdf6967621c445147fa968782f295fb3c6c4cf605ba536764ec0` |
| patched `tiled_topk.py` (pinned blob plus patch) | `f2d0fadae632f4b2af233ca2c53122e7fbcb78a92dc5d16479e541330cf9052d` |

## 1. What the patch changes

Only `tiled_topk.py`. The selection algorithm (10-bit coarse histogram on the
fp16 key, four 8-bit refine rounds on the fp32 key, exact overflow rescan) is
unchanged. Three things are new:

1. **Deterministic tie admission.** Exact fp32 ties at the final selection
   boundary were admitted by shared-memory atomic arrival order in two places:
   the buffered arm's round 3 (`_smem_xadd(lr, 0, -1)`) and the exact overflow
   fallback (`ex_key == ex_pivot` loop). Both now call
   `_admit_smallest_index_ties`, which selects the `rem` tied candidates with
   the smallest *logical* K index by a four-round ascending MSD radix over the
   32-bit index, using deterministic integer histograms in the existing
   `hist0/hist1` shared arrays. The buffered arm passes the round-3 low byte as
   the tie key (its buffer already shares the upper 24 key bits); the fallback
   passes the full pivot. The fallback keeps its old all-ties loop when the tie
   population does not exceed the remaining slots (that case is already
   deterministic), reading the population from the final round's prefix counts
   without a rescan.

   Consumer scope: `_exact_overflow_fallback` is also imported by
   `fused_indexer.py` (line 1461 of the pin), a consumer this kit does not
   test. That shared function stays byte-identical (the check script verifies
   it against the pinned blob), so the fused indexer keeps its original
   arrival-order tie admission and its compile identity. The patched kernel
   class calls a new copy, `_exact_overflow_fallback_stable`, which carries the
   two extra parameters (`carry_indices`, `output_index_offset`) and the tie
   stage. Only `DSATiledTopkKernel` (behind `run_row_topk` and
   `run_tiled_topk`) is therefore changed; the fused radix path is not.
2. **Logical tie key across folds.** `_logical_index_virtual` returns
   `row_start + vidx + output_index_offset` for local candidates and the carried
   global logical index for carried slots, so a supertile fold chunk ranks ties
   exactly like one global selection would (carried candidates come from earlier
   K and win ties, matching a stable global sort). Physical page mapping and the
   `output_gather_table` mapping are applied after selection as before, so the
   mapped row path resolves ties on logical position.
3. **Canonical output order.** `_sort_out_by_index` bitonic-sorts the selected
   virtual indices ascending before write-out. Admission positions still come
   from atomic counters, so without this the emitted order (not the set) would
   remain arrival-dependent; downstream kernels that consume indices in emitted
   order would then not be bitwise reproducible.

Compile identities are bumped, not reused: `attention.indexer.tiled_topk`
version 4 to 5 with policy token `tiled_topk_v2_stable_index_tiebreak`, and
`attention.indexer.row_topk` version 4 to 5 with policy string
`row_topk_v11_stable_index_tiebreak`. The kernel class now also asserts a
power-of-two top-k (512, 1024, 2048 are the only supported values).

Cost: the tie stage runs only on rows that actually have an exact-key tie at
the boundary (which is precisely the previously nondeterministic case). In the
buffered arm it scans at most the candidate buffer (1024 entries for k=512,
8192 otherwise) eight times with 8-stage prefixes; in the fallback it rescans
the row eight more times, on top of the six passes the fallback already makes.
The bitonic sort adds 45 (k=512) to 66 (k=2048) barrier stages per row.

## 2. DS4.1 dispatch, confirmed from the live serving chain (corrected)

Earlier drafts of this note claimed a live tiled fold path; that was derived
from generic `_impl.py` dispatch and is wrong for DS4.1. The serving chain in
vLLM `1794dcf` `vllm/models/deepseek_v4_1/attention.py` is:

- `_forward` picks `index_mode = "decode" if swa.is_decode else "prefill"`
  and quantizes the query through `_index_plan(mode, rows)`; both modes are
  MXFP4 plans (`dsa_indexer.Caps(cache_format="mxfp4", topk=512, ...)` in
  `_declare_index_plan`), differing only in row capacity (`DECODE_CHUNK` versus
  `INDEX_CHUNK`) and the caps `mode`.
- `forward_mqa` binds each row chunk with `dsa_indexer.bind(...)` at a
  `score_width`, then calls `dsa_indexer.score(binding)` (`score_mxfp4`) and
  `dsa_indexer.select(binding)` (`select_mxfp4`). `select_mxfp4` calls
  `run_row_topk` on the fp32 `logits`, then the index-ordering `sort` launch,
  and at the candidate source layer a second `run_row_topk` over block logits
  (`candidate_topk_blocks=2048`) with expand. Layers after the source layer
  select over `max_candidates=16384` positions and emit through the
  `output_gather_table` mapping.
- `mxfp4.py` contains no reference to `run_tiled_topk`; the supertile fold
  and `extent_splits` belong to the fp8 cache-format state in `_impl.py` and
  `paged.py`, which DS4.1 does not plan. The `_index_call` preparation and the
  gate render both prepare only `score_mxfp4`/`select_mxfp4`.

Consequences for this patch: the three live DS4.1 configurations are all the
row kernel (`is_first`, no carry), in full-width, block-stage and mapped forms,
and all are covered by the buffered-arm and fallback tie stages plus the
canonical sort. The `_logical_index_virtual` carry branch and the fold and
extent-split tests remain correct for the fp8 path but are inert for DS4.1;
they are kept so the kernel stays deterministic for every caller.

## 3. Review of `deterministic_topk.py` (the diagnostic repair)

Sound as a control: it takes the kth value from the stock selection (a valid
top-k set, only its tie choice differs), counts greater/equal per tile with
integer sums, prefix-scans deterministically, and admits the lowest logical
positions among equals; scores are copied from the input, not perturbed;
gather mapping is reapplied. Points that keep it diagnostic rather than final:

- It wraps only `run_row_topk`, which (section 2) is the selector for both
  DS4.1 prefill and decode, so it does cover the live chain. It does not cover
  `run_tiled_topk` (fp8 cache format, not planned by DS4.1) or the fused radix
  path.
- Three extra launches and four `torch.empty` allocations per selection call,
  including inside graph capture (works under the graph pool, but churns the
  allocator in eager steps and must be placed inside `select_mxfp4` before the
  `sort` launch to take effect in serving).
- Rows with fewer than k valid entries rely on the stock kernel writing `-inf`
  into unused value slots; that holds today but is an implicit contract.
- Output order within the greater/equal partitions is deterministic, but the
  stock kernel's own order is not, so bitwise reproducibility of consumers that
  read the stock order is not established by it.

## 4. Other order-dependent paths to control in the determinism qualification

- `B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1`: MoE combine (dynamic family and the
  `w4a8_phase2`, `nvfp4_phase2`, `dynamic` phase-2 variants). Changes the compile
  cache namespace; confirm engagement from the ordered-combine kernel names.
- Micro MoE decode kernel. Checkpoint geometry (verified 2026-09-25 on the
  pinned config): 384 routed experts, 6 experts per token, hidden 5120, MoE
  intermediate 2304 (576 per TP4 rank). The micro/dynamic cutover is 64 routed
  pairs, so an 8-row DSpark verify step is 48 pairs and dispatches to the
  compact micro path, which owns 1 to 8 live tokens (`_MICRO_MAX_TOKENS`);
  only steps above 8 rows reach the dynamic family. Consequently
  `B12X_DYNAMIC_DETERMINISTIC_OUTPUT` (dynamic only, see `_impl.py` line 3237)
  never touches a decode or verify step; it covers prefill, where the compact
  M16 split-materialized dynamic kernels run the deterministic phase 2 and the
  ordered top-k sum. Micro's only floating-point atomic is the trellis K-split
  merge (`red_add_global_f32`, trellis layouts only); the pinned modelopt
  `fp4_e8m0_k32` layout compiles it away, so micro is deterministic by
  inspection for this checkpoint.
- `tiny_decode` (1 to 4 tokens, bf16 atomics, no deterministic branch): not
  selected with K7, but would be by a K0 control; set `B12X_W4A8_TINY_DECODE=0`
  for any such arm.
- `B12X_DENSE_SPLITK_TURBO=0`: dense split-K exists only in plans whose row
  capacity is at most 8 (tuner legality `query.max_rows > min(8, tile_m)`
  rejects it; the static policy agrees), so it can only act in decode or verify
  steps, never in prefill. Four-way atomic BF16 slices need a step of at most 6
  rows on block-FP8 or MXFP8 shapes with wide N and K between 4096 and 6144;
  the retained tuning receipt shows 21 four-way and 8 two-way selections among
  those small plans. A verify step with the full 7 drafts has 8 rows and takes
  no split, so a 3-token repeatability matrix never exercises this path; only
  steps with 6 or fewer rows (partial drafts, adaptive verification) do.
  With turbo off the tuner rejects four-way and the policy clamps to two slices
  through the exact FP32 reducer; note this changes the legal tactic set, so
  the tuning cache is not reusable across the two settings.
- Engram overlap (`VLLM_DS41_ENGRAM_OVERLAP`): the spin-wait timeout writes an
  unread `failed` flag; disable for the determinism boot or surface the flag.
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`, persisted FlashInfer autotune cache.
- Fixed by inspection: RoCEnante fixed rank order, NCCL fixed topology, indexer
  scorer fixed-order reduction, greedy sampling and greedy DSpark drafting.

## 5. Integration into this kit

The patch is Python (CuTe DSL) only: no native rebuild, but the JIT namespace
for these two kernels changes through the bumped compile identities. Apply it
the way PR865 is applied: extend `prepare.py` with a `patched_b12x` step that
`git apply --check --whitespace=error`s the patch in scratch against the
tracked B12X file, records the patch digest and the resulting file digest in
`runtime.lock.json` (a `b12x_patches` entry beside `ds41_nccl`), and lets
`install.py` place the changed file through the ordinary tracked refresh.
`preflight.py` should add the patch to its manifest. The cache fingerprint
should include the patch digest so old plans cannot be reused.

## 6. Tests and acceptance

1. `claude_topk_patch_check.sh` (workstation): apply, compile, digests.
2. `claude_test_topk_tiebreak.py` inside the candidate image with a GPU
   (`--quick` first, then full): row path across widths 17 to 524288, k in
   512/1024/2048, scenarios `fp32_random` (must equal `torch.topk`), `bf16`,
   `equal`, `ties_at_rank`, `ties_overflow` (forces the fallback), `neg_inf_mix`,
   `two_levels`; lengths per row of full, minus 31, 0, k, k+1, 513. Each case
   checks exact index-set equality with the stable reference, unchanged values,
   ascending emitted order, bitwise identity over 8 repeats, and graph replay.
   Then the mapped path (4096 with k 512/1024, and the live DS4.1 shape 16384
   with k=512), the supertile fold with carries at 4096/32768/524288, and
   `extent_splits=2`. The row widths include 16384 (live candidate width) and
   75000 (block-stage width at 600000 context, k=2048) so both live DS4.1
   selections are exercised at their real shapes; the block stage's fp32 block
   logits are covered by the `fp32_random` and `two_levels` kinds.
3. `claude_test_topk_tiebreak.py --stock-module <pinned tiled_topk.py>` records
   the pre-patch failures (report only) for the receipt.
4. Serving level, after the runtime build: `probe_repeatability.py` at 256, 513,
   514, 1024, 16384 and then 131K, 262K, 524K, with the MoE, split-K and Engram
   switches above; require bitwise-identical top-20 logprobs across cold salted
   repeats. Only then replay the historical input. A deterministic outcome may
   still be the wrong answer; that is a separate finding.

## 7. Known risks to watch in the first GPU compile

The patch was verified only as Python syntax and as a clean `git apply`; CuTe
DSL tracing runs on the GPU host. Idioms were copied from the surrounding
kernel, but three constructs are new here and should be checked first if the
build gate fails: `Uint32`-typed jit parameters (`tie_key`, `tie_mask`),
`Int32(Uint32(a) ^ Uint32(b))` in the bitonic sort, and passing `s_out` twice
to the fallback's tie stage (as the unused buffer argument and as the output).
