# Decision-row capture and reference audit (schema v3; hook kit repaired 2026-09-26, rebuild required)

Files (all `claude_*` or `claude-*`, in `ds41-20260924/`):
- `claude_decision_row_audit.py` with `claude_test_decision_row_audit.py`:
  the offline audit, 16 CPU tests;
- `claude-decision-row-capture.py`: the in-image helper, installed as
  `vllm/models/deepseek_v4_1/claude_decision_row.py`;
- `claude_prepare_decision_row.py`, `claude_install_decision_row.py`,
  `Dockerfile.claude-decision-row`, `claude-decision-row.lock.json`,
  `.ignore`, `.patch` and `claude-decision-row-attention.py`: the
  diagnostic image recipe over e06df11a;
- `claude_test_decision_row_capture.py`: 11 CPU tests of the helper, the
  patch, the lock and the installer;
- pinned inputs `claude-pinned-b12x-compressed_reference.py` (a7d7d29b
  verbatim, sha 123afd75) and `claude-ds41-attention-config.json` (a
  subset of the HF fb2764a5 config, fetched sha 8be45ce0).

Nothing has been built, launched or run on a node. The user owns execution.

## Purpose and stated limits

The audit asks whether production operators at the decision row (the last
prompt position of the 524288-token original dual-needle request on B2)
deviate from references computed over the exact bytes they read. Every
report carries these limits:
1. The references share the cache formats and the published rounding with
   production. Agreement means the kernels compute what the format defines
   at this row. It is **not an independent model oracle**, and a shared
   systematic bias is not detectable here.
2. The attention envelope is a conformance/sensitivity measure from
   production-plan runs, not a certification.
3. Byte-identical full responses between the capture run and B2 show **no
   observed output perturbation** by the hooks. They do not prove hidden
   numerics were untouched.
4. The KV and index-key contents written by the 63 earlier prefill chunks
   are inputs here, not validated.

## Verified facts (vLLM 1794dcf1 attention.py = installed d3b92eb0; b12x a7d7d29b; config fb2764a5)

- 40 target layers. The 3 DSpark nextn layers (40-42) are skipped;
  `is_draft` and `layer_id >= 40` are excluded in the helper.
- Compress ratio: layers 0-1 are 0; 2-19 are 2; 20-39 are 1. The page is
  `256 // max(ratio, 1)`: 128 or 256 states.
- KV owners are `max(s <= layer)` over [2, 8, 14, 20]. Index sources are
  [2, 8, 14, 20, 24, 28, 32, 36]; others reuse their source's
  `topk_indices_buffer`. Candidate source is 20; index layers 24-36 score
  only its candidates.
- The indexer is replicated on every rank (`ReplicatedLinear`, all 32
  heads). Attention has 16 local heads per rank; sm_scale is 512^-0.5,
  with a per-head sink.
- MXFP4 score (`dsa_indexer/mxfp4.py` `_PagedScore`):
  - q and k (E2M1 x UE8M0/32) are dequantized to BF16;
  - per head: BF16(dot), then BF16(max(BF16(dot), 0) x FP32(w));
  - the FP32 head sum is rounded to BF16.
  The selector takes the FP32 of that value. Exact ties go to the lowest
  logical index (`determinism-tiled-topk.py`, `_admit_smallest_index_ties`,
  used by `run_row_topk`).
- The shipped `dsa_indexer/reference.py` implements the FP8 128+4-byte
  layout, not DS4.1's 68-byte MXFP4. It is not used.
- Decision chunk: 524288 = 64 x 8192. It is identified on the host from
  the original SWA metadata: `is_decode` False, `num_reqs` 1,
  `num_actual_tokens` 8192 and `max_seq_len` 524288. There is no device
  read.

## Open items, resolved from source

- **O1 (mapped slots):** after `mla.run`, `binding.scratch.mapped_indices`
  holds the physical slots the kernel read. The native `map_indexed_pages`
  (`compressed_sparse_mla/_metadata.py` 114-145, `_preparation.py`
  424-449, `compressed_api.py` 212-235) maps them with
  `page_size = indexed_page_size = _main_page` and
  `num_pages = indexed_k_cache.shape[0]`. The capture records both the
  logical indices plus the page-table row and the mapped slots. The audit
  checks `table[l // page] * page + l % page` against them.
- **O2 (SWA slots):** the installed `sparse_mla.py` (ring fix) `_chunk`
  writes `swa_indices[r, c] = page * 128 + logical % 128` over
  `logical = start + c`, with `start = max(pos + 1 - 128, 0)`. Page 0 and
  columns at or beyond length are -1. `swa_lengths[r]` is the valid count,
  and `top_lengths[r] = min(visible, 512)`. The gather rule is
  `kv_cache[slot // 128, (slot % 128) * 528 : +528]`, the same rule the
  reference's `_gather_cache_reference` applies.
- **O3 (candidates):** the layer-20 selector (`select_mxfp4`, block
  prepare in `_SelectPrepare`) works as follows:
  - 8-entry blocks;
  - block logit = max of the valid FP32(BF16 score) values;
  - the block holding the last visible entry is forced to +inf;
  - top-2048 blocks (lowest block index on exact ties), expanded to
    ascending positions, at most 16384.
  The audit reproduces this rule.
- **O4 (index pages):** `index_pages[row]` is the request's physical
  block-table row, filled by the `_pages` kernel (holes are -1, block 0
  reserved). The page size is `_index_page = 256 // ratio`, and the layout
  is `[page, 64]` data then `[page, 4]` UE8M0 (`index_mxfp4_page_bytes`).
  The pages are immutable once full. Decode appends start past 524288,
  which falls on a page boundary for both page sizes.
- **O5 (memory):** all of it is allocated at arm time:
  - about 10 MB of pinned per-layer staging;
  - about 100 MB of pinned key storage (full page-table rows of owners 2,
    8, 14 and 20);
  - one 41 MB device key buffer.
  On GB10 unified memory this is about 150 MB per rank, so budget it
  against MemAvailable. The owner-20 pages are shared by layers 20-36 and
  saved once. Ranks 1-3 hash them; only rank 0 stores them.
- **O6 (eager):** attention is the opaque custom op
  `vllm::dsv41_b12x_attention`, whose body calls `layer._forward` in
  Python. The capture sizes are 1..32 (FULL_AND_PIECEWISE), so the
  8192-row decision chunk runs eagerly. The helper also refuses to act
  while a stream is capturing.

## CED row geometry (root cause of the 2026-09-26 abort, verified in source)

The first live capture aborted on every rank with `IndexError: index 8191 is
out of bounds for dimension 0 with size 128` at the first CED decoder layer.
`ced.py ced_decoder_start` is the first full-resolution KV source after a
compressed encoder: layer 20. Layers 20..39 set `is_ced_decoder`; in prefill
the model gathers only the last `CED_WINDOW = 128` rows of each request at
the boundary (`_decoder_indices`: `original_end - compact_end + row`), and
`attention._query_metadata` returns `metadata.decoder` (num_actual_tokens
128, positions gathered) for those layers while the original SWA metadata
still reports 8192 tokens. So q, output, swa_indices, top_lengths, the page
table, topk_indices_buffer[:rows], iq_data/iq_scale/iw and the candidate
buffers all have 128 rows there, and the decision row is row 127, not 8191.
Layers 0..19 keep 8192 rows and row 8191. The helper now resolves the rows
per layer through `layer._query_metadata` (as production does), indexes row
`query_rows - 1` of every tensor it is handed, checks each tensor's leading
dimension, stages that row's position and reports any position other than
`prompt_tokens - 1` as a problem; any other geometry aborts (fail open) and
writes a durable `<token>-rank<k>.aborted` receipt.

## Capture contract (schema `claude-decision-row-v3`, one file per rank)

- **`meta`, known in-process:** rank, node (`DS41_NODE`), kit sha
  (`DS41_KIT_SHA256`), source trees (the installed lock), generation
  (token), prompt_tokens, row_position (staged position of layer 0's
  decision row), chunk_rows 8192, row_index 8191, decoder_rows 128,
  batch_requests, and `problems` (which must be empty).
- **per layer entry (v3):** `ced_decoder`, `query_rows` (8192 or 128),
  `row` (query_rows - 1) and `position` (must be prompt_tokens - 1); the
  audit's `validate` checks all four against the layer table.
- **Observed identity, produced by the external driver, not in-process:**
  image id, kit, source trees, the decoded sparse-MLA regime (from the
  post-boot selection snapshot and `claude_decode_sparse_mla`), and the
  full response signature.
- **The audit requires:**
  - observed regime and response signature equal B2's;
  - an equal B12X tree (image id, kit and vLLM tree legitimately differ);
  - every capture's kit and trees equal the observed values;
  - exactly ranks 0..3.
- **`layers[L]` for L in 0..39:**
  - `q`, `out` (BF16 [16,512]), `attn_sink`, `plan` (mode, config);
  - `swa_len`, `swa_slots`, `swa_records` [128,528];
  - for compressed layers: `indexed_len`, `indexed_logical`,
    `indexed_page_table_row`, `indexed_slots`, `indexed_records` [512,288];
  - for index sources: `indexer` = `q_data`, `q_scales`, `weights`,
    `cache_length`, `page_size`, `index_pages`, `topk`,
    `key_pages_sha256`, `key_pages` (rank 0), plus `candidates` and
    `candidate_len` (layer 20).

## Hook discipline (as implemented)

- **Four guarded call sites in `forward_mqa`,** added only. The patch test
  proves nothing is removed or changed:
  - `begin()` after `require_prepared`;
  - `indexer()` after `dsa_indexer.select` in the chunk with
    `end == rows`;
  - `attention()` after `mla.run`;
  - the import.
- **`begin()` refuses under graph capture:** its first action is
  `torch.cuda.is_current_stream_capturing()`. It returns before any
  trigger read, arming or allocation, so no `cudaHostAlloc` or device
  allocation can occur inside CUDA graph capture. FULL-graph replay runs
  no Python hooks at all.
- **Unarmed cost:** the capture check, a boolean, and at most one trigger
  `stat` per second.
- **Arm time (the first eager, non-capturing forward after the
  trigger):**
  - pinned staging for all 40 layers (about 10 MB);
  - pinned key storage per key owner, sized for the full index page
    table (about 100 MB);
  - one device key-staging buffer (the largest owner, about 41 MB).
  Arm before sending the request.
- **In the decision forward only,** on the current stream, immediately
  after each producer:
  - non_blocking D2H of the small tensors;
  - record gathers;
  - at each key owner (the indexer owns its k cache: layers 2, 8, 14
    and 20), `index_select(k_cache, page_table_row.clamp_min(0),
    out=device_buffer)` followed by non_blocking D2H into that owner's
    pinned storage.
  Stream order serializes reuse of the device buffer. There is no
  `.item()`, `.cpu()`, `.tolist()`, synchronize, wait or new stream (AST
  test, including `stage_keys` and `finish`).
- **Completion without a later forward:**
  - layer 39's attention hook records an event and starts a dedicated
    waiter thread that blocks on that event only;
  - it then builds the file from host memory and writes it atomically,
    with a one-shot `.consumed` receipt carrying the file sha and any
    problems;
  - it disarms.
  The waiter never touches device caches (AST test). Block reuse after
  the request ends therefore cannot reach the capture (test: every cache
  overwritten right after the forward, with zero further `begin()` calls;
  the saved bytes equal the pre-reuse bytes).
- **Key dedupe:** a consumer index layer (24-36) references its owner's
  staged pages only when its complete used page-table row and cache
  length equal the owner's. Otherwise `meta.problems` records it, the
  entry carries no key pages, and the audit rejects the capture (test).
- **Incomplete forward:** if layer 39 runs without all 40 layers seen,
  nothing is saved and the session disarms (test).
- Timing changes are expected; see limit 3.

## Rules and verdicts

| Check | Rule | Classification |
| --- | --- | --- |
| provenance | exactly ranks 0..3; identities as above | failure: `invalid-capture` (never a pass) |
| structure | layer and role set, dtypes, geometry, page size, key-page hash | failure: `invalid-capture` |
| attention | pinned reference with FP32 q holding the BF16 values (output FP32); per-head relative L2 within E[mode] | outside: `deviation-found` |
| selection structure | exactly min(512, pool) unique ascending in-range in-pool entries, then a pure -1 tail | failure: `deviation-found` |
| selection, exact | reference top-k by (-score, index) equals production | differs within band only: `unresolved-boundary-order` (not a pass) |
| selection, approximate | differences beyond 1.01 BF16 ulp of the threshold | `deviation-found` |
| candidates (layer 20) | the same structure, exact and band rules at block level | as above |
| mapper | independent page-table resolution equals `indexed_slots` | mismatch: `deviation-found` |
| replicated indexer | per-rank hashes of q, scales, weights, key pages, cache length and candidates | inputs differ: `replicated_input_divergence`; inputs equal but selections differ: `replicated_selection_mismatch` (both deviations) |

A clean result is `within-conformance-envelope`, suffixed
` (provisional envelope)` until the option-2 conformance envelope JSON is
supplied. It is not labelled "correct".

## Remaining before execution (user-owned)

1. **Conformance envelope:** run `claude_conformance_mla.py` (option 2;
   CPU tests in `claude_test_conformance_mla.py`) on one idle GPU in the
   router or diagnostic image:
   - `--regime` is the decoder JSON; `--blocks 80927`;
   - it needs about 18.4 GB for the production-shaped pool plus 8 GB
     margin, and `--out` must be writable (not the read-only /gate
     mount);
   - it fails closed unless each harness plan's selection key equals the
     production key, executes the regime's configs as overrides, and
     requires bitwise repeatability;
   - its GPU path is untested here, so start with
     `--trials 1 --sample-rows 4`;
   - it writes `<out>.envelope.json` for the audit's `--envelope`, which
     is an empirical conformance measure, not an oracle.
2. **Build:** `build_decision_row.py`, then distribution and a
   select/restore amendment of kind `decision-row-capture`.
3. **Declared capacity:** `--num-gpu-blocks-override 80927`, which reuses
   B1's measured R_B1 records. Verify after boot with
   `claude_decode_sparse_mla`.
4. **Arm and run:**
   - write the trigger on all 4 hosts;
   - send one short, unrelated warm-up request (its eager prefill forward
     arms the process and allocates about 150 MB per rank) and confirm
     `CLAUDE-DECISION-ROW armed` in every rank's log. An idle server runs
     no forwards, so without a warm-up the arming allocation lands in the
     target request's first chunk;
   - send the frozen original input once, cold;
   - the waiter saves without any later forward; confirm
     `CLAUDE-DECISION-ROW saved` and the `.consumed` receipts;
   - an `aborted` log means a hook error was contained, serving continued
     and nothing was saved.
5. **Audit:** build the observed identity (response signature, regime,
   image, kit, trees) and run the audit with B2's expected identity and
   the envelope.
6. **Needle spans:** tokenize the frozen input offline. The late marker
   is at token 366986.

## Tie exposure (added 2026-09-25, attribution context only)

`tie_exposure(values, positions, selected, k, bound)` reports, per index
source at rank 0 and per layer-20 candidate block selection: the exact
rank-k threshold, the number of pool entries scoring exactly that
threshold (`tie_population`), `strictly_above`, `tie_admitted` (tied
entries production selected), `opposite_rule_changes` (entries the mirror
rule, highest index first, would swap) and the position quartiles of the
admitted ties. With `--spans`, needle coverage is reported under both tie
rules (`lowest_index_rule`, `highest_index_rule`). The summary carries
`tie_exposed_layers` and `needle_coverage_rule_dependent`; neither affects
the verdict. Ties are a property of BF16 scores over the pool, not a
deviation; a rule-dependent needle coverage says the selected set is
index-order dependent by design at that row.

## Same-boot control identity (added 2026-09-26)

`audit(..., control=...)` / `--control control.json`: the identity of two
unarmed cold requests on the very same capture boot whose full-response
signatures were identical (driver-verified before arming). When supplied,
the armed capture must equal the control in image_id, kit_sha256, full
source_trees, regime and response_signature (else `invalid-capture`), and
the B2 comparison is reported explicitly and separately in
`report['comparisons']['b2']`: `regime_equal` and `b12x_tree_equal` stay
validity requirements (plan equivalence), `response_signature_equal` is
reported with a `response_note` that names a historical B2 full-response
mismatch as observed across the runs' differences (KV capacity, boot,
image) without attributing it, and `accuracy`
restates that the original 524K retrieval remains a failed gate. Without a
control the legacy gate applies: the B2 full response must match.
