# DRAFT — issue for local-inference-lab/vllm (+ companion digest for Luke/Martin)

---

## Issue title

FlashInfer decode builder reuses frozen-shape persistent wrappers for
reduced-depth spec-as-decode steps → EngineDeadError

## Environment

- Tree: `dev/gilded-gnosis` @ `3003860` (r24 DS4-runtime lineage; our SM121
  composed image `gilded-gnosis-v20p3-spark`, vLLM tree `92b27a47`)
- Hardware: 2× DGX Spark GB10 (SM121, aarch64), TP2 over RoCE
- Model: `poolside/Laguna-S-2.1-FP8` + `Laguna-S-2.1-DFlash-FP8`, K=5,
  `--attention-backend FLASHINFER`, kv fp8, block 128, chunked prefill +
  async scheduling, `cudagraph_mode FULL_AND_PIECEWISE`
- Not model-specific: any GQA target + dflash/eagle-family drafting on the
  FlashInfer decode path is exposed. MLA paths (`deepseek_v4_attention`)
  are not affected.

## Two crash signatures, same defect

Both kill the engine (`EngineDeadError`) via
`flashinfer/decode.py: fast_decode_plan`:

1. `ValueError: q_len_per_req is part of the frozen cudagraph shape: this
   wrapper was planned with 6, got 8` — a lone request straddling a
   chunked-prefill boundary: the short prefill tail is fused with the spec
   step (tail + 1 + K tokens). Scheduler dump excerpt from the live hit:
   `num_computed_tokens=[8192], num_scheduled_tokens={req: 8},
   num_spec_tokens_to_schedule=5` (prompt 8194, budget 8224 block-aligned
   to 8192 → tail 2).
2. `... planned with 6, got 5` — spec depth truncated as a request
   approaches `max_tokens` (5 tokens remaining → 1 + 4 drafts).

Signature 2 is reachable by **any length-capped request whose budget is not
a multiple of K+1** — i.e. ordinary production traffic. Capture-ladder or
batch-budget changes only reshape the window; they cannot close it.

## Root cause (traced end-to-end)

1. `config/speculative.py:1090` — `method in ("dflash","dspark")` forces
   `parallel_drafting=True`.
2. `attention/backend.py:888` — `_init_reorder_batch_threshold` therefore
   sets `reorder_batch_threshold = 1 + 2K` (11 at K=5). This ceiling
   *deliberately* admits reduced-depth lone steps as decodes so
   spec-as-decode kernels can serve them.
3. `attention/backends/utils.py: split_decodes_and_prefills`
   (`require_uniform=True`) — a single-request batch is trivially uniform,
   so lone steps with q_len ∈ {1..11} classify as decode at their actual
   q_len.
4. **Defect — `attention/backends/flashinfer.py:1510`**: the builder's
   persistent-wrapper eligibility predicate checks only capacity:
   `enable_cuda_graph and pure_decode and
   num_decode_tokens <= _decode_cudagraph_max_bs`. It never checks the
   per-request shape invariant. The comment two lines below states the
   assumption ("Spec-as-decode verify batches carry a uniform ... tokens
   per request") without enforcing it.
5. The persistent wrappers are planned during capture with frozen
   `q_len_per_req = 1 + K`; FlashInfer correctly refuses the mismatched
   replan and the engine dies.

Note the dispatcher side already does this correctly —
`cudagraph_utils._is_compatible` requires
`desc.uniform_token_count == uniform_token_count`, so the *model* runs the
step piecewise while the *attention builder* independently grabs the frozen
wrapper: two sources of truth that disagree.

The invariant, stated once: **a persistent FlashInfer decode wrapper may be
reused only when the runtime q_len_per_req equals the value frozen during
wrapper planning.** The classification ceiling (1+2K) and the planned shape
(1+K) are different quantities by design; substituting
`reorder_batch_threshold` for the planned length would look right and be
wrong.

## Deterministic reproducers

With any dflash GQA deployment, K=5, FULL decode graphs:

- A: chat completion, `max_tokens: 8` (any small value ≢ 0 mod K+1) →
  crash within one request (signature 2).
- B: prompt of `chunk_size + small tail` tokens (e.g. 8,194–8,200 at an
  8,192-token prefill chunk) → crash at the boundary step (signature 1).

## Fix (patch available; carried in our v20p3p1 overlay)

Complete the predicate at the site that owns the decision:

- store `self._planned_decode_q_len = 1 + num_spec_tokens` in the builder
  (`__init__`, same quantity that already sizes `_decode_cudagraph_max_bs`);
- factor the eligibility check into a pure function
  (`persistent_decode_wrapper_eligible`) whose docstring carries the
  invariant and the 1+K vs 1+2K warning;
- require `decode_q_len == planned_decode_q_len` for the persistent
  wrapper; mismatched steps take the existing dynamic wrapper
  (`_get_decode_wrapper(..., use_cudagraph=False)`), which replans per call.

One file, +46/−5. Steady-state verification (q_len = 1+K) keeps FULL
graphs; only the rare boundary/truncation steps run piecewise.

### Validation

- Unit matrix (in our build gate): threshold derivation 11-with /
  6-without parallel drafting; q_len 6 → persistent (×1 and ×8 requests);
  every q_len in {1..11}\{6} → dynamic; q_len 12 → prefill classification
  via the real splitter; capacity/pure-decode terms; no-spec planned=1.
- E2E on the pair: both reproducers complete with the engine alive;
  outputs token-exact vs PIECEWISE at temp 0; 163 "Replaying FULL CUDA
  graph" lines confirm the persistent path still carries normal decode;
  full bench sweep (the original crash trigger) survives.
- Perf: FULL(capture 96) vs PIECEWISE at identical sampling: +28.8%
  15-cell decode geomean, +49% coding-peak (32.8 vs 22.0 tok/s), cc1 ITL
  55→32 ms.

## Suggested longer-term design (beyond the correctness fix)

- Per-q_len persistent wrapper families (what FlashInfer's error message
  itself suggests) so reduced-depth steps can also replay FULL graphs —
  a performance extension, not needed for correctness (boundary steps are
  rare).
- Make the dispatcher's batch descriptor the single source of truth for
  graph eligibility instead of re-deriving it in the attention builder.
- Rename `_decode_cudagraph_max_bs` — it is a token count, not a batch
  size ("planned with 6, got 8" took longer to find because of it).

---

## Companion digest (accumulated findings, separate from the issue)

1. **FlashInfer pin**: r24 pins `801d57a`, which predates the
   flashinfer-ai#3932 merge (2026-07-31). We measured the pre-merge b12x
   MoE NVFP4 arithmetic at 0.61× (normal-scale) / 0.24× (subnormal-scale)
   vs reference on SM12x. We carry `7ad08da`; the pin retires when GG's
   FlashInfer advances past the merge (our arithmetic gate verifies).
2. **tvm-ffi build/runtime skew**: the build stage floating
   `apache-tvm-ffi >= 0.1.6` (resolved 0.1.13) against runtime 0.1.10
   breaks xgrammar at import (`TVMFFIGetCustomAllocator`). Pin the build
   stage to the runtime version.
3. **Reproduce-mode hazard**: rebuilding a release from branch refs breaks
   when branches move; our Dockerfiles now sha-fetch pinned commits.
   Recommend the same for GG release reproduce paths.
4. **Draft-depth guidance for GB10**: K=5 beat K=7 on both DS4-Flash-0731
   (dspark) and Laguna-S-2.1 (dflash) by ~10% coding-peak; deep positions
   accept 22%/19% (K7 p5–6) and worse at temp 1.0. Both model cards
   recommend deeper (7/15). Community NVFP4 data agrees (K6–15 ≈ 0%).
5. **EXL3 is not aarch64-portable** (immintrin/`__builtin_cpu_supports`,
   incl. upstream turboderp): we build with `SKIP_EXLLAMAV3` + stubs.
6. **r16 DS4-0731 sizing on 2x RTX PRO 6000 (96 GB)**: the KV-check
   requirement decomposes as a fixed DSpark verify cache of roughly
   2 x MAX_NUM_BATCHED_TOKENS x 452 KB (length- and seqs-independent)
   plus ~4.4 KB/token MLA KV. At the release defaults (batched 8192)
   the fixed block is ~6.8 GiB, which makes the documented 131072
   envelope knife-edge at 0.975 on headless cards and unbootable next to
   any desktop residency; batched 4096 collapses it to ~3.4 GiB and the
   full envelope boots at 0.950 with a compositor resident (validated
   2026-08-04: KV pool 238,998 tokens, 1.82x). Suggest documenting
   MAX_NUM_BATCHED_TOKENS as the primary sizing lever in the r16 page;
   vLLM's "estimated maximum model length" hint is misleading here (it
   divides by the fixed-block rate as if per-token).
7. **RETRACTED (2026-08-15), replaced by a harness lesson + an API
   footgun report**: we previously reported the a1-retile ENCODER
   (`quantize_tiles`) as inconsistent with the 3INST/MCG codebooks
   fork-wide. That finding was OUR harness's bug, not an encoder defect:
   `quantize_tiles` selects codebooks by KEY PRESENCE
   (`mcg = "mcg" in quant_args`), and our oracle passed BOTH keys with
   boolean values, so every arm encoded with one codebook - the "identical
   mismatch counts on SM120 and SM121" were the harness reproducing
   itself, which we misread as cross-platform confirmation. The DECODE
   family findings stand unchanged (exhaustive oracle: `decode`,
   `reconstruct`, GEMM bit-exact on both platforms, all three codebooks,
   all 65,536 indices). What we now ship and recommend upstream:
   (a) value-based codebook selection with mutual-exclusion rejection
   (`bool(quant_args.get("mcg", False))` etc.) - the key-presence
   semantics is a footgun that silently reinterprets explicit False;
   (b) a corrected oracle testing each codebook separately at K3-K8
   including the production K6/MCG arm, encoder/decode round-trip, and
   both-selected rejection; (c) an online-K6 production-path pipeline
   gate (meta-Hessian fallback, output scaling, packed tensors, cache-hit
   byte-identity). Our earlier claim that stock
   `tests/test_quant_fn.py::test_encode_ideal` fails on SM12x is
   RETRACTED after re-measurement (2026-08-14, GG r34-spark image,
   value-based selection patch applied): 48/48 parametrizations pass
   (3 codebooks x K1-8 x bsz {1,64}), stock test body verbatim including
   its `{"mcg": mcg, "mul1": mul1}` both-keys-boolean convention. The
   observed "failure" was the same key-presence collapse: under presence
   semantics all three cb arms select one codebook, so ideal encodings
   for the other two cannot round-trip. Note the stock test is therefore
   itself a victim of the footgun on unpatched trees - with value-based
   selection it works as written.
8. **r33 dropped `return_prompt_logits`**: the KLD runner
   (`scripts/glm52_exl3_shared_h_kld.py`) feature-detects
   `return_prompt_logits` in `SamplingParams`, but neither the r28 nor r33
   composed trees carry it - the harness silently falls back to the
   `prompt_logprobs=-1` container path. On 121 GB UMA hosts (DGX Spark)
   that fallback OOM-kills rank 0: the model-runner side accumulates
   compact tensors, but the engine-to-frontend conversion into
   Python-object logprob containers peaks at ~28 GB RSS for a 2,047 x
   154,880 capture (kernel oom-kill confirmed). Restoring the dense/tensor
   return path would make KLD capture feasible on unified-memory hardware;
   until then we adopt your published KLD figures (justified by the decode
   bit-exactness above).
9. **r31+ native L2 offload cache poisoning** (community report, 08-08
   summary): failed disk read permanently marks blocks missing with no
   cleanup path (PR #252/#254 interaction). We do not run offload; flagged
   for awareness.
10. **For poolside (separate HF discussions, FYI)**: RC2 config ships only
   the 1M rope profile (card's 256K profile: factor 32, af 1.34657 — not
   shipped); laguna SWA layers reuse global rope
   (`laguna.py:389` warning); tokenizer snapshot trips the transformers
   Mistral-regex warning (`fix_mistral_regex=True` not plumbable through
   vLLM today). Also: `generation_config.json` embeds
   `speculative_config` K=15 with the BF16 draft — stock `vllm serve`
   silently inherits it.
