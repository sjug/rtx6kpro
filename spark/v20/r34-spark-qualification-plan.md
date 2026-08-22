# GG r34-spark: R7 serving qualification plan (SM121)

Status: build in progress on dusty (2026-08-14). This plan gates the
"serving-qualified" declaration; build-time gates are separate and live in
`build-gilded-gnosis-v20-r34-spark-sm121-cu132.sh` (oracle, GEMM parity,
execution contract, scratch matrix, graph capture).

Release criterion (agreed 2026-08-15): K6 works without any override,
passes the corrected encoder and full-pipeline gates, survives cold and
cached boots, and shows no material quality regression on SM121.

## Qualification boundary: online-K6 (K,N) signature enumeration

The build-time graph gate covers the two known R7 TP4 shared-expert tile
families exactly:

- gate/up merged shard: 6144 -> 1024
- down shard: 512 -> 6144

The eligible-projection set is checkpoint-driven, so the full selected
(K,N) set cannot be hand-derived safely. Agreed boundary (2026-08-15):

1. On the FIRST real R7 TP4 boot, record the complete set of (K,N)
   signatures the online-K6 policy selects (boot log / cache-key dump).
2. Diff against the two gated families above.
3. Any additional family gets added to
   `tests/verify_b12x_trellis_graph_capture.py` and re-run in-image
   BEFORE declaring the image serving-qualified.

This is a qualification boundary, not a build blocker.

## Serving qualification checklist (user's step 6)

- [ ] Stage R7 checkpoint; cluster window scheduled by user.
- [ ] Cold boot with empty K6 online cache: full encode path, boot time,
      no capture errors across the whole graph ladder.
- [ ] Record selected (K,N) set; execute the enumeration boundary above.
- [ ] Warm restart: all cache hits, byte-identical load, boot time.
- [ ] MTP0 and MTP3 serving paths.
- [ ] Quality: online-K6 vs VLLM_EXL3_ONLINE_QUANT=none KLD comparison;
      no material regression.
- [ ] Decode/prefill throughput, graph capture, memory headroom on the
      real NCCL/RoCE TP4 topology.
- [ ] Soak.
- [x] Post-build in-image re-run of stock
      `tests/test_quant_fn.py::test_encode_ideal` (re-measure the
      retraction-flagged upstream-report item 7 claim). DONE 2026-08-14
      in the r34-spark image: 48/48 parametrizations pass (3 codebooks x
      K1-8 x bsz {1,64}), verbatim stock body incl. its both-keys-boolean
      quant_args convention; only deviations device cuda:0 (stock
      hardcodes cuda:2) and skipping the module-level Llama-8B load used
      by other tests. Under upstream key-presence selection the three cb
      arms collapsed to one codebook (the original false "failure"); under
      the patched value-based contract each arm selects its own codebook
      and all ideal encodings round-trip losslessly.
