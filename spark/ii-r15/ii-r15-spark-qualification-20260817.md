# II r15 Spark/SM121 qualification record (2026-08-17)

Image: `infernal-invocation-r15-spark-sm121-vllm8ad1330-b12x96e5d3d-fi1ac6942-cu133-torch213-20260817`
(30.8 GB; build + full gate battery PASS on dusty 03:26). First II image
with the corrected EXL3 program: online K6 OPEN (no override anywhere),
value-based codebook contract, capability-gated B12X graph scratch.
Plan: `qualification-plan.md`. Bench JSONs: `bench/`.

## Phase 1 - DS4-Flash-0731 TP2 (dusty/kirby): PASS

Boot contract:
- Cold boot (fresh r15 cache fingerprint): 306s launch-to-ready,
  KV pool 1,082,207 tokens (largest DS4 pool recorded on this pair).
- Warm restart: 197s, pool 1,083,468 (stable). Later boots 198-286s
  across arms.
- Both served names; correct arithmetic on both; truncation honored.
- Reasoning contract: r15 == r33 production BYTE-level on the control
  probe (effort=high deterministic 40-token terse answer, low/medium
  ~103-117; reasoning_content empty at all efforts on BOTH lines -
  the deepseek_v4 parser only fills it when the model emits think tags,
  which DS4-0731 skips on trivial prompts). Probe-design lesson: the
  contract gate is cross-release consistency, not an assumed monotonic
  shape.

Correctness battery (signature-based per the r10 methodology):
- probe-298: 0/200 + 0/800-soak corruption hits (validates the MERGED
  #298 on SM121/RoCE).
- decode-cells: 41.1/64.1/91.8 tok/s cc1/2/4 - at or above the r34
  baseline cells.
- 524k needle: 519,440-token prompt, total exactly 524,288; exact
  recall, 356s prefill, clean stop. Cap re-earned on r15 (#289/#290
  moved context paths since r10).
- Strict tools (#320, NEW): 24/24 exact-sum schema JSON; 24/24 c1 +
  64/64 c8 conformant required tool calls; 225k-context (231,641
  tokens) tool call conformant with EXACT needle recall into the
  arguments (126s). STRICT-TOOLS-OK.

## Full backend sweep (6 arms x 15 cells, acceptance-normalized) vs r34

| arm | r34 steps-geo | r15 steps-geo | delta |
|---|---|---|---|
| b12x-a8 (prod) | 22.24 (lucky boot; ABA 21.67) | 21.45 / 21.30 (2 boots) | parity, see ABA |
| b12x-a16 | 22.27 | 21.95 | -1.4% |
| b12x-a8-dglin | 21.83 | 21.59 | -1.1% |
| lucifer-cutlass (autotune OFF) | 19.58 | 19.50 | -0.4% |
| lucifer-default | 18.63 | 18.65 | +0.1% |
| b12x-a8 + native indexer | 21.58 | 21.44 | -0.7% |

**A/B/A resolution**: r15 a8 first read -3.6% vs r34; two r15 boots
reproduced each other (21.45/21.30), then an r34 re-boot on the same
pair measured 21.67 - r34's original 22.24 was a +2.6% lucky boot.
Like-for-like residual ~1-1.5% = inside the boot band. **Verdict:
engine parity r15 vs r34 across all six arms.**

Other sweep observations:
- Backend ordering identical to r34: a8 best cc1, a16 best cc4 (+6%
  over a8 at cc4), lucifer arms 10-15% behind.
- lucifer-cutlass still requires ENABLE_FLASHINFER_AUTOTUNE=0 (same
  FlashInfer pin; TMA tactic failure documented in the r34 record).
- KV pools: a8 1,082-1,083k; a16 1,032k; dglin 936k; cutlass 1,052k;
  lucifer-default 1,148k; native-idx 1,081k. All above their r34
  counterparts by ~60-70k tokens (r15/cu133 side memory win).
- b12x #214 (inactive routes): no measurable decode win on our config -
  our capture ladder includes exact verify multiples, so padding rows
  are already rare; the fix is defensive here.

## Production config confirmed: b12x-a8, b12x indexer, K5 probabilistic,
## 524288, c4 channel pin - unchanged from the GG contract.

## Not covered (open)
- Phase 2 GLM-EXL3 TP4 re-test + K6-open serving program: needs the
  user-scheduled 4-node window (see plan).
- Long soak beyond the 800-request probe (folded into any cutover
  decision).
- DS4 II-cutover decision: user's call. Evidence: 4-gate parity, #320
  structured output proven, biggest KV pools recorded, boot receipts.

State at record time: dusty/kirby serving r15 production defaults
(staging); rusty/toby production DS4 stays GG r33; GLM cluster stays GG
r33. The release-source ceremony completed on 2026-08-21: the r15 recipe
is commit `ff4183531798d5cbff741b7d8d9c9601ccbbe541`, and the benchmark
tool used for the later acceptance-normalized comparisons is commit
`88acca8b8e5a5f13b24c494d56f35ea2f0b4375f`.
