# II r15 Spark/SM121 qualification plan (drafted 2026-08-17)

Image: `infernal-invocation-r15-spark-sm121-vllm8ad1330-b12x96e5d3d-fi1ac6942-cu133-torch213-20260817`
(build gates ALL PASS on dusty 2026-08-17 03:26; corrected EXL3 program:
online K6 open, no override, capability-gated graph scratch).

Probes reused from `spark/ii-r10/` unless noted. Methodology carried
forward: DSpark-probabilistic output is nondeterministic at temp 0 even
solo, so DS4 gates are signature/distribution-based, never token-exact;
token-exact remains valid only on greedy paths.

## Phase 1 - DS4-Flash-0731 TP2 (dusty/kirby)

1. Boot contract. Cold boot on the fresh r15 cache fingerprint (record
   boot time), then warm restart (JIT cache hits, boot time). Both served
   names resolve; reasoning-contract probe (default effort=high); required
   tool-call probe; truncation probe. Runner: run-ds4-ii-r10-tp2-node.sh
   with IMAGE=r15 (pass-through env, c4 channel pin, GPU_MEM 0.87,
   MAX_MODEL_LEN 524288).
2. Correctness battery (signature-based):
   - probe-298: 200 mixed-cc + 800-request soak, corruption-signature
     watchdog, 0 hits required. #298 is MERGED in the r15 base, so this
     validates the upstream fix on SM121 rather than our workaround.
   - decode-cells-probe vs the r34-spark pair baselines.
   - deep-ctx-needle: needle ~519k exact, boundary admission/rejection at
     524288, 4/4 concurrent at long context.
   - Structured output (NEW, r14 #320 + #294/#295): strict tool-grammar
     probes under DSpark - valid-JSON arithmetic check (mirrors the r15
     upstream receipt), grammar conformance at c1 and c8, and a
     225k-context strict-tools case (r14 receipt pattern, 2-GPU scale).
   - Inactive-route invariant (NEW, r15 b12x #214): decode cells A/B vs
     r34 - equal-or-better cc1 steps/s expected (padding-row work
     removed); identical corruption-signature cleanliness across cc 1-4.
3. Performance: 15-cell grid (0/16k/32k/64k/128k x cc 1/2/4) with
   acceptance-normalized steps/s (bench 0b4185b+), compared against the
   r34-spark b12x-a8 grid on this same pair. Record KV pool.
4. Soak: multi-hour mixed load, watchdog on, 0 errors.

## Phase 2 - GLM-5.2 EXL3 TP4 (the r10 blocker re-test; needs a
## user-scheduled 4-node window)

- Prefill re-test FIRST: r10 plateaued at 169-299 tok/s vs 767-798 on GG
  (hardware exonerated). r15 postdates #300/#148 projection-mixed R7,
  #301 sparse-MLA contracts, and r13 native SQG - re-measure before any
  upstreaming of the r10 finding.
- K6-open program on SM121 (first time on the II line):
  - Cold boot, empty online-K6 cache: full encode path, complete FULL
    graph ladder with no capture errors (the scratch fix's real test).
  - Record the complete selected (K,N) signature set on first boot; diff
    against the two gated TP4 shared-expert families (6144->1024,
    512->6144); any new family goes into the graph gate BEFORE the
    serving-qualified declaration (same boundary as r34).
  - Warm restart: all cache hits byte-identical. r10-era caches must MISS
    cleanly (encoder identity now carries +p.<patch-sha>).
  - Quality: online-K6 vs VLLM_EXL3_ONLINE_QUANT=none KLD; MTP0 and
    MTP3; acceptance parity with the r33 record (~3.20).
  - Targets: decode >= 23.7, prefill 767-798 band, pool >= 309k @262k.
- DCP: start DCP1 (the upstream-qualified shape); DCP2 is a separate arm
  (r14 #309 deferred DCP workspace in play) before it can replace the GG
  DCP2 production config.
- Runner: run-glm52-ii-r10-tp4-node.sh updated to r15; pass-through-only
  env (no r33-era EXL3 tuning pins); GPU_MEM 0.84 + explicit KV pin.

## Phase 3 - Laguna: PARKED (user decision). r10 receipts stand.

## Cutover criteria

- DS4: all Phase-1 gates green -> cutover is a user decision (arguments:
  #214 inactive-route perf, structured-output correctness, 4-gate parity).
- GLM: only after Phase 2 fully green including soak; release criterion
  as r34: K6 works with no override, corrected gates pass, cold and warm
  boots survive, no material quality regression.
- Fleet fan-out (beyond dusty/kirby staging) only after the relevant
  cutover decision.
