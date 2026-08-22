# vLLM shm-broadcast spin-wait on GB10: reproduced, fixed, zero cost (2026-08-17)

Source: nacyot.github.io/artifacts/vllm-spin-wait-gb10-en/ - vLLM
`SpinCondition.wait()` (vllm/distributed/device_communicators/
shm_broadcast.py) spins sched_yield for up to `busy_loop_s=1` s between
queue messages; decode messages arrive every few ms, so the sleep path
never engages and reader threads peg P-cores continuously.

Our whole fleet carries the same default (`busy_loop_s: float = 1`,
line 134 in the II r15 tree; generic vLLM code, present in GG r33/r34
too). On UMA GB10 the spinning CPU shares the thermal budget with the
GPU, so this is not cosmetic.

## A/B on dusty/kirby (II r15, DS4 TP2, identical 4-stream decode load,
## measured via nv-monitor :9101, 60 s windows)

| | dusty (head) stock | dusty fix | kirby (worker) stock | kirby fix |
|---|---|---|---|---|
| CPU avg | 15.0% (~3 cores) | 5.4% (~1 core) | 5.3% | 5.2% |
| hot cores >90% | 3 | 1 | 1 | 1 |
| CPU temp avg/max | 82.9/85 C | 75.0/77 C | 75.6/78 C | 73.2/76 C |
| GPU temp avg | 69.5 C | 65.5 C | 67.2 C | 64.8 C |

Decode parity (decode-cells, same boot pair): stock 37.9/59.6/95.8 vs
fix 37.7/59.4/96.4 tok/s at cc1/2/4 - identical within noise, matching
the article's claim.

Platform nuance vs the article: the spin concentrates on the HEAD node
(engine writer + API server + local worker on the broadcast queue -> 3
pegged X925 cores, -8 C CPU / -4 C GPU when fixed); the headless worker
node barely spins (1 hot core either way).

## Fix

One line: `busy_loop_s: float = 1` -> `0.002` (2 ms grace keeps burst
latency, sleeps through steady decode). Deployed on the staging pair as
a labeled read-only overlay:
  dusty:~/ii-r15-hotfix/shm_broadcast-spinfix.py
  kirby:~/ii-r15-hotfix-shm_broadcast-spinfix.py
mounted over $TARGET via the runner's EXTRA_PODMAN_ARGS. Staging pair
LEFT RUNNING with the overlay (running code differs from image digest;
all results after 2026-08-17 ~10:00 labeled accordingly).

## Implications / open decisions

1. PRODUCTION IS AFFECTED TODAY: rusty (DS4 head) and sparky (GLM head)
   spin ~3 P-cores whenever decoding, at production duty cycle. Overlay
   adoption there is a user decision (needs a restart window).
2. Respin candidates: add the one-liner to the spark overlays of the
   next r15-spark/r34-spark builds (or push env-tunability upstream -
   the fork could default 0.002 on aarch64/GB10).
3. Upstream report item (vLLM proper + local-inference fork): default
   busy_loop_s=1 is pathological on thermally-coupled SoCs.
4. The r15-vs-r34 latency comparison's +32 ms cc1 TTFT note is
   unrelated (both arms measured with stock spin behavior).

## AMENDMENT 2026-08-18: GLM-on-r34p first boot HUNG - cause UNRESOLVED

First-EVER GLM-5.2 boot on any r34 build (TP4/DCP2, 4 nodes) hung
post-init: engine init completed (190.8 s), all four workers reached
final init lines, then zero GPU util while the EngineCore writer failed
to acquire a broadcast block for 11+ min (writer-starvation message
repeating). GLM rolled back to r33 (restored, smoke-verified). Hang
logs: spark/v20/glm-r34p-hang-logs/.

SUSPECT SET - every r33->r34 delta GLM exercises and DS4 does not, PLUS
the spin fix: #280 EXL3 runtime rewrite (largest delta, GLM-only),
#281 InstantTensor borrowed-buffer (GLM loads via InstantTensor),
the spark scratch-planner overlay (GLM trellis shapes), busy_loop_s
0.002. The writer-starvation log is a GENERIC symptom of ANY stalled
reader, not specific to the spin change ([[harness-false-positives]]:
do not promote component-name pattern-matching to diagnosis). Weak
circumstantial pointer to the spin fix: the stall window (post-init,
pre-API) is IPC-heavy and the fix is the only change in that file.

DISCRIMINATOR (pending user-scheduled window): boot GLM on r34p with
STOCK shm_broadcast.py bind-mounted over the image. Boots clean ->
spin fix implicated (then also reassess DS4 prod). Hangs identically ->
spin fix exonerated; investigate #280/#281/overlay. EITHER WAY: py-spy
the stalled rank BEFORE teardown (lesson: this hang was torn down
without live stack capture - logs are a weak substitute).

DS4 rusty/toby remain on r34p, serving clean (TP2 has ~7 clean boots +
battery + live traffic on this change). GLM prod: r33.
