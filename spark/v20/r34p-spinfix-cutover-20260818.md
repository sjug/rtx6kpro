# r34-spark spin-fix amendment + DS4 production cutover (2026-08-18)

Image: `gilded-gnosis-v20-r34-spark-sm121-vllm9eb28db-b12xcd3ce19-fi1ac6942-cu132-20260817`
= r34-spark amended IN PLACE (pre-ceremony) with a 4th spark-overlay
file: vllm shm_broadcast.py `busy_loop_s: 1 -> 0.002` (SpinCondition
sched_yield grace never expires under decode cadence; ~3 head-node
P-cores pegged, -8C CPU / -4C GPU when fixed; A/B receipts in
spark/ii-r15/spin-wait-finding-20260817.md). vLLM tree
c3ffb74f -> 9eb28db0, integration patch sha bde4fb5c, DATE_TAG 20260817.
Build: all 5 GPU gates green; in-image busy_loop_s=0.002 verified.

## Staging lean-qual (dusty/kirby)

Boot 373s, KV 1,026,870; spin check IN-IMAGE (no overlay): 5.6% CPU,
1 hot core, 71.6C (stock same load: 15%, 3 cores, 82.9C); decode cells
43.1/60.4/86.6 (cc1 on the 43.7 contract band); probe-298 0/200.

## Production cutover (rusty/toby, user-authorized 2026-08-18)

Distribution over the 200G mesh per standing rule (dusty->10.11.11.5,
rusty->10.11.11.6; first-contact ssh-keyscan needed). Graceful stop
worker-first (podman stop -t 60), relaunch via deployed
run-ds4-v20-tp2-node.sh with IMAGE= override. Receipts:
- Boot 335s, KV pool 979,341.
- Smoke both served names correct; decode 41.7/59.4/91.3 (contract band).
- Spin check LIVE under load: rusty 5.7% CPU / 1 hot core / 72C,
  toby 5.5% / 1 / 69C. The 3-core production spin is retired.

Rollback: r33 image retained on both nodes AND remains the runner
default - a bare ROLE= relaunch restores prior production.

## Scope notes

- GLM cluster NOT cut over: #280 rewrote the EXL3 runtime GLM serves on;
  GLM-on-r34 has no qualification receipt. Image STAGED to all four GLM
  nodes (2-wide mesh fan from dusty) awaiting the user-scheduled window.
- II r15 staging pair (dusty/kirby): idle post-build; its next respin
  should bake the same spin fix (currently proven there via overlay).
- The 2026-08-21 source ceremony records corrected r34, including this
  amendment and the later complete queue correction, as commit
  `a565fa90df2bb0ac4480233cf7cd909001f8e32d`.
