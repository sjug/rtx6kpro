# v16 accuracy sweep — rusty/toby pair, 2026-07-14/15 (stopped early)

Accuracy/correctness validation of the v16 image cells
(`fathomless-firmament-v16-spark-sm121-...-20260714`), two-node TP2,
`MAX_NUM_SEQS=8`, `GPU_MEM=0.85`, temp 0 for dataset profiles, benchmarks via
`llm-inference-bench` profiles at cc8 (estonia cc4). Goal: does the fastest
serving cell (lucifer-cutlass dspark) pay for its decode speed with accuracy?

**Stopped by user on the morning of 07-15** after cell 1 of 4 — only the
lucifer-cutlass cell has complete results. No paired comparison is possible
yet; the numbers below stand alone against public reference scores.

## Results — lucifer-cutlass dspark (serving-default candidate)

| Test | Result | Notes |
|---|---|---|
| gsm8k, full 1,319 items | **97.0%** (1279/1319) | p50 226 completion tokens; exactly 1 item hit the 16k cap; 0 request errors |
| estonia (177k-token context) | **100%** (30/30) | p50 2,535 tokens; FlashInfer/CUTLASS long-context path fully healthy |
| lavd-test (ledger consistency) | 90% (9/10) | p50 18k tokens |
| gsm8k 25-item slice | 100% (25/25) | calibration run |
| hotel-lights | 7/8 completed correct | partial: 22/30 requests hit the venv connect failure (see incidents) |

97% GSM8K is consistent with DeepSeek-V4-class reference scores — no evidence
that the MXFP4×MXFP8 CUTLASS MoE path degrades accuracy, and estonia says the
sparse long-context path is exact. **Verdict pending the mtp0 baseline pair**
for a same-hardware A/B.

## Not completed

- mtp0 baseline: hotel-lights timed out 3× (45 min timeout vs ~20k-token
  answers); gsm8k was ~25 min in when stopped. No usable files.
- b12x-a16 dspark and mtp2 cells: never started.
- To resume: `scripts/overnight_accuracy_sweep_v3.sh` is the corrected
  orchestrator (300-item gsm8k subsets, no hotel timeout trap); or run
  individual cells via `run_bench.sh --test-profile gsm8k --profile-concurrency 8
  --max-tokens 16384` against a relaunched cell.

## Incidents (cost ~5 h of the night)

1. **uv portable-python connect failure** (~23:37): the bench venv's
   uv-managed cpython lost the ability to complete TCP connects to
   rusty:8000 — SYNs unanswered for that binary only, while system python and
   curl connected fine in the same second. Kernel-level, per-binary,
   unexplained, and transient: by morning a freshly recreated `uv venv`
   (same managed cpython 3.12.10) connected normally, and the venv now runs
   on the uv-managed interpreter as usual. (The overnight unblock
   temporarily used system python; recreated properly the next morning.)
   Probe evidence: `connect-probe.log`.
2. **hotel-lights timeout miscalibration**: the model spends ~20k reasoning
   tokens per hotel answer; the 45-min timeout (calibrated on the
   connect-corrupted first run) killed healthy runs on both cells it touched.
   The corrected v3 script existed by 00:50 but a Claude-harness permission
   outage prevented swapping it in overnight; the v2 script ran to plan
   otherwise.

## Files

- `benchmark_results-v16-lucifer-*.json` — raw per-run data (item_id per run;
  gsm8k JSON supports paired `--compare-baseline` analysis later)
- `sweep-steps.log` — orchestrator timeline; `bench-full.log.gz` — full bench output
- `connect-probe.log` — 15 s connect probes during/after incident 1
- `scripts/` — orchestrator v2 (as run), v3 (corrected, unused), summarizer
