# Fresh R32 Qwen return benchmark, September 21

Requested after restoring R32 to dusty/kirby. Standard unchanged run_bench.sh,
MTP3/aligned, 30 seconds per cell, contexts 0/16K/32K/64K/128K, C1/C2/C4.
Both harness digests, checkpoint and image/serving policy were checked.
The endpoint was idle before the run. All 15 cells passed the standard validator.
Execution ran 09:09:54 to 09:23:00 EDT and exited zero. Raw result SHA256:
`6c126f12132cf78150835f52b618d373e7142396042130c931c0e9d07f23244e`.

## Results

Geometric means over five contexts. Karmic repetitions share one boot; R32 is
the fresh restored boot. This is not an independently repeated multi-boot study.

| Concurrency | Karmic r1 tok/s | Karmic r2 tok/s | R32 tok/s | Karmic r1 steps/s | Karmic r2 steps/s | R32 steps/s |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 47.23 | 46.44 | 44.19 | 22.86 | 22.80 | 22.07 |
| C2 | 72.97 | 75.84 | 74.92 | 36.62 | 36.76 | 36.58 |
| C4 | 116.82 | 120.46 | 115.79 | 55.84 | 55.53 | 55.79 |

R32 engine steps versus Karmic: C1 -3.46%/-3.23%, C2 -0.11%/-0.48%,
C4 -0.09%/+0.47%. R32 effective acceptance is 2.002/2.048/2.075 at C1/C2/C4;
Karmic r2 was 2.037/2.063/2.169. The C4 output difference versus r2 is primarily
acceptance, not slower R32 engine steps.

| Prefill | Karmic r1 tok/s | Karmic r2 tok/s | R32 tok/s | R32 change vs r1 / r2 |
|---|---:|---:|---:|---:|
| 8K | 3058 | 3089 | 3124 | +2.16% / +1.13% |
| 16K | 2972 | 2994 | 3004 | +1.08% / +0.33% |
| 32K | 2910 | 2896 | 2918 | +0.27% / +0.76% |
| 64K | 2687 | 2674 | 2764 | +2.87% / +3.37% |
| 128K | 2479 | 2441 | 2501 | +0.89% / +2.46% |

## Verdict

This does not establish that R32 is broadly faster than Karmic. R32 has modestly
better prefill in this comparison, Karmic has higher C1 engine steps, and C2/C4
engine steps are within half a percent. The prior roughly 3% C4 advantage of the
historical R32 receipt does not reproduce on this fresh R32 boot. Keep the
historical measurements, but retract any claim of an established Karmic C4
regression. No DS4 conclusion transfers to Qwen.

R32 remains serving per the rollback decision; no deployment change was made
by this benchmark. No benchmark repository edits or commits were made.

## Health and limitations

Both nodes had zero NVRM allocation warnings, Xids, tracebacks and active
clock-event samples during this window. Logs and one-second GPU samples are
retained. First-use Triton compilation warnings occurred at 09:09:56 to
09:09:59 during initial preparation, and at 09:15:58 as the first C2 cell was
being admitted. These are not hidden or described as a JIT-free run. The grid
passed its warmup, error, capacity and completeness checks. All inference POSTs
in the captured head log originated from the benchmark client.

`comparison-karmic-r1.json` and `comparison-karmic-r2.json` hold protocol-checked
per-cell comparisons, acceptance and prefill details. In those files baseline
means Karmic and candidate means the fresh R32 return run.
