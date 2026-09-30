# DS4 Vision full qualification on the Karmic beta image, September 29

Image `sha256:500ae05b98da0658c1a5e1820387f96f2121c5659bd7954ad1c5861f20934f05`
(`karmic-beta-20260929-spark-sm121`), rusty head, toby worker, user-authorized window.
Goal: the best serving configuration, not R38 reproduction. Fresh boot for this pass.
The quick prefill test that preceded it is in `PREFILL-20260929.md`.

## Configuration

Upstream beta defaults from the image launcher and recipe `ds4-vision` profile (FlashInfer
autotune on, `reasoning_effort` high, default multimodal processor cache, AOT compile,
async scheduling, prefix-cache retention 4096, multi-stream GEMM threshold 1024). GB10
choices: utilization 0.85, `NCCL_PROTO=LL,Simple` on the pair rails, 524,288 context,
TP2/DCP1, DSpark K3 probabilistic, 4 sequences, 4,096 batched tokens, capture to 16.

Backend `b12x-a8` (B12X attention, MoE and dense linears). Upstream's profile describes
native dense (`b12x-a8-dglin`); `b12x-a8` was chosen because the September 28 GB10
evaluation measured it about 3 percent faster in engine steps on R38p (its `b12x-k3` arm),
and the September 28 depth evaluation found K3 better than K6.

## Correctness

- Pre-launch: image ID, runtime-file checksums and model manifest verified on both nodes.
- First completion `333`, stop; model list exact.
- `VISION-QUALIFICATION-PASS`: nonthinking and default-effort arithmetic, three vision
  cases and three vision continuations, tool call and tool round trip, four concurrent
  mixed requests, exact retrieval at 16,384 (7.12 s, 0 cached) and 524,000 tokens
  (334.4 s, `739184`, stop). The 524K request reused 16,128 prefix tokens, the same reuse
  as JJ-main (336.2 s) and R38 (336.5 s); it is not a cold prefill measurement.
- `STRUCTURED-STRESS-PASS 32`: all 32 constrained-JSON cases in waves of four with K3.
  This does not replay the unknown request behind the historical R38 grammar failure.
- Post-grid semantic battery passed; `DS4-KARMIC-BETA-QUALIFICATION-COMPLETE`.

## Capacity

KV pool **1,525,210 tokens** (2.91 full 524K requests): R38 1,516,572 (+0.6%), JJ-main
711,894, the quick-test boot of this image with `b12x-a8-dglin` 1,182,996. Graph capture
took 0.36 + 0.24 GiB here against 1.23 + 0.20 GiB on the quick-test boot. The two boots
also differ in cold versus warm kernel caches (JJ-main's loss was traced to cold-boot
preparation retention), so backend and cache state are not separated.

## Performance

Grid `20260929T124135-0400__karmic-beta-20260929-tp2-dspark-k3-b12x-a8-524k__r01.json`,
15 valid cells (sha256 `c1ac7177...`), unchanged `run_bench.sh`.

Against the R38p `b12x-k3` arm (September 28, same backend), via `compare.py`:

| | c1 | c2 | c4 |
| --- | ---: | ---: | ---: |
| Output tok/s | +2.7% | +0.8% | +4.2% |
| Engine steps/s | -0.2% (17.76 to 17.73) | -1.5% (27.25 to 26.85) | +3.2% (37.60 to 38.80) |
| Acceptance length | +2.9% | +2.3% | +1.0% |

Prefill (grid scouts): 8K 2,231 vs 2,177 (+2.5%); 16K to 128K 2,223 / 2,183 / 2,111 / 1,976
vs 2,293 / 2,252 / 2,177 / 2,040 (-3.0% to -3.1%). Against the September 22 R38p
qualification scouts (2,169 / 2,269 / 2,230 / 2,154 / 2,021): +2.9% at 8K, about -2% from
16K to 128K. That grid cannot be compared cell by cell because its c4/16K cell has
`warmup_timed_out` (it was rechecked separately); `compare.py` correctly refuses it.

Reading: decode at parity or better, prefill about 2 to 3 percent below R38p from 16K up,
KV at parity. The September 20 Karmic long-context prefill collapse (-11.5% at 64K, -17.9%
at 128K) is gone. One candidate boot against historical baselines; no cause asserted for
the residual prefill gap.

## Health

Zero kernel NV_ERR / Xid / OOM events across the whole window, including startup.
Minimum MemAvailable 4.98 GiB rusty, 6.38 GiB toby. Containers running, not OOM-killed.
GPU clock events: `0x4` (software power cap) during model load on both nodes; on rusty only,
`0x20` (software thermal slowdown) four samples and one `0x68` during the 524K prefill, and
two `0x20` samples during the grid (12:47:22 and 12:47:32). Peak temperature 87 C rusty,
86 C toby. The rusty thermal samples match the September 20 pattern and are a caveat on
the affected cells, not an attribution.

## Status

Qualified for correctness; performance at or near R38p. **Promoted to production by the
user on 2026-09-29**: serving DS4 Vision on rusty/toby as `ds4-vision-karmic-beta-20260929-tp2`. The previous
`ds4-vision-perf-20260928-b12x-k3` (R38p) containers are stopped and retained for
rollback. Receipts: `../qualification/ds4-vision/`
(git-ignored); campaign
`runs/deepseek-v4-flash/vision-exp/2026-09-karmic-beta-sm121-qualification/`.

## Native context amendment, 2026-09-29

User rule: always serve at the model's maximum native context. The checkpoint's
`max_position_embeddings` is 1,048,576 (YaRN factor 16 over 65,536); the 524,288 limit was an
envelope inherited from the September 10 migration. `run-node.sh` now sets
`MAX_MODEL_LEN=1048576`; nothing else changed (same image, backend and settings). By user
direction no 1M needle was run; the qualification above (retrieval to 524,000) stands, and
retrieval between 524K and 1M is not separately tested.

Restart 2026-09-29 20:29 UTC: the 524K production container was stopped worker first and removed after
archiving its logs (`~/logs/ds4-vision-karmic-beta-20260929-tp2-524k-*`); the R38p rollback
containers were untouched. Boot on warm caches: engine reports `max model len 1048576`;
weights 81.05 GiB in 83.5 s; graphs 0.27 + 0.23 GiB; available KV memory 14.62 / 14.59 GiB
(rusty / toby); engine init 100.6 s. KV pool **2,200,602 tokens**, 2.10 full 1M requests. This
token count is not directly comparable with the 1,525,210 reported at 524K, because the
reported token capacity of DS4's mixed sliding-window and compressed cache depends on the
configured maximum length. Completion `333`, stop. Zero kernel NV_ERR / Xid / OOM since start,
restarts 0, MemAvailable 7.0 / 8.0 GB.
