# v16.3 source validation — 2026-07-16

This image is the original Spark v16 recipe with only the paired
FlashInfer and vLLM fixes changed. All other pins remain inherited from
`../v16/`, including B12X `fe06f49`.

## FlashInfer

- v16 base: `voipmonitor/flashinfer:codex/sm120-dspark-stack-20260711`
  at `801d57a08958c13d375ddbb6be3be4808f48a708`
- candidate: `sjug/flashinfer:codex/sm120-dspark-stack-pr3932-20260716`
  at `c0700ff92ddbc32bd452d9d03cf75d9a8f780fe5`
- live branch tip matched the candidate pin.
- the merge base is exactly the v16 pin, and the candidate is a linear
  three-commit series on that base.
- `git range-diff` against FlashInfer PR #3932 found the first two commits
  patch-identical. The third differs only by omitting two test imports:
  the v16 base already imports `torch.nn.functional` and provides its own
  local `quant_dequant_fp4_reference` helper.

Candidate commits:

1. `c3fc5a7a` — correct precise-path FP4 quant multiplier and subnormal
   E4M3 scale decode
2. `f71e913c` — add the B12X benchmark quant-mode selector
3. `c0700ff9` — add `input_global_scale` to decouple FC1 input quantization
   from `w1_alpha`

## vLLM

- v16 base:
  `local-inference-lab/vllm:codex/fathomless-firmament-v16-unified-20260712`
  at `8f86f425102cee08745462615d54115eee275f9f`
- candidate:
  `sjug/vllm:codex/fathomless-firmament-v16-unified-pr48536-20260716`
  at `6a5a106b3f783be87e01e15ad99e590b740f3945`
- live branch tip matched the candidate pin.
- the merge base is exactly the v16 pin, and the candidate is a linear
  two-commit series on that base.
- `git range-diff` found candidate commit `47d25466d` patch-identical to
  vLLM PR #48536 head `73a1251bd`.
- candidate commit `6a5a106b3` is an additional test-only commit affecting
  `tests/kernels/moe/test_flashinfer_b12x_moe.py`; it changes no runtime
  source.

## Build

- recipe: `build-ds4dspark-v16.3-spark-sm121-cu132.sh`
- image:
  `localhost/voipmonitor/vllm:fathomless-firmament-v16p3-spark-sm121-vllm6a5a106-b12xfe06f49-fic0700ff-cu132-20260716`
- image ID: `f48058737ff938f8c8a557828c1ea4468359578b0adb4fb817536b1e35719980`
- log: `/home/jugs/logs/build-v16p3-20260716_161655.log`
- all 46 build steps and the recipe's final post-build checks completed.
- the bundled verifier reported Torch `2.12.0+cu132`, CUDA `13.2`, NCCL
  `2.29.7`, FlashInfer `0.6.15+cu132`, vLLM
  `0.11.2.dev280+fathomless.firmament.v16p3.vllm6a5a106.b12xfe06f49.fic0700ff.sm121.cu132.20260716`,
  B12X `0.30.0`, and DeepGEMM `2.5.0`.

## Transfer and deployment

- transferred directly from `rusty` (`10.11.1.1`) to `toby-ib`
  (`10.11.1.2`) over the 200G interface.
- transfer and load completed in 117 seconds; log:
  `/home/jugs/logs/transfer-v16p3-to-toby-ib-20260716_165757.log`.
- the loaded image on `toby` has the same image ID and passed the same
  bundled verifier.
- the stopped v16.2 containers were inspected before replacement. The
  reproduced launch settings were DSpark, B12X A16, TP=2, NCCL, max model
  length 262,144, max sequences 4, max batched tokens 8,192, and GPU memory
  utilization 0.87.
- runtime logs:
  - `rusty`: `/home/jugs/logs/v16p3-b12x-head-20260716_170318.log`
  - `toby`: `/home/jugs/logs/v16p3-b12x-worker-20260716_170318.log`

Both ranks joined the NCCL world, loaded the main and DSpark draft weights,
completed B12X mHC warmup, FlashInfer sparse-MLA warmup/autotuning, and CUDA
graph capture. The head reached `Application startup complete`; `/health`
returned HTTP 200 and `/v1/models` advertised `DeepSeek-V4-Flash-DSpark`.
Neither log contained a traceback, CUDA error, OOM, or engine-init failure.

This is the important functional result: the paired FlashInfer/vLLM fixes
clear the startup path that failed in the earlier image.

## B12X benchmark

Result:

`/home/jugs/git/llm-inference-bench/benchmark_results-ds4dspark-v16p3-b12xfe06f49-b12x-a16-dspark_20260716_182142.json`

The run used `llm-decode-bench` 0.4.29, 30 seconds per sustained-decode
cell, concurrency 1/2/4, contexts 0/16K/32K/64K/128K, integrated prefill
scouts, and the normal `max_tokens=8192` default. All 15 cells completed with
zero request errors, underfilled cells, or capacity-limited cells. The service
remained healthy on both nodes after the 13m11s run.

### Prefill

Client-observed scout throughput, tokens/s:

| Context | v16 | v16.1 | v16.2 | v16.3 |
| ---: | ---: | ---: | ---: | ---: |
| 8K | 2,207 | 2,161 | 2,140 | 2,114 |
| 16K | 2,245 | 2,195 | 2,178 | 2,233 |
| 32K | 2,227 | 2,171 | 2,158 | 2,209 |
| 64K | 2,137 | 2,105 | 2,093 | 2,137 |
| 128K | 2,002 | 1,976 | 1,928 | 2,001 |

At 16K through 128K, v16.3 is within 0.0-0.8% of original v16 and is
2.1-3.8% faster than v16.2. The 8K scout remains 4.2% below v16 and 1.2%
below v16.2.

### Sustained decode

v16.3 aggregate tokens/s:

| Context | CC1 | CC2 | CC4 |
| ---: | ---: | ---: | ---: |
| 0 | 35.0 | 61.5 | 86.3 |
| 16K | 46.9 | 88.0 | 96.9 |
| 32K | 53.4 | 66.7 | 95.3 |
| 64K | 37.7 | 67.0 | 87.4 |
| 128K | 69.7 | 79.6 | 86.0 |

Context-average throughput and the geometric mean over all 15 cells:

| Image | CC1 avg | CC2 avg | CC4 avg | 15-cell geometric mean |
| --- | ---: | ---: | ---: | ---: |
| v16 B12X DSpark | 44.3 | 72.6 | 95.1 | 66.7 |
| v16.1 B12X DSpark | 45.0 | 65.7 | 97.6 | 65.8 |
| v16.2 B12X DSpark | 42.3 | 66.3 | 94.6 | 63.8 |
| v16.3 B12X DSpark | 48.6 | 72.6 | 90.4 | 67.4 |

The geometric mean makes v16.3 look 1.1% faster than v16 and 5.6% faster
than v16.2, but this is not a broad improvement. The median per-cell change
is -0.7% versus v16 and -2.0% versus v16.2; the positive mean is pulled up by
the 128K/CC1 cell, which is 57% above v16 and 105% above v16.2.

CC2 recovers on average: it essentially matches original v16 and is 9.4%
above v16.2. It is still highly context-sensitive. Relative to v16, v16.3
is -22.6% at 64K but +26.2% at 128K.

CC4 is the clear regression. Its context average is 5.0% below v16, 7.4%
below v16.1, and 4.4% below v16.2. It loses every CC4 cell to v16.2, by
1.8-9.3%; versus original v16, the largest losses are 16.5% at 64K and
11.6% at 128K.

### Cross-backend context

These are not causal comparisons, but they locate the new B12X result among
the completed Cutlass runs:

| Run | CC1 avg | CC2 avg | CC4 avg | 15-cell geometric mean | Prefill avg |
| --- | ---: | ---: | ---: | ---: | ---: |
| v16 Lucifer/Cutlass DSpark | 52.4 | 65.6 | 99.9 | 69.5 | 2,234 |
| v16.1 Lucifer/Cutlass DSpark | 53.2 | 71.8 | 93.6 | 70.1 | 2,281 |
| v16.3 B12X DSpark | 48.6 | 72.6 | 90.4 | 67.4 | 2,139 |

v16.3 B12X is roughly level with v16.1 Cutlass at CC2 (+1.1%), but trails it
at CC1 (-8.7%), CC4 (-3.4%), overall geometric mean (-3.9%), and prefill
(-6.2%).

### Full rerun

A second identical run completed in 13m17s:

`/home/jugs/git/llm-inference-bench/benchmark_results-ds4dspark-v16p3-b12xfe06f49-b12x-a16-dspark_20260716_194455.json`

It again completed all 15 cells with zero errors, underfilled cells, or
capacity-limited cells. Both ranks remained up and `/health` returned HTTP
200 afterward.

| Context | Run 1 CC1 | Run 2 CC1 | Run 1 CC2 | Run 2 CC2 | Run 1 CC4 | Run 2 CC4 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 35.0 | 38.9 | 61.5 | 59.1 | 86.3 | 84.7 |
| 16K | 46.9 | 56.2 | 88.0 | 64.3 | 96.9 | 98.3 |
| 32K | 53.4 | 50.5 | 66.7 | 70.3 | 95.3 | 93.8 |
| 64K | 37.7 | 53.9 | 67.0 | 88.0 | 87.4 | 100.6 |
| 128K | 69.7 | 40.2 | 79.6 | 69.4 | 86.0 | 94.6 |

The largest run-to-run changes were 64K/CC1 +42.9%, 64K/CC2 +31.3%,
128K/CC1 -42.3%, and 16K/CC2 -26.9%. The first run's complete 64K-row
regression therefore did not reproduce, and its exceptional 128K/CC1 result
also did not reproduce.

Prefill was stable within 1.1% at 8K/16K/32K, but run 2 was 7.8% slower at
64K and 10.0% slower at 128K. Its prefill rates were 2,136, 2,230, 2,208,
1,971, and 1,801 tokens/s respectively.

Summary versus the single v16 run:

| Metric | v16 | v16.3 run 1 | v16.3 run 2 | v16.3 two-run mean | Mean delta vs v16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CC1 average | 44.3 | 48.6 | 47.9 | 48.2 | +8.9% |
| CC2 average | 72.6 | 72.6 | 70.2 | 71.4 | -1.6% |
| CC4 average | 95.1 | 90.4 | 94.4 | 92.4 | -2.9% |
| 15-cell geometric mean | 66.7 | 67.4 | 67.7 | 67.9 | +1.9% |
| Prefill average | 2,163.6 | 2,138.8 | 2,069.2 | 2,104.0 | -2.8% |

The benchmark leaves `temperature` unset, so vLLM uses the model's default
stochastic sampling (`temperature=1.0`, `top_p=1.0`). Each run also uses a
different cache-busting prompt marker. The reported speculative-acceptance
rate moved with several of the large throughput swings: 128K/CC1 changed
from 1.000 to 0.585 while throughput fell from 69.7 to 40.2 tokens/s, and
16K/CC2 changed from 0.764 to 0.355 while throughput fell from 88.0 to 64.3.
This makes one-run cell deltas unsafe to attribute solely to the image.

### Third full pass

A third identical pass completed in 13m21s:

`/home/jugs/git/llm-inference-bench/benchmark_results-ds4dspark-v16p3-b12xfe06f49-b12x-a16-dspark_20260716_204955.json`

It also completed all 15 cells with zero errors, underfilled cells, or
capacity-limited cells. Both containers remained up and the API remained
healthy.

| Context | CC1 | CC2 | CC4 |
| ---: | ---: | ---: | ---: |
| 0 | 35.7 | 60.9 | 80.5 |
| 16K | 51.3 | 77.8 | 88.2 |
| 32K | 49.7 | 70.0 | 92.3 |
| 64K | 65.1 | 81.0 | 85.6 |
| 128K | 48.6 | 66.0 | 91.4 |

Pass-three prefill was 2,179, 2,234, 1,871, 1,926, and 1,742 tokens/s at
8K, 16K, 32K, 64K, and 128K respectively.

Three-run summary versus the single v16 result:

| Metric | v16 | Run 1 | Run 2 | Run 3 | Three-run mean | Mean delta vs v16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CC1 average | 44.3 | 48.6 | 47.9 | 50.1 | 48.9 | +10.3% |
| CC2 average | 72.6 | 72.6 | 70.2 | 71.1 | 71.3 | -1.7% |
| CC4 average | 95.1 | 90.4 | 94.4 | 87.6 | 90.8 | -4.6% |
| 15-cell geometric mean | 66.7 | 67.4 | 67.7 | 67.3 | 67.5 | +1.2% |
| Prefill average | 2,163.6 | 2,138.8 | 2,069.2 | 1,990.4 | 2,066.1 | -4.5% |

The widest decode spreads across the three passes are 64K/CC1 (37.7-65.1,
a range equal to 52% of its mean), 128K/CC1 (40.2-69.7, 56%), 16K/CC2
(64.3-88.0, 31%), and 64K/CC2 (67.0-88.0, 27%). CC4 is more stable than
CC1/CC2 at most contexts, but its three-run average remains 4.6% below the
single v16 run.

Prefill is repeatable at 16K, but the three-run ranges grow to 16% at 32K,
10% at 64K, and 14% at 128K. The three-run mean is 4.5% below v16 overall.

### Interpretation

- Startup correctness is fixed.
- Decode and long-prefill results show material run-to-run variability.
- Across the three v16.3 runs, CC1 is higher, CC2 is slightly lower, and
  CC4 is lower than the single v16 run, while the overall geometric mean is
  roughly flat/slightly higher.
- A matched v16 rerun, or a deterministic benchmark series using the same
  generation settings for every image, is needed before treating the
  remaining throughput differences as stable image regressions.
- The clean causal comparison for the paired fixes is v16.3 versus v16:
  both use B12X `fe06f49`. v16.1 and v16.2 use B12X `1bcc652`, so comparisons
  against those images also include a B12X change.
