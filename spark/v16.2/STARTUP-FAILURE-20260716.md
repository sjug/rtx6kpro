# v16.2 FlashInfer CUTLASS startup failure

Date: 2026-07-16

## Summary

The v16.2 image is not generally broken. It starts and serves requests successfully
with the B12X MoE backend. The observed startup failure is specific to the
`lucifer-cutlass` configuration and occurs while FlashInfer autotunes the CUTLASS
fused-MoE GEMM1 path for the DeepSeek V4 MXFP4 model on SM121.

The regression boundary is upstream FlashInfer commit `5823159c` (PR #3738),
which landed between the v16.1 and v16.2 FlashInfer pins. The custom sparse-MLA
commits in the v16.2 branch do not modify fused-MoE or autotuner code.

## Images and FlashInfer pins

- v16.1 image: `fi801d57a`
- v16.2 image: `fi3245a18`
- v16.2 upstream base: `b072eaac`
- v16.2 custom commits:
  - `e100f0c0` — PR #3817 decode fix
  - `c39c6b8f` — PR #3817 tests
  - `3245a18e` — PR #3834 BF16 prefill top-k 256 instantiation

The source diff from `b072eaac` to `3245a18` contains only:

- `csrc/sparse_mla_sm120_decode_dsv4.cu`
- `csrc/sparse_mla_sm120_prefill.cu`
- `flashinfer/mla/_sparse_mla_sm120.py`
- `tests/attention/test_sparse_mla_sm120.py`

## Reproduction and logs

The failed v16.2 launch used:

```text
mode=dspark
backend=lucifer-cutlass
--attention-backend FLASHINFER_MLA_SPARSE_DSV4
--kernel-config.moe_backend flashinfer_cutlass
```

Recovered failure log on `rusty`:

```text
~/logs/v16p2-failed-recovered-20260716_1347.log
```

Successful v16.2 B12X startup log on `rusty`:

```text
~/logs/v16p2-b12x-head-20260716_151335.log
```

The successful B12X run completed sparse-MLA metadata warmup, sparse decode
autotuning, graph capture, API startup, health checks, and a chat-completion
request.

## Failure sequence

The failed CUTLASS run completed all of the following before the crash:

1. Loaded the base and draft model weights.
2. Completed FlashInfer sparse-MLA prefill metadata warmup.
3. Completed sparse-MLA decode autotuning.
4. Entered `trtllm::fused_moe::gemm1` autotuning.

During the first GEMM1 optimization profile, v16.2 reported:

```text
[Autotuner]: Skipped 20 unsupported tactic(s) for trtllm::fused_moe::gemm1
Error: Failed to initialize the TMA descriptor 715
torch.AcceleratorError: CUDA error: an illegal instruction was encountered
```

The Python traceback eventually surfaced at `torch.rand` while preparing the
next autotuning inputs. CUDA execution is asynchronous, so that allocation is
where the poisoned CUDA context was observed, not the operation that originally
executed the illegal instruction. The call path immediately before the failure
was:

```text
vLLM flashinfer_cutlass_moe.py
  -> flashinfer.fused_moe.core.cutlass_fused_moe
  -> flashinfer.autotuner.AutoTuner.choose_one
  -> trtllm::fused_moe::gemm1 profiling
```

For comparison, v16.1 reported 14 unsupported GEMM1 tactics per optimization
profile, continued through all 21 GEMM1 profiles and GEMM2 tuning, saved 66
fused-MoE configurations, and completed server startup.

The `/21` progress indicator in these logs counts optimization profiles, not
the number of tactics. The directly supported comparison is therefore 20
failed tactics during v16.2's first profile versus 14 failed tactics per profile
in the successful v16.1 run.

## Local source-history result

The old v16.1 pin `801d57a` does not contain either of these commits:

- `5823159c` — `perf: optimize MXFP4xBF16 & INT4xFP8 and add MXFP4xFP8 CUTLASS MoE backend for SM90 (#3738)`
- `b35396c1` — `Yanqinz/autotuner tactic (#3707)`

The v16.2 base contains both. Of these, `b35396c1` changes generic autotuner
tactic typing and GEMM code but does not change the fused-MoE kernel path.
Commit `5823159c` changes the CUTLASS fused-MoE implementation, profiler,
tactic validation, TMA setup, and associated kernels. It also introduces
`get_valid_tactics_for_shape` in `flashinfer/fused_moe/core.py` and the native
binding.

This makes `5823159c` / PR #3738 the source-level regression boundary for the
SM121 MXFP4/BF16 CUTLASS fused-MoE startup failure. Identifying the individual
bad tactic or exact hunk still requires a focused reproduction.

## Recommended isolation steps

1. Build a v16.2 test image with `5823159c` reverted and repeat the identical
   `lucifer-cutlass` launch. This is the strongest commit-level confirmation.
2. For tactic-level isolation, rerun once with FlashInfer debug logging and
   `CUDA_LAUNCH_BLOCKING=1` so the failing tactic and launch site are recorded.
3. If only a small number of tactics are bad, generate a FlashInfer tactics
   blocklist as a temporary workaround while the kernel regression is fixed.

Until then, the v16.2 B12X backend is a working path and independently exercises
the custom sparse-MLA changes successfully.
