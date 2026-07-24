# Fix Int32 overflow in direct-K fused-indexer page offsets

## Summary

The warp-owns-tokens direct-L2 score path (`39d7404`) computes the per-warp K
fragment byte offset in Int32:

```python
k_off = (
    pid * Int32(self.k_quant_page_stride)
    + (tip0 + lane // Int32(4)) * Int32(_INDEX_HEAD_DIM)
    + lane4 * Int32(4)
)
```

Serving integrations hand this kernel **strided views of a packed per-page
record** — we observe `k_quant_bytes.stride(0) = 1,077,120 B` in a vLLM
DSV4-Flash deployment, vs the 8448 of a bare `[pages, 64, 128]`+scales
tensor. With that stride the Int32 product wraps at
`pid >= ceil(2^31 / 1,077,120) = 1994`; the wrapped value sign-extends and
the `ld.global.nc` lands ~1.2 GB *before* the tensor →
`CUDA_ERROR_ILLEGAL_ADDRESS`, killing the CUDA context.

Because standalone tensors need `pid >= 254,187` to overflow, unit tests and
benchmarks never see it. In serving it presents as: *stable for hours at
short context, then the first request past ~50k tokens kills the engine* —
page ids only climb past 1994 once enough KV has been allocated.

## Who hits it

- **GB10 / DGX Spark (48 SMs):** `ctas_per_group = 48 // rows`, so **every
  speculative-verify batch (rows ≥ 3) takes the direct-K path**. DSpark K=5/6
  serving dies on the first long-context request. mtp0 (rows 1) stays on the
  staged pipeline and is unaffected — which made this look like a spec-decode
  bug for a while.
- **SM120 / RTX PRO 6000 (188 SMs):** the default resolver keeps small-row
  batches on the staged path, but decode batches of **rows ≥ 12** (or any
  caller passing `ctas_per_group <= 16`) reach the same code. Reproduced on
  RTX PRO 6000 with `ctas_per_group=8`: identical fault.

## Evidence

GPU core dump from a 2-node GB10 TP2 pair
(`CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1`, cuda-gdb): Warp MMU Fault in
`SparseNSAFusedIndexerKernel`, faulting SASS

```
IMAD R6, R41, 0x106f80, R6      ; offset = pid * 1,077,120 (32-bit)
SHF.R.S32.HI R7, RZ, 0x1f, R6   ; sign-extend
IADD.64 R14, R6, UR8            ; base + wrapped offset
LDG.E.CONSTANT R12, [R14.64]    ; MMU fault
```

with `R41 = pid = 6839` (6839 × 1,077,120 ≈ 6.9 GB, truncates to ≈ −1.2 GB).

## Fix

Compute the page term as `Int64(pid) * Int64(stride)` and carry the offset
through `_score_tokens_direct_k` as Int64 — the same convention the staged
pipeline already uses (`Int64(page_id) * page_stride_bytes` in
`_load_permute_k_page_g2s` / `_stream_issue_k_page_cp_async`). Intra-page
terms are Int32-safe and only widened for the final add. 11 lines.

The neighboring `k_scales[pid, tip0 + col]` access goes through cute layout
algebra which compiles 64-bit — verified clean to pid 15,900 (a 4.28-billion-
element scale index) with the fix applied.

## Validation

- **Regression test** (included, house style): reconstructs the packed
  stride with `pid >= 2000`, forces `ctas_per_group=8`. Fails on the unfixed
  kernel in ~4 s with the CUDA error; passes with the fix. Verified on
  SM120 (RTX PRO 6000) and SM121 (GB10).
- **Full `tests/test_fused_indexer.py` suite:** 73/73 pass with the fix
  (RTX PRO 6000, torch 2.13/cu130, cutlass-dsl 4.6).
- **Serving (2× GB10 TP2, DSV4-Flash DSpark K=6, b12x-a8):** stock IMAs at
  49,637 real tokens every attempt; with the fix (direct-K active) survives
  49,637 / 66,179 / 88,234 / 132,345 tokens eager AND with FULL_AND_PIECEWISE
  cudagraphs through **239,628 tokens** (≈ max_model_len), spec decoding
  healthy at depth 6 throughout.
- Standalone parity repro also available (stock faults at pid 2000, fixed
  passes through pid 15,900).
