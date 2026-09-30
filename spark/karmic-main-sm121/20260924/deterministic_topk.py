"""Diagnostic stable tie repair for B12X row top-k, without score perturbation.

The existing selector supplies the exact kth score. Re-scan the input and keep
all scores above it plus the lowest logical indices equal to it. Three GPU
passes use integer prefix counts, not atomic arrival order. Outputs are ordered
by logical index within the greater/equal partitions; the existing MXFP4 sort
retains responsibility for final attention-index ordering and physical mapping.
This is a diagnostic implementation pending full serving qualification.
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _counts(X, Length, Old, Count, Threshold, W: tl.constexpr, K: tl.constexpr,
            T: tl.constexpr, B: tl.constexpr, KB: tl.constexpr):
    r, t = tl.program_id(0), tl.program_id(1)
    k = tl.arange(0, KB)
    threshold = tl.min(tl.load(Old + r * K + k, k < K, other=float('inf')), 0)
    if t == 0:
        tl.store(Threshold + r, threshold)
    i = t * B + tl.arange(0, B)
    n = tl.minimum(tl.maximum(tl.load(Length + r), 0), W)
    x = tl.load(X + r.to(tl.int64) * W + i, i < W, other=-float('inf'))
    valid = (i < n) & (x != -float('inf')) & (x == x)
    above = valid & (x > threshold)
    equal = valid & (x == threshold)
    tl.store(Count + (r * 2) * T + t, tl.sum(above.to(tl.int32), 0))
    tl.store(Count + (r * 2 + 1) * T + t, tl.sum(equal.to(tl.int32), 0))


@tr.jit
def _prefix(Count, Prefix, Total, Values, Indices, T: tl.constexpr,
            K: tl.constexpr, TB: tl.constexpr, KB: tl.constexpr):
    r = tl.program_id(0)
    t = tl.arange(0, TB)
    a = tl.load(Count + (r * 2) * T + t, t < T, other=0)
    e = tl.load(Count + (r * 2 + 1) * T + t, t < T, other=0)
    tl.store(Prefix + (r * 2) * T + t, tl.cumsum(a, 0) - a, t < T)
    tl.store(Prefix + (r * 2 + 1) * T + t, tl.cumsum(e, 0) - e, t < T)
    tl.store(Total + r, tl.sum(a, 0))
    k = tl.arange(0, KB)
    tl.store(Values + r * K + k, -float('inf'), k < K)
    tl.store(Indices + r * K + k, -1, k < K)


@tr.jit
def _scatter(X, Length, Threshold, Prefix, Total, Values, Indices, Gather,
             W: tl.constexpr, K: tl.constexpr, T: tl.constexpr, B: tl.constexpr,
             HAS_GATHER: tl.constexpr, OFFSET: tl.constexpr):
    r, t = tl.program_id(0), tl.program_id(1)
    i = t * B + tl.arange(0, B)
    n = tl.minimum(tl.maximum(tl.load(Length + r), 0), W)
    x = tl.load(X + r.to(tl.int64) * W + i, i < W, other=-float('inf'))
    threshold = tl.load(Threshold + r)
    valid = (i < n) & (x != -float('inf')) & (x == x)
    above = valid & (x > threshold)
    equal = valid & (x == threshold)
    ap = tl.load(Prefix + (r * 2) * T + t)
    ep = tl.load(Prefix + (r * 2 + 1) * T + t)
    total = tl.load(Total + r)
    ai = ap + tl.cumsum(above.to(tl.int32), 0) - 1
    ei = ep + tl.cumsum(equal.to(tl.int32), 0) - 1
    pos = tl.where(above, ai, total + ei)
    selected = above | (equal & (ei < K - total))
    selected = selected & (pos >= 0) & (pos < K)
    index = i + OFFSET
    if HAS_GATHER:
        index = tl.load(Gather + r.to(tl.int64) * W + i, selected, other=-1)
    tl.store(Values + r * K + pos, x, selected)
    tl.store(Indices + r * K + pos, index, selected)


def repair_topk(logits, lengths, values, indices, gather=None, offset=0):
    if logits.ndim != 2 or values.ndim != 2 or indices.shape != values.shape:
        raise ValueError('Invalid deterministic top-k shapes')
    rows, width = logits.shape
    k = values.shape[1]
    if values.shape[0] != rows or lengths.shape != (rows,) or lengths.dtype != torch.int32:
        raise ValueError('Invalid deterministic top-k row metadata')
    if not logits.is_cuda or any(x.device != logits.device for x in (lengths, values, indices)):
        raise ValueError('Deterministic top-k tensors must share one CUDA device')
    if not all(x.is_contiguous() for x in (logits, lengths, values, indices)):
        raise ValueError('Deterministic top-k requires contiguous tensors')
    if logits.dtype != torch.float32 or values.dtype != torch.float32 or indices.dtype != torch.int32:
        raise ValueError('Invalid deterministic top-k dtypes')
    if gather is not None and (gather.shape != logits.shape or gather.dtype != torch.int32 or not gather.is_contiguous()):
        raise ValueError('Invalid deterministic top-k gather table')
    if gather is not None and gather.device != logits.device:
        raise ValueError('Gather table must share the logits device')
    if not rows or not width or not k:
        raise ValueError('Empty deterministic top-k dimensions are not admitted')
    block = 1024
    tiles = tr.cdiv(width, block)
    counts = torch.empty((rows, 2, tiles), device=logits.device, dtype=torch.int32)
    prefix = torch.empty_like(counts)
    total = torch.empty((rows,), device=logits.device, dtype=torch.int32)
    threshold = torch.empty((rows,), device=logits.device, dtype=torch.float32)
    _counts[(rows, tiles)](logits, lengths, values, counts, threshold, width, k,
                           tiles, block, tr.next_power_of_2(k))
    _prefix[(rows,)](counts, prefix, total, values, indices, tiles, k,
                     tr.next_power_of_2(tiles), tr.next_power_of_2(k))
    _scatter[(rows, tiles)](logits, lengths, threshold, prefix, total, values,
                            indices, gather if gather is not None else indices,
                            width, k, tiles, block, gather is not None, offset)
