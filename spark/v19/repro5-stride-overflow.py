"""Int32 page-offset overflow repro with SERVING-REALISTIC strides.

The serving index cache stores each 64-token page inside a 1,077,120-byte
record (quant view stride(0)=1,077,120 B; scales f32 view stride(0)=269,280
elements). The direct-K body computes byte offsets as pid * Int32(stride):
  - quant read overflows at pid >= ceil(2^31 / 1,077,120) = 1994  (bug #1)
  - scales cute-indexing, if Int32, overflows at pid >= 7976       (bug #2?)

Usage: repro5-stride-overflow.py PID_LO [ROWS]
Allocates a real pool big enough for the pid range, builds a page table whose
entries are [PID_LO, PID_LO+192), runs the fused indexer once. Exit 0 = pass.
Run one trial per process under CUDA_LAUNCH_BLOCKING=1.
"""
import sys

import torch

from b12x.attention.indexer.fused_indexer import run_fused_paged_indexer

PID_LO = int(sys.argv[1])
ROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 7
P_USED = 192
REC = 1077120  # bytes per page record (measured in serving: SASS 0x106f80)
POOL_PAGES = PID_LO + P_USED + 8

dev = torch.device("cuda", 0)
torch.manual_seed(0)

pool = torch.zeros(POOL_PAGES * REC, dtype=torch.uint8, device=dev)
quant = pool.as_strided((POOL_PAGES, 64, 128), (REC, 128, 1))
# fill only the used pages' quant rows with data
quant[PID_LO : PID_LO + P_USED] = 3
poolf = pool.view(torch.float32)
scales = poolf.as_strided((POOL_PAGES, 64), (REC // 4, 1), storage_offset=2048)
scales[PID_LO : PID_LO + P_USED] = 1.0

q = torch.full((ROWS, 64, 128), 3, dtype=torch.uint8, device=dev)
weights = torch.rand(ROWS, 64, device=dev)
pt = (
    torch.arange(PID_LO, PID_LO + P_USED, dtype=torch.int32, device=dev)
    .repeat(ROWS, 1)
    .contiguous()
)
W = P_USED * 64
sl = (W - (ROWS - 1) + torch.arange(ROWS, dtype=torch.int32, device=dev)).contiguous()

print(
    f"pool={POOL_PAGES} pages x {REC} B = {POOL_PAGES*REC/2**30:.2f} GiB, "
    f"pids [{PID_LO}, {PID_LO+P_USED}), quant stride {quant.stride(0)} B, "
    f"scales stride {scales.stride(0)} el, rows={ROWS}",
    flush=True,
)
torch.cuda.synchronize()
idx, val = run_fused_paged_indexer(
    q_bytes=q,
    weights=weights,
    k_quant_bytes=quant,
    k_scales=scales,
    real_page_table=pt,
    seqlens=sl,
    num_heads=64,
    topk=512,
    output_physical_slots=True,
)
torch.cuda.synchronize()
lo, hi = PID_LO * 64, (PID_LO + P_USED) * 64
inrange = ((idx >= lo) & (idx < hi)) | (idx == -1)
print(
    f"PASS pid_lo={PID_LO}: out={tuple(idx.shape)} "
    f"phys-slot-range-ok={bool(inrange.all())}",
    flush=True,
)
