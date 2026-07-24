"""Correctness validation of the b12x fused paged indexer vs a torch reference.

The serving crash path consumes fused-indexer topk indices in the unified
decode gather. H1 says the rewritten direct-K fused indexer emits wrong or
out-of-range indices at verify row counts (rows>=3 on 48-SM GB10). Validate:
score = sum_h w_h * relu(q_h . k) * k_scale, top-512 per row.

Checks per trial: (a) all indices in [0, P*64) or -1; (b) recall of the
reference top-512 set (ties tolerated via score threshold); (c) determinism
across 3 repeats.
"""
import sys

import torch

from b12x.attention.indexer.fused_indexer import run_fused_paged_indexer

ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 7
W = int(sys.argv[2]) if len(sys.argv) > 2 else 12409
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0

dev = torch.device("cuda", 0)
torch.manual_seed(SEED)
P = (W + 63) // 64

# fp8 e4m3 values via clean quantization from bf16 (avoids NaN byte patterns)
q_f = torch.randn(ROWS, 64, 128, device=dev, dtype=torch.float32)
k_f = torch.randn(P * 64, 128, device=dev, dtype=torch.float32)
q8 = q_f.to(torch.float8_e4m3fn)
k8 = k_f.to(torch.float8_e4m3fn)
weights = torch.rand(ROWS, 64, device=dev) + 0.1
ks = (torch.rand(P, 64, device=dev) + 0.5)
pt = torch.arange(P, dtype=torch.int32, device=dev).repeat(ROWS, 1).contiguous()
sl = (W - (ROWS - 1) + torch.arange(ROWS, dtype=torch.int32, device=dev)).contiguous()

# reference: score[r, t] = sum_h w[r,h] * relu(q[r,h,:] . k[t,:]) * ks[t]
qd = q8.float()
kd = k8.float()
scores = torch.einsum("rhd,td->rht", qd, kd).relu_()
scores = torch.einsum("rht,rh->rt", scores, weights)
scores = scores * ks.reshape(-1)[None, :]
for r in range(ROWS):
    scores[r, sl[r]:] = float("-inf")

ref_val, ref_idx = torch.topk(scores, 512, dim=1)

outs = []
for rep in range(3):
    idx, val = run_fused_paged_indexer(
        q_bytes=q8.view(torch.uint8),
        weights=weights,
        k_quant_bytes=k8.view(P, 64, 128).view(torch.uint8),
        k_scales=ks,
        real_page_table=pt,
        seqlens=sl,
        num_heads=64,
        topk=512,
        output_physical_slots=False,
    )
    torch.cuda.synchronize()
    outs.append((idx.clone(), val.clone()))

idx, val = outs[0]
det = all(torch.equal(idx, o[0]) for o in outs[1:])
oob = ((idx < -1) | (idx >= P * 64)).sum().item()
neg = (idx == -1).sum().item()

# recall vs reference set with tie tolerance: accept any index whose ref score
# >= the 512th ref score minus epsilon
eps = 1e-3
bad_rows = []
for r in range(ROWS):
    thr = ref_val[r, -1].item() - eps
    got = idx[r][idx[r] >= 0]
    got_scores = scores[r, got.long()]
    n_bad = (got_scores < thr).sum().item()
    ref_set = set(ref_idx[r].tolist())
    got_set = set(got.tolist())
    overlap = len(ref_set & got_set)
    bad_rows.append((r, n_bad, overlap))

print(f"rows={ROWS} W={W} seed={SEED} deterministic={det} oob={oob} negpad={neg}")
for r, n_bad, overlap in bad_rows:
    flag = "  <-- MISMATCH" if n_bad > 0 or overlap < 500 else ""
    print(f"  row {r}: below-threshold={n_bad} overlap-with-ref={overlap}/512{flag}")
worst = max(b[1] for b in bad_rows)
sys.exit(1 if (oob or worst > 4) else 0)
