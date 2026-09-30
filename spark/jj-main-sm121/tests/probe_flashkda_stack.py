#!/usr/bin/env python3
"""Fresh-process FlashKDA stack measurement, not a kernel patch or model gate.

Derived from the first-launch probe in Karmic 57a80980 tests/models/kimi_k3/test_kda.py.
A CUDA failure or wrong zero result fails. Stack growth is retained as information
for review before model qualification, rather than automatically changing sources.
"""
import json

import torch
import vllm._flashkda_C  # noqa: F401
from cuda.bindings import runtime


def stack_limit():
    status, limit = runtime.cudaDeviceGetLimit(runtime.cudaLimit.cudaLimitStackSize)
    if int(status) != 0:
        raise RuntimeError(f'cudaDeviceGetLimit failed: {status}')
    return limit


if torch.cuda.get_device_capability() != (12, 1):
    raise RuntimeError('Physical SM121 required')
q = torch.zeros((1, 32, 2, 128), dtype=torch.bfloat16, device='cuda')
beta = torch.zeros((1, 32, 2), dtype=q.dtype, device=q.device)
state = torch.zeros((1, 2, 128, 128), device=q.device)
final = torch.full_like(state, float('nan'))
out = torch.full_like(q, float('nan'))
a_log = torch.zeros(2, device=q.device)
bias = torch.zeros((2, 128), device=q.device)
lengths = torch.tensor([0, 32], dtype=torch.int32, device=q.device)
scratch = torch.empty(torch.ops._flashkda_C.get_workspace_size(32, 2, 1),
                      dtype=torch.uint8, device=q.device)
torch.cuda.synchronize()
before = stack_limit()
torch.ops._flashkda_C.fwd(q, q, q, q, beta, 128**-0.5, out, scratch,
                        a_log, bias, -5.0, state, final, lengths)
torch.cuda.synchronize()
after = stack_limit()
if not (out == 0).all() or not (final == 0).all():
    raise RuntimeError('FlashKDA zero-input reference failed')
print(json.dumps({'stack_before_bytes': before, 'stack_after_bytes': after,
                  'grew': after > before, 'scope': 'diagnostic-not-model-qualification'}), flush=True)
