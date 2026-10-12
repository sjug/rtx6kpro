#!/usr/bin/env python3
"""Verify the actual source-first Spark runtime before starting vLLM."""
import os
from pathlib import Path

import b12x
import torch
import vllm

if not __debug__:
    raise RuntimeError("Runtime verification requires assertions")
assert Path(vllm.__file__).is_relative_to("/opt/jovian-judgement/vllm/vllm")
assert Path(b12x.__file__).is_relative_to("/opt/jovian-judgement/b12x")
assert Path(os.environ["VLLM_NCCL_SO_PATH"]).is_file()
assert os.environ["VLLM_NCCL_SO_PATH"] in os.environ["LD_PRELOAD"].split(":")
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert torch.cuda.get_device_capability() == (12, 1)
import vllm._C_stable_libtorch  # noqa: E402, F401
import vllm._moe_C_stable_libtorch  # noqa: E402, F401

x = torch.arange(512, device="cuda", dtype=torch.float32).reshape(2, 256) / 256
x = x.to(torch.bfloat16)
w = torch.ones(256, device="cuda", dtype=torch.bfloat16)
out = torch.empty_like(x)
torch.ops._C.rms_norm(out, x, w, 1e-6)
ref = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
torch.testing.assert_close(out.float(), ref, atol=0.02, rtol=0.02)
torch.cuda.synchronize()
print("DS4-VISION-SM121-NATIVE-PREFLIGHT-PASS", flush=True)
