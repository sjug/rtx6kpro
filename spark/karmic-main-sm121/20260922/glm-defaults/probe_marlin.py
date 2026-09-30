#!/usr/bin/env python3
"""Small real Marlin repack probe; retain driver mappings before any failure."""
import ctypes
import json
import os
from pathlib import Path

import torch
from vllm import _custom_ops as ops


def identity():
    library = ctypes.CDLL('libcuda.so.1')
    version = ctypes.c_int()
    status = library.cuDriverGetVersion(ctypes.byref(version))
    mappings = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                       if '/libcuda.so' in line or '/libnvidia-ptxjitcompiler.so' in line})
    print(json.dumps({'driver_api_status': status, 'driver_api_version': version.value,
                      'mappings': mappings, 'LD_LIBRARY_PATH': os.environ.get('LD_LIBRARY_PATH'),
                      'CUDA_MODULE_LOADING': os.environ.get('CUDA_MODULE_LOADING'),
                      'torch_cuda': torch.version.cuda}), flush=True)


identity()
if torch.cuda.get_device_capability() != (12, 1):
    raise RuntimeError('Expected SM121')
weight = torch.zeros((16, 64), device='cuda', dtype=torch.int32)
torch.cuda.synchronize()
print('INPUT-READY', flush=True)
try:
    result = ops.gptq_marlin_repack(b_q_weight=weight, size_k=64, size_n=64, num_bits=8)
    runtime = ctypes.CDLL('libcudart.so.13')
    status = runtime.cudaGetLastError()
    print('REPACK-LAUNCH-STATUS', status, flush=True)
    if status:
        raise RuntimeError(f'Marlin repack CUDA launch failed: {status}')
    torch.cuda.synchronize()
    if result.numel() != weight.numel() or torch.count_nonzero(result).item() != 0:
        raise RuntimeError('Incorrect repack of zero input')
    print('MARLIN-REPACK-PASS', flush=True)
finally:
    identity()
