"""DIAGNOSTIC ONLY: BF16 reduced-precision reduction off, once per GPU worker process.

Installed as vllm/v1/worker/ds41_precision.py and called exactly once from the
top of Worker.init_device, before distributed init, model load, B12X
preparation and graph capture. It sets one process-wide torch global,
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction, to False,
verifies the read-back and prints one marker line per rank. Nothing on any
forward path imports or calls this module; there is no per-call toggling, no
other math setting changes, and native and B12X kernels are untouched.

Scope of the observed effect (receipts/indexer-gpu-replay-20260926T215626Z):
with the flag off, PyTorch sets CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION
on every bf16 cuBLAS GEMM it issues, so split-K partials can no longer be
folded into a bf16 destination. This is a diagnostic arm, not a determinism,
batch-invariance, retrieval or qualification claim.
"""
import os
import sys
import time

import torch

SCHEMA = 'ds41-precision-v1'
FLAG = 'allow_bf16_reduced_precision_reduction'
MARKER = 'DS41-PRECISION-APPLIED'
SOURCE_LOCK = '/opt/ds41-precision/ds41-precision.lock.json'
_state = {'applications': []}


def backend():
    return torch.backends.cuda.matmul


def apply(*, rank, stream=None):
    """Set the flag False for this process, verify it, print the marker; raise if it did not take."""
    matmul = backend()
    before = getattr(matmul, FLAG)
    setattr(matmul, FLAG, False)
    after = getattr(matmul, FLAG)
    if after is not False:
        raise RuntimeError(f'{FLAG} did not read back False after assignment: {after!r}')
    record = {'rank': rank, 'before': bool(before), 'after': after, 'torch': torch.__version__,
              'pid': os.getpid(), 'applied_unix': time.time(), 'count': len(_state['applications']) + 1}
    _state['applications'].append(record)
    print(f'{MARKER} rank={rank} {FLAG}={after} before={before} torch={torch.__version__} pid={record["pid"]}',
          file=stream or sys.stdout, flush=True)
    return record


def status():
    """No-hot-path gate callable: the live flag and every application in this process."""
    return {'schema': SCHEMA, FLAG: getattr(backend(), FLAG), 'applications': list(_state['applications'])}
