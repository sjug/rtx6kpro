#!/usr/bin/env python3
"""Run the patched upstream packed-state tests against vLLM's actual extension.

Only the Python package adapter is substituted. Test inputs, reference
recurrence and checkpoint comparisons are upstream's. Both graph tests also
poison output and final state before replay so stale eager output cannot pass.
"""

if not __debug__:
    raise RuntimeError("R28 verification requires Python assertions enabled")

import sys
import types

import torch
import vllm._flashkda_C  # noqa: F401
from flashkda_counts import RequiredMatrix

assert torch.cuda.get_device_capability() == (12, 1)
assert "checkpoint_indptr" in str(torch.ops._flashkda_C.fwd.default._schema)
adapter = types.ModuleType("flash_kda")
adapter.get_workspace_size = torch.ops._flashkda_C.get_workspace_size


def fwd(
    q,
    k,
    v,
    g,
    beta,
    scale,
    out,
    A_log,
    dt_bias,
    lower_bound,
    initial_state=None,
    final_state=None,
    cu_seqlens=None,
    workspace=None,
    checkpoint_state=None,
    checkpoint_offsets=None,
    checkpoint_indptr=None,
):
    if workspace is None:
        sequences = q.shape[0] if cu_seqlens is None else cu_seqlens.numel() - 1
        size = adapter.get_workspace_size(
            q.shape[0] * q.shape[1], q.shape[2], sequences
        )
        workspace = torch.empty(size, dtype=torch.uint8, device=q.device)
    torch.ops._flashkda_C.fwd(
        q,
        k,
        v,
        g,
        beta,
        scale,
        out,
        workspace,
        A_log,
        dt_bias,
        lower_bound,
        initial_state,
        final_state,
        cu_seqlens,
        checkpoint_state,
        checkpoint_offsets,
        checkpoint_indptr,
    )


adapter.fwd = fwd
sys.modules["flash_kda"] = adapter
directory = "/opt/local-inference/r28-native/flashkda-tests"
sys.path.insert(0, directory)
import pytest

matrix = RequiredMatrix()
code = pytest.main(
    [
        "-s",
        "-vv",
        directory + "/test_vsplit.py",
        "-k",
        "packed_checkpoints or dense_checkpoints",
    ],
    plugins=[matrix],
)
matrix.verify(code)
print("FLASHKDA-R28-GATE-OK collected=12 passed=12 skipped=0", flush=True)
