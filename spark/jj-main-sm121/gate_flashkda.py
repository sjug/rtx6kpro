#!/usr/bin/env python3
"""Execute the selected native build's packed/dense checkpoint test matrix."""
import argparse
import sys
import types

import pytest
import torch
import vllm._flashkda_C  # noqa: F401

parser = argparse.ArgumentParser()
parser.add_argument('tests')
parser.add_argument('--collect-only', action='store_true')
args = parser.parse_args()
if torch.cuda.get_device_capability() != (12, 1):
    raise RuntimeError('This qualification requires GB10')

adapter = types.ModuleType('flash_kda')
adapter.get_workspace_size = torch.ops._flashkda_C.get_workspace_size


def fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
        initial_state=None, final_state=None, cu_seqlens=None, workspace=None,
        checkpoint_state=None, checkpoint_offsets=None, checkpoint_indptr=None):
    if workspace is None:
        sequences = q.shape[0] if cu_seqlens is None else cu_seqlens.numel() - 1
        size = adapter.get_workspace_size(q.shape[0] * q.shape[1], q.shape[2], sequences)
        workspace = torch.empty(size, dtype=torch.uint8, device=q.device)
    torch.ops._flashkda_C.fwd(q, k, v, g, beta, scale, out, workspace, A_log,
                            dt_bias, lower_bound, initial_state, final_state,
                            cu_seqlens, checkpoint_state, checkpoint_offsets,
                            checkpoint_indptr)


class RequiredCases:
    def __init__(self):
        self.collected = 0
        self.passed = 0
        self.invalid = []

    def pytest_collection_finish(self, session):
        self.collected = len(session.items)

    def pytest_collectreport(self, report):
        if report.skipped or report.failed:
            self.invalid.append(report.nodeid)

    def pytest_runtest_logreport(self, report):
        if report.skipped or report.failed or hasattr(report, 'wasxfail'):
            self.invalid.append(report.nodeid)
        if report.when == 'call' and report.passed:
            self.passed += 1


adapter.fwd = fwd
sys.modules['flash_kda'] = adapter
sys.path.insert(0, args.tests)
matrix = RequiredCases()
command = ['-s', '-vv', args.tests + '/test_vsplit.py', '-k', 'packed_checkpoints or dense_checkpoints']
if args.collect_only:
    command.append('--collect-only')
code = pytest.main(command, plugins=[matrix])
if code or matrix.collected != 12 or matrix.invalid or (not args.collect_only and matrix.passed != 12):
    raise RuntimeError(f'Incomplete FlashKDA gate: collected={matrix.collected}, passed={matrix.passed}, invalid={matrix.invalid}, exit={code}')
print(f'FLASHKDA-JJ-MAIN-{"COLLECTION" if args.collect_only else "GATE"}-PASS collected=12 passed={matrix.passed}', flush=True)
