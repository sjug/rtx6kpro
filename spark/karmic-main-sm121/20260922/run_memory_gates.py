#!/usr/bin/env python3
"""New memory contracts, each collected and executed in its own process."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'inherited/tests'))
from run_r38_regressions import RequiredCases

CASES = {
    'verified_counts': ('tests/v1/worker/test_async_verified_draft_counts.py', 2),
    'roce_health': ('tests/v1/worker/test_b12x_roce_health.py', 5),
    'restore_scalars': ('tests/v1/worker/test_boundary_checkpoint_restore_scalars.py', 19),
    'mhc': ('tests/model_executor/kernels/test_b12x_linear.py::test_b12x_mhc_preparation_reuses_outputs_with_exact_graph_replay', 8),
    'envelope': ('tests/v1/attention/test_ds4_workspace_envelope.py', 8),
    'lane_reserve': ('tests/v1/worker/test_workspace.py::test_preparation_reservation_preserves_per_lane_profile_capacity', 2),
    'metadata_free': ('tests/v1/worker/test_workspace.py::test_dsv4_metadata_free_profile_does_not_reserve_split_attention', 2),
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=CASES)
    parser.add_argument('--collect-only-count', action='store_true')
    args = parser.parse_args()
    if not args.case:
        for case in CASES:
            subprocess.run([sys.executable, __file__, '--case', case] +
                           (['--collect-only-count'] if args.collect_only_count else []), check=True)
        return
    import pytest
    os.chdir('/opt/jovian-judgement/vllm')
    path, count = CASES[args.case]
    gate = RequiredCases(count)
    argv = ['-s', '-vv', '--confcutdir=' + path.rsplit('/', 1)[0], path]
    if args.collect_only_count:
        argv.append('--collect-only')
    status = pytest.main(argv, plugins=[gate, __import__('workspace_fixture')])
    if args.collect_only_count:
        gate.verify_collection(status, count)
    else:
        gate.verify(status)
    print(f'MEMORY-GATE-PASS {args.case} count={count} collect_only={args.collect_only_count}', flush=True)

if __name__ == '__main__':
    main()
