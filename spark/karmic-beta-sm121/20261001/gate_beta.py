#!/usr/bin/env python3
"""Run beta's own regression tests for the fixes that motivated this build, in-image.

Every pytest session runs in its own process: first a collection pass, then a run that
must pass exactly the collected cases with no skip, xfail or error. Counts are not pinned
in advance (no local vLLM environment); they are printed for the build receipt.
"""
import os
import subprocess
import sys

if not __debug__:
    raise RuntimeError('Regression gates require assertions')
FILES = (
    'tests/models/test_deepseek_v4_1_compressor.py',              # vllm #943 compressor ring writes
    'tests/v1/worker/test_gpu_block_table.py',                    # circular-buffer slot mapping
    'tests/kernels/mamba/test_aligned_state_indices_padding.py',  # NULL_BLOCK_ID padding (638f90a1fc)
    'tests/v1/attention/test_dcp_correct_attn_out.py',            # DCP LSE for non-power-of-two ranks
    'tests/v1/executor/test_multiproc_gather_responses.py',       # failed-rank gather
    'tests/entrypoints/launchers/test_engine_failure_exit.py',    # serve exits 1 on engine failure
    'tests/v1/engine/test_structured_output_draft_handoff.py',    # MTP draft rows stay constrained
    'tests/v1/core/test_prefill_compute_share_scheduler.py',     # stalled prefill lanes (#959/#960)
    'tests/v1/core/test_boundary_admission.py',                  # full-KV preemption (#946)
    'tests/kernels/layers/test_qwen_gdn_linear_attn.py',          # bound state pool
    'tests/models/qwen4_exp/test_ple_shared_table.py',            # shared PLE table (#961)
)


def session(path, plugins, *extra):
    import pytest
    os.chdir('/opt/jovian-judgement/vllm')
    return pytest.main(['-p', 'no:cacheprovider', '--confcutdir=' + path.rsplit('/', 1)[0], *extra, path],
                       plugins=plugins)


def child(mode, index, expected=None):
    path = FILES[index]
    if mode == 'collect':
        class Counter:
            count = 0

            def pytest_collection_modifyitems(self, items):
                Counter.count = len(items)
        code = session(path, [Counter()], '-q', '--collect-only')
        if code != 0 or Counter.count == 0:
            raise RuntimeError(f'Collection failed or empty: {path} exit={code} count={Counter.count}')
        print(f'COLLECTED {Counter.count}', flush=True)
        return
    sys.path.insert(0, '/gate/inherited/tests')
    from run_r38_regressions import RequiredCases
    gate = RequiredCases(expected)
    gate.verify(session(path, [gate, __import__('workspace_fixture')], '-s', '-vv'))
    print(f'KARMIC-BETA-CASE-PASS {path} count={expected}', flush=True)


def main():
    total = 0
    for index, path in enumerate(FILES):
        out = subprocess.run([sys.executable, __file__, 'collect', str(index)], check=True,
                             capture_output=True, text=True)
        lines = [line for line in out.stdout.splitlines() if line.startswith('COLLECTED ')]
        if len(lines) != 1:
            raise RuntimeError(f'No collection count for {path}:\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}')
        count = int(lines[0].split()[1])
        subprocess.run([sys.executable, __file__, 'run', str(index), str(count)], check=True)
        total += count
    print(f'KARMIC-BETA-GATE-PASS files={len(FILES)} cases={total}', flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    else:
        mode, index = sys.argv[1], int(sys.argv[2])
        if mode not in ('collect', 'run'):
            raise RuntimeError(f'Unknown mode: {mode}')
        child(mode, index, int(sys.argv[3]) if mode == 'run' else None)
