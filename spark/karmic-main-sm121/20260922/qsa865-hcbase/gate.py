#!/usr/bin/env python3
"""PR865 gates only, in addition to the unchanged parent image gate suite."""
import os
import subprocess
import sys

if not __debug__:
    raise RuntimeError('Regression gates require assertions')
CASES = (
    ('tests/models/qwen4_exp/test_b12x_qsa.py::test_qsa_capture_metadata_covers_max_model_len', 1),
    ('tests/v1/cudagraph/test_cudagraph_manager.py::test_capture_marks_dummy_batch_for_every_graph_mode', 2),
)
if len(sys.argv) == 1:
    for index in range(len(CASES)):
        subprocess.run([sys.executable, __file__, str(index)], check=True)
else:
    sys.path.insert(0, '/gate/inherited/tests')
    from run_r38_regressions import RequiredCases
    import pytest
    import workspace_fixture
    os.chdir('/opt/jovian-judgement/vllm')
    path, count = CASES[int(sys.argv[1])]
    gate = RequiredCases(count)
    result = pytest.main(['-s', '-vv', '-p', 'no:cacheprovider',
                          '--confcutdir=' + path.rsplit('/', 1)[0], path],
                         plugins=[gate, workspace_fixture])
    gate.verify(result)
    print(f'QSA865-GATE-PASS count={count}', flush=True)
