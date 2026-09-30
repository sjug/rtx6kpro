#!/usr/bin/env python3
"""Run each upstream regression in a fresh process; reject absent/skipped cases."""
import os
import subprocess
import sys
import tempfile

if not __debug__:
    raise RuntimeError('Regression gates require assertions')
CASES = (
    ('tests/models/qwen4_exp/test_b12x_qsa.py::test_qsa_capture_metadata_covers_max_model_len', 1),
    ('tests/v1/cudagraph/test_cudagraph_manager.py::test_capture_marks_dummy_batch_for_every_graph_mode', 2),
    ('tests/models/kimi_k3/test_recoverssm_boundary.py', 10),
    ('tests/model_executor/kernels/test_b12x_linear.py::test_b12x_fp8_unplanned_rows_reuse_capacity_or_default_and_replay', 2),
    ('tests/v1/attention/test_b12x_sparse_mla_api.py::test_deepseek_v4_wo_preparation_runs_and_replays_native_projection', 2),
    ('lmcache-tests/tests/v1/multiprocess/test_mq.py::test_server_survives_malformed_request_headers', 1),
)
if len(sys.argv) == 1:
    # The loader links libcuda, so compile and load only with CDI attached.
    # A unique cache forces compilation from the refreshed C sources.
    from b12x.loader import _native
    previous_cache = os.environ.get('XDG_CACHE_HOME')
    with tempfile.TemporaryDirectory(prefix='qsa865-loader-') as cache:
        os.environ['XDG_CACHE_HOME'] = cache
        module = _native.load()
        if module.ABI_VERSION != 1:
            raise RuntimeError('Wrong loader ABI')
        linked = subprocess.check_output(['ldd', module.__file__], text=True)
        print(linked, flush=True)
        if 'not found' in linked or 'liburing.so' not in linked:
            raise RuntimeError('Loader dependencies unresolved or liburing missing')
        print('QSA865-LOADER-COMPILE-LINK-PASS', flush=True)
    if previous_cache is None:
        del os.environ['XDG_CACHE_HOME']
    else:
        os.environ['XDG_CACHE_HOME'] = previous_cache
    for index in range(len(CASES)):
        subprocess.run([sys.executable, __file__, str(index)], check=True)
else:
    sys.path.insert(0, '/gate/inherited/tests')
    from run_r38_regressions import RequiredCases
    import pytest
    os.chdir('/opt/jovian-judgement/vllm')
    path, count = CASES[int(sys.argv[1])]
    plugins = []
    if path.startswith('lmcache-tests/'):
        os.chdir('/gate/qsa865/lmcache-tests')
        sys.path.insert(0, os.getcwd())
        path = path.removeprefix('lmcache-tests/')
    else:
        plugins.append(__import__('workspace_fixture'))
    gate = RequiredCases(count)
    result = pytest.main(['-s', '-vv', '-p', 'no:cacheprovider', '--confcutdir=' + path.rsplit('/', 1)[0], path], plugins=[gate, *plugins])
    gate.verify(result)
    print(f'QSA865-GATE-PASS count={count}', flush=True)
