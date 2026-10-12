#!/usr/bin/env python3
"""Run the upstream compiler corpus and MXFP8 numerical/graph suite on SM121.

B12X ships inside FlashInfer 0.7.1; its tests moved to tests/experimental/b12x of the FlashInfer
tree and the archived repository's validation/cutlass_migration/corpus.txt is gone. The corpus
is pinned here explicitly (the archived tip's twelve entries at their new paths) and runs from
the kit's frozen, host-unpacked FlashInfer source against the installed B12X root.
"""
import importlib.metadata as metadata
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path('/kit/build/flashinfer')
TESTS = 'tests/experimental/b12x/'
B12X_ROOT = Path('/opt/jovian-judgement/b12x')


CORPUS = (  # validation/cutlass_migration/corpus.txt of the archived B12X tip 236ddff0, relocated
    '_lib/test_cutlass_runtime_patches.py',
    'gemm/test_blockscaled.py',
    'gemm/test_tensor_fp8_linear.py',
    'moe/test_fused_moe.py',
    'attention/test_attention_paged_page128_high_pid.py',
    'gemm/test_cute_migration_gemm_corpus.py',
    'moe/test_w4a8_migration_corpus.py',
    'attention/test_cute_migration_paged_corpus.py::test_paged_fp8_prefill_size_graph_oracle',
    'attention/test_cute_migration_paged_corpus.py::test_paged_verify_split_merge_graph_oracle',
    'attention/test_cute_migration_paged_corpus.py::test_paged_decode_direct_graph_oracle',
    'attention/test_attention_contiguous_torch_ref.py',
    'preparation/test_startup_capture.py::test_graph_replay_survives_compiler_cache_eviction',
)


def cases():
    return [TESTS + case for case in CORPUS] + [TESTS + case for case in (
        'moe/test_mxfp8_w8a8.py',
        'sequence/test_ple_embedding.py',  # caller-owned mapped-host storage (#454)
        'moe/test_w4a16_e2e.py::test_w4a16_skipped_empty_row_blocks_are_bit_identical',  # #469
        'moe/test_w4a16_a4_prefill.py',  # a4 prefill, now per call (#472/#473/#476, FlashInfer 3294b3ad)
        'moe/test_nvfp4_split_dispatch_policy.py::test_preparation_resolves_determinism_before_selecting_split',  # #464
        'preparation/test_session.py::test_preparation_import_ignores_unused_barrier_timeout')]  # #467


def child(mode, expected=0):
    import pytest
    os.chdir(ROOT)
    # The corpus imports FlashInfer's `benchmarks.experimental` helpers from the source tree.
    # vLLM's own `benchmarks` package (first on PYTHONPATH) owns that name, so the FlashInfer
    # directory is added to its package path; the source tree itself stays off sys.path so it
    # cannot shadow the installed flashinfer package or the B12X root.
    import benchmarks
    if str(ROOT / 'benchmarks') not in benchmarks.__path__:
        benchmarks.__path__.append(str(ROOT / 'benchmarks'))
    args = ['-p', 'no:cacheprovider', '--confcutdir=' + TESTS.rstrip('/'),
            # test_a4_prefill_launchers_are_owned_by_each_device needs two CUDA devices (GB10 has one).
            '-k', 'not matches_flashinfer_cudnn and not test_laguna_gqa6_extend_prepared_graph_replay_high_page_ids_and_tails'
                  ' and not test_a4_prefill_launchers_are_owned_by_each_device', *cases()]
    if mode == 'collect':
        class Count:
            total = 0

            def pytest_collection_finish(self, session):
                Count.total = len(session.items)
        result = pytest.main([*args, '-q', '--collect-only'], plugins=[Count()])
        if result or Count.total == 0:
            raise RuntimeError('Empty or failed compiler gate collection')
        print(f'COLLECTED {Count.total}', flush=True)
    else:
        sys.path.insert(0, '/gate/inherited/tests')
        from run_r38_regressions import RequiredCases
        gate = RequiredCases(expected)
        import torch
        import flashinfer
        if not Path(flashinfer.__file__).resolve().is_relative_to('/opt/venv'):
            raise RuntimeError(f'Source tree shadows the installed FlashInfer: {flashinfer.__file__}')
        torch.set_float32_matmul_precision('highest')
        import compiler_oracle
        gate.verify(pytest.main([*args, '-vv', '-s'], plugins=[gate, compiler_oracle]))


def main():
    import cutlass
    from verify_compiler import validate_cutlass, validate_compiler_metadata
    import b12x
    # The root initializer executes _api.py as the package (b12x_root_init.py).
    if Path(b12x.__file__).resolve() != (B12X_ROOT / 'b12x/_api.py').resolve():
        raise RuntimeError(f'Compiler gate must exercise the B12X root: {b12x.__file__}')
    if not (ROOT / TESTS / 'conftest.py').is_file():
        raise RuntimeError('Frozen FlashInfer source with tests/experimental/b12x is not mounted at /kit/build/flashinfer')
    print('CUTLASS-IMPORT-PASS', validate_cutlass(cutlass), flush=True)
    print('COMPILER-CONSUMER-METADATA-PASS', validate_compiler_metadata(), flush=True)
    import torch
    if torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError('Compiler gate requires SM121')
    lock = json.loads(Path('/kit/inputs.lock.json').read_text())
    for name, row in lock['wheels'].items():
        if metadata.version(name) != row['version']:
            raise RuntimeError(f'Dependency version mismatch: {name}')
    # Check the updated distributions' dependency constraints, including extras without
    # extras-only requirements; retain inherited foundation's unrelated package policy.
    from packaging.requirements import Requirement
    for name in (*lock['wheels'], 'flashinfer-python', 'flashinfer-jit-cache', 'flashinfer-jit-cache-sm121a', 'vllm', 'b12x'):
        for text in metadata.requires(name) or ():
            req = Requirement(text)
            if name in ('vllm', 'b12x') and not req.name.startswith('nvidia-cutlass-dsl'):
                continue
            if req.marker and not req.marker.evaluate({'extra': ''}):
                continue
            installed = metadata.version(req.name)
            if req.specifier and not req.specifier.contains(installed, prereleases=True):
                raise RuntimeError(f'Unsatisfied dependency: {name} requires {text}, found {installed}')
    # The tests import the installed b12x (the root, first on PYTHONPATH) and the installed
    # flashinfer; the child extends vLLM's `benchmarks` package path for the corpus helpers only.
    env = dict(os.environ)
    out = subprocess.check_output([sys.executable, __file__, 'collect'], env=env, text=True)
    counts = [line for line in out.splitlines() if line.startswith('COLLECTED ')]
    if len(counts) != 1:
        raise RuntimeError(f'Malformed compiler gate collection:\n{out}')
    count = int(counts[0].split()[1])
    subprocess.run([sys.executable, __file__, 'run', str(count)], env=env, check=True)
    print(f'KARMIC-COMPILER-471-GATE-PASS cases={count}', flush=True)


if __name__ == '__main__':
    if not __debug__:
        raise RuntimeError('Compiler gates require assertions')
    if len(sys.argv) == 1:
        main()
    elif sys.argv[1] in ('collect', 'run'):
        child(sys.argv[1], int(sys.argv[2]) if len(sys.argv) == 3 else 0)
    else:
        raise RuntimeError('Unknown compiler gate mode')
