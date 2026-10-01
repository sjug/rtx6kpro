#!/usr/bin/env python3
"""Run the upstream compiler corpus and MXFP8 numerical/graph suite on SM121."""
import importlib.metadata as metadata
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path('/opt/jovian-judgement/b12x')


def cases():
    corpus = ROOT / 'validation/cutlass_migration/corpus.txt'
    return [line.strip() for line in corpus.read_text().splitlines() if line.strip()] + [
        'tests/moe/test_mxfp8_w8a8.py',
        'tests/sequence/test_ple_embedding.py']  # caller-owned mapped-host storage (#454)


def child(mode, expected=0):
    import pytest
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    args = ['-p', 'no:cacheprovider', '--confcutdir=tests',
            '-k', 'not matches_flashinfer_cudnn', *cases()]
    if mode == 'collect':
        class Count:
            total = 0

            def pytest_collection_modifyitems(self, items):
                Count.total = len(items)
        result = pytest.main([*args, '-q', '--collect-only'], plugins=[Count()])
        if result or Count.total == 0:
            raise RuntimeError('Empty or failed compiler gate collection')
        print(f'COLLECTED {Count.total}', flush=True)
    else:
        sys.path.insert(0, '/gate/inherited/tests')
        from run_r38_regressions import RequiredCases
        gate = RequiredCases(expected)
        gate.verify(pytest.main([*args, '-vv', '-s'], plugins=[gate]))


def main():
    import cutlass
    from verify_compiler import validate_cutlass
    print('CUTLASS-IMPORT-PASS', validate_cutlass(cutlass), flush=True)
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
    for name in (*lock['wheels'], 'flashinfer-python', 'flashinfer-jit-cache', 'vllm', 'b12x'):
        for text in metadata.requires(name) or ():
            req = Requirement(text)
            if name in ('vllm', 'b12x') and not req.name.startswith('nvidia-cutlass-dsl'):
                continue
            if req.marker and not req.marker.evaluate({'extra': ''}):
                continue
            installed = metadata.version(req.name)
            if req.specifier and not req.specifier.contains(installed, prereleases=True):
                raise RuntimeError(f'Unsatisfied dependency: {name} requires {text}, found {installed}')
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', ''))
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
