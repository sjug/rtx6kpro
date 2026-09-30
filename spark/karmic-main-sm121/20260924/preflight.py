#!/usr/bin/env python3
"""Check every runtime assembly input before entering a GPU build window."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FOUNDATION = ROOT.parent / '20260922'
sys.path.insert(0, str(FOUNDATION))
from contracts import file_sha, require, sha
from check_nccl import check


def validate():
    require(__debug__, 'Optimized Python is not admitted')
    check()
    pin = json.loads((ROOT / 'build.lock.json').read_text())
    for key, path in [('base_lock_sha256', FOUNDATION / 'qsa865/runtime.lock.json'),
                      ('runtime_lock_sha256', ROOT / 'runtime.lock.json'),
                      ('refresh_sha256', ROOT / 'refresh.tar'),
                      ('nccl_lock_sha256', ROOT / 'nccl.lock.json')]:
        require(pin[key] == file_sha(path), f'Input drift: {path}')
    require(pin['base_image_id'] == '1a7a8acff71044e3b8bf97dacb7617cd1bdb50d0ab2fc2e35364a4d4dc165dcc', 'Base drift')
    component = pin['nccl_component']
    for key in ('image_id', 'library_sha256'):
        value = component[key]
        require(len(value) == 64 and set(value) <= set('0123456789abcdef'), f'Invalid component {key}')
    lock = json.loads((ROOT / 'runtime.lock.json').read_text())
    for name, expected in lock['assets'].items():
        require(file_sha(FOUNDATION / name) == expected, f'Inherited gate asset drift: {name}')
    require(lock['sources']['vllm']['refreshed_tree'] == 'a906a6d9f8843a177c7e627c592894163add1f3b', 'vLLM tree drift')
    require(lock['sources']['b12x']['refreshed_tree'] == '19e40c53ad4d8fa52d32177e4f72c73c780c3cce', 'B12X tree drift')
    files = ['Dockerfile', '.dockerignore', 'install.py', 'prepare.py', 'preflight.py', 'build.sh',
             'gate_ds41.py', 'launch_contract.py', 'upstream-launch.json', 'prepare_launcher.py',
             'test_launch_contract.py', 'test_wheel_package.py', 'gate_engram.py', 'seccomp-io-uring.json',
             'build.lock.json', 'runtime.lock.json', 'nccl.lock.json',
             'refresh.tar', 'pin_component.py', 'check_nccl.py']
    manifest = {name: file_sha(ROOT / name) for name in files}
    # Reused gates and their imports are recipe inputs, not mutable external tests.
    for path in FOUNDATION.rglob('*'):
        parts = path.relative_to(FOUNDATION).parts
        if path.is_file() and path.suffix in ('.py', '.sh') and (
            len(parts) == 1 or parts[0] == 'inherited'
        ):
            manifest['../20260922/' + str(path.relative_to(FOUNDATION))] = file_sha(path)
    values = [file_sha(ROOT / 'build.lock.json'), pin['runtime_lock_sha256'], pin['cache_fingerprint'],
              lock['sources']['vllm']['refreshed_tree'], lock['sources']['b12x']['refreshed_tree'],
              sha(json.dumps(manifest, sort_keys=True).encode()), component['image_id'], component['library_sha256']]
    return manifest, values


if __name__ == '__main__':
    manifest, values = validate()
    if sys.argv[1:] == ['--values']:
        print('\n'.join(values))
    elif sys.argv[1:] == ['--manifest']:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        require(not sys.argv[1:], 'Unknown argument')
        print('DS41-RUNTIME-INPUTS-PASS')
