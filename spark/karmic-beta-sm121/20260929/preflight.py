#!/usr/bin/env python3
"""Check every Karmic beta assembly input before entering a GPU build window."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# SM121 foundation tree: base image 1a7a8acf, its source lock, shared contracts and gates.
FOUNDATION = ROOT.parents[1] / 'karmic-main-sm121/20260922'
sys.path.insert(0, str(FOUNDATION))
from contracts import file_sha, require, sha

BASE = '1a7a8acff71044e3b8bf97dacb7617cd1bdb50d0ab2fc2e35364a4d4dc165dcc'
VLLM_TREE = '5a01c2d4f2822b2e2b39c0422b555f8aa59c9bb8'
B12X_TREE = '641d413e96d96bf673b34d573c0e27d6acb3a9ca'
FILES = ['Dockerfile', '.containerignore', 'install.py', 'prepare.py', 'preflight.py', 'build.sh',
         'run-qwen.sh', 'gate_beta.py', 'test_kit.py', 'build.lock.json', 'runtime.lock.json', 'refresh.tar']


def validate():
    require(__debug__, 'Optimized Python is not admitted')
    pin = json.loads((ROOT / 'build.lock.json').read_text())
    for key, path in [('base_lock_sha256', FOUNDATION / 'qsa865/runtime.lock.json'),
                      ('runtime_lock_sha256', ROOT / 'runtime.lock.json'),
                      ('refresh_sha256', ROOT / 'refresh.tar')]:
        require(pin[key] == file_sha(path), f'Input drift: {path}')
    require(pin['base_image_id'] == BASE, 'Base drift')
    lock = json.loads((ROOT / 'runtime.lock.json').read_text())
    require(lock['base_image_id'] == BASE, 'Runtime lock base drift')
    for name, expected in lock['assets'].items():
        require(file_sha(FOUNDATION / name) == expected, f'Inherited gate asset drift: {name}')
    require(lock['sources']['vllm']['refreshed_tree'] == VLLM_TREE, 'vLLM tree drift')
    require(lock['sources']['b12x']['refreshed_tree'] == B12X_TREE, 'B12X tree drift')
    require(lock['sources']['vllm']['commit'] == pin['vllm_commit'], 'vLLM commit drift')
    require(lock['sources']['b12x']['commit'] == pin['b12x_commit'], 'B12X commit drift')
    manifest = {name: file_sha(ROOT / name) for name in FILES}
    # Reused foundation gates and their imports are recipe inputs, not mutable external tests.
    for path in FOUNDATION.rglob('*'):
        parts = path.relative_to(FOUNDATION).parts
        if path.is_file() and path.suffix in ('.py', '.sh') and (
            len(parts) == 1 or parts[0] == 'inherited' or parts[:2] == ('qsa865', 'gate.py')
        ):
            manifest['karmic-main-sm121/20260922/' + str(path.relative_to(FOUNDATION))] = file_sha(path)
    values = [file_sha(ROOT / 'build.lock.json'), pin['runtime_lock_sha256'], pin['cache_fingerprint'],
              VLLM_TREE, B12X_TREE, sha(json.dumps(manifest, sort_keys=True).encode())]
    return manifest, values


if __name__ == '__main__':
    manifest, values = validate()
    if sys.argv[1:] == ['--values']:
        print('\n'.join(values))
    elif sys.argv[1:] == ['--manifest']:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        require(not sys.argv[1:], 'Unknown argument')
        print('KARMIC-BETA-INPUTS-PASS')
