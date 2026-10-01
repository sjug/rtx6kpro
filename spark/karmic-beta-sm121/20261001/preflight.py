#!/usr/bin/env python3
"""Check every Karmic beta assembly input before entering a GPU build window."""
import json
import sys
from pathlib import Path
from unpack_sources import artifact_sha

ROOT = Path(__file__).resolve().parent
# SM121 foundation tree: base image 1a7a8acf, its source lock, shared contracts and gates.
FOUNDATION = ROOT.parents[1] / 'karmic-main-sm121/20260922'
sys.path.insert(0, str(FOUNDATION))
from contracts import file_sha, require, sha

BASE = '1a7a8acff71044e3b8bf97dacb7617cd1bdb50d0ab2fc2e35364a4d4dc165dcc'
VLLM_TREE = '96971e1c01f2ed035ee73cceb71c144de76ede06'
B12X_TREE = '5561aa253c6853dab3d9041b70e74828fde44dad'
FILES = ['Dockerfile', '.containerignore', 'install.py', 'prepare.py', 'preflight.py', 'build.sh',
         'run-qwen.sh', 'gate_beta.py', 'gate_compiler.py', 'test_kit.py', 'build.lock.json', 'runtime.lock.json', 'refresh.tar',
         'prepare_inputs.py', 'inputs.lock.json', 'compiler-arm64.lock', 'unpack_sources.py', 'upgrade_dependencies.py',
         'publication.json', 'glm/run-glm-tp4-node.sh', 'glm/templates/glm53-flash.jinja',
         'ds4-vision/run-node.sh', 'ds4-vision/launch-in-container.sh', 'ds4-vision/runtime-preflight.py',
         'ds4-vision/verify-model.py', 'ds4-vision/runtime-files.sha256', 'ds4-vision/model-manifest.json',
         'verify_compiler.py', 'build_flashinfer.py', 'verify_flashinfer_component.py', 'gate_flashinfer.py', 'aot_guard.py', 'compare-production.py', 'capture-baseline.sh',
         'runner-baseline.json', 'test_review.py', 'distribute.sh', 'execute.sh', 'benchmark.sh',
         'ds4-vision/execute.sh', 'ds4-vision/benchmark.sh', 'ds4-vision/compare.py',
         'ds4-vision/probe-structured.py', 'ds4-vision/probe-prefill.py', 'glm/execute-glm.sh',
         'glm/qualify-glm.sh', 'glm/confirm-transport.sh', 'glm/probe-fresh-prefill.py', 'glm/probe-acceptance-ab.py']


def validate():
    require(__debug__, 'Optimized Python is not admitted')
    pin = json.loads((ROOT / 'build.lock.json').read_text())
    require(file_sha(ROOT / 'inputs.lock.json') == pin['inputs_lock_sha256'], 'Dependency lock drift')
    require(file_sha(ROOT / 'compiler-arm64.lock') == pin['compiler_lock_sha256'], 'Compiler wheel lock drift')
    dependencies = json.loads((ROOT / 'inputs.lock.json').read_text())
    for row in [*dependencies['wheels'].values(), *dependencies['sources'].values()]:
        require(artifact_sha(ROOT / 'inputs' / row['file'], row) == row['sha256'], 'Dependency artifact drift')
        if 'tree_file' in row:
            require(file_sha(ROOT / 'inputs' / row['tree_file']) == row['tree_sha256'], 'Source inventory drift')
    publication = json.loads((ROOT / 'publication.json').read_text())
    require(publication['components']['vllm']['source_commit'] == pin['vllm_commit'], 'Published vLLM pin drift')
    require(publication['components']['b12x']['source_commit'] == pin['b12x_commit'], 'Published B12X pin drift')
    require(publication['recipe_commit'] == pin['recipe_commit'], 'Publication recipe drift')
    require(publication['components']['flashinfer']['source_commit'] == dependencies['sources']['flashinfer']['commit'],
            'Published FlashInfer pin drift')
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
    for path in (ROOT / 'inputs').iterdir():
        require(path.is_file() and not path.is_symlink(), f'Invalid dependency input: {path}')
        row = next((row for row in dependencies['sources'].values() if row['file'] == path.name), {})
        manifest['inputs/' + path.name] = artifact_sha(path, row)
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
