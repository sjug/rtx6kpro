"""Compose the reviewed Engram repair over the exact clean ring-fix sources."""
import argparse
import hashlib
import json
from pathlib import Path

import prepare as runtime_prepare
from contracts import git_tree, manifest

ROOT = Path(__file__).resolve().parent
BASE = 'e7b273407022ae74726d6c7b7465aa567b2c5a0594d5e0b46002a742a11f5146'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_lock(name):
    return json.loads((ROOT / name).read_text())


def replace(files, path, source, expected):
    data = (ROOT / source).read_bytes()
    if sha(data) != expected:
        raise RuntimeError('Source drift: ' + source)
    files[path] = (files[path][0], data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--native-lock-sha256', required=True)
    args = parser.parse_args()
    native_bytes = (ROOT / 'claude-ple-batch.lock.json').read_bytes()
    if sha(native_bytes) != args.native_lock_sha256:
        raise RuntimeError('Native lock differs from reviewed identity')
    native = json.loads(native_bytes)
    for name, expected in native['inputs'].items():
        if sha((ROOT / name).read_bytes()) != expected:
            raise RuntimeError('Native recipe input drift: ' + name)
    failclosed = read_lock('claude-engram-failclosed.lock.json')
    progress = read_lock('engram-progress-vllm.lock.json')
    if progress['native_lock_sha256'] != args.native_lock_sha256:
        raise RuntimeError('Integration and native API locks disagree')
    if sha((ROOT / 'claude-engram-failclosed.lock.json').read_bytes()) != progress['failclosed_lock_sha256']:
        raise RuntimeError('Fail-closed parent drift')
    baseline = read_lock('runtime.lock.json')
    sources = {'vllm': runtime_prepare.patched_vllm(runtime_prepare.VLLM),
               'b12x': runtime_prepare.previous.source('b12x', runtime_prepare.B12X)}
    for component, files in sources.items():
        if manifest(files) != baseline['sources'][component]['files']:
            raise RuntimeError('Original Spark source reconstruction failed: ' + component)
    prior = read_lock('determinism.lock.json')
    dense = read_lock('dense-release.lock.json')
    ring = read_lock('ring-fix.lock.json')
    for path, source, expected in (
        (prior['moe_dependency']['target_path'], 'moe-dependency-preparation.py', prior['moe_dependency']['output_sha256']),
        (prior['selector_path'], 'determinism-tiled-topk.py', prior['selector_output_sha256']),
        (dense['target_path'], 'dense_gemm-fence-before-release.py', dense['output_sha256']),
    ):
        replace(sources['b12x'], path, source, expected)
    if git_tree(sources['b12x']) != ring['b12x_tree']:
        raise RuntimeError('Clean base B12X reconstruction failed')
    replace(sources['vllm'], ring['source_path'], 'ring-fix-sparse_mla.py', ring['output_sha256'])
    before = {component: manifest(files) for component, files in sources.items()}
    targets = {}
    for component, changes in [('vllm', progress['targets']), ('b12x', native['targets'])]:
        for path, target in changes.items():
            expected_input = (failclosed['targets'][path]['input_sha256']
                              if component == 'vllm' else target['input_sha256'])
            if sha(sources[component][path][1]) != expected_input:
                raise RuntimeError('Repair does not apply to clean base: ' + path)
            replace(sources[component], path, target['source'], target['output_sha256'])
            targets[component + '/' + path] = dict(target, input_sha256=expected_input)
    support = [
        'Dockerfile.engram-repair', 'install_engram_repair.py', 'build_engram_repair.py',
        'prepare_engram_repair.py', 'test_engram_enqueue_gpu.py', 'test_engram_epoch_gpu.py',
        'verify_engram_repair.py',
        'engram-progress-engram.py', 'claude-ple-batch.lock.json',
        'engram-progress-vllm.lock.json', 'claude-engram-failclosed.lock.json',
        'engram-progress-vllm.patch', 'claude-engram-failclosed.patch', 'claude-ple-batch.patch',
        'seccomp-io-uring.json', 'test_compressor_ring_mapping.py', 'run_ring_regression.py',
    ]
    inputs = sorted(set(support + [item['source'] for item in targets.values()]))
    record = {
        'base_image_id': BASE, 'base_b12x_tree': ring['b12x_tree'],
        'scope': 'native prequeued Engram I/O plus sticky cross-rank timeout rejection',
        'status': 'build-candidate-not-model-qualified',
        'reviewed_native_lock_sha256': args.native_lock_sha256,
        'before': before, 'after': {c: manifest(f) for c, f in sources.items()},
        'trees': {c: git_tree(f) for c, f in sources.items()},
        'targets': targets,
        'inputs': {name: sha((ROOT / name).read_bytes()) for name in inputs},
    }
    record['cache_fingerprint'] = 'ds41-engram-' + sha(json.dumps(
        {'trees': record['trees'], 'inputs': record['inputs']}, sort_keys=True).encode())[:20]
    (ROOT / 'engram-repair.lock.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    (ROOT / 'engram-repair.ignore').write_text('**\n' + ''.join('!' + name + '\n' for name in inputs + ['engram-repair.lock.json']))
    print('ENGRAM-REPAIR-PREPARED', sha((ROOT / 'engram-repair.lock.json').read_bytes()), record['trees'])


if __name__ == '__main__':
    main()
