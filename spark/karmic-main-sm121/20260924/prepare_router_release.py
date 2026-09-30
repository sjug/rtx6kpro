"""Freeze the single router fence over the clean, instrument-free repair parent."""
import hashlib
import json
from pathlib import Path

import prepare
from contracts import git_tree, manifest

ROOT = Path(__file__).resolve().parent
BASE = 'ee03505df7a3b5b251d657ab201f62ac9d74eb2aee033cc5a094d7996caab4c5'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parent_bytes = (ROOT / 'engram-repair.lock.json').read_bytes()
    parent = json.loads(parent_bytes)
    files = prepare.previous.source('b12x', prepare.B12X)
    det = json.loads((ROOT / 'determinism.lock.json').read_text())
    dense = json.loads((ROOT / 'dense-release.lock.json').read_text())
    changes = [(det['moe_dependency']['target_path'], 'moe-dependency-preparation.py', det['moe_dependency']['output_sha256']),
               (det['selector_path'], 'determinism-tiled-topk.py', det['selector_output_sha256']),
               (dense['target_path'], 'dense_gemm-fence-before-release.py', dense['output_sha256'])]
    changes += [(path.removeprefix('b12x/'), target['source'], target['output_sha256'])
                for path, target in parent['targets'].items() if path.startswith('b12x/')]
    for path, source, expected in changes:
        data = (ROOT / source).read_bytes()
        if sha(data) != expected:
            raise RuntimeError('Parent source drift: ' + source)
        files[path] = (files[path][0], data)
    if manifest(files) != parent['after']['b12x'] or git_tree(files) != parent['trees']['b12x']:
        raise RuntimeError('Clean repair parent reconstruction differs')
    delta = json.loads((ROOT / 'router-prefill-release-before.json').read_text())
    target = delta['path']
    if sha(files[target][1]) != delta['base_sha256']:
        raise RuntimeError('Router input differs')
    updated = (ROOT / 'router-prefill-release-before.py').read_bytes()
    if sha(updated) != delta['source_sha256']:
        raise RuntimeError('Router output differs')
    files[target] = (files[target][0], updated)
    inputs = ['Dockerfile.router-release', 'prepare_router_release.py', 'install_router_release.py',
              'build_router_release.py', 'test_router_prefill_gpu.py', 'verify_engram_repair.py',
              'test_engram_epoch_gpu.py', 'engram-progress-engram.py',
              'test_engram_enqueue_gpu.py', 'run_ring_regression.py',
              'test_compressor_ring_mapping.py', 'seccomp-io-uring.json',
              'router-prefill-release-before.py', 'router-prefill-release-before.patch',
              'router-prefill-release-before.json', 'prepare_router_release_probe.py',
              'test_router_release_contract.py', 'test_prepare_router_release_probe.py',
              'claude-pinned-bf16_gemv-_prefill.py']
    inputs = sorted(inputs)
    lock = {'base_image_id': BASE, 'base_lock_sha256': sha(parent_bytes),
            'base_trees': parent['trees'], 'before': parent['after'],
            'after': {'vllm': parent['after']['vllm'], 'b12x': manifest(files)},
            'trees': {'vllm': parent['trees']['vllm'], 'b12x': git_tree(files)},
            'target': target, 'input_sha256': delta['base_sha256'],
            'output_sha256': delta['source_sha256'],
            'status': 'candidate-not-qualified',
            'inputs': {name: sha((ROOT / name).read_bytes()) for name in inputs}}
    lock['cache_fingerprint'] = 'ds41-router-' + sha(json.dumps(lock['trees'], sort_keys=True).encode())[:20]
    (ROOT / 'router-release.lock.json').write_text(json.dumps(lock, sort_keys=True, indent=2) + '\n')
    (ROOT / 'router-release.ignore').write_text('**\n' + ''.join('!' + n + '\n' for n in inputs + ['router-release.lock.json']))
    print('ROUTER-RELEASE-PREPARED', lock['trees'], flush=True)


if __name__ == '__main__':
    main()
