"""Compose reviewed selector and MoE declaration patches without checkout writes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from prepare_deterministic_diagnostic import COMMIT, REPO, ROOT, object_id, tree_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch-sha256', required=True)
    parser.add_argument('--result-sha256', required=True)
    args = parser.parse_args()
    patch = ROOT / 'claude-dsa-topk-tiebreak.patch'
    if hashlib.sha256(patch.read_bytes()).hexdigest() != args.patch_sha256:
        raise RuntimeError('Selector patch differs from reviewed identity')
    path = 'b12x/attention/dsa_indexer/tiled_topk.py'
    original = subprocess.check_output(['git', '-C', str(REPO), 'show', f'{COMMIT}:{path}'])
    with tempfile.TemporaryDirectory(prefix='ds41-compose-') as temporary:
        target = Path(temporary) / path
        target.parent.mkdir(parents=True)
        target.write_bytes(original)
        subprocess.run(['git', 'apply', '--check', '--whitespace=error', str(patch)], cwd=temporary, check=True)
        subprocess.run(['git', 'apply', '--whitespace=error', str(patch)], cwd=temporary, check=True)
        result = target.read_bytes()
    if hashlib.sha256(result).hexdigest() != args.result_sha256:
        raise RuntimeError('Patched selector differs from reviewed identity')
    listing = subprocess.check_output(['git', '-C', str(REPO), 'ls-tree', '-rz', COMMIT])
    entries = {}
    for raw in listing.split(b'\0'):
        if raw:
            metadata, name = raw.split(b'\t', 1)
            mode, _, digest = metadata.decode().split()
            entries[name.decode()] = (mode, digest)
    moe = json.loads((ROOT / 'moe-dependency.lock.json').read_text())
    moe_source = (ROOT / 'moe-dependency-preparation.py').read_bytes()
    if hashlib.sha256(moe_source).hexdigest() != moe['output_sha256']:
        raise RuntimeError('MoE dependency repair drift')
    entries[moe['target_path']] = ('100644', object_id('blob', moe_source))
    if tree_id(entries) != moe['b12x_tree']:
        raise RuntimeError('MoE control base tree mismatch')
    entries[path] = ('100644', object_id('blob', result))
    lock = {'base_image_id': 'd880b1297eda5d788bd45c2f85a0b42e9dbcdde856117a10ec6ec3f92d7b5937',
            'upstream_b12x_commit': COMMIT, 'base_b12x_tree': moe['b12x_tree'],
            'b12x_tree': tree_id(entries),
            'selector_path': path, 'selector_input_sha256': hashlib.sha256(original).hexdigest(),
            'selector_output_sha256': args.result_sha256, 'selector_patch_sha256': args.patch_sha256,
            'moe_dependency': moe,
            'scope': 'Deterministic DSA tie selection plus declaration of existing deterministic MoE reducer'}
    (ROOT / 'determinism-tiled-topk.py').write_bytes(result)
    (ROOT / 'determinism.lock.json').write_text(json.dumps(lock, indent=2, sort_keys=True) + '\n')
    print(json.dumps(lock, indent=2))


if __name__ == '__main__':
    main()
