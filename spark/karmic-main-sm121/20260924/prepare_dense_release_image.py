"""Compose the measured two-line fence over the uninstrumented repair image."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from prepare_deterministic_diagnostic import COMMIT, REPO, object_id, tree_id

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--refresh-unbuilt', action='store_true')
args = parser.parse_args()
if args.refresh_unbuilt and (list((root / 'receipts').glob('dense-release-build-*'))):
    raise RuntimeError('Refuse to refresh a recipe with build receipts')
prior = json.loads((root / 'determinism.lock.json').read_text())
variant = json.loads((root / 'dense-fence-before-release.json').read_text())
target = 'b12x/_lib/dense_gemm.py'
data = (root / 'dense_gemm-fence-before-release.py').read_bytes()
original = subprocess.check_output(['git', '-C', str(REPO), 'show', f'{COMMIT}:{target}'])
if hashlib.sha256(data).hexdigest() != variant['source_sha256']:
    raise RuntimeError('Measured source changed')
entries = {}
for row in subprocess.check_output(['git', '-C', str(REPO), 'ls-tree', '-rz', COMMIT]).split(b'\0'):
    if row:
        metadata, name = row.split(b'\t', 1)
        mode, _, digest = metadata.decode().split()
        entries[name.decode()] = (mode, digest)
baseline = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', f'{COMMIT}^{{tree}}'], text=True).strip()
if tree_id(entries) != baseline:
    raise RuntimeError('Original tree reconstruction failed')
for path, file, expected in [
    (prior['moe_dependency']['target_path'], 'moe-dependency-preparation.py', prior['moe_dependency']['output_sha256']),
    (prior['selector_path'], 'determinism-tiled-topk.py', prior['selector_output_sha256']),
]:
    content = (root / file).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        raise RuntimeError('Prior repair changed: ' + file)
    entries[path] = ('100644', object_id('blob', content))
if tree_id(entries) != prior['b12x_tree']:
    raise RuntimeError('Uninstrumented base tree mismatch')
entries[target] = ('100644', object_id('blob', data))
lock = {
    'base_image_id': '30ef9c2d529e9df0a3ea348c773d8ff55284389a90a40922ac1b0d93ccffdd9b',
    'base_b12x_tree': prior['b12x_tree'], 'b12x_tree': tree_id(entries),
    'upstream_b12x_commit': COMMIT, 'target_path': target,
    'input_sha256': hashlib.sha256(original).hexdigest(),
    'output_sha256': hashlib.sha256(data).hexdigest(),
    'scope': 'Order shared-memory reads before both dense GEMM consumer stage releases; arithmetic and tactics unchanged',
    'inputs': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in (
        'dense_gemm-fence-before-release.py', 'dense-fence-before-release.patch',
        'determinism.lock.json', 'runtime.lock.json', 'install_dense_release.py',
        'Dockerfile.dense-release', 'prepare_dense_release_image.py')},
    'build_inputs': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in (
        'build_dense_release.py', 'dense-release.ignore', 'extract_dense_cubin.py',
        'verify_dense_release_sass.py', 'probe_engram_dense_replay.py',
        'test_moe_dependency.py', 'claude_test_topk_tiebreak.py')},
}
path = root / 'dense-release.lock.json'
with path.open('w' if args.refresh_unbuilt else 'x') as stream:
    stream.write(json.dumps(lock, indent=2, sort_keys=True) + '\n')
print(json.dumps(lock, indent=2))
