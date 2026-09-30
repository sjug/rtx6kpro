"""Pin the built combined candidate, preserving the previous pin and manifest."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
receipt = root / 'receipts/determinism-build'
image = (receipt / 'BUILD-OK').read_text().strip().removeprefix('sha256:')
info = json.loads((receipt / 'image-inspect.json').read_text())[0]
raw = (root / 'determinism.lock.json').read_bytes()
lock = json.loads(raw)
digest = hashlib.sha256(raw).hexdigest()
if info['Id'].removeprefix('sha256:') != image:
    raise RuntimeError('Build image mismatch')
labels = info['Config']['Labels']
if labels['local-inference.ds41.diagnostic.lock.sha256'] != digest or labels['b12x.source-tree'] != lock['b12x_tree']:
    raise RuntimeError('Image composition mismatch')
for name, marker in [('test_moe_dependency.py.log', 'DETERMINISTIC-MOE-DEPENDENCY-PASS'),
                     ('claude_test_topk_tiebreak.py.log', 'CLAUDE-TOPK-TIEBREAK-PASS checks=10598')]:
    if marker not in (receipt / name).read_text().splitlines():
        raise RuntimeError(f'Missing gate: {name}')
old = json.loads((root / 'candidate.json').read_text())
manifest = json.loads((root / 'runtime-files.json').read_text())
if old['image_id'] != lock['base_image_id']:
    raise RuntimeError('Unexpected preceding candidate')
for name, expected in manifest.items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError(f'Existing runtime drift: {name}')
pin = dict(old, image_id=image, diagnostic={'kind': 'determinism',
    'lock_sha256': digest, 'b12x_tree': lock['b12x_tree']})
data = (json.dumps(pin, indent=2, sort_keys=True) + '\n').encode()
updated = dict(manifest, **{'candidate.json': hashlib.sha256(data).hexdigest()})
with (root / 'determinism-candidate-amendment.json').open('x') as f:
    f.write(json.dumps({'prior_candidate': old, 'prior_manifest': manifest,
                       'candidate': pin, 'manifest': updated}, indent=2, sort_keys=True) + '\n')
(root / 'candidate.json').write_bytes(data)
(root / 'runtime-files.json').write_text(json.dumps(updated, indent=2, sort_keys=True) + '\n')
print('DETERMINISM-CANDIDATE-FROZEN', image)
