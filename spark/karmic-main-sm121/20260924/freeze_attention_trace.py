"""Retain the prior kit and pin the identical four-node diagnostic build."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
receipt = Path(json.loads((root / 'receipts/attention-trace-build-receipt.json').read_text())['directory'])
image = (receipt / 'BUILD-OK').read_text().strip()
lock_bytes = (root / 'attention-trace.lock.json').read_bytes()
lock = json.loads(lock_bytes)
sha = hashlib.sha256(lock_bytes).hexdigest()
old = json.loads((root / 'candidate.json').read_text())
manifest = json.loads((root / 'runtime-files.json').read_text())
if old['image_id'] != lock['base_image_id']:
    raise RuntimeError('Unexpected preceding image')
for name, expected in manifest.items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError(f'Existing runtime drift: {name}')
for node in ('dusty', 'toby', 'rusty', 'kirby'):
    info = json.loads((receipt / f'{node}-inspect.json').read_text())[0]
    if info['Id'].removeprefix('sha256:') != image:
        raise RuntimeError('Build identities differ')
    labels = info['Config']['Labels']
    if (labels['local-inference.ds41.diagnostic.lock.sha256'] != sha
            or labels['b12x.source-tree'] != old['diagnostic']['b12x_tree']
            or labels['local-inference.cache.fingerprint'] != 'ds41-trace-' + sha[:20]):
        raise RuntimeError('Diagnostic provenance mismatch')
pin = dict(old, image_id=image,
           diagnostic=dict(old['diagnostic'], kind='attention-trace', lock_sha256=sha))
data = (json.dumps(pin, indent=2, sort_keys=True) + '\n').encode()
updated = dict(manifest, **{'candidate.json': hashlib.sha256(data).hexdigest()})
with (root / 'receipts/attention-trace-amendment.json').open('x') as stream:
    stream.write(json.dumps({'prior_candidate': old, 'prior_manifest': manifest,
                            'candidate': pin, 'manifest': updated}, indent=2) + '\n')
(root / 'candidate.json').write_bytes(data)
(root / 'runtime-files.json').write_text(json.dumps(updated, indent=2, sort_keys=True) + '\n')
print('TRACE-CANDIDATE-FROZEN', image)
