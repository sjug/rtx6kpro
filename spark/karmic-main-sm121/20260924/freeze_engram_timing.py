"""Pin the diagnostic derivative without changing the serving launch contract."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
build = json.loads((root / 'receipts/engram-timing-build-receipt.json').read_text())
receipt = root / 'receipts' / Path(build['directory']).name
image = (receipt / 'BUILD-OK').read_text().strip()
info = json.loads((receipt / 'image-inspect.json').read_text())[0]
lock_bytes = (root / 'claude-engram-timing.lock.json').read_bytes()
lock = json.loads(lock_bytes)
digest = hashlib.sha256(lock_bytes).hexdigest()
if image != build['image_id'] or info['Id'].removeprefix('sha256:') != image:
    raise RuntimeError('Build receipt identity mismatch')
for name, expected in lock['inputs'].items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError('Built input drift: ' + name)
old = json.loads((root / 'candidate.json').read_text())
if old['image_id'] != lock['base_image_id']:
    raise RuntimeError('Expected the ring-fix candidate as parent')
labels = info['Config']['Labels']
for key, expected in {
    'local-inference.ds41.diagnostic.kind': 'engram-host-timing',
    'local-inference.ds41.diagnostic.lock.sha256': digest,
    'b12x.source-tree': old['diagnostic']['b12x_tree'],
    'local-inference.status': 'diagnostic-only-not-qualified',
    'local-inference.ds41-build-lock.sha256': old['build_lock_sha256'],
    'local-inference.source-lock.sha256': old['source_lock_sha256'],
    'local-inference.serving.recipe.sha256': old['recipe_sha256'],
}.items():
    if labels.get(key) != expected:
        raise RuntimeError('Image label mismatch: ' + key)
manifest = json.loads((root / 'runtime-files.json').read_text())
for name, expected in manifest.items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError('Prior runtime drift: ' + name)
pin = dict(old, image_id=image, diagnostic={
    'kind': 'engram-host-timing', 'lock_sha256': digest,
    'b12x_tree': old['diagnostic']['b12x_tree']})
payload = json.dumps(pin, indent=2, sort_keys=True) + '\n'
updated = dict(manifest, **{'candidate.json': hashlib.sha256(payload.encode()).hexdigest()})
with (root / f'engram-timing-{image[:12]}-candidate-amendment.json').open('x') as stream:
    stream.write(json.dumps({'prior_candidate': old, 'prior_manifest': manifest,
        'candidate': pin, 'manifest': updated, 'status': 'diagnostic only'},
        indent=2, sort_keys=True) + '\n')
(root / 'candidate.json').write_text(payload)
(root / 'runtime-files.json').write_text(json.dumps(updated, indent=2, sort_keys=True) + '\n')
print('ENGRAM-TIMING-CANDIDATE-FROZEN', image)
