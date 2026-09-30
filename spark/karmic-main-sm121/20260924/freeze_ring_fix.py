"""Preserve the old kit and pin the build-gated, uninstrumented ring repair."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
build = json.loads((root / 'receipts/ring-fix-build-receipt.json').read_text())
receipt = root / 'receipts' / Path(build['directory']).name
image = (receipt / 'BUILD-OK').read_text().strip()
info = json.loads((receipt / 'image-inspect.json').read_text())[0]
lock_bytes = (root / 'ring-fix.lock.json').read_bytes()
lock = json.loads(lock_bytes)
digest = hashlib.sha256(lock_bytes).hexdigest()
if image != build['image_id'] or info['Id'].removeprefix('sha256:') != image:
    raise RuntimeError('Build receipt identity mismatch')
for name, expected in lock['inputs'].items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError('Built input drift: ' + name)
old_pin = json.loads((root / 'candidate.json').read_text())
labels = info['Config']['Labels']
for key, expected in {
    'local-inference.ds41.diagnostic.kind': 'compressor-ring-mapping-fix',
    'local-inference.ds41.diagnostic.lock.sha256': digest,
    'b12x.source-tree': lock['b12x_tree'],
    'local-inference.status': 'candidate-not-qualified',
    'local-inference.ds41-build-lock.sha256': old_pin['build_lock_sha256'],
    'local-inference.source-lock.sha256': old_pin['source_lock_sha256'],
    'local-inference.serving.recipe.sha256': old_pin['recipe_sha256'],
}.items():
    if labels.get(key) != expected:
        raise RuntimeError('Image label mismatch: ' + key)
old_manifest = json.loads((root / 'runtime-files.json').read_text())
for name, expected in old_manifest.items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError('Prior runtime drift: ' + name)
pin = dict(old_pin, image_id=image, diagnostic={
    'kind': 'compressor-ring-mapping-fix', 'lock_sha256': digest,
    'b12x_tree': lock['b12x_tree']})
pin.pop('diagnostic_overlay', None)
payload = json.dumps(pin, indent=2, sort_keys=True) + '\n'
manifest = dict(old_manifest, **{'candidate.json': hashlib.sha256(payload.encode()).hexdigest()})
with (root / f'ring-fix-{image[:12]}-candidate-amendment.json').open('x') as stream:
    stream.write(json.dumps({'prior_candidate': old_pin, 'prior_manifest': old_manifest,
        'candidate': pin, 'manifest': manifest, 'status': 'build gated; model qualification pending'},
        indent=2, sort_keys=True) + '\n')
(root / 'candidate.json').write_text(payload)
(root / 'runtime-files.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
print('RING-FIX-CANDIDATE-FROZEN', image)
