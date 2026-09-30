"""Pin the reviewed diagnostic image, retaining the complete preceding kit."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
build = json.loads((root / 'receipts/stale-probe-build-receipt.json').read_text())
receipt = root / 'receipts' / Path(build['directory']).name
image = (receipt / 'BUILD-OK').read_text().strip().removeprefix('sha256:')
info = json.loads((receipt / 'image-inspect.json').read_text())[0]
lock_bytes = (root / 'stale-probe.lock.json').read_bytes()
lock = json.loads(lock_bytes)
digest = hashlib.sha256(lock_bytes).hexdigest()
if image != build['image_id'] or info['Id'].removeprefix('sha256:') != image:
    raise RuntimeError('Diagnostic build receipt mismatch')
for name, expected in lock['inputs'].items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError('Build input drift: ' + name)
old_pin = json.loads((root / 'candidate.json').read_text())
if (old_pin['image_id'] != lock['base_image_id'] and
        old_pin.get('diagnostic', {}).get('kind') != 'stale-state-probe'):
    raise RuntimeError('Prior candidate is not the diagnostic base')
labels = info.get('Labels') or info['Config']['Labels']
for label, value in {
    'local-inference.ds41.diagnostic.lock.sha256': digest,
    'local-inference.ds41.diagnostic.kind': 'stale-state-probe',
    'b12x.source-tree': lock['b12x_tree'],
    'local-inference.status': 'diagnostic-only-not-qualified',
    'local-inference.ds41-build-lock.sha256': old_pin['build_lock_sha256'],
    'local-inference.source-lock.sha256': old_pin['source_lock_sha256'],
    'local-inference.serving.recipe.sha256': old_pin['recipe_sha256'],
}.items():
    if labels.get(label) != value:
        raise RuntimeError('Diagnostic image label mismatch: ' + label)
old_manifest = json.loads((root / 'runtime-files.json').read_text())
for name, expected in old_manifest.items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError('Prior kit drift: ' + name)
pin = dict(old_pin, image_id=image, diagnostic={
    'kind': 'stale-state-probe', 'lock_sha256': digest,
    'b12x_tree': lock['b12x_tree']})
pin.pop('diagnostic_overlay', None)
payload = json.dumps(pin, indent=2, sort_keys=True) + '\n'
manifest = dict(old_manifest, **{'candidate.json': hashlib.sha256(payload.encode()).hexdigest()})
with (root / f'stale-probe-{image[:12]}-candidate-amendment.json').open('x') as stream:
    stream.write(json.dumps({'prior_candidate': old_pin, 'prior_manifest': old_manifest,
        'candidate': pin, 'manifest': manifest, 'status': 'diagnostic only; not qualified'},
        indent=2, sort_keys=True) + '\n')
(root / 'candidate.json').write_text(payload)
(root / 'runtime-files.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
print('STALE-PROBE-CANDIDATE-FROZEN', image)
