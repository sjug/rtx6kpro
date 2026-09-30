"""Pin the build-gated uninstrumented image, preserving the prior runtime kit."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
build = json.loads((root / 'receipts/dense-release-build-receipt.json').read_text())
receipt = root / 'receipts' / Path(build['directory']).name
image = (receipt / 'BUILD-OK').read_text().strip().removeprefix('sha256:')
info = json.loads((receipt / 'image-inspect.json').read_text())[0]
lock_bytes = (root / 'dense-release.lock.json').read_bytes()
lock = json.loads(lock_bytes)
expected = hashlib.sha256(lock_bytes).hexdigest()
labels = info.get('Labels') or info['Config']['Labels']
if image != build['image_id'] or info['Id'].removeprefix('sha256:') != image:
    raise RuntimeError('Build receipt image mismatch')
for label, value in {
    'local-inference.ds41.diagnostic.lock.sha256': expected,
    'local-inference.ds41.diagnostic.kind': 'dense-release-fence',
    'b12x.source-tree': lock['b12x_tree'],
}.items():
    if labels.get(label) != value:
        raise RuntimeError('Built identity mismatch: ' + label)
old_pin = json.loads((root / 'candidate.json').read_text())
for label, value in {
    'local-inference.ds41-build-lock.sha256': old_pin['build_lock_sha256'],
    'local-inference.source-lock.sha256': old_pin['source_lock_sha256'],
    'local-inference.serving.recipe.sha256': old_pin['recipe_sha256'],
    'local-inference.status': 'candidate-not-qualified',
}.items():
    if labels.get(label) != value:
        raise RuntimeError('Inherited build identity mismatch: ' + label)
old_manifest = json.loads((root / 'runtime-files.json').read_text())
for name, digest in old_manifest.items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
        raise RuntimeError('Prior runtime file drift: ' + name)
pin = dict(old_pin, image_id=image, diagnostic={
    'kind': 'dense-release-fence', 'lock_sha256': expected,
    'b12x_tree': lock['b12x_tree']})
pin.pop('diagnostic_overlay', None)
payload = json.dumps(pin, indent=2, sort_keys=True) + '\n'
manifest = dict(old_manifest, **{'candidate.json': hashlib.sha256(payload.encode()).hexdigest()})
with (root / 'dense-release-candidate-amendment.json').open('x') as stream:
    stream.write(json.dumps({'prior_candidate': old_pin, 'prior_manifest': old_manifest,
        'candidate': pin, 'manifest': manifest, 'status': 'build-gated; serving qualification pending'},
        indent=2, sort_keys=True) + '\n')
(root / 'candidate.json').write_text(payload)
(root / 'runtime-files.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
print('DENSE-RELEASE-CANDIDATE-FROZEN', image)
