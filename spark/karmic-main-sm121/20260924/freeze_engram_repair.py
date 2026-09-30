"""Pin the gated repair, preserving the timing arm's prior kit for audit."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
build = json.loads((root / 'receipts/engram-repair-build-receipt.json').read_text())
receipt = root / 'receipts' / Path(build['directory']).name
image = (receipt / 'BUILD-OK').read_text().strip()
info = json.loads((receipt / 'image-inspect.json').read_text())[0]
raw = (root / 'engram-repair.lock.json').read_bytes()
lock = json.loads(raw)
digest = hashlib.sha256(raw).hexdigest()
if image != build['image_id'] or info['Id'].removeprefix('sha256:') != image or digest != build['lock_sha256']:
    raise RuntimeError('Build receipt identity mismatch')
for name, expected in lock['inputs'].items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError('Built input drift: ' + name)
old = json.loads((root / 'candidate.json').read_text())
if old['image_id'] != '32f93547a3c50a8b72ecfcbd8f0f2f0685bff34cfb4213b77748db80b5a6bebb':
    raise RuntimeError('Expected the last timing candidate pin')
labels = info['Config']['Labels']
for key, expected in {
    'local-inference.ds41.diagnostic.kind': 'engram-prequeued-native-io-fix',
    'local-inference.ds41.diagnostic.lock.sha256': digest,
    'b12x.source-tree': lock['trees']['b12x'],
    'vllm.source-tree': lock['trees']['vllm'],
    'local-inference.status': 'candidate-not-qualified',
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
    'kind': 'engram-prequeued-native-io-fix', 'lock_sha256': digest,
    'b12x_tree': lock['trees']['b12x'], 'vllm_tree': lock['trees']['vllm']})
payload = json.dumps(pin, indent=2, sort_keys=True) + '\n'
updated = dict(manifest, **{'candidate.json': hashlib.sha256(payload.encode()).hexdigest()})
with (root / f'engram-repair-{image[:12]}-candidate-amendment.json').open('x') as stream:
    stream.write(json.dumps({'prior_candidate': old, 'prior_manifest': manifest,
        'candidate': pin, 'manifest': updated, 'status': 'build-gated, model qualification pending'},
        indent=2, sort_keys=True) + '\n')
(root / 'candidate.json').write_text(payload)
(root / 'runtime-files.json').write_text(json.dumps(updated, indent=2, sort_keys=True) + '\n')
print('ENGRAM-REPAIR-CANDIDATE-FROZEN', image)
