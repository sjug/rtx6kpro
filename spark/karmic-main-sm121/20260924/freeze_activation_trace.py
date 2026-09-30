"""Freeze the GPU-gated diagnostic without changing model launch settings."""
import hashlib
import argparse
import json
from pathlib import Path

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
modes = parser.add_mutually_exclusive_group()
modes.add_argument('--moe-seams', action='store_true')
modes.add_argument('--moe-capture', action='store_true')
args = parser.parse_args()
prefix = 'moe-seams' if args.moe_seams else 'activation-trace'
stem = 'claude-moe-seams' if args.moe_seams else 'claude-act-trace'
kind = 'activation-digest-moe-seams' if args.moe_seams else 'activation-digest'
if args.moe_capture:
    prefix, stem, kind = 'moe-capture', 'claude-moe-capture', 'activation-digest-moe-capture'
build = json.loads((root / ('receipts/' + prefix + '-build-receipt.json')).read_text())
receipt = root / 'receipts' / Path(build['directory']).name
image = (receipt / 'BUILD-OK').read_text().strip()
info = json.loads((receipt / 'image-inspect.json').read_text())[0]
raw = (root / (stem + '.lock.json')).read_bytes()
lock = json.loads(raw)
digest = hashlib.sha256(raw).hexdigest()
if image != build['image_id'] or digest != build['lock_sha256'] or info['Id'].removeprefix('sha256:') != image:
    raise RuntimeError('Build receipt identity mismatch')
for name, expected in lock['inputs'].items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError('Built input drift: ' + name)
if 'ACTIVATION-TRACE-GPU-PASS' not in (receipt / 'gpu.log').read_text().splitlines():
    raise RuntimeError('GPU transport gate missing')
if args.moe_capture and 'CLAUDE-MOE-CAPTURE-CUDA-GATE-PASS' not in (receipt / 'capture-gpu.log').read_text().splitlines():
    raise RuntimeError('Capture GPU gate missing')
old = json.loads((root / 'candidate.json').read_text())
if old['image_id'] != lock['base_image_id']:
    raise RuntimeError('Candidate does not match the declared diagnostic parent')
labels = info['Config']['Labels']
for key, expected in {
    'local-inference.ds41.diagnostic.kind': kind,
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
    'kind': kind, 'lock_sha256': digest,
    'b12x_tree': old['diagnostic']['b12x_tree']})
payload = json.dumps(pin, indent=2, sort_keys=True) + '\n'
updated = dict(manifest, **{'candidate.json': hashlib.sha256(payload.encode()).hexdigest()})
with (root / f'{prefix}-{image[:12]}-candidate-amendment.json').open('x') as stream:
    stream.write(json.dumps({'prior_candidate': old, 'prior_manifest': manifest,
        'candidate': pin, 'manifest': updated, 'status': 'diagnostic only'},
        indent=2, sort_keys=True) + '\n')
(root / 'candidate.json').write_text(payload)
(root / 'runtime-files.json').write_text(json.dumps(updated, indent=2, sort_keys=True) + '\n')
print('ACTIVATION-TRACE-CANDIDATE-FROZEN', image)
