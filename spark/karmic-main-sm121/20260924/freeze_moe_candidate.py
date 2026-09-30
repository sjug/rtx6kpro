"""Pin the gated dependency-only image, retaining the preceding runtime pins."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
receipt = root / 'receipts/moe-dependency-build'
image = (receipt / 'BUILD-OK').read_text().strip().removeprefix('sha256:')
info = json.loads((receipt / 'image-inspect.json').read_text())[0]
lock_bytes = (root / 'moe-dependency.lock.json').read_bytes()
lock = json.loads(lock_bytes)
labels = info.get('Labels') or info['Config']['Labels']
if info['Id'].removeprefix('sha256:') != image:
    raise RuntimeError('Build receipt image mismatch')
if 'DETERMINISTIC-MOE-DEPENDENCY-PASS' not in (receipt / 'dependency-gate.log').read_text():
    raise RuntimeError('Dependency gate missing')
expected = hashlib.sha256(lock_bytes).hexdigest()
if labels['local-inference.ds41.diagnostic.lock.sha256'] != expected:
    raise RuntimeError('Image diagnostic lock mismatch')
old_pin = json.loads((root / 'candidate.json').read_text())
old_manifest = json.loads((root / 'runtime-files.json').read_text())
if old_pin['image_id'] != lock['base_image_id']:
    raise RuntimeError('Unexpected prior candidate')
pin = dict(old_pin, image_id=image, diagnostic={
    'kind': 'moe-dependency', 'lock_sha256': expected, 'b12x_tree': lock['b12x_tree']})
(root / 'candidate.json').write_text(json.dumps(pin, indent=2, sort_keys=True) + '\n')
manifest = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in old_manifest}
if {name for name in manifest if manifest[name] != old_manifest[name]} != {'candidate.json', 'run_node.py'}:
    raise RuntimeError('Unexpected runtime update')
with (root / 'moe-candidate-amendment.json').open('x') as output:
    output.write(json.dumps({'prior_candidate': old_pin, 'prior_manifest': old_manifest,
                            'candidate': pin, 'manifest': manifest}, indent=2, sort_keys=True) + '\n')
(root / 'runtime-files.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
print('MOE-CANDIDATE-FROZEN', image)
