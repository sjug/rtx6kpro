"""Freeze the reviewed, diagnostic-only helper mount without changing the image."""
import hashlib
import json
from pathlib import Path
from diagnostic_overlay import IMAGE, SOURCE, TARGET, validate

root = Path(__file__).resolve().parent
pin = json.loads((root / 'candidate.json').read_text())
if pin['image_id'] != IMAGE or 'diagnostic_overlay' in pin:
    raise RuntimeError('Unexpected previous candidate')
prior = json.loads((root / 'receipts/pre-ops-candidate.json').read_text())
if pin != prior:
    raise RuntimeError('Candidate changed since preservation')
pin['diagnostic_overlay'] = {'source': SOURCE, 'target': TARGET,
    'sha256': hashlib.sha256((root / SOURCE).read_bytes()).hexdigest()}
validate(pin, root)
manifest = json.loads((root / 'receipts/pre-ops-runtime-files.json').read_text())
changed = {'run_node.py', 'runtime.py', 'candidate.json'}
for name, expected in manifest.items():
    if name not in changed and hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError('Unexpected runtime drift: ' + name)
(root / 'candidate.json').write_text(json.dumps(pin, indent=2, sort_keys=True) + '\n')
for name in changed | {'diagnostic_overlay.py', SOURCE}:
    manifest[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
(root / 'runtime-files.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
print('TRACE-OPS-PINNED', pin['diagnostic_overlay']['sha256'])
