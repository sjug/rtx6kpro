"""Pin the reviewed in-process replay helper without unrelated runtime changes."""
import hashlib
import json
from pathlib import Path
from diagnostic_overlay import SOURCE, validate

root = Path(__file__).resolve().parent
pin = json.loads((root / 'candidate.json').read_text())
manifest = json.loads((root / 'runtime-files.json').read_text())
prior = root / 'receipts/pre-replay-trace-helper.py'
if hashlib.sha256(prior.read_bytes()).hexdigest() != pin['diagnostic_overlay']['sha256']:
    raise RuntimeError('Previous helper identity mismatch')
for name, expected in manifest.items():
    if name != SOURCE and hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError('Unrelated kit drift: ' + name)
pin['diagnostic_overlay']['sha256'] = hashlib.sha256((root / SOURCE).read_bytes()).hexdigest()
validate(pin, root)
(root / 'candidate.json').write_text(json.dumps(pin, indent=2, sort_keys=True) + '\n')
for name in (SOURCE, 'candidate.json'):
    manifest[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
(root / 'runtime-files.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
print('REPLAY-TRACE-FROZEN', pin['diagnostic_overlay']['sha256'])
