"""Refresh only the reviewed Engram diagnostic helper and candidate hashes."""
import hashlib
import json
from pathlib import Path
from diagnostic_overlay import SOURCE, validate

root = Path(__file__).resolve().parent
pin = json.loads((root / 'candidate.json').read_text())
manifest = json.loads((root / 'runtime-files.json').read_text())
old = (root / 'receipts/pre-engram-trace-helper.py').read_bytes()
if hashlib.sha256(old).hexdigest() != pin['diagnostic_overlay']['sha256']:
    raise RuntimeError('Previous helper receipt mismatch')
for name, expected in manifest.items():
    if name != SOURCE and hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError('Unrelated kit drift: ' + name)
pin['diagnostic_overlay']['sha256'] = hashlib.sha256((root / SOURCE).read_bytes()).hexdigest()
validate(pin, root)
(root / 'candidate.json').write_text(json.dumps(pin, indent=2, sort_keys=True) + '\n')
for name in (SOURCE, 'candidate.json'):
    manifest[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
(root / 'runtime-files.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
print('ENGRAM-TRACE-FROZEN', pin['diagnostic_overlay']['sha256'])
