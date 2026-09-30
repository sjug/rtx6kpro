"""Record the narrow approved MoE diagnostic change without losing baseline pins."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
old_bytes = (root / 'runtime-files.json').read_bytes()
old = json.loads(old_bytes)
new = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in old}
changed = [name for name in old if old[name] != new[name]]
if changed != ['launch_contract.py']:
    raise RuntimeError(f'Unexpected diagnostic runtime changes: {changed}')
record = {'authorization': 'User requested completing determinism investigation, fix, image and qualification on four DS41 nodes',
          'scope': 'Optional B12X_DYNAMIC_DETERMINISTIC_OUTPUT=0|1; all model arguments unchanged',
          'prior_manifest_sha256': hashlib.sha256(old_bytes).hexdigest(),
          'prior_manifest': old, 'updated_manifest': new}
with (root / 'determinism-profile-amendment.json').open('x') as out:
    out.write(json.dumps(record, indent=2, sort_keys=True) + '\n')
(root / 'runtime-files.json').write_text(json.dumps(new, indent=2, sort_keys=True) + '\n')
print('DETERMINISM-PROFILE-FROZEN')
