"""Materialize the reviewed patch for an isolated GPU test, no checkout writes."""
import hashlib
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent
patch = root / 'claude-dsa-topk-tiebreak.patch'
if hashlib.sha256(patch.read_bytes()).hexdigest() != '7d77a7fb6f2abdf6967621c445147fa968782f295fb3c6c4cf605ba536764ec0':
    raise RuntimeError('Claude patch changed; review new identity before testing')
path = 'b12x/attention/dsa_indexer/tiled_topk.py'
source = subprocess.check_output(['git', '-C', str(Path.home() / 'git/b12x'), 'show',
                                  'a7d7d29b2ef8869086e0ceaa787321f17544e3c9:' + path])
with tempfile.TemporaryDirectory(prefix='ds41-topk-') as temporary:
    target = Path(temporary) / path
    target.parent.mkdir(parents=True)
    target.write_bytes(source)
    subprocess.run(['git', 'apply', '--check', '--whitespace=error', str(patch)], cwd=temporary, check=True)
    subprocess.run(['git', 'apply', '--whitespace=error', str(patch)], cwd=temporary, check=True)
    data = target.read_bytes()
if hashlib.sha256(data).hexdigest() != 'f2d0fadae632f4b2af233ca2c53122e7fbcb78a92dc5d16479e541330cf9052d':
    raise RuntimeError('Claude patched source mismatch')
(root / 'claude-topk-candidate.py').write_bytes(data)
print('CLAUDE-CANDIDATE-EXTRACTED')
