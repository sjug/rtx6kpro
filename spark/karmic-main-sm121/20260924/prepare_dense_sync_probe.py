"""Generate a one-variable, isolated-only warp-release diagnostic from the pin."""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
revision = 'a7d7d29b2ef8869086e0ceaa787321f17544e3c9'
path = 'b12x/_lib/dense_gemm.py'
source = subprocess.check_output(['git', '-C', '/home/jugs/git/b12x', 'show', f'{revision}:{path}'], text=True)
needle = '                        mainloop_pipeline.consumer_release(mainloop_consumer_state)'
if source.count(needle) != 2:
    raise RuntimeError('Unexpected release-site count')
# Preserve indentation at each site: the in-loop site has four extra spaces.
lines = source.splitlines(keepends=True)
result = []
for line in lines:
    if line.strip() == 'mainloop_pipeline.consumer_release(mainloop_consumer_state)':
        indent = line[:len(line) - len(line.lstrip())]
        result.append(indent + 'cute.arch.sync_warp()\n')
    result.append(line)
updated = ''.join(result)
patch = ''.join(difflib.unified_diff(source.splitlines(keepends=True), updated.splitlines(keepends=True),
                                  fromfile='a/' + path, tofile='b/' + path))
target = root / 'dense_gemm-warp-sync.py'
if target.exists():
    raise RuntimeError('Diagnostic already generated')
target.write_text(updated)
(root / 'dense-warp-sync.patch').write_text(patch)
identity = {'base_revision': revision, 'base_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'source_sha256': hashlib.sha256(updated.encode()).hexdigest(),
            'patch_sha256': hashlib.sha256(patch.encode()).hexdigest(), 'release_sites': 2,
            'scope': 'isolated diagnostic only; not a qualified fix'}
(root / 'dense-warp-sync.json').write_text(json.dumps(identity, indent=2) + '\n')
print(json.dumps(identity), flush=True)
