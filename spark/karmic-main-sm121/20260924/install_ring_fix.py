"""Apply the pinned Python-only fix and prove no other shipped files changed."""
import hashlib
import json
from pathlib import Path
import shutil

if not __debug__:
    raise RuntimeError('Optimized Python is not admitted')
here = Path('/opt/ds41-ring-fix')
lock = json.loads((here / 'ring-fix.lock.json').read_text())
tree = Path('/opt/jovian-judgement/vllm')
target = tree / lock['source_path']
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
for name, expected in lock['inputs'].items():
    if sha(here / name) != expected:
        raise RuntimeError('Recipe input drift: ' + name)
if sha(target) != lock['input_sha256']:
    raise RuntimeError('Base sparse MLA source mismatch')
def inventory():
    return {str(p): sha(p) for base in (tree, Path('/opt/jovian-judgement/b12x'))
            for p in base.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
before = inventory()
if not any(p.endswith('.so') for p in before):
    raise RuntimeError('Missing native artifacts')
for helper in ('claude_stale_state_probe.py', 'trace_attention.py', 'trace_attention_ops.py'):
    if list(tree.rglob(helper)):
        raise RuntimeError('Diagnostic helper must not ship: ' + helper)
shutil.copyfile(here / 'ring-fix-sparse_mla.py', target)
compile(target.read_bytes(), str(target), 'exec')
after = inventory()
expected = dict(before, **{str(target): lock['output_sha256']})
if after != expected:
    raise RuntimeError('Change was not confined to the pinned sparse MLA file')
(here / 'before-files.json').write_text(json.dumps(before, indent=2, sort_keys=True) + '\n')
(here / 'after-files.json').write_text(json.dumps(after, indent=2, sort_keys=True) + '\n')
print('RING-FIX-INSTALL-PASS', flush=True)
