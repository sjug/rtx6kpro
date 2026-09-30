"""Install a hash-gated, explicitly diagnostic attention wrapper."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/opt/ds41-attention-trace')
lock = json.loads((ROOT / 'attention-trace.lock.json').read_text())
tree = Path('/opt/jovian-judgement/vllm')
target = tree / lock['source_path']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


for path, expected in ((target, lock['input_sha256']),
                       (ROOT / 'trace-patched-attention.py', lock['output_sha256']),
                       (ROOT / 'trace_attention.py', lock['helper_sha256'])):
    if digest(path) != expected:
        raise RuntimeError(f'Trace input mismatch: {path}')
native = {str(p): digest(p) for p in tree.rglob('*.so')}
shutil.copyfile(ROOT / 'trace-patched-attention.py', target)
shutil.copyfile(ROOT / 'trace_attention.py', target.with_name('trace_attention.py'))
for path in (target, target.with_name('trace_attention.py')):
    compile(path.read_bytes(), str(path), 'exec')
if native != {str(p): digest(p) for p in tree.rglob('*.so')}:
    raise RuntimeError('Native objects changed during trace install')
print('ATTENTION-TRACE-INSTALL-PASS', flush=True)
