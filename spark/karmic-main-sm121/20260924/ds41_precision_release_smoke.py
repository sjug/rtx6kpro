"""GPU smoke for the precision release image, run from /gate by build_precision_release.py.

Gates: the imported worker and helper are the locked files and linked; capture helpers are
absent; the router, engram and parent locks are untouched; the helper sets the flag and
prints the exact marker. The cuBLAS observation afterwards is descriptive only: it records
whether flag-off outputs of one bf16 shape match across two row counts, never a gate.
"""
import contextlib
import hashlib
import importlib
import io
import json
from pathlib import Path

import torch

ROOT = Path('/opt/jovian-judgement')
lock_raw = Path('/opt/ds41-precision-release/ds41-precision-release.lock.json').read_bytes()
lock = json.loads(lock_raw)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
    raise RuntimeError('A visible GB10 is required')
loaded = {}
for path, entry in lock['targets'].items():
    name = path.removesuffix('.py').replace('/', '.')
    module = importlib.import_module(name)
    if (Path(module.__file__).resolve() != (ROOT / 'vllm' / path).resolve()
            or digest(module.__file__) != entry['output_sha256']):
        raise RuntimeError('Wrong imported module: ' + name)
    loaded[name] = module
worker, helper = loaded['vllm.v1.worker.gpu_worker'], loaded[lock['helper_module']]
if worker._ds41_precision is not helper:
    raise RuntimeError('Precision hook not linked')
for path in lock['capture_paths_absent']:
    if (ROOT / 'vllm' / path).exists():
        raise RuntimeError('Capture helper present: ' + path)
if digest('/opt/ds41-router-release/' + lock['base_lock']) != lock['base_lock_sha256']:
    raise RuntimeError('Router lock changed')
if any(Path(p).exists() for p in ('/opt/ds41-decision-row', '/opt/ds41-window', '/opt/ds41-indexer', '/opt/ds41-precision')):
    raise RuntimeError('Diagnostic install directory present')
before = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
out = io.StringIO()
with contextlib.redirect_stdout(out):
    helper.apply(rank='smoke')
line = out.getvalue().strip()
if (before is not True or not line.startswith(lock['apply_marker'] + ' rank=smoke allow_bf16_reduced_precision_reduction=False before=True ')
        or torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is not False):
    raise RuntimeError('Precision flag did not take: ' + line)
print(line, flush=True)

generator = torch.Generator(device='cuda').manual_seed(0)
x = torch.randn((8192, 5120), generator=generator, device='cuda').to(torch.bfloat16)
w = torch.randn((32, 5120), generator=generator, device='cuda').to(torch.bfloat16) / 64
rows = {m: torch.mm(x[-m:], w.T)[-128:] for m in (8192, 4096)}
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
default = torch.mm(x, w.T)[-128:]
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
print('PRECISION-RELEASE-GEMM-OBSERVATION', json.dumps({
    'shape': [8192, 5120, 32], 'flag_off_rows_equal_8192_vs_4096': bool(torch.equal(rows[8192], rows[4096])),
    'flag_off_equals_default_8192': bool(torch.equal(rows[8192], default)),
    'scope': 'descriptive; random inputs, one shape, one process; not a gate'}), flush=True)
print(lock['smoke_pass'], flush=True)
