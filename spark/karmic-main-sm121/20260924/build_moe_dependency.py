"""Build and gate the one-file dependency repair on idle dusty; no publication."""
import hashlib
import json
from pathlib import Path
import socket
import subprocess

ROOT = Path(__file__).resolve().parent
if socket.gethostname().split('.')[0] != 'dusty':
    raise RuntimeError('Build requires dusty')
for node in ('dusty', 'toby', 'rusty', 'kirby'):
    command = ['podman', 'ps', '-q'] if node == 'dusty' else [
        'ssh', '-o', 'BatchMode=yes', node, 'podman ps -q']
    result = subprocess.run(command, text=True, capture_output=True, timeout=30)
    if result.returncode:
        raise RuntimeError(f'{node}: idle probe failed: {result.stderr}')
    if result.stdout.strip():
        raise RuntimeError(f'{node} is not idle; build refused')
lock_bytes = (ROOT / 'moe-dependency.lock.json').read_bytes()
lock = json.loads(lock_bytes)
digest = hashlib.sha256(lock_bytes).hexdigest()
out = ROOT / 'receipts/moe-dependency-build'
out.mkdir(parents=True, exist_ok=False)
files = ['Dockerfile.moe-dependency', 'moe-dependency.lock.json', 'moe-dependency.patch',
         'moe-dependency-preparation.py', 'install_moe_dependency.py', 'runtime.lock.json',
         'build_moe_dependency.py', 'test_moe_dependency.py']
(out / 'inputs.json').write_text(json.dumps({p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                            for p in files}, indent=2) + '\n')
subprocess.run(['podman', 'build', '--pull=never', '--network=none', '--format=docker',
                '--tag', 'localhost/voipmonitor/build-components:ds41-moe-dependency-20260925',
                '--iidfile', str(out / 'image.id'), '-f', str(ROOT / 'Dockerfile.moe-dependency'),
                '--build-arg', 'PATCH_TREE=' + lock['b12x_tree'],
                '--build-arg', 'PATCH_LOCK=' + digest,
                '--build-arg', 'PATCH_CACHE=ds41-moe-dependency-' + digest[:20], str(ROOT)], check=True)
image = (out / 'image.id').read_text().strip()
inspection = subprocess.check_output(['podman', 'inspect', image], text=True)
(out / 'image-inspect.json').write_text(inspection)
with (out / 'dependency-gate.log').open('w') as output:
    subprocess.run(['podman', 'run', '--rm', '--pull=never', '--device', 'nvidia.com/gpu=all',
                    '-e', 'PYTHONUNBUFFERED=1', '-e', 'B12X_PRINT_COMPILE_PROGRESS=1',
                    '-v', f'{ROOT}:/diag:ro', '--entrypoint', '/opt/venv/bin/python',
                    image, '/diag/test_moe_dependency.py'], stdout=output, stderr=subprocess.STDOUT, check=True)
if 'DETERMINISTIC-MOE-DEPENDENCY-PASS' not in (out / 'dependency-gate.log').read_text():
    raise RuntimeError('Dependency gate did not execute')
(out / 'BUILD-OK').write_text(image + '\n')
print('MOE-DEPENDENCY-BUILD-OK', image, flush=True)
