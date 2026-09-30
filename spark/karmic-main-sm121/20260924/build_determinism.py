"""Build reviewed Python-only fix; retain candidate, publish only after GPU gates."""
import hashlib
import json
from pathlib import Path
import socket
import subprocess

ROOT = Path(__file__).resolve().parent
if socket.gethostname().split('.')[0] != 'dusty':
    raise RuntimeError('Build requires dusty')


def idle():
    for node in ('dusty', 'toby', 'rusty', 'kirby'):
        command = ['podman', 'ps', '-q'] if node == 'dusty' else [
            'ssh', '-o', 'BatchMode=yes', node, 'podman ps -q']
        result = subprocess.run(command, text=True, capture_output=True, timeout=30)
        if result.returncode or result.stdout.strip():
            raise RuntimeError(f'{node}: idle check failed: {result.stdout} {result.stderr}')


idle()
raw = (ROOT / 'determinism.lock.json').read_bytes()
lock = json.loads(raw)
digest = hashlib.sha256(raw).hexdigest()
base = json.loads(subprocess.check_output(['podman', 'inspect', lock['base_image_id']], text=True))[0]
if base['Id'].removeprefix('sha256:') != lock['base_image_id'] or base['Config']['Labels']['b12x.source-tree'] != lock['base_b12x_tree']:
    raise RuntimeError('Base image identity or tree differs from lock')
out = ROOT / 'receipts/determinism-build'
out.mkdir(parents=True, exist_ok=False)
files = ['Dockerfile.determinism', 'determinism.lock.json', 'determinism-tiled-topk.py',
         'claude-dsa-topk-tiebreak.patch', 'install_determinism.py', 'runtime.lock.json',
         'build_determinism.py', 'claude_test_topk_tiebreak.py', 'test_moe_dependency.py']
(out / 'inputs.json').write_text(json.dumps({p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                            for p in files}, indent=2) + '\n')
subprocess.run(['podman', 'build', '--pull=never', '--network=none', '--format=docker',
                '--tag', 'localhost/voipmonitor/build-components:ds41-determinism-20260925',
                '--iidfile', str(out / 'image.id'), '-f', str(ROOT / 'Dockerfile.determinism'),
                '--build-arg', 'PATCH_TREE=' + lock['b12x_tree'],
                '--build-arg', 'PATCH_LOCK=' + digest,
                '--build-arg', 'PATCH_CACHE=ds41-determinism-' + digest[:20], str(ROOT)], check=True)
image = (out / 'image.id').read_text().strip()
inspection = subprocess.check_output(['podman', 'inspect', image], text=True)
(out / 'image-inspect.json').write_text(inspection)
labels = json.loads(inspection)[0]['Config']['Labels']
if labels['b12x.source-tree'] != lock['b12x_tree'] or labels['local-inference.ds41.diagnostic.lock.sha256'] != digest:
    raise RuntimeError('Candidate labels differ from composition')
idle()
for script, marker in [('test_moe_dependency.py', 'DETERMINISTIC-MOE-DEPENDENCY-PASS'),
                       ('claude_test_topk_tiebreak.py', 'CLAUDE-TOPK-TIEBREAK-PASS checks=')]:
    log = out / (script + '.log')
    with log.open('w') as output:
        subprocess.run(['podman', 'run', '--rm', '--pull=never', '--device', 'nvidia.com/gpu=all',
                        '-e', 'PYTHONUNBUFFERED=1', '-e', 'B12X_PRINT_COMPILE_PROGRESS=1',
                        '-v', f'{ROOT}:/diag:ro', '--entrypoint', '/opt/venv/bin/python',
                        image, '/diag/' + script], stdout=output, stderr=subprocess.STDOUT, check=True)
    if marker not in log.read_text():
        raise RuntimeError(f'Gate did not execute: {script}')
    print('GATE-PASS', script, flush=True)
subprocess.run(['podman', 'tag', image, 'localhost/voipmonitor/vllm:karmic-main-ds41-deterministic-spark-sm121'], check=True)
(out / 'BUILD-OK').write_text(image + '\n')
print('DETERMINISM-BUILD-OK', image, flush=True)
