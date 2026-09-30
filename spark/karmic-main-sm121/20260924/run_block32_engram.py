"""Isolated Engram projection repeatability on an idle authorized node."""
import argparse
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument('--trace', action='store_true')
p.add_argument('--poison', action='store_true')
p.add_argument('--checkpoint', action='store_true')
p.add_argument('--racecheck', action='store_true')
p.add_argument('--racecheck-all', action='store_true')
a = p.parse_args()
image = json.loads((root / 'candidate.json').read_text())['image_id']
remote = '/home/jugs/git/ds41-r38/karmic-main-20260924'
cache = '/home/jugs/.cache/vllm-jj-ds41-tp4'
node = 'toby'
for host in ('toby', 'rusty', 'kirby', 'dusty'):
    running = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', host, 'podman ps -q'], text=True)
    if running.strip():
        raise RuntimeError('GPU is not idle: ' + host)
subprocess.run(['scp', str(root / 'probe_block32_tactic.py'), f'{node}:{remote}/'], check=True)
variant = 'trace' if a.trace else 'synthetic'
if a.poison:
    variant += '-poison'
if a.checkpoint:
    variant += '-checkpoint'
if a.racecheck:
    variant += '-racecheck'
if a.racecheck_all:
    variant += '-racecheck-all'
log = root / 'receipts' / f'engram256-{variant}.log'
if log.exists():
    raise RuntimeError('Receipt exists: ' + str(log))
cmd = [
    'podman', 'run', '--rm', '--pull=never', '--network=none',
    '--device', 'nvidia.com/gpu=all', '--name', 'ds41-engram-isolated',
    '-v', f'{remote}:/diag:ro', '-v', f'{cache}:/cache',
    '-v', '/home/jugs/.cache/huggingface:/root/.cache/huggingface:ro',
    '-e', 'PYTHONUNBUFFERED=1', '-e', 'B12X_PRINT_COMPILE_PROGRESS=1',
    '-e', 'B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1', '-e', 'B12X_DENSE_SPLITK_TURBO=1',
    '-e', 'B12X_COMPILE_CACHE_DIR=/cache/jit/ds41-trace-9e561cc431981197a101/b12x',
    '-e', 'B12X_CUTE_COMPILE_CACHE_DIR=/cache/jit/ds41-trace-9e561cc431981197a101/b12x/cute',
    '--entrypoint', '/opt/venv/bin/python', image, '-u', '/diag/probe_block32_tactic.py', '--engram256',
]
if a.trace:
    cmd += ['--trace', '/cache/ds41-attention-trace/engram-ops-192-0-toby.pt']
if a.poison:
    cmd += ['--poison']
if a.checkpoint:
    cmd += ['--checkpoint', '/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/fb2764a5cf321eaa5070ca8f9e892818f477c16d']
if a.racecheck or a.racecheck_all:
    entry = cmd.index('--entrypoint')
    cmd[entry + 1] = '/usr/local/cuda/bin/compute-sanitizer'
    pos = cmd.index(image) + 1
    sanitizer = ['--tool', 'racecheck', '--racecheck-report', 'all',
                    '--error-exitcode', '86', '--target-processes', 'application-only',
                    '--check-tensor-ops', 'yes']
    if not a.racecheck_all:
        sanitizer += ['--kernel-name', 'regex=.*DenseGemm.*', '--launch-count', '4']
    cmd[pos:pos] = sanitizer + ['/opt/venv/bin/python']
    cmd += ['--repeats', '3', '--selected-only']
import shlex
print('RUN', node, shlex.join(cmd), flush=True)
with log.open('w') as output:
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', node, shlex.join(cmd)], stdout=output, stderr=subprocess.STDOUT)
print('FINISHED', result.returncode, log, flush=True)
raise SystemExit(result.returncode)
