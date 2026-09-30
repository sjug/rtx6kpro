"""CPU-only error-pattern fit against retained tensors, no inference or GPU."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--node', choices=('dusty', 'toby', 'rusty', 'kirby'), required=True)
p.add_argument('--length', type=int, choices=(129, 160, 192), default=129)
p.add_argument('--repeat', type=int, default=0)
p.add_argument('--checkpoint', action='store_true')
p.add_argument('--label', default='')
p.add_argument('--capture', help='Outputs-only first-mismatch basename in the retained trace directory')
a = p.parse_args()
root = Path(__file__).resolve().parent
remote = '/home/jugs/git/ds41-r38/karmic-main-20260924'
cache = '/home/jugs/.cache/vllm-jj-ds41-tp4'
image = json.loads((root / 'candidate.json').read_text())['image_id']
name = f'replay-engram-{a.length}-{a.repeat}-{a.node}'
if a.capture:
    if Path(a.capture).name != a.capture or not a.capture.endswith('.first-mismatch.pt'):
        raise RuntimeError('Require a retained first-mismatch basename')
    if a.checkpoint:
        raise RuntimeError('Outputs-only mode does not use checkpoint weights')
    name = a.capture
if a.label and not a.label.replace('-', '').isalnum():
    raise RuntimeError('Invalid receipt label')
log = root / 'receipts' / f'{name}-cpu-fit{"-checkpoint" if a.checkpoint else ""}{"-" + a.label if a.label else ""}.log'
if log.exists() or a.repeat < 0 or a.repeat > 3:
    raise RuntimeError('Require a fresh receipt and valid repeat')
subprocess.run(['scp', str(root / 'claude_fit_error_structure.py'), f'{a.node}:{remote}/'], check=True)
if a.checkpoint:
    subprocess.run(['scp', str(root / 'fit_engram_with_checkpoint.py'), f'{a.node}:{remote}/'], check=True)
cmd = ['podman', 'run', '--rm', '--pull=never', '--network=none',
       '-v', f'{remote}:/diag:ro', '-v', f'{cache}/ds41-attention-trace:/traces:ro',
       '-v', '/home/jugs/.cache/huggingface:/root/.cache/huggingface:ro',
       '--entrypoint', '/opt/venv/bin/python', image, '-u']
if a.capture:
    cmd += ['/diag/claude_fit_error_structure.py', 'replace', '--capture', f'/traces/{a.capture}']
elif a.checkpoint:
    cmd += ['/diag/fit_engram_with_checkpoint.py', '--checkpoint',
            '/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/fb2764a5cf321eaa5070ca8f9e892818f477c16d/model-00047-of-00048.safetensors']
else:
    cmd += ['/diag/claude_fit_error_structure.py', 'trace']
if not a.capture:
    cmd += ['--trace', f'/traces/{name}.pt', '--observed-key', 'dense_replay_synchronized',
            '--reference-key', 'projected_kv']
with log.open('w') as f:
    process = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', a.node, shlex.join(cmd)],
                               text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in process.stdout:
        f.write(line)
        f.flush()
        print(line, end='', flush=True)
    rc = process.wait()
print('CPU-FIT-FINISHED', rc, log, flush=True)
raise SystemExit(rc)
