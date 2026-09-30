"""Run the exact-identity isolated reproducer after all four ranks are idle."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--node', choices=('dusty', 'toby', 'rusty', 'kirby'), default='dusty')
p.add_argument('--repeats', type=int, default=64)
p.add_argument('--length', type=int, choices=(129, 160, 192), default=129)
p.add_argument('--reference-key', choices=('projected_kv', 'dense_replay_synchronized'), default='projected_kv')
p.add_argument('--delay-ms', type=float, default=0)
p.add_argument('--label', default='baseline')
p.add_argument('--warp-sync', action='store_true')
p.add_argument('--fresh-compile', action='store_true')
p.add_argument('--sanitizer', choices=('racecheck', 'memcheck', 'synccheck', 'initcheck'))
p.add_argument('--column-slice')
p.add_argument('--poison-output', action='store_true')
p.add_argument('--source-variant', choices=('warp-sync', 'no-peek', 'unroll2', 'late-release', 'reinit', 'no-reg-realloc', 'prologue-barrier', 'reinit-runtime', 'init-runtime-before', 'init-negative-zero', 'first-mma-overwrite', 'grid-all-tiles', 'grid300', 'all-release-after-mma', 'main-release-after-mma', 'epilogue-barrier', 'proxy-before-release', 'group-before-release', 'group-main-release', 'group-tail-release', 'fence-before-release', 'fence-after-release', 'all-lanes-release'))
a = p.parse_args()
if a.warp_sync and a.source_variant:
    raise RuntimeError('Select one independent source variant')
variant = 'warp-sync' if a.warp_sync else a.source_variant
root = Path(__file__).resolve().parent
remote = '/home/jugs/git/ds41-r38/karmic-main-20260924'
cache = '/home/jugs/.cache/vllm-jj-ds41-tp4'
image = json.loads((root / 'candidate.json').read_text())['image_id']
data = json.loads((root / f'receipts/replay-engram-trace-requests/{a.node}-{a.length}-engram-analysis.json').read_text())
projection = next(o for o in data[0]['engram_baseline'] if o['name'] == 'engram_projection')
references = set()
for node in ('dusty', 'toby', 'rusty', 'kirby'):
    other = json.loads((root / f'receipts/replay-engram-trace-requests/{node}-{a.length}-engram-analysis.json').read_text())
    op = next(o for o in other[0]['engram_baseline'] if o['name'] == 'engram_projection')
    references.add(op['outputs'][a.reference_key]['sha256'])
if len(references) != 1:
    raise RuntimeError(f'No unanimous cross-rank reference for request {a.length}-0')
reference_sha = references.pop()
key = projection['dense_programs'][0]['key']
manifest_path = f'jit/ds41-trace-9e561cc431981197a101/b12x/{key[:2]}/{key}.json'
if not a.label.replace('-', '').isalnum() or not 0 <= a.delay_ms <= 1000:
    raise RuntimeError('Invalid diagnostic label or delay')
stem = f'engram-dense-isolated-{a.node}-{a.label}'
log = root / 'receipts' / (stem + '.log')
if log.exists():
    raise RuntimeError('Receipt already exists')
for node in ('dusty', 'toby', 'rusty', 'kirby'):
    running = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', node, 'podman ps -q'], text=True)
    if running.strip():
        raise RuntimeError('GPU not idle: ' + node)
raw = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', a.node,
                               f'cat {cache}/{manifest_path}'], text=True)
manifest = json.loads(raw)
if manifest['cache_key'] != key:
    raise RuntimeError('Manifest identity mismatch')
(root / 'receipts' / (stem + '-manifest.json')).write_text(raw)
subprocess.run(['scp', str(root / 'probe_engram_dense_replay.py'), f'{a.node}:{remote}/'], check=True)
if variant:
    subprocess.run(['scp', str(root / f'dense_gemm-{variant}.py'), f'{a.node}:{remote}/'], check=True)
report = f'/cache/ds41-attention-trace/{stem}.json'
cmd = ['podman', 'run', '--rm', '--pull=never', '--network=none',
       '--device', 'nvidia.com/gpu=all', '--name', 'ds41-dense-isolated',
       '-v', f'{remote}:/diag:ro', '-v', f'{cache}:/cache',
       '-v', '/home/jugs/.cache/huggingface:/root/.cache/huggingface:ro',
       '-e', 'PYTHONUNBUFFERED=1',
       '--entrypoint', '/opt/venv/bin/python', image, '-u', '/diag/probe_engram_dense_replay.py',
       '--trace', f'/cache/ds41-attention-trace/replay-engram-{a.length}-0-{a.node}.pt',
       '--manifest', f'/cache/{manifest_path}', '--report', report,
       '--reference-sha256', reference_sha, '--reference-key', a.reference_key,
       '--checkpoint', '/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/fb2764a5cf321eaa5070ca8f9e892818f477c16d/model-00047-of-00048.safetensors',
       '--repeats', str(a.repeats), '--delay-ms', str(a.delay_ms)]
if variant:
    identity = json.loads((root / f'dense-{variant}.json').read_text())
    entry = cmd.index('--entrypoint')
    cmd[entry:entry] = ['-v', f'{remote}/dense_gemm-{variant}.py:/opt/jovian-judgement/b12x/b12x/_lib/dense_gemm.py:ro']
    cmd += ['--source-sha256', identity['source_sha256']]
if a.fresh_compile:
    cmd += ['--fresh-cache-dir', f'/cache/ds41-attention-trace/{stem}-compile']
if a.column_slice:
    cmd += ['--column-slice', a.column_slice]
if a.poison_output:
    cmd += ['--poison-output']
if a.sanitizer:
    cmd[cmd.index('--entrypoint') + 1] = '/usr/local/cuda/bin/compute-sanitizer'
    position = cmd.index(image) + 1
    options = ['--tool', a.sanitizer, '--error-exitcode', '86',
               '--target-processes', 'application-only']
    if a.sanitizer == 'racecheck':
        options += ['--racecheck-report', 'all', '--check-tensor-ops', 'yes']
    cmd[position:position] = options + ['/opt/venv/bin/python']
(root / 'receipts' / (stem + '-command.json')).write_text(json.dumps(cmd, indent=2) + '\n')
with log.open('w') as f:
    process = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', a.node, shlex.join(cmd)],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        f.write(line)
        f.flush()
        print(line, end='', flush=True)
    rc = process.wait()
raw = subprocess.run(['ssh', '-o', 'BatchMode=yes', a.node,
                      f'cat {cache}/ds41-attention-trace/{stem}.json'], capture_output=True, text=True)
if raw.returncode == 0:
    json.loads(raw.stdout)
    (root / 'receipts' / (stem + '.json')).write_text(raw.stdout)
print('ISOLATED-REPLAY-FINISHED', rc, log, flush=True)
raise SystemExit(rc)
