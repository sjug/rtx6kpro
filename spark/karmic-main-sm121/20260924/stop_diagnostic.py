"""Stop only the pinned four-node diagnostic, workers first, retaining evidence."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
from cache_metrics import idle_snapshot

root = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument('--label', required=True)
p.add_argument('--failed-engine', action='store_true',
               help='Archive a failed diagnostic only after the API container has exited')
a = p.parse_args()
if not a.label.replace('-', '').isalnum():
    raise RuntimeError('Invalid retained-container label')
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
out = root / 'receipts' / f'stop-{a.label}-{stamp}'
out.mkdir(exist_ok=False)
if not a.failed_engine:
    (out / 'idle.json').write_text(json.dumps(idle_snapshot('http://dusty:8000'), indent=2))
image = json.loads((root / 'candidate.json').read_text())['image_id']
name = 'ds41-flash-karmic-main-tp4'
nodes = ('toby', 'rusty', 'kirby', 'dusty')


def remote(node, command, timeout=120):
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', node, command],
                            text=True, capture_output=True, timeout=timeout)
    with (out / 'commands.log').open('a') as f:
        f.write(f'{node}: {command}\n{result.stdout}\n{result.stderr}\nrc={result.returncode}\n')
    if result.returncode:
        raise RuntimeError(f'{node}: stop action failed; inspect {out}')
    return result.stdout


if a.failed_engine:
    head = json.loads(remote('dusty', f'podman inspect {name}'))[0]
    if head['State']['Running'] or head['Image'].removeprefix('sha256:') != image:
        raise RuntimeError('Failed-engine archival requires the pinned API container to have exited')
for node in nodes:
    info = json.loads(remote(node, f'podman inspect {name}'))[0]
    (out / f'{node}-before.json').write_text(json.dumps(info, indent=2))
    if info['Image'].removeprefix('sha256:') != image or (not a.failed_engine and not info['State']['Running']):
        raise RuntimeError(f'{node}: unexpected live identity')
    (out / f'{node}-before.log').write_text(remote(node, f'podman logs {name} 2>&1'))
for node in nodes:
    remote(node, f'podman stop -t 60 {name}')
    info = json.loads(remote(node, f'podman inspect {name}'))[0]
    (out / f'{node}-stopped.json').write_text(json.dumps(info, indent=2))
    if info['State']['Running'] or info['State']['OOMKilled'] or (not a.failed_engine and info['State']['ExitCode'] != 0):
        raise RuntimeError(f'{node}: unclean stop, no further action')
    remote(node, f'podman rename {name} {name}-{a.label}-{stamp.lower()}')
    print('STOPPED-RETAINED', node, flush=True)
print('STOP-COMPLETE', out, flush=True)
