"""Run the small four-rank discriminator only against an idle diagnostic pair-set."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import shlex
import subprocess
import time
import datetime

from cache_metrics import idle_snapshot
from launch_contract import render

ROOT = Path(__file__).resolve().parent
NODES = ('dusty', 'toby', 'rusty', 'kirby')
NAME = 'ds41-flash-karmic-main-tp4'
REMOTE = '/home/jugs/git/ds41-r38/karmic-main-20260924'
p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
p.add_argument('--proto', choices=('LL,Simple', 'Simple'))
p.add_argument('--idle-images', action='store_true', help='Use retained standalone test containers after serving has stopped')
a = p.parse_args()
a.out.mkdir(exist_ok=False, parents=True)
if not a.idle_images:
    idle_snapshot('http://dusty:8000')
image = json.loads((ROOT / 'candidate.json').read_text())['image_id']
test_name = 'ds41-nccl-probe-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S').lower()

def preflight(node):
    if a.idle_images:
        busy = subprocess.run(['ssh', '-o', 'BatchMode=yes', node, 'podman ps -q'],
                              text=True, capture_output=True, check=True, timeout=30)
        if busy.stdout.strip():
            raise RuntimeError(f'{node}: serving must be stopped for standalone probe')
        subprocess.run(['ssh', '-o', 'BatchMode=yes', node, f'podman image exists {image}'], check=True, timeout=30)
        subprocess.run(['scp', str(ROOT / 'probe_nccl_repeatability.py'), f'{node}:{REMOTE}/'], check=True, timeout=30)
        return
    command = f'podman inspect {NAME}; awk \'/^MemAvailable:/ {{print $2}}\' /proc/meminfo'
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', node, command],
                            capture_output=True, text=True, check=True, timeout=30)
    raw, free = result.stdout.rsplit('\n', 2)[:2]
    info = json.loads(raw)[0]
    if info['Image'].removeprefix('sha256:') != image or not info['State']['Running']:
        raise RuntimeError(f'{node}: unexpected serving identity')
    if int(free) < 6 * 1024 * 1024:
        raise RuntimeError(f'{node}: insufficient headroom for separate communicator')
    (a.out / f'{node}-inspect.json').write_text(json.dumps(info, indent=2) + '\n')
    subprocess.run(['scp', str(ROOT / 'probe_nccl_repeatability.py'), f'{node}:{REMOTE}/'],
                   check=True, timeout=30)

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(preflight, NODES))
if not a.idle_images:
    idle_snapshot('http://dusty:8000')
processes = []
try:
    for rank, node in enumerate(NODES):
        stream = (a.out / f'{node}.log').open('w')
        argv = ['podman', 'exec']
        if a.idle_images:
            argv = ['podman', 'run', '--pull=never', '--name', test_name,
                    '--network=host', '--device=nvidia.com/gpu=all', '--ipc=private',
                    '--device=/dev/infiniband',
                    '--shm-size=64g', '--pids-limit=-1',
                    '-v', f'{REMOTE}:/opt/ds41-adapter:ro', '--entrypoint=/bin/bash']
            env = render(node)['env']
            for key, value in env.items():
                argv += ['-e', f'{key}={value}']
        argv += ['-e', f'RANK={rank}', '-e', 'MASTER_ADDR=10.11.11.7',
                '-e', 'MASTER_PORT=29876', '-e', 'NCCL_DEBUG=INFO',
                '-e', 'NCCL_DEBUG_SUBSYS=INIT,NET',
                '-e', 'PYTHONUNBUFFERED=1']
        if a.proto:
            argv += ['-e', f'NCCL_PROTO={a.proto}']
        if a.idle_images:
            argv += [image, '-c', 'export LD_LIBRARY_PATH=/opt/nccl-2.30.7/lib:${LD_LIBRARY_PATH:-}; '
                     'exec /opt/venv/bin/python /opt/ds41-adapter/probe_nccl_repeatability.py']
        else:
            argv += [NAME, '/opt/venv/bin/python', '/opt/ds41-adapter/probe_nccl_repeatability.py']
        (a.out / f'{node}-command.json').write_text(json.dumps(argv, indent=2) + '\n')
        process = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', node, shlex.join(argv)],
                                   stdout=stream, stderr=subprocess.STDOUT)
        processes.append((node, process, stream))
    deadline = time.monotonic() + 300
    while any(proc.poll() is None for _, proc, _ in processes):
        if time.monotonic() > deadline:
            raise RuntimeError('NCCL probe exceeded five-minute window')
        time.sleep(1)
    codes = {node: proc.returncode for node, proc, _ in processes}
    (a.out / 'exit-codes.json').write_text(json.dumps(codes, indent=2) + '\n')
    if any(codes.values()):
        raise RuntimeError(f'NCCL probe failed: {codes}')
    for node in NODES:
        log = (a.out / f'{node}.log').read_text()
        if 'NET/IB' not in log or 'NCCL-REPEATABILITY-PASS' not in log:
            raise RuntimeError(f'{node}: missing transport or completion evidence')
    print('FOUR-RANK-NCCL-REPEATABILITY-PASS', flush=True)
finally:
    for node, proc, stream in processes:
        if proc.poll() is None:
            # Exact diagnostic argv only, never a server/worker process.
            cleanup = ['podman', 'exec', NAME, 'pkill', '-TERM', '-f',
                       '^/opt/venv/bin/python /opt/ds41-adapter/probe_nccl_repeatability.py$']
            if a.idle_images:
                cleanup = ['podman', 'stop', '-t', '30', test_name]
            subprocess.run(['ssh', '-o', 'BatchMode=yes', node, shlex.join(cleanup)], timeout=30)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=10)
        stream.close()
