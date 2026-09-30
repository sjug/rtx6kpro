#!/usr/bin/env python3
"""Observe all four ranks until API startup, without changing their state."""
import concurrent.futures
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
nodes = ('dusty', 'toby', 'rusty', 'kirby')
name = 'ds41-flash-karmic-main-tp4'
deadline = time.monotonic() + 3600


def poll(node):
    command = f'podman inspect {name} --format "{{{{json .State}}}}" && podman logs --tail 35 {name} 2>&1'
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', node, command],
                            capture_output=True, text=True, timeout=45)
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (out / f'{node}-startup.log').open('a') as stream:
        stream.write(f'\n{stamp}\n{result.stdout}\n{result.stderr}')
    if result.returncode:
        raise RuntimeError(f'{node}: observation failed: {result.stderr}')
    state = json.loads(result.stdout.splitlines()[0])
    if not state['Running']:
        raise RuntimeError(f'{node}: container stopped: {state}')
    text = '\n'.join(result.stdout.splitlines()[1:])
    print(stamp, node, '\n'.join(text.splitlines()[-3:]), flush=True)
    if 'Traceback (most recent' in text or 'EngineDeadError' in text:
        raise RuntimeError(f'{node}: engine error; inspect full logs')
    return node == 'dusty' and 'Application startup complete' in text


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    while time.monotonic() < deadline:
        if any(list(pool.map(poll, nodes))):
            print('API-STARTED: real completion gate still required', flush=True)
            break
        time.sleep(15)
    else:
        raise RuntimeError('Bounded one-hour startup window expired; no automatic restart')
