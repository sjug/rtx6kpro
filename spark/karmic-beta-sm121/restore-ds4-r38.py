#!/usr/bin/env python3
"""Restore the retained, user-approved DS4 Vision pair without changing it."""
import json
import shlex
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'qualification/ds4-r38-restored-20260921'
NODES = ('toby', 'rusty')
TARGET = 'ds4-vision-jj-r38-tp2'
IMAGE = 'ea031e1d3d051033f077fc986bf6f8fce04cf9ab52483d5a719ba13114567fc5'
MODEL = 'DeepSeek-V4-Flash-Vision-Exp'

def remote(node, command):
    return subprocess.check_output(['ssh', '-n', '-o', 'BatchMode=yes', '-o',
        'ConnectTimeout=10', node, command], text=True, timeout=100)

def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2) + '\n')

def inspect(node):
    return json.loads(remote(node, f'podman inspect {TARGET}'))[0]

def main():
    resume = '--monitor-only' in sys.argv
    OUT.mkdir(exist_ok=resume)
    started = datetime.now(timezone.utc).isoformat()
    try:
        for node in NODES:
            info = inspect(node)
            save(f'{node}-before.json', info)
            if info['Image'] != IMAGE or info['State']['Running'] != resume:
                raise RuntimeError(f'Unexpected identity/state on {node}')
            remote(node, f'podman image exists {IMAGE}')
            running = remote(node, "podman ps --format '{{.Names}}'").strip()
            if running != (TARGET if resume else ''):
                raise RuntimeError(f'Unexpected running container on {node}')
            for mount in info['Mounts']:
                remote(node, f"test -e {shlex.quote(mount['Source'])}")
        for node in NODES:
            if not resume:
                print(node, remote(node, f'podman start {TARGET}').strip(), 'started', flush=True)
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            for node in NODES:
                state = inspect(node)['State']
                if not state['Running'] or state['OOMKilled']:
                    raise RuntimeError(f'{node} stopped or OOM-killed')
                print(node, remote(node, f'podman logs --since 30s {TARGET} 2>&1')[-1600:], flush=True)
            try:
                with urllib.request.urlopen('http://rusty:8000/v1/models', timeout=10) as response:
                    models = json.load(response)
            except OSError as error:
                print(str(error), flush=True)
                time.sleep(15)
                continue
            save('models.json', models)
            if [m['id'] for m in models['data']] != [MODEL]:
                raise RuntimeError('Wrong served alias')
            payload = {'model': MODEL, 'messages': [{'role': 'user',
                'content': 'Calculate 17 times 23 minus 58. Reply with only the resulting integer.'}],
                'chat_template_kwargs': {'thinking': False}, 'temperature': 0, 'max_tokens': 128}
            request = urllib.request.Request('http://rusty:8000/v1/chat/completions',
                json.dumps(payload).encode(), {'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.load(response)
            save('completion.json', result)
            choice = result['choices'][0]
            if choice['finish_reason'] != 'stop' or choice['message']['content'].strip() != '333':
                raise RuntimeError('Real completion failed')
            for node in NODES:
                save(f'{node}-final.json', inspect(node))
            save('status.json', {'status': 'R38-RESTORED', 'finished': datetime.now(timezone.utc).isoformat()})
            print('R38-RESTORED: real completion 333, expected alias', flush=True)
            return
        raise RuntimeError('Readiness deadline exceeded')
    finally:
        for node in NODES:
            for suffix, command in [('container', f'podman logs --timestamps --since {shlex.quote(started)} {TARGET} 2>&1'),
                                    ('kernel', f'journalctl -k --since {shlex.quote(started)} --no-pager')]:
                try:
                    (OUT / f'{node}-{suffix}.log').write_text(remote(node, command))
                except Exception as error:
                    print(f'Log capture failed: {node} {suffix}: {error}', flush=True)

if __name__ == '__main__':
    main()
