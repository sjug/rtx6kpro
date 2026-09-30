#!/usr/bin/env python3
"""User-authorized restoration of retained R32 Qwen containers only."""
import json
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

out = Path(__file__).resolve().parent / 'qualification/qwen-r32-restored-20260921'
out.mkdir(exist_ok=False)
old = 'qwen38-flash-next-nvfp4-karmic-tp2'
target = 'qwen38-flash-next-nvfp4-jj-r32-tp2'
image = '74e53e710bef141f6f68e722582569f9c6aa388bce405ad6f1423566a2300c9c'
started = datetime.now(timezone.utc).isoformat()

def remote(node, command):
    return subprocess.check_output(['ssh', '-n', '-o', 'BatchMode=yes', '-o',
                                   'ConnectTimeout=10', node, command], text=True, timeout=100)

def save(name, data):
    (out / name).write_text(json.dumps(data, indent=2) + '\n')

def inspect(node, name):
    return json.loads(remote(node, f'podman inspect {name}'))[0]

try:
    for node in ('kirby', 'dusty'):
        info = inspect(node, target)
        save(f'{node}-r32-before.json', info)
        save(f'{node}-karmic-before.json', inspect(node, old))
        if info['Image'] != image or info['State']['Running']:
            raise RuntimeError('Unexpected rollback identity/state')
        remote(node, f'podman image exists {image}')
        for mount in info['Mounts']:
            remote(node, f"test -d '{mount['Source']}'")
    for node in ('kirby', 'dusty'):
        print(node, remote(node, f'podman stop -t 60 {old}').strip(), 'stopped', flush=True)
    for node in ('kirby', 'dusty'):
        if remote(node, "podman ps --format '{{.Names}}'").strip():
            raise RuntimeError(f'Unexpected serving container on {node}')
        print(node, remote(node, f'podman start {target}').strip(), 'started', flush=True)
    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline:
        for node in ('kirby', 'dusty'):
            state = inspect(node, target)['State']
            if not state['Running'] or state['OOMKilled']:
                raise RuntimeError(f'{node} stopped or OOM-killed')
            log = remote(node, f'podman logs --tail 8 {target} 2>&1')
            print(node, log[-1200:], flush=True)
        try:
            payload = {'model': 'Qwen3.8-Flash-Next',
                       'messages': [{'role': 'user', 'content': 'Calculate 17 times 23 minus 58. Reply with only the resulting integer.'}],
                       'chat_template_kwargs': {'enable_thinking': False},
                       'temperature': 0, 'max_tokens': 128}
            request = urllib.request.Request('http://dusty:8000/v1/chat/completions',
                json.dumps(payload).encode(), {'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.load(response)
        except (OSError, TimeoutError) as error:
            print(type(error).__name__, str(error), flush=True)
            time.sleep(15)
            continue
        save('completion.json', result)
        choice = result['choices'][0]
        if choice['finish_reason'] != 'stop' or choice['message']['content'].strip() != '333':
            raise RuntimeError('Restored service failed real completion')
        with urllib.request.urlopen('http://dusty:8000/v1/models', timeout=10) as response:
            models = json.load(response)
        save('models.json', models)
        if [m['id'] for m in models['data']] != ['Qwen3.8-Flash-Next']:
            raise RuntimeError('Wrong served alias')
        for node in ('kirby', 'dusty'):
            save(f'{node}-r32-final.json', inspect(node, target))
            save(f'{node}-karmic-final.json', inspect(node, old))
        save('status.json', {'status': 'R32-RESTORED', 'finished': datetime.now(timezone.utc).isoformat()})
        print('R32-RESTORED real completion 333; alias Qwen3.8-Flash-Next', flush=True)
        break
    else:
        raise RuntimeError('Readiness deadline exceeded')
finally:
    for node in ('kirby', 'dusty'):
        (out / f'{node}-r32.log').write_text(remote(node, f"podman logs --timestamps --since '{started}' {target} 2>&1"))
        (out / f'{node}-kernel.log').write_text(remote(node, f"journalctl -k --since '{started}' --no-pager"))
