#!/usr/bin/env python3
"""Diagnostic only: repeat prepared scout shapes without prefix-cache reuse.

Read the unchanged benchmark's prompt helpers, but do not modify its repository.
This is not another standard grid and does not replace the qualification receipt.
"""
import ast
import hashlib
import json
import re
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
import urllib.request

BASE = 'http://rusty:8000'
MODEL = 'DeepSeek-V4-Flash-Vision-Exp'
IMAGE = 'f30dc6d9a2a6f6fc0ac9ff8cddb04d9f631f4a87b48ca7cac69254802fe83233'
NAME = 'ds4-vision-karmic-tp2'
OUT = Path(__file__).resolve().parents[1] / 'qualification/ds4-vision/warm-prefill'
HARNESS = Path('/home/jugs/git/llm-inference-bench/llm_decode_bench.py')
COUNTERS = {
    'count': 'vllm:request_prefill_time_seconds_count',
    'seconds': 'vllm:request_prefill_time_seconds_sum',
    'computed': 'vllm:request_prefill_kv_computed_tokens_sum',
    'completed': 'vllm:request_success_total',
    'running': 'vllm:num_requests_running',
    'waiting': 'vllm:num_requests_waiting',
}


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')


def post(path, payload):
    request = urllib.request.Request(BASE + path, json.dumps(payload).encode(),
                                     {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.load(response)


def metrics():
    with urllib.request.urlopen(BASE + '/metrics', timeout=15) as response:
        raw = response.read().decode()
    result = {}
    for key, name in COUNTERS.items():
        values = re.findall(r'^' + re.escape(name) + r'(?:\{[^\n]*\})? ([^\n]+)$', raw, re.M)
        if not values:
            raise RuntimeError(f'Missing counter: {name}')
        result[key] = sum(float(value) for value in values)
    return result, raw


def ssh(node, command):
    return subprocess.check_output(['ssh', '-n', '-o', 'BatchMode=yes', '-o',
                                    'ConnectTimeout=10', node, command], text=True, timeout=45)


def main():
    OUT.mkdir(exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    source = HARNESS.read_text()
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != '2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3':
        raise RuntimeError('Benchmark source changed')
    tree = ast.parse(source)
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
                ('PADDING_SENTENCES', 'GENERATION_PROMPT', 'CHARS_PER_TOKEN') for t in node.targets):
            selected.append(node)
        if isinstance(node, ast.FunctionDef) and node.name in ('generate_padding_text', 'build_messages'):
            selected.append(node)
    helpers = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(HARNESS), 'exec'), helpers)
    document = helpers['generate_padding_text'](400000)
    build_messages = helpers['build_messages']
    snapshots = {}
    monitors = []
    files = []
    rows = []
    try:
        for node in ('rusty', 'toby'):
            inspect = json.loads(ssh(node, f'podman inspect {NAME}'))[0]
            if inspect['Image'].removeprefix('sha256:') != IMAGE or not inspect['State']['Running']:
                raise RuntimeError('Serving identity mismatch')
            snapshots[node] = {k: inspect[k] for k in ('Id', 'Image', 'State')}
            handle = (OUT / f'{node}-gpu.csv').open('w')
            files.append(handle)
            monitors.append(subprocess.Popen(['ssh', '-n', '-o', 'BatchMode=yes', node,
                'nvidia-smi --query-gpu=timestamp,clocks.sm,clocks_event_reasons.active,temperature.gpu,power.draw --format=csv -l 1'],
                stdout=handle, stderr=subprocess.STDOUT))
        save('identity.json', {'started': started, 'nodes': snapshots, 'harness_sha256': digest,
             'scope': 'Prepared-shape diagnostic; fresh nonce prefixes, unchanged benchmark prompt helpers; no cache flush or restart'})
        # Alternate direction to avoid confounding length with a single warmup order.
        lengths = [8198, 16147, 32077, 63932, 127645]
        for rep in range(3):
            for length in (lengths if rep != 1 else list(reversed(lengths))):
                label = f'r{rep + 1}-{length}'
                prefix = f'[BENCH_{uuid.uuid4().hex}_CTX_{length}] '
                def messages(chars):
                    return build_messages(length, prefix + document[:chars])
                def count(chars):
                    return post('/tokenize', {'model': MODEL, 'messages': messages(chars)})['count']
                low, high = 0, min(len(document), length * 8)
                found = None
                while low <= high:
                    middle = (low + high) // 2
                    n = count(middle)
                    if n == length:
                        found = middle
                        break
                    if n < length:
                        low = middle + 1
                    else:
                        high = middle - 1
                if found is None:
                    raise RuntimeError(f'Cannot construct exact prompt: {length}')
                payload = {'model': MODEL, 'messages': messages(found), 'max_tokens': 1,
                           'stream': True, 'ignore_eos': True, 'stream_options': {'include_usage': True}}
                save(f'{label}-request.json', payload)
                before, raw = metrics()
                (OUT / f'{label}-before.prom').write_text(raw)
                if before['running'] or before['waiting']:
                    raise RuntimeError('Foreign request active; refusing diagnostic overlap')
                stamp = datetime.now(timezone.utc).isoformat()
                print(json.dumps({'start': label, 'tokens': length, 'tail': length % 4096, 'at': stamp}), flush=True)
                request = urllib.request.Request(BASE + '/v1/chat/completions', json.dumps(payload).encode(),
                                                 {'Content-Type': 'application/json'})
                t0 = time.monotonic()
                first = None
                events = []
                usage = None
                done = False
                with urllib.request.urlopen(request, timeout=900) as response:
                    for line in response:
                        if not line.startswith(b'data: '):
                            continue
                        content = line[6:].strip()
                        if content == b'[DONE]':
                            done = True
                            break
                        event = json.loads(content)
                        events.append(event)
                        if event.get('usage'):
                            usage = event['usage']
                        for choice in event.get('choices', []):
                            delta = choice.get('delta', {})
                            if first is None and any(delta.get(k) for k in ('content', 'reasoning', 'reasoning_content')):
                                first = time.monotonic() - t0
                save(f'{label}-response.json', events)
                deadline = time.monotonic() + 15
                while True:
                    after, raw = metrics()
                    if after['completed'] > before['completed'] and not after['running'] and not after['waiting']:
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError('Completion counters did not settle')
                    time.sleep(.25)
                (OUT / f'{label}-after.prom').write_text(raw)
                changes = {k: after[k] - before[k] for k in COUNTERS}
                cached = (usage or {}).get('prompt_tokens_details', {}).get('cached_tokens')
                valid = (done and first is not None and usage['prompt_tokens'] == length and cached == 0
                         and changes['count'] == 1 and changes['completed'] == 1
                         and changes['computed'] == length and changes['seconds'] > 0)
                row = {'label': label, 'started': stamp, 'tokens': length, 'tail': length % 4096,
                       'valid': valid, 'ttft_s': first, 'client_tps': length / first if first else None,
                       'server_prefill_s': changes['seconds'], 'server_tps': length / changes['seconds'],
                       'cached_tokens': cached, 'usage': usage, 'counter_deltas': changes}
                rows.append(row)
                save('results.json', rows)
                print(json.dumps(row), flush=True)
                if not valid:
                    raise RuntimeError('Invalid or cache-contaminated sample; receipts retained')
        save('summary.json', [{'tokens': n, 'samples': 3,
            **{k: median(r[k] for r in rows if r['tokens'] == n)
               for k in ('ttft_s', 'client_tps', 'server_prefill_s', 'server_tps')}} for n in lengths])
        print('WARM-PREFILL-DIAGNOSTIC-COMPLETE', flush=True)
    finally:
        for monitor in monitors:
            monitor.terminate()
            monitor.wait(timeout=15)
        for handle in files:
            handle.close()
        for node in snapshots:
            (OUT / f'{node}-container.log').write_text(ssh(node, f"podman logs --timestamps --since '{started}' {NAME} 2>&1"))
            (OUT / f'{node}-kernel.log').write_text(ssh(node, f"journalctl -k --since '{started}' --no-pager"))
            final = json.loads(ssh(node, f'podman inspect {NAME}'))[0]
            save(f'{node}-final.json', {k: final[k] for k in ('Id', 'Image', 'State')})
            if final['Id'] != snapshots[node]['Id'] or final['State']['StartedAt'] != snapshots[node]['State']['StartedAt']:
                raise RuntimeError('Boot changed during diagnostic')


if __name__ == '__main__':
    main()
