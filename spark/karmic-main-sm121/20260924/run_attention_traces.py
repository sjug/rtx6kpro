"""Collect three cold 300-row traces and two 256-row controls on the idle pair."""
import hashlib
import argparse
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import uuid

from cache_metrics import idle_snapshot, finish

ROOT = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument('--operators', action='store_true')
p.add_argument('--prefix', help='Unique retained diagnostic trace name')
p.add_argument('--pairs', help='Comma-separated length:repeat diagnostic pairs')
p.add_argument('--dense-replay', action='store_true', help='Capture a second Engram GEMM without replacing its result')
a = p.parse_args()
prefix = a.prefix or ('ops' if a.operators else 'trace')
if not prefix.replace('-', '').isalnum():
    raise RuntimeError('Unsafe trace prefix')
pairs = ((256, 3), (300, 2)) if a.operators else ((300, 3), (256, 2))
if a.pairs:
    pairs = tuple(tuple(map(int, item.split(':'))) for item in a.pairs.split(','))
    if any(len(item) != 2 or not 1 <= item[0] <= 384 or not 2 <= item[1] <= 8 for item in pairs):
        raise RuntimeError('Invalid trace pairs')
OUT = ROOT / ('receipts/' + prefix + '-trace-requests' if a.operators else 'receipts/attention-trace-requests')
OUT.mkdir(exist_ok=False)
REMOTE = '/home/jugs/git/ds41-r38/karmic-main-20260924'
CACHE = '/home/jugs/.cache/vllm-jj-ds41-tp4'
NAME = 'ds41-flash-karmic-main-tp4'
NODES = ('dusty', 'toby', 'rusty', 'kirby')
image = json.loads((ROOT / 'candidate.json').read_text())['image_id']
corpus = json.loads((ROOT / 'receipts/determinism-corpus.json').read_text())
requests = []


def remote(node, command):
    return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', node, command], text=True, timeout=120)


for node in NODES:
    info = json.loads(remote(node, f'podman inspect {NAME}'))[0]
    if not info['State']['Running'] or info['Image'].removeprefix('sha256:') != image:
        raise RuntimeError('Unexpected serving image')
    remote(node, f'test ! -e {CACHE}/ds41-trace-control.json')
    subprocess.run(['scp', str(ROOT / 'analyze_attention_trace.py'), f'{node}:{REMOTE}/'], check=True)

try:
    for length, count in pairs:
        for repeat in range(count):
            run = f'{prefix}-{length}-{repeat}'
            control = OUT / (run + '-control.json')
            control.write_text(json.dumps({'run': run, 'rows': length, 'dense_replay': a.dense_replay}) + '\n')
            for node in NODES:
                subprocess.run(['scp', str(control), f'{node}:{CACHE}/ds41-trace-control.json'], check=True)
            body = {'model': 'DeepSeek-V4.1-Flash', 'messages': corpus[str(length)]['messages'],
                    'temperature': 0, 'max_tokens': 1, 'logprobs': True, 'top_logprobs': 20,
                    'chat_template_kwargs': {'thinking': False}, 'cache_salt': uuid.uuid4().hex}
            before = idle_snapshot('http://dusty:8000')
            start = time.monotonic()
            req = urllib.request.Request('http://dusty:8000/v1/chat/completions',
                                         json.dumps(body).encode(), {'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=600) as response:
                result = json.load(response)
            (OUT / (run + '.json')).write_text(json.dumps({'request': body, 'response': result,
                'elapsed_s': time.monotonic() - start}, indent=2) + '\n')
            if result['usage']['prompt_tokens'] != length:
                raise RuntimeError('Unexpected prompt length')
            if finish('http://dusty:8000', before, length, OUT / (run + '-cache.json')) != 0:
                raise RuntimeError('Trace request had cache reuse')
            for node in NODES:
                remote(node, f'test -s {CACHE}/ds41-attention-trace/{run}-{node}.pt')
            requests.append({'length': length, 'repeat': repeat, 'choice': result['choices'][0]})
            print('TRACE-CAPTURED', run, flush=True)
    for node in NODES:
        for length, count in pairs:
            paths = ' '.join(f'/cache/ds41-attention-trace/{prefix}-{length}-{r}-{node}.pt' for r in range(count))
            command = (f'podman exec {NAME} /opt/venv/bin/python '
                       f'/opt/ds41-adapter/analyze_attention_trace.py {paths}')
            raw = remote(node, command)
            (OUT / f'{node}-{length}-analysis.json').write_text(raw)
            print('TRACE-ANALYSIS', node, length,
                  [(r['first_differing_input'], r['first_differing_output']) for r in json.loads(raw)], flush=True)
    (OUT / 'response-comparison.json').write_text(json.dumps(requests, indent=2) + '\n')
finally:
    for node in NODES:
        # Preserve the last control but disable tracing before returning.
        remote(node, f'if test -e {CACHE}/ds41-trace-control.json; then '
                     f'test ! -e {CACHE}/ds41-attention-trace/{prefix}-control-final.json && '
                     f'mv {CACHE}/ds41-trace-control.json {CACHE}/ds41-attention-trace/{prefix}-control-final.json; fi')
print('TRACE-COLLECTION-COMPLETE', flush=True)
