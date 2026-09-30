"""One non-streaming request with a reviewed, one-shot rank-3 publication skip.

Requires the dedicated fault image on all four nodes. Never arms the clean
candidate. Keeps the request, response, consumed epoch and per-rank logs.
"""
import argparse
import datetime
import hashlib
import http.client
import json
from pathlib import Path
import shlex
import subprocess
import time
import urllib.error
import urllib.request
import uuid

from cache_metrics import idle_snapshot
from claude_classify_fault_inject import classify, CODES
from runtime import kit_digest

ROOT = Path(__file__).resolve().parent
REMOTE = '/home/jugs/git/ds41-r38/karmic-main-20260924'
NAME = 'ds41-flash-karmic-main-tp4'
NODES = ('dusty', 'toby', 'rusty', 'kirby')


def validate(info, image, digest, node):
    if info['Image'].removeprefix('sha256:') != image or not info['State']['Running']:
        raise RuntimeError('Wrong live fault diagnostic: ' + node)
    labels = info['Config']['Labels']
    if labels.get('local-inference.ds41.kit.sha256') != digest:
        raise RuntimeError('Wrong live runtime kit: ' + node)


def remote(node, command):
    return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', node, command], text=True, timeout=60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    pin = json.loads((ROOT / 'candidate.json').read_text())
    build = json.loads((ROOT / 'receipts/engram-fault-build-receipt.json').read_text())
    if (pin.get('diagnostic', {}).get('kind') != 'engram-fault-inject' or
            pin['image_id'] != build['image_id'] or
            pin['diagnostic']['lock_sha256'] != build['lock_sha256']):
        raise RuntimeError('Fault test requires the installation-gated injection image')
    digest = kit_digest()
    for node in NODES:
        info = json.loads(remote(node, 'podman inspect ' + NAME))[0]
        validate(info, pin['image_id'], digest, node)
        (args.out / (node + '-before.json')).write_text(json.dumps(info, indent=2) + '\n')
    baseline = idle_snapshot('http://dusty:8000')
    (args.out / 'idle.json').write_text(json.dumps(baseline, indent=2) + '\n')
    corpus = json.loads((ROOT / 'receipts/determinism-corpus.json').read_text())['1024']
    if corpus['tokenize']['count'] != 1024:
        raise RuntimeError('Frozen diagnostic prompt length differs')
    token = 'engram-fault-' + uuid.uuid4().hex
    trigger = {'token': token, 'rank': 3, 'node': 'kirby', 'min_tokens': 1024}
    trigger_path = args.out / 'trigger.json'
    trigger_path.write_text(json.dumps(trigger) + '\n')
    body = {'model': 'DeepSeek-V4.1-Flash', 'messages': corpus['messages'],
            'temperature': 0, 'max_tokens': 8, 'stream': False,
            'cache_salt': uuid.uuid4().hex, 'chat_template_kwargs': {'thinking': False}}
    (args.out / 'request.json').write_text(json.dumps(body, indent=2) + '\n')
    (args.out / 'probe.py').write_bytes(Path(__file__).read_bytes())
    (args.out / 'classifier.py').write_bytes((ROOT / 'claude_classify_fault_inject.py').read_bytes())
    since = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (args.out / 'identity.json').write_text(json.dumps({'image_id': pin['image_id'], 'kit': digest,
        'since': since, 'classifier_sha256': hashlib.sha256((ROOT / 'claude_classify_fault_inject.py').read_bytes()).hexdigest()}, indent=2) + '\n')
    staged = REMOTE + '/' + token + '.json'
    subprocess.run(['scp', str(trigger_path), 'kirby:' + staged], check=True)
    # A fresh token and a trigger created after readiness prevent firing at boot.
    remote('kirby', 'podman cp ' + shlex.quote(staged) + ' ' + NAME + ':/cache/ds41-engram-fault-inject.json')
    (args.out / 'idle-before-send.json').write_text(json.dumps(idle_snapshot('http://dusty:8000'), indent=2) + '\n')
    started = time.monotonic()
    response = {'status': None, 'body': None}
    timed_out = False
    try:
        request = urllib.request.Request('http://dusty:8000/v1/chat/completions',
            json.dumps(body).encode(), {'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=120) as result:
            response['status'] = result.status
            response['body'] = result.read().decode()
    except urllib.error.HTTPError as error:
        response.update(status=error.code, body=error.read().decode())
    except (OSError, urllib.error.URLError, http.client.HTTPException) as error:
        timed_out = isinstance(error, TimeoutError) or isinstance(getattr(error, 'reason', None), TimeoutError)
        response['error'] = type(error).__name__ + ': ' + str(error)
    response['elapsed_s'] = time.monotonic() - started
    if isinstance(response['body'], str):
        try:
            response['body'] = json.loads(response['body'])
        except json.JSONDecodeError:
            pass
    (args.out / 'response.json').write_text(json.dumps(response, indent=2) + '\n')
    receipt_name = 'ds41-engram-fault-inject-' + token + '.consumed'
    copy = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'kirby',
        'podman cp ' + NAME + ':/cache/' + receipt_name + ' -'], capture_output=True, timeout=60)
    receipt = None
    if copy.returncode == 0:
        import io
        import tarfile
        with tarfile.open(fileobj=io.BytesIO(copy.stdout)) as archive:
            files = [m for m in archive.getmembers() if m.isfile()]
            if len(files) != 1:
                raise RuntimeError('Unexpected consumed-receipt archive')
            receipt = json.load(archive.extractfile(files[0]))
        (args.out / 'consumed.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (args.out / 'receipt-copy.log').write_text(copy.stderr.decode())
    host_path = '/home/jugs/.cache/vllm-jj-ds41-tp4/' + receipt_name
    host = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'kirby', 'cat ' + shlex.quote(host_path)],
                          text=True, capture_output=True, timeout=60)
    (args.out / 'host-receipt-copy.log').write_text(host.stderr)
    host_receipt = json.loads(host.stdout) if host.returncode == 0 else None
    if host_receipt is not None:
        (args.out / 'consumed-host.json').write_text(json.dumps(host_receipt, indent=2) + '\n')
        if receipt is None:
            receipt = host_receipt
    # Preserve shutdown behavior independently from the API rejection verdict.
    deadline = time.monotonic() + 180
    states = {}
    with (args.out / 'shutdown.jsonl').open('x') as stream:
        while True:
            states = {node: json.loads(remote(node, 'podman inspect ' + NAME))[0]['State'] for node in NODES}
            stream.write(json.dumps({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                     'states': states}) + '\n')
            stream.flush()
            running = [node for node, state in states.items() if state['Running']]
            print('FAULT-SHUTDOWN running=' + ','.join(running), flush=True)
            if not running or time.monotonic() >= deadline:
                break
            time.sleep(5)
    logs = {}
    for node in NODES:
        logs[node] = remote(node, 'podman logs --since ' + shlex.quote(since) + ' ' + NAME + ' 2>&1')
        (args.out / (node + '-injection.log')).write_text(logs[node])
        (args.out / (node + '-after.json')).write_text(remote(node, 'podman inspect ' + NAME))
        (args.out / (node + '-kernel.log')).write_text(remote(node,
            'journalctl -k --since ' + shlex.quote(since) + ' --no-pager'))
    verdict, reasons = classify(receipt, logs['dusty'], response, 1024)
    if receipt is not None and (receipt.get('token') != token or receipt.get('rank') != 3 or receipt.get('node') != 'kirby'):
        verdict, reasons = 'MISTARGETED', ['Receipt token, rank or node differs from the trigger']
    elif host_receipt is not None and receipt != host_receipt:
        verdict, reasons = 'INCONCLUSIVE', ['Container and host consumed receipts differ']
    if timed_out:
        verdict = 'NO-REJECTION-WITHIN-DEADLINE' if receipt is not None else 'INCONCLUSIVE'
        reasons = ['No API rejection observed within 120 seconds after sending the request']
    report = {'verdict': verdict, 'reasons': reasons, 'receipt': receipt,
              'still_running_after_shutdown_window': [node for node, state in states.items() if state['Running']],
              'scope': 'one injected missing publication; not retrieval or model qualification'}
    (args.out / 'verdict.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    return CODES.get(verdict, 6)


if __name__ == '__main__':
    raise SystemExit(main())
