"""Build one canonical diagnostic image on idle dusty for archive distribution."""
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent
REMOTE = '/home/jugs/git/ds41-r38/karmic-main-20260924'
NODES = ('dusty', 'toby', 'rusty', 'kirby')
OUT = ROOT / 'receipts' / ('attention-trace-build-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
OUT.mkdir(exist_ok=False)
lock_bytes = (ROOT / 'attention-trace.lock.json').read_bytes()
lock = json.loads(lock_bytes)
sha = hashlib.sha256(lock_bytes).hexdigest()
files = ['attention-trace.lock.json', 'trace-patched-attention.py', 'trace_attention.py',
         'install_attention_trace.py', 'Dockerfile.attention-trace', 'build_attention_trace.py',
         'attention-trace.ignore']
(OUT / 'inputs.json').write_text(json.dumps({p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                           for p in files}, indent=2))


def ssh(node, command):
    return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', node, command], text=True, timeout=180)


for node in NODES:
    if ssh(node, 'podman ps -q').strip():
        raise RuntimeError(f'{node} is serving')
    info = json.loads(ssh(node, 'podman image inspect ' + lock['base_image_id']))[0]
    if info['Id'].removeprefix('sha256:') != lock['base_image_id']:
        raise RuntimeError(f'{node}: base mismatch')


def build(node):
    subprocess.run(['scp', *[str(ROOT / p) for p in files], f'{node}:{REMOTE}/'], check=True)
    command = (f'cd {REMOTE} && podman build --pull=never --network=none --format=docker '
               '--timestamp=0 --tag localhost/voipmonitor/build-components:ds41-attention-trace '
               '--iidfile attention-trace-image.id -f Dockerfile.attention-trace '
               '--ignorefile attention-trace.ignore '
               f'--build-arg TRACE_LOCK={sha} --build-arg TRACE_CACHE=ds41-trace-{sha[:20]} .')
    with (OUT / f'{node}-build.log').open('w') as stream:
        subprocess.run(['ssh', '-o', 'BatchMode=yes', node, command], stdout=stream,
                       stderr=subprocess.STDOUT, check=True, timeout=600)
    image = ssh(node, f'cat {REMOTE}/attention-trace-image.id').strip().removeprefix('sha256:')
    raw = ssh(node, 'podman image inspect ' + shlex.quote(image))
    (OUT / f'{node}-inspect.json').write_text(raw)
    info = json.loads(raw)[0]
    labels = info['Config']['Labels']
    if labels['local-inference.ds41.diagnostic.lock.sha256'] != sha:
        raise RuntimeError('Trace label mismatch')
    print('TRACE-BUILT', node, image, flush=True)
    return image


image = build('dusty')
(OUT / 'BUILD-OK').write_text(image + '\n')
(ROOT / 'receipts/attention-trace-build-receipt.json').write_text(json.dumps({'directory': str(OUT)}) + '\n')
print('TRACE-BUILD-OK', image, flush=True)
