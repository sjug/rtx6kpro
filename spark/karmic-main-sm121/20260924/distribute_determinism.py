"""Distribute the gated candidate as an unchanged Docker archive on the fabric."""
import concurrent.futures
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess

ROOT = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument('--receipt', default='receipts/determinism-build/BUILD-OK')
p.add_argument('--tag', default='localhost/voipmonitor/vllm:karmic-main-ds41-deterministic-spark-sm121')
p.add_argument('--prefix', default='determinism')
a = p.parse_args()
if not a.prefix.replace('-', '').isalnum():
    raise RuntimeError('Invalid receipt prefix')
if socket.gethostname().split('.')[0] != 'dusty':
    raise RuntimeError('Distribution requires dusty')
image = (ROOT / a.receipt).read_text().strip().removeprefix('sha256:')
tag = a.tag
info = json.loads(subprocess.check_output(['podman', 'image', 'inspect', tag]))[0]
if info['Id'].removeprefix('sha256:') != image:
    raise RuntimeError('Tag differs from gated image')
peers = {'toby': '10.11.11.6', 'rusty': '10.11.11.5', 'kirby': '10.11.11.8'}
for node in ('dusty', *peers):
    command = f'podman ps -q && df -B1 --output=avail {ROOT}'
    args = ['bash', '-c', command] if node == 'dusty' else ['ssh', '-o', 'BatchMode=yes', node, command]
    result = subprocess.check_output(args, text=True).strip().splitlines()
    if len(result) != 2 or int(result[-1]) < info['Size'] * 2 + 20_000_000_000:
        raise RuntimeError(f'{node}: not idle or inadequate disk capacity: {result}')
archive = ROOT / (a.prefix + '-image.tar')
if archive.exists():
    raise RuntimeError('Archive exists; inspect before overwriting')
subprocess.run(['podman', 'save', '--format', 'docker-archive', '-o', str(archive), tag], check=True)
with archive.open('rb') as stream:
    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
(ROOT / 'receipts' / (a.prefix + '-archive.json')).write_text(json.dumps({
    'image_id': image, 'sha256': digest, 'bytes': archive.stat().st_size}, indent=2))
print('ARCHIVE-SAVED', digest, flush=True)


def transfer(item):
    node, address = item
    route = json.loads(subprocess.check_output(['ip', '-j', 'route', 'get', address]))[0]
    if route.get('dev') != 'enp1s0f0np0' or route.get('prefsrc') != '10.11.11.7':
        raise RuntimeError(f'{node}: non-fabric route: {route}')
    subprocess.run(['rsync', '--whole-file', '--info=progress2', '-e',
                    'ssh -o BatchMode=yes -o Compression=no -c aes128-gcm@openssh.com',
                    str(archive), f'{address}:{archive}'], check=True)
    actual = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', node, f'sha256sum {archive}'], text=True).split()[0]
    if actual != digest:
        raise RuntimeError(f'{node}: archive hash differs')
    subprocess.run(['ssh', '-o', 'BatchMode=yes', node, f'podman load -i {archive}'], check=True)
    inspection = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', node, f'podman image inspect {image}'], text=True)
    if json.loads(inspection)[0]['Id'].removeprefix('sha256:') != image:
        raise RuntimeError(f'{node}: loaded image differs')
    (ROOT / f'receipts/{a.prefix}-{node}-image.json').write_text(inspection)
    print('IMAGE-VERIFIED', node, image, flush=True)


with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(transfer, peers.items()))
print('DETERMINISM-DISTRIBUTED', flush=True)
