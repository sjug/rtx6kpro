"""Transfer the build-gated diagnostic as an unchanged Docker archive on 200G."""
import concurrent.futures
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--arm', choices=('stale-probe', 'ring-fix', 'engram-timing'), default='stale-probe')
arm = parser.parse_args().arm
if socket.gethostname().split('.')[0] != 'dusty':
    raise RuntimeError('Run archive distribution on dusty')
build = json.loads((root / f'receipts/{arm}-build-receipt.json').read_text())
image = build['image_id']
receipt = root / 'receipts' / Path(build['directory']).name
if (receipt / 'BUILD-OK').read_text().strip() != image:
    raise RuntimeError('Missing build gate')
tag = 'localhost/voipmonitor/build-components:ds41-' + arm
info = json.loads(subprocess.check_output(['podman', 'image', 'inspect', tag]))[0]
if info['Id'].removeprefix('sha256:') != image:
    raise RuntimeError('Component tag changed')
archive = root / (arm + '-' + image[:12] + '.tar')
peers = {'toby': '10.11.11.6', 'rusty': '10.11.11.5', 'kirby': '10.11.11.8'}
for node in ('dusty', *peers):
    command = f'podman ps -q && df -B1 --output=avail {root}'
    argv = ['bash', '-c', command] if node == 'dusty' else ['ssh', '-o', 'BatchMode=yes', node, command]
    result = subprocess.check_output(argv, text=True)
    lines = result.strip().splitlines()
    if len(lines) != 2 or int(lines[-1]) < info['Size'] * 2 + 20_000_000_000:
        raise RuntimeError(f'{node}: busy or inadequate disk capacity')
if archive.exists():
    raise RuntimeError('Archive exists; inspect before retry')
subprocess.run(['podman', 'save', '--format', 'docker-archive', '--output', str(archive), tag], check=True)
with archive.open('rb') as stream:
    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
(root / f'receipts/{arm}-archive.json').write_text(json.dumps({
    'image_id': image, 'sha256': digest, 'bytes': archive.stat().st_size}, indent=2) + '\n')
print('ARCHIVE-SAVED', image, digest, archive.stat().st_size, flush=True)

def transfer(item):
    node, address = item
    route = json.loads(subprocess.check_output(['ip', '-j', 'route', 'get', address]))[0]
    if route.get('dev') != 'enp1s0f0np0' or route.get('prefsrc') != '10.11.11.7':
        raise RuntimeError('Unexpected bulk-transfer route: ' + str(route))
    subprocess.run(['rsync', '--whole-file', '--info=progress2', '-e',
        'ssh -o BatchMode=yes -o Compression=no -c aes128-gcm@openssh.com',
        str(archive), f'{address}:{archive}'], check=True)
    actual = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', node,
        f'sha256sum {archive}'], text=True).split()[0]
    if actual != digest:
        raise RuntimeError(node + ': archive digest mismatch')
    subprocess.run(['ssh', '-o', 'BatchMode=yes', node, f'podman load -i {archive}'], check=True)
    raw = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', node,
        'podman image inspect ' + image], text=True)
    if json.loads(raw)[0]['Id'].removeprefix('sha256:') != image:
        raise RuntimeError(node + ': image identity mismatch')
    (root / f'receipts/{arm}-{node}-image.json').write_text(raw)
    print('IMAGE-VERIFIED', node, image, flush=True)

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(transfer, peers.items()))
print(arm.upper() + '-DISTRIBUTED', flush=True)
