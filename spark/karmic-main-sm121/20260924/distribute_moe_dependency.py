"""Plain Docker archive over switched 200G; no conversion or compression."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import socket
import subprocess

ROOT = Path(__file__).resolve().parent
IMAGE = 'd880b1297eda5d788bd45c2f85a0b42e9dbcdde856117a10ec6ec3f92d7b5937'
TAG = 'localhost/voipmonitor/build-components:ds41-moe-dependency-20260925'
PEERS = {'toby': '10.11.11.6', 'rusty': '10.11.11.5', 'kirby': '10.11.11.8'}
archive = ROOT / 'moe-dependency-image.tar'
if socket.gethostname().split('.')[0] != 'dusty':
    raise RuntimeError('Archive distributor requires dusty')
if not (ROOT / 'receipts/moe-dependency-build/BUILD-OK').is_file():
    raise RuntimeError('Dependency gate missing')
if subprocess.check_output(['podman', 'ps', '-q'], text=True).strip():
    raise RuntimeError('Build host is serving')
size = json.loads(subprocess.check_output(['podman', 'image', 'inspect', IMAGE]))[0]['Size']
for node in PEERS:
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', node,
                             f'podman ps -q && df -B1 --output=avail {ROOT}'],
                            text=True, capture_output=True, check=True)
    lines = result.stdout.strip().splitlines()
    if len(lines) != 2 or int(lines[-1]) < size * 2 + 20_000_000_000:
        raise RuntimeError(f'{node}: not idle or inadequate staging capacity: {lines}')
if archive.exists():
    raise RuntimeError('Archive already exists; inspect rather than overwrite')
subprocess.run(['podman', 'save', '--format', 'docker-archive', '--output', str(archive), TAG], check=True)
with archive.open('rb') as source:
    digest = hashlib.file_digest(source, 'sha256').hexdigest()
(ROOT / 'receipts/moe-dependency-archive.json').write_text(json.dumps({
    'image': IMAGE, 'archive_sha256': digest, 'bytes': archive.stat().st_size}, indent=2) + '\n')
print('ARCHIVE-SAVED', digest, archive.stat().st_size, flush=True)
def transfer(item):
    node, address = item
    route = json.loads(subprocess.check_output(['ip', '-j', 'route', 'get', address]))[0]
    if route.get('dev') != 'enp1s0f0np0' or route.get('prefsrc') != '10.11.11.7':
        raise RuntimeError(f'Not on switched fabric: {route}')
    subprocess.run(['rsync', '--whole-file', '--info=progress2',
                    '-e', 'ssh -o BatchMode=yes -o Compression=no -c aes128-gcm@openssh.com',
                    str(archive), f'{address}:{archive}'], check=True)
    # Control/verification uses hostnames; only bulk bytes use the fabric IPs.
    script = f'sha256sum {archive}'
    actual = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', node, script], text=True).split()[0]
    if actual != digest:
        raise RuntimeError(f'{node}: archive checksum mismatch')
    subprocess.run(['ssh', '-o', 'BatchMode=yes', node, f'podman load -i {archive}'], check=True)
    inspection = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', node,
                                          f'podman image inspect {IMAGE}'], text=True)
    if json.loads(inspection)[0]['Id'].removeprefix('sha256:') != IMAGE:
        raise RuntimeError(f'{node}: loaded image identity mismatch')
    (ROOT / f'receipts/moe-dependency-{node}-image.json').write_text(inspection)
    print('IMAGE-VERIFIED', node, IMAGE, flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(transfer, PEERS.items()))
print('MOE-DEPENDENCY-DISTRIBUTED', flush=True)
