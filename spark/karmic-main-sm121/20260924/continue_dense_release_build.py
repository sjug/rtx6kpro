"""Start the reviewed build as soon as the last isolated matrix receipt closes."""
import json
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parent
remote = '/home/jugs/git/ds41-r38/karmic-main-20260924'
last = root / 'receipts/engram-dense-isolated-kirby-release-matrix-m192.json'
deadline = time.monotonic() + 1800
while not last.exists():
    if time.monotonic() >= deadline:
        raise RuntimeError('Matrix completion deadline exceeded; no build started')
    time.sleep(5)
data = json.loads(last.read_text())
if len(data['results']) != 4096 or not all(row['same_as_serving'] and row['same_as_first'] for row in data['results']):
    raise RuntimeError('Final matrix receipt failed; no build started')
receipts = []
for node in ('dusty', 'kirby'):
    for length in (129, 160, 192):
        label = 'fence-before-release-confirm' if (node, length) == ('kirby', 129) else f'release-matrix-m{length}'
        receipts.append(root / 'receipts' / f'engram-dense-isolated-{node}-{label}.json')
subprocess.run(['scp', *map(str, receipts), f'dusty:{remote}/receipts/'], check=True)
print('MATRIX-CLOSED: starting reviewed image build and gates', flush=True)
result = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'dusty',
    f'cd {remote} && python3 -u build_dense_release.py'])
if result.returncode:
    raise RuntimeError('Build failed; inspect retained remote build receipt')
subprocess.run(['scp', f'dusty:{remote}/receipts/dense-release-build-receipt.json', str(root / 'receipts')], check=True)
receipt = json.loads((root / 'receipts/dense-release-build-receipt.json').read_text())
directory = Path(receipt['directory']).name
if not directory.startswith('dense-release-build-'):
    raise RuntimeError('Unexpected build receipt path')
subprocess.run(['rsync', '-a', '-e', 'ssh -o BatchMode=yes -o Compression=no',
    f'dusty:{remote}/receipts/{directory}', str(root / 'receipts') + '/'], check=True)
print('DENSE-RELEASE-BUILD-RECEIPTS-RETAINED', receipt['image_id'], flush=True)
