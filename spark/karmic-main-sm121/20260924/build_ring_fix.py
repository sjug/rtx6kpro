"""Build only on idle hosts, preserve evidence before gates, then publish."""
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parent
remote = '/home/jugs/git/ds41-r38/karmic-main-20260924'
lock_bytes = (root / 'ring-fix.lock.json').read_bytes()
lock = json.loads(lock_bytes)
digest = hashlib.sha256(lock_bytes).hexdigest()
for name, expected in lock['inputs'].items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError('Unfrozen input: ' + name)
def ssh(node, command):
    return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', node, command],
                                   text=True, stderr=subprocess.STDOUT, timeout=60)
def idle():
    for node in ('dusty', 'toby', 'rusty', 'kirby'):
        for command in ('podman ps -q', 'nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits'):
            if ssh(node, command).strip():
                raise RuntimeError('Host busy: ' + node)
idle()
base = json.loads(ssh('dusty', 'podman image inspect ' + lock['base_image_id']))[0]
if base['Id'].removeprefix('sha256:') != lock['base_image_id'] or base['Config']['Labels']['b12x.source-tree'] != lock['b12x_tree']:
    raise RuntimeError('Unqualified base identity')
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
out = root / 'receipts' / ('ring-fix-build-' + stamp)
out.mkdir(exist_ok=False)
(out / 'inputs').mkdir()
files = [*lock['inputs'], 'ring-fix.lock.json', 'ring-fix.ignore']
for name in files:
    (out / 'inputs' / name).write_bytes((root / name).read_bytes())
(out / 'BUILD-STATUS').write_text('building; not qualified\n')
subprocess.run(['scp', *[str(root / name) for name in files], f'dusty:{remote}/'], check=True)
def run(command, filename, expected_code=0):
    with (out / filename).open('x') as log:
        process = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', 'dusty', shlex.join(command)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end='', flush=True)
        code = process.wait()
    (out / (filename + '.exit-code')).write_text(str(code) + '\n')
    if code != expected_code:
        raise RuntimeError(f'{filename}: exit {code}, expected {expected_code}')
common = ['podman', 'run', '--rm', '--pull=never', '--network=none',
          '--device', 'nvidia.com/gpu=all', '-v', remote + ':/diag:ro',
          '-e', 'PYTHONUNBUFFERED=1', '-e', 'B12X_PRINT_COMPILE_PROGRESS=1',
          '--entrypoint', '/opt/venv/bin/python']
try:
    # This named continuation must fail on its numerical assertion, not collection
    # or import. Keep the complete assertion log for audit.
    run(common + [lock['base_image_id'], '-u', '-m', 'pytest', '-s', '-vv',
        '-p', 'no:cacheprovider', '/diag/test_compressor_ring_mapping.py',
        '-k', 'odd_continuation_loads_own_saved_projection and False and 385', '--maxfail=1'],
        'base-red.log', expected_code=1)
    red = (out / 'base-red.log').read_text()
    if '1 failed, 28 deselected' not in red or 'Tensor-likes are not equal' not in red:
        raise RuntimeError('Base did not reproduce the intended saved-projection failure')
    idle()
    run(['podman', 'build', '--pull=never', '--network=none', '--format=docker',
         '--timestamp=0', '--tag', 'localhost/voipmonitor/build-components:ds41-ring-fix',
         '--iidfile', remote + '/ring-fix-image.id', '--ignorefile', remote + '/ring-fix.ignore',
         '-f', remote + '/Dockerfile.ring-fix', '--build-arg', 'RING_LOCK=' + digest,
         '--build-arg', 'RING_CACHE=ds41-ring-' + digest[:20], remote], 'build.log')
    image = ssh('dusty', 'cat ' + remote + '/ring-fix-image.id').strip().removeprefix('sha256:')
    raw = ssh('dusty', 'podman image inspect ' + image)
    (out / 'image-inspect.json').write_text(raw)
    info = json.loads(raw)[0]
    labels = info['Config']['Labels']
    for key, expected in {
        'local-inference.ds41.diagnostic.kind': 'compressor-ring-mapping-fix',
        'local-inference.ds41.diagnostic.lock.sha256': digest,
        'b12x.source-tree': lock['b12x_tree'],
        'local-inference.status': 'candidate-not-qualified',
    }.items():
        if labels.get(key) != expected:
            raise RuntimeError('Candidate label mismatch: ' + key)
    idle()
    run(common + [image, '-u', '/diag/run_ring_regression.py'], 'ring-regression.log')
    if 'RING-REGRESSION-PASS cases=29' not in (out / 'ring-regression.log').read_text().splitlines():
        raise RuntimeError('Missing exact regression marker')
    tag = 'localhost/voipmonitor/vllm:karmic-main-ds41-deterministic-spark-sm121'
    ssh('dusty', shlex.join(['podman', 'tag', image, tag]))
    (out / 'BUILD-OK').write_text(image + '\n')
    (out / 'BUILD-STATUS').write_text('build gates passed; model qualification pending\n')
    (root / 'receipts/ring-fix-build-receipt.json').write_text(json.dumps({
        'directory': str(out), 'image_id': image, 'lock_sha256': digest}, indent=2) + '\n')
    print('RING-FIX-BUILD-OK', image, flush=True)
except BaseException:
    (out / 'BUILD-STATUS').write_text('failed; component retained; not qualified\n')
    raise
