"""Build one diagnostic image on idle dusty; retain inputs, image and gate logs."""
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parent
remote = '/home/jugs/git/ds41-r38/karmic-main-20260924'
lock_bytes = (root / 'stale-probe.lock.json').read_bytes()
lock = json.loads(lock_bytes)
digest = hashlib.sha256(lock_bytes).hexdigest()
for name, expected in lock['inputs'].items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError('Unfrozen input: ' + name)

def ssh(node, command, timeout=120):
    return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', node, command],
                                   text=True, stderr=subprocess.STDOUT, timeout=timeout)

def idle():
    for node in ('dusty', 'toby', 'rusty', 'kirby'):
        for command in ('podman ps -q', 'nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits'):
            if ssh(node, command, timeout=30).strip():
                raise RuntimeError('Host is busy: ' + node)

idle()
base = json.loads(ssh('dusty', 'podman image inspect ' + lock['base_image_id']))[0]
if (base['Id'].removeprefix('sha256:') != lock['base_image_id'] or
        base['Config']['Labels']['b12x.source-tree'] != lock['b12x_tree']):
    raise RuntimeError('Base image identity mismatch')
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
out = root / 'receipts' / ('stale-probe-build-' + stamp)
out.mkdir(exist_ok=False)
(out / 'inputs').mkdir()
files = [*lock['inputs'], 'stale-probe.lock.json', 'stale-probe.ignore']
for name in files:
    (out / 'inputs' / name).write_bytes((root / name).read_bytes())
(out / 'BUILD-STATUS').write_text('building diagnostic; not qualified\n')
subprocess.run(['scp', *[str(root / name) for name in files], f'dusty:{remote}/'], check=True)

def run(command, filename):
    with (out / filename).open('x') as log:
        process = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', 'dusty', command],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end='', flush=True)
        if process.wait():
            raise RuntimeError('Diagnostic command failed: ' + filename)

try:
    cmd = ['podman', 'build', '--pull=never', '--network=none', '--format=docker',
           '--timestamp=0', '--tag', 'localhost/voipmonitor/build-components:ds41-stale-probe',
           '--iidfile', 'stale-probe-image.id', '--ignorefile', 'stale-probe.ignore',
           '-f', 'Dockerfile.stale-probe', '--build-arg', 'PROBE_LOCK=' + digest,
           '--build-arg', 'PROBE_CACHE=ds41-stale-' + digest[:20], '.']
    run(f'cd {remote} && ' + shlex.join(cmd), 'build.log')
    image = ssh('dusty', f'cat {remote}/stale-probe-image.id').strip().removeprefix('sha256:')
    raw = ssh('dusty', 'podman image inspect ' + image)
    (out / 'image-inspect.json').write_text(raw)
    info = json.loads(raw)[0]
    labels = info['Config']['Labels']
    if (info['Id'].removeprefix('sha256:') != image or
            labels['local-inference.ds41.diagnostic.lock.sha256'] != digest or
            labels['local-inference.ds41.diagnostic.kind'] != 'stale-state-probe' or
            labels['b12x.source-tree'] != lock['b12x_tree']):
        raise RuntimeError('Built diagnostic identity mismatch')
    idle()
    # CPU Torch smoke of actual byte views and indexing in the pinned runtime.
    code = '''import sys, unittest
import torch
from vllm.v1.kv_cache_interface import CircularBufferSpec
if not __debug__:
    raise RuntimeError('Assertions must be enabled')
sys.path.insert(0, '/opt/ds41-stale-probe')
suite = unittest.defaultTestLoader.loadTestsFromName('claude_test_stale_state_probe_torch')
result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
if result.testsRun != 12 or result.skipped or not result.wasSuccessful():
    raise RuntimeError('Required twelve CPU Torch tests did not all pass')
print('STALE-PROBE-RUNTIME-SMOKE-PASS', flush=True)
'''
    run(shlex.join(['podman', 'run', '--rm', '--pull=never', '--network=none',
        '--entrypoint', '/opt/venv/bin/python', image, '-u', '-c', code]), 'runtime-smoke.log')
    if 'STALE-PROBE-RUNTIME-SMOKE-PASS' not in (out / 'runtime-smoke.log').read_text().splitlines():
        raise RuntimeError('Missing runtime smoke marker')
    (out / 'BUILD-OK').write_text(image + '\n')
    (out / 'BUILD-STATUS').write_text('diagnostic build gates passed; not model qualification\n')
    (root / 'receipts/stale-probe-build-receipt.json').write_text(json.dumps({
        'directory': str(out), 'image_id': image, 'lock_sha256': digest}, indent=2) + '\n')
    print('STALE-PROBE-BUILD-OK', image, flush=True)
except BaseException:
    (out / 'BUILD-STATUS').write_text('failed; image retained; not qualified\n')
    raise
