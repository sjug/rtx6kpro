"""Build a reviewed, diagnostic-only timing image on the idle DS4.1 hosts."""
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parent
remote = '/home/jugs/git/ds41-r38/karmic-main-20260924'
raw_lock = (root / 'claude-engram-timing.lock.json').read_bytes()
lock = json.loads(raw_lock)
digest = hashlib.sha256(raw_lock).hexdigest()


def ssh(node, command):
    return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', node, command],
                                   text=True, stderr=subprocess.STDOUT, timeout=60)


def idle():
    for node in ('dusty', 'toby', 'rusty', 'kirby'):
        for cmd in ('podman ps -q', 'nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits'):
            if ssh(node, cmd).strip():
                raise RuntimeError('Host busy: ' + node)


for name, expected in lock['inputs'].items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError('Diagnostic input drift: ' + name)
idle()
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
out = root / 'receipts' / ('engram-timing-build-' + stamp)
out.mkdir(exist_ok=False)
(out / 'inputs').mkdir()
files = [*lock['inputs'], 'claude-engram-timing.lock.json', 'claude-engram-timing.ignore']
for name in files:
    (out / 'inputs' / name).write_bytes((root / name).read_bytes())
(out / 'inputs/build_engram_timing.py').write_bytes(Path(__file__).read_bytes())
(out / 'BUILD-STATUS').write_text('building diagnostic; never model-qualified\n')
subprocess.run(['scp', *[str(root / name) for name in files], f'dusty:{remote}/'], check=True)
command = ['podman', 'build', '--pull=never', '--network=none', '--format=docker',
    '--timestamp=0', '--tag', 'localhost/voipmonitor/build-components:ds41-engram-timing',
    '--iidfile', remote + '/engram-timing-image.id',
    '--ignorefile', remote + '/claude-engram-timing.ignore',
    '-f', remote + '/Dockerfile.claude-engram-timing',
    '--build-arg', 'TIMING_LOCK=' + digest,
    '--build-arg', 'TIMING_CACHE=ds41-engram-timing-' + digest[:20], remote]
try:
    with (out / 'build.log').open('x') as stream:
        process = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', 'dusty', shlex.join(command)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(line, end='', flush=True)
        if process.wait():
            raise RuntimeError('Diagnostic build failed; see build.log')
    image = ssh('dusty', 'cat ' + remote + '/engram-timing-image.id').strip().removeprefix('sha256:')
    inspect = ssh('dusty', 'podman image inspect ' + image)
    (out / 'image-inspect.json').write_text(inspect)
    labels = json.loads(inspect)[0]['Config']['Labels']
    for key, value in {
        'local-inference.ds41.diagnostic.kind': 'engram-host-timing',
        'local-inference.ds41.diagnostic.lock.sha256': digest,
        'local-inference.status': 'diagnostic-only-not-qualified',
    }.items():
        if labels.get(key) != value:
            raise RuntimeError('Unexpected diagnostic label: ' + key)
    if 'ENGRAM-TIMING-INSTALL-PASS' not in (out / 'build.log').read_text().splitlines():
        raise RuntimeError('Missing source/native inventory gate')
    idle()
    check = ssh('dusty', shlex.join(['podman', 'run', '--rm', '--pull=never',
        '--network=none', '--entrypoint', '/opt/venv/bin/python', image, '-c',
        'from vllm.models.deepseek_v4_1 import claude_engram_timing as t; '
        't.install_b12x(); assert t._installed == [True]; print("TIMING-BIND-PASS")']))
    (out / 'binding.log').write_text(check)
    if 'TIMING-BIND-PASS' not in check.splitlines():
        raise RuntimeError('Timing binding gate did not pass')
    (out / 'BUILD-OK').write_text(image + '\n')
    (out / 'BUILD-STATUS').write_text('diagnostic build gates passed; never model-qualified\n')
    (root / 'receipts/engram-timing-build-receipt.json').write_text(json.dumps({
        'directory': str(out), 'image_id': image, 'lock_sha256': digest}, indent=2) + '\n')
    print('ENGRAM-TIMING-BUILD-OK', image, flush=True)
except BaseException:
    (out / 'BUILD-STATUS').write_text('failed; protected diagnostic image retained\n')
    raise
