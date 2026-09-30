"""Build the precision release candidate on idle dusty and gate it before tagging.

Same shape as build_router_release.py: frozen inputs and allowlist, exact parent image and
labels, network-less build, installer marker, label check, then the router's own GPU gates
and the precision smoke, each by exact marker. Writes receipts/precision-release-build-receipt.json
only after every gate passes. It does not select, distribute or launch anything.
"""
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from build_activation_trace import idle, ssh, REMOTE

ROOT = Path(__file__).resolve().parent
STEM = 'ds41-precision-release'
TAG = 'localhost/voipmonitor/vllm:karmic-main-ds41-precision-release-spark-sm121'


def frozen(root=ROOT):
    raw = (root / (STEM + '.lock.json')).read_bytes()
    lock = json.loads(raw)
    for name, expected in lock['inputs'].items():
        if Path(name).name != name or hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Unfrozen input: ' + name)
    if (root / (STEM + '.ignore')).read_text() != '**\n' + ''.join('!' + n + '\n' for n in [*lock['inputs'], STEM + '.lock.json']):
        raise RuntimeError('Build context allowlist differs')
    base = json.loads((root / 'receipts/router-release-build-receipt.json').read_text())
    if base['image_id'] != lock['base_image_id'] or base['lock_sha256'] != lock['base_lock_sha256']:
        raise RuntimeError('Release is not based on the recorded router parent')
    if hashlib.sha256((root / lock['base_lock']).read_bytes()).hexdigest() != lock['base_lock_sha256']:
        raise RuntimeError('Local router lock differs from the declared base')
    return lock, hashlib.sha256(raw).hexdigest()


def expected_labels(lock, digest):
    return {'local-inference.ds41.diagnostic.kind': lock['kind'],
            'local-inference.ds41.diagnostic.lock.sha256': digest,
            'local-inference.ds41.release.base-image': lock['base_image_id'],
            'local-inference.ds41.release.base-lock.sha256': lock['base_lock_sha256'],
            'local-inference.ds41.release.precision-lock.sha256': lock['precision_lock_sha256'],
            'local-inference.cache.fingerprint': lock['cache_fingerprint'],
            'local-inference.status': lock['status'],
            **{c + '.source-tree': t for c, t in lock['trees'].items()}}


def main():
    lock, digest = frozen()
    idle()
    info = json.loads(ssh('dusty', 'podman image inspect ' + lock['base_image_id']))[0]
    labels = info['Config']['Labels']
    if (info['Id'].removeprefix('sha256:') != lock['base_image_id']
            or labels.get('local-inference.ds41.diagnostic.kind') != lock['base_kind']
            or labels.get('local-inference.ds41.diagnostic.lock.sha256') != lock['base_lock_sha256']
            or any(labels.get(c + '.source-tree') != t for c, t in lock['base_trees'].items())):
        raise RuntimeError('Router parent provenance differs')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = ROOT / 'receipts' / ('precision-release-build-' + stamp)
    (out / 'inputs').mkdir(parents=True, exist_ok=False)
    names = [*lock['inputs'], STEM + '.lock.json', STEM + '.ignore']
    for name in names:
        (out / 'inputs' / name).write_bytes((ROOT / name).read_bytes())
    (out / 'BUILD-STATUS').write_text('building; not model-qualified\n')
    subprocess.run(['scp', *[str(ROOT / n) for n in names], 'dusty:' + REMOTE + '/'], check=True)

    def run(command, name):
        with (out / name).open('x') as log:
            process = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', 'dusty', shlex.join(command)],
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
            code = process.wait()
        (out / (name + '.exit')).write_text(str(code) + '\n')
        if code:
            raise RuntimeError(name + ' failed')

    try:
        run(['podman', 'build', '--pull=never', '--network=none', '--format=docker', '--timestamp=0',
             '--tag', 'localhost/voipmonitor/build-components:' + STEM,
             '--iidfile', REMOTE + '/precision-release-image.id', '--ignorefile', REMOTE + '/' + STEM + '.ignore',
             '-f', REMOTE + '/Dockerfile.' + STEM, '--build-arg', 'PRECISION_LOCK=' + digest,
             '--build-arg', 'PRECISION_CACHE=' + lock['cache_fingerprint'], REMOTE], 'build.log')
        image = ssh('dusty', 'cat ' + REMOTE + '/precision-release-image.id').strip().removeprefix('sha256:')
        raw_inspect = ssh('dusty', 'podman image inspect ' + image)
        (out / 'image-inspect.json').write_text(raw_inspect)
        built = json.loads(raw_inspect)[0]
        inherited = info['RootFS']['Layers']
        if not inherited or built['RootFS']['Layers'][:len(inherited)] != inherited:
            raise RuntimeError('Release layer ancestry differs')
        for key, value in expected_labels(lock, digest).items():
            if built['Config']['Labels'].get(key) != value:
                raise RuntimeError('Image label mismatch: ' + key)
        if lock['install_pass'] not in (out / 'build.log').read_text().splitlines():
            raise RuntimeError('Missing installer gate')
        common = ['podman', 'run', '--rm', '--pull=never', '--network=none', '--device', 'nvidia.com/gpu=all',
                  '--security-opt', 'seccomp=' + REMOTE + '/seccomp-io-uring.json',
                  '-e', 'PYTHONUNBUFFERED=1', '-e', 'B12X_PRINT_COMPILE_PROGRESS=1', '-e', 'CUTE_DSL_ARCH=sm_121a',
                  '-v', REMOTE + ':/gate:ro', '--entrypoint', '/opt/venv/bin/python', image]
        for script, marker in [*lock['gates'], ['ds41_precision_release_smoke.py', lock['smoke_pass']]]:
            idle()
            name = script + '.log'
            run(common + ['-u', '/gate/' + script], name)
            if marker not in (out / name).read_text().splitlines():
                raise RuntimeError('Missing exact marker: ' + marker)
        ssh('dusty', shlex.join(['podman', 'tag', image, TAG]))
        (out / 'BUILD-OK').write_text(image + '\n')
        (out / 'BUILD-STATUS').write_text('build gates passed; full-model qualification pending\n')
        (ROOT / 'receipts/precision-release-build-receipt.json').write_text(json.dumps(
            {'directory': str(out), 'image_id': image, 'lock_sha256': digest}, indent=2) + '\n')
        print('PRECISION-RELEASE-BUILD-OK', image, flush=True)
    except BaseException:
        (out / 'BUILD-STATUS').write_text('failed; candidate retained, not qualified\n')
        raise


if __name__ == '__main__':
    main()
