"""Build the ratio-1 extend BF16 diagnostic over the exact precision release on idle dusty.

Same shape as build_precision_release.py: frozen inputs and allowlist, exact parent image and
labels, network-less build, layer ancestry, installer marker, labels, then the container smoke
by exact marker. Natives and B12X are the release's (installer-proven), so the release's GPU
gates are not repeated. Writes receipts/ratio1-build-receipt.json only after every gate.
"""
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent
STEM = 'ds41-ratio1'
TAG = 'localhost/voipmonitor/build-components:ds41-ratio1-diagnostic'


def frozen(root=ROOT):
    raw = (root / (STEM + '.lock.json')).read_bytes()
    lock = json.loads(raw)
    for name, expected in lock['inputs'].items():
        if Path(name).name != name or hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Unfrozen input: ' + name)
    if (root / (STEM + '.ignore')).read_text() != '**\n' + ''.join('!' + n + '\n' for n in [*lock['inputs'], STEM + '.lock.json']):
        raise RuntimeError('Build context allowlist differs')
    base = json.loads((root / 'receipts/precision-release-build-receipt.json').read_text())
    if base['image_id'] != lock['base_image_id'] or base['lock_sha256'] != lock['base_lock_sha256']:
        raise RuntimeError('Diagnostic is not based on the recorded precision release')
    if hashlib.sha256((root / lock['base_lock']).read_bytes()).hexdigest() != lock['base_lock_sha256']:
        raise RuntimeError('Local release lock differs from the declared base')
    return lock, hashlib.sha256(raw).hexdigest()


def expected_labels(lock, digest):
    return {'local-inference.ds41.diagnostic.kind': lock['kind'],
            'local-inference.ds41.diagnostic.lock.sha256': digest,
            'local-inference.ds41.diagnostic.base-image': lock['base_image_id'],
            'local-inference.ds41.diagnostic.base-kind': lock['base_kind'],
            'local-inference.ds41.diagnostic.base-lock.sha256': lock['base_lock_sha256'],
            'local-inference.cache.fingerprint': lock['cache_fingerprint'],
            'local-inference.status': lock['status'],
            **{c + '.source-tree': t for c, t in lock['trees'].items()}}


def main():
    from build_activation_trace import idle, ssh, REMOTE
    lock, digest = frozen()
    idle()
    info = json.loads(ssh('dusty', 'podman image inspect ' + lock['base_image_id']))[0]
    labels = info['Config']['Labels']
    if (info['Id'].removeprefix('sha256:') != lock['base_image_id']
            or labels.get('local-inference.ds41.diagnostic.kind') != lock['base_kind']
            or labels.get('local-inference.ds41.diagnostic.lock.sha256') != lock['base_lock_sha256']
            or any(labels.get(c + '.source-tree') != t for c, t in lock['base_trees'].items())):
        raise RuntimeError('Release parent provenance differs')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = ROOT / 'receipts' / ('ratio1-build-' + stamp)
    (out / 'inputs').mkdir(parents=True, exist_ok=False)
    names = [*lock['inputs'], STEM + '.lock.json', STEM + '.ignore']
    for name in names:
        (out / 'inputs' / name).write_bytes((ROOT / name).read_bytes())
    (out / 'BUILD-STATUS').write_text('building diagnostic; not qualified\n')
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
             '--tag', TAG, '--iidfile', REMOTE + '/ratio1-image.id', '--ignorefile', REMOTE + '/' + STEM + '.ignore',
             '-f', REMOTE + '/Dockerfile.' + STEM, '--build-arg', 'RATIO1_LOCK=' + digest,
             '--build-arg', 'RATIO1_CACHE=' + lock['cache_fingerprint'], REMOTE], 'build.log')
        image = ssh('dusty', 'cat ' + REMOTE + '/ratio1-image.id').strip().removeprefix('sha256:')
        raw_inspect = ssh('dusty', 'podman image inspect ' + image)
        (out / 'image-inspect.json').write_text(raw_inspect)
        built = json.loads(raw_inspect)[0]
        inherited = info['RootFS']['Layers']
        if not inherited or built['RootFS']['Layers'][:len(inherited)] != inherited:
            raise RuntimeError('Diagnostic layer ancestry differs')
        for key, value in expected_labels(lock, digest).items():
            if built['Config']['Labels'].get(key) != value:
                raise RuntimeError('Image label mismatch: ' + key)
        if lock['install_pass'] not in (out / 'build.log').read_text().splitlines():
            raise RuntimeError('Missing installer gate')
        idle()
        run(['podman', 'run', '--rm', '--pull=never', '--network=none', '--device', 'nvidia.com/gpu=all',
             '-e', 'PYTHONUNBUFFERED=1', '-v', REMOTE + ':/gate:ro', '--entrypoint', '/opt/venv/bin/python',
             image, '-u', '/gate/ds41_ratio1_smoke.py'], 'smoke.log')
        if lock['smoke_pass'] not in (out / 'smoke.log').read_text().splitlines():
            raise RuntimeError('Missing exact marker: ' + lock['smoke_pass'])
        (out / 'BUILD-OK').write_text(image + '\n')
        (out / 'BUILD-STATUS').write_text('build gates passed; diagnostic only, not qualified\n')
        (ROOT / 'receipts/ratio1-build-receipt.json').write_text(json.dumps(
            {'directory': str(out), 'image_id': image, 'lock_sha256': digest}, indent=2) + '\n')
        print('RATIO1-BUILD-OK', image, flush=True)
    except BaseException:
        (out / 'BUILD-STATUS').write_text('failed; diagnostic retained, not qualified\n')
        raise


if __name__ == '__main__':
    main()
