"""Build the one-site router candidate on idle dusty and gate before serving tag."""
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from build_activation_trace import idle, ssh, REMOTE

ROOT = Path(__file__).resolve().parent


def main():
    raw = (ROOT / 'router-release.lock.json').read_bytes()
    lock = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    for name, expected in lock['inputs'].items():
        if Path(name).name != name or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Unfrozen input: ' + name)
    expected_ignore = '**\n' + ''.join('!' + n + '\n' for n in [*lock['inputs'], 'router-release.lock.json'])
    if (ROOT / 'router-release.ignore').read_text() != expected_ignore:
        raise RuntimeError('Build context allowlist differs')
    idle()
    info = json.loads(ssh('dusty', 'podman image inspect ' + lock['base_image_id']))[0]
    if info['Id'].removeprefix('sha256:') != lock['base_image_id']:
        raise RuntimeError('Wrong base image')
    for component in ('vllm', 'b12x'):
        if info['Config']['Labels'].get(component + '.source-tree') != lock['base_trees'][component]:
            raise RuntimeError('Wrong base source: ' + component)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = ROOT / 'receipts' / ('router-release-build-' + stamp)
    (out / 'inputs').mkdir(parents=True, exist_ok=False)
    names = [*lock['inputs'], 'router-release.lock.json', 'router-release.ignore']
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
             '--tag', 'localhost/voipmonitor/build-components:ds41-router-release',
             '--iidfile', REMOTE + '/router-release-image.id', '--ignorefile', REMOTE + '/router-release.ignore',
             '-f', REMOTE + '/Dockerfile.router-release', '--build-arg', 'ROUTER_LOCK=' + digest,
             '--build-arg', 'ROUTER_CACHE=' + lock['cache_fingerprint'],
             '--build-arg', 'ROUTER_B12X_TREE=' + lock['trees']['b12x'], REMOTE], 'build.log')
        image = ssh('dusty', 'cat ' + REMOTE + '/router-release-image.id').strip().removeprefix('sha256:')
        raw_inspect = ssh('dusty', 'podman image inspect ' + image)
        (out / 'image-inspect.json').write_text(raw_inspect)
        labels = json.loads(raw_inspect)[0]['Config']['Labels']
        expected_labels = {'local-inference.ds41.diagnostic.lock.sha256': digest,
                           'local-inference.ds41.diagnostic.kind': 'router-stage-release-candidate',
                           'local-inference.cache.fingerprint': lock['cache_fingerprint'],
                           'local-inference.status': 'candidate-not-qualified',
                           **{c + '.source-tree': t for c, t in lock['trees'].items()}}
        for key, value in expected_labels.items():
            if labels.get(key) != value:
                raise RuntimeError('Image label mismatch: ' + key)
        if 'ROUTER-RELEASE-INSTALL-PASS' not in (out / 'build.log').read_text().splitlines():
            raise RuntimeError('Missing installer gate')
        common = ['podman', 'run', '--rm', '--pull=never', '--network=none', '--device', 'nvidia.com/gpu=all',
                  '--security-opt', 'seccomp=' + REMOTE + '/seccomp-io-uring.json',
                  '-e', 'PYTHONUNBUFFERED=1', '-e', 'B12X_PRINT_COMPILE_PROGRESS=1', '-e', 'CUTE_DSL_ARCH=sm_121a',
                  '-v', REMOTE + ':/gate:ro', '--entrypoint', '/opt/venv/bin/python', image]
        for script, marker in (
            ('verify_engram_repair.py', 'ENGRAM-NATIVE-GATE-PASS'),
            ('test_router_prefill_gpu.py', 'ROUTER-PREFILL-GPU-PASS cases=4'),
            ('test_engram_epoch_gpu.py', 'ENGRAM-EPOCH-GPU-PASS 8'),
            ('test_engram_enqueue_gpu.py', 'ENGRAM-ENQUEUE-GPU-PASS 16'),
            ('run_ring_regression.py', 'RING-REGRESSION-PASS cases=29'),
        ):
            idle()
            name = script + '.log'
            run(common + ['-u', '/gate/' + script], name)
            if marker not in (out / name).read_text().splitlines():
                raise RuntimeError('Missing exact marker: ' + marker)
        tag = 'localhost/voipmonitor/vllm:karmic-main-ds41-router-fence-spark-sm121'
        ssh('dusty', shlex.join(['podman', 'tag', image, tag]))
        (out / 'BUILD-OK').write_text(image + '\n')
        (out / 'BUILD-STATUS').write_text('build gates passed; full-model causal qualification pending\n')
        (ROOT / 'receipts/router-release-build-receipt.json').write_text(json.dumps(
            {'directory': str(out), 'image_id': image, 'lock_sha256': digest}, indent=2) + '\n')
        print('ROUTER-RELEASE-BUILD-OK', image, flush=True)
    except BaseException:
        (out / 'BUILD-STATUS').write_text('failed; protected candidate retained, not qualified\n')
        raise


if __name__ == '__main__':
    main()
