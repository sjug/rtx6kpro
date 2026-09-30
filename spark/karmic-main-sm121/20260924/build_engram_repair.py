"""Build on idle DS4.1 hosts, retain before gates, publish only after GPU gates."""
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def main():
    root = Path(__file__).resolve().parent
    remote = '/home/jugs/git/ds41-r38/karmic-main-20260924'
    raw = (root / 'engram-repair.lock.json').read_bytes()
    lock = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    for name, expected in lock['inputs'].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Unfrozen input: ' + name)

    def ssh(node, command):
        return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', node, command], text=True, timeout=60)

    def idle():
        for node in ('dusty', 'toby', 'rusty', 'kirby'):
            for command in ('podman ps -q', 'nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits'):
                if ssh(node, command).strip():
                    raise RuntimeError('Host busy: ' + node)

    idle()
    info = json.loads(ssh('dusty', 'podman image inspect ' + lock['base_image_id']))[0]
    if info['Id'].removeprefix('sha256:') != lock['base_image_id'] or info['Config']['Labels']['b12x.source-tree'] != lock['base_b12x_tree']:
        raise RuntimeError('Clean base identity differs')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = root / 'receipts' / ('engram-repair-build-' + stamp)
    out.mkdir(exist_ok=False)
    (out / 'inputs').mkdir()
    files = [*lock['inputs'], 'engram-repair.lock.json', 'engram-repair.ignore']
    for name in files:
        (out / 'inputs' / name).write_bytes((root / name).read_bytes())
    (out / 'BUILD-STATUS').write_text('building; not model-qualified\n')
    subprocess.run(['scp', *[str(root / name) for name in files], 'dusty:' + remote + '/'], check=True)

    def run(command, name):
        with (out / name).open('x') as log:
            proc = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', 'dusty', shlex.join(command)],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
            code = proc.wait()
        (out / (name + '.exit')).write_text(str(code))
        if code:
            raise RuntimeError(name + ': exit ' + str(code))

    try:
        run(['podman', 'build', '--pull=never', '--network=none', '--format=docker', '--timestamp=0',
             '--tag', 'localhost/voipmonitor/build-components:ds41-engram-repair',
             '--iidfile', remote + '/engram-repair-image.id', '--ignorefile', remote + '/engram-repair.ignore',
             '-f', remote + '/Dockerfile.engram-repair', '--build-arg', 'ENGRAM_LOCK=' + digest,
             '--build-arg', 'ENGRAM_CACHE=' + lock['cache_fingerprint'],
             '--build-arg', 'ENGRAM_VLLM_TREE=' + lock['trees']['vllm'],
             '--build-arg', 'ENGRAM_B12X_TREE=' + lock['trees']['b12x'], remote], 'build.log')
        image = ssh('dusty', 'cat ' + remote + '/engram-repair-image.id').strip().removeprefix('sha256:')
        inspection = ssh('dusty', 'podman image inspect ' + image)
        (out / 'image-inspect.json').write_text(inspection)
        labels = json.loads(inspection)[0]['Config']['Labels']
        for key, expected in {'local-inference.ds41.diagnostic.lock.sha256': digest,
                              'vllm.source-tree': lock['trees']['vllm'],
                              'b12x.source-tree': lock['trees']['b12x'],
                              'local-inference.cache.fingerprint': lock['cache_fingerprint']}.items():
            if labels.get(key) != expected:
                raise RuntimeError('Label mismatch: ' + key)
        idle()
        common = ['podman', 'run', '--rm', '--pull=never', '--network=none',
                  '--device', 'nvidia.com/gpu=all', '--security-opt', 'seccomp=' + remote + '/seccomp-io-uring.json',
                  '-e', 'PYTHONUNBUFFERED=1', '-e', 'B12X_PRINT_COMPILE_PROGRESS=1',
                  '-v', remote + ':/gate:ro', '--entrypoint', '/opt/venv/bin/python', image]
        for script, name, marker in [
            ('verify_engram_repair.py', 'native-gpu.log', 'ENGRAM-NATIVE-GATE-PASS'),
            ('test_engram_epoch_gpu.py', 'epoch-gpu.log', 'ENGRAM-EPOCH-GPU-PASS 8'),
            ('test_engram_enqueue_gpu.py', 'enqueue-gpu.log', 'ENGRAM-ENQUEUE-GPU-PASS 16'),
            ('run_ring_regression.py', 'ring-regression.log', 'RING-REGRESSION-PASS cases=29'),
        ]:
            run(common + ['-u', '/gate/' + script], name)
            if marker not in (out / name).read_text().splitlines():
                raise RuntimeError('Missing exact GPU marker: ' + marker)
        tag = 'localhost/voipmonitor/vllm:karmic-main-ds41-deterministic-spark-sm121'
        ssh('dusty', shlex.join(['podman', 'tag', image, tag]))
        (out / 'BUILD-OK').write_text(image + '\n')
        (out / 'BUILD-STATUS').write_text('build gates passed; full model qualification pending\n')
        (root / 'receipts/engram-repair-build-receipt.json').write_text(json.dumps(
            {'directory': str(out), 'image_id': image, 'lock_sha256': digest}, indent=2) + '\n')
        print('ENGRAM-REPAIR-BUILD-OK', image, flush=True)
    except BaseException:
        (out / 'BUILD-STATUS').write_text('failed; protected candidate retained; not qualified\n')
        raise


if __name__ == '__main__':
    main()
