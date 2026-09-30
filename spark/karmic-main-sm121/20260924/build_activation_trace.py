"""Build the reviewed activation diagnostic only when all four DS4.1 nodes are idle."""
import datetime
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent
REMOTE = '/home/jugs/git/ds41-r38/karmic-main-20260924'


def ssh(node, command):
    return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', node, command],
                                   text=True, timeout=60)


def idle():
    for node in ('dusty', 'toby', 'rusty', 'kirby'):
        for command in ('podman ps -q', 'nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits'):
            if ssh(node, command).strip():
                raise RuntimeError('Host is not idle: ' + node)


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--moe-seams', action='store_true')
    modes.add_argument('--moe-capture', action='store_true')
    args = parser.parse_args()
    stem = 'claude-moe-seams' if args.moe_seams else 'claude-act-trace'
    prefix = 'moe-seams' if args.moe_seams else 'activation-trace'
    kind = 'activation-digest-moe-seams' if args.moe_seams else 'activation-digest'
    marker = 'MOE-SEAMS-INSTALL-PASS' if args.moe_seams else 'ACT-TRACE-INSTALL-PASS'
    if args.moe_capture:
        stem, prefix = 'claude-moe-capture', 'moe-capture'
        kind, marker = 'activation-digest-moe-capture', 'MOE-CAPTURE-INSTALL-PASS'
    raw = (ROOT / (stem + '.lock.json')).read_bytes()
    lock = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    for name, expected in lock['inputs'].items():
        if Path(name).name != name or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Unfrozen diagnostic input: ' + name)
    idle()
    base = json.loads(ssh('dusty', 'podman image inspect ' + shlex.quote(lock['base_image_id'])))[0]
    if base['Id'].removeprefix('sha256:') != lock['base_image_id']:
        raise RuntimeError('Base image differs')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = ROOT / 'receipts' / (prefix + '-build-' + stamp)
    (out / 'inputs').mkdir(parents=True, exist_ok=False)
    names = [*lock['inputs'], stem + '.lock.json', stem + '.ignore',
             'test_activation_trace_gpu.py', 'build_activation_trace.py']
    for name in names:
        (out / 'inputs' / name).write_bytes((ROOT / name).read_bytes())
    (out / 'BUILD-STATUS').write_text('building diagnostic; not model-qualified\n')
    subprocess.run(['scp', *[str(ROOT / n) for n in names], 'dusty:' + REMOTE + '/'], check=True)

    def run(command, filename):
        with (out / filename).open('x') as log:
            proc = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', 'dusty', shlex.join(command)],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
            code = proc.wait()
        (out / (filename + '.exit')).write_text(str(code))
        if code:
            raise RuntimeError('Failed: ' + filename)

    try:
        run(['podman', 'build', '--pull=never', '--network=none', '--format=docker', '--timestamp=0',
             '--tag', 'localhost/voipmonitor/build-components:ds41-' + prefix,
             '--iidfile', REMOTE + '/' + prefix + '-image.id',
             '--ignorefile', REMOTE + '/' + stem + '.ignore',
             '-f', REMOTE + '/Dockerfile.' + stem,
             '--build-arg', 'TRACE_LOCK=' + digest,
             '--build-arg', 'TRACE_CACHE=ds41-' + prefix + '-' + digest[:20], REMOTE], 'build.log')
        image = ssh('dusty', 'cat ' + REMOTE + '/' + prefix + '-image.id').strip().removeprefix('sha256:')
        raw_inspect = ssh('dusty', 'podman image inspect ' + shlex.quote(image))
        (out / 'image-inspect.json').write_text(raw_inspect)
        labels = json.loads(raw_inspect)[0]['Config']['Labels']
        for key, value in {'local-inference.ds41.diagnostic.kind': kind,
                           'local-inference.ds41.diagnostic.lock.sha256': digest,
                           'local-inference.status': 'diagnostic-only-not-qualified'}.items():
            if labels.get(key) != value:
                raise RuntimeError('Diagnostic label mismatch: ' + key)
        if marker not in (out / 'build.log').read_text().splitlines():
            raise RuntimeError('Missing exact installation gate')
        idle()
        run(['podman', 'run', '--rm', '--pull=never', '--network=none',
             '--device', 'nvidia.com/gpu=all', '-e', 'PYTHONUNBUFFERED=1',
             '-v', REMOTE + ':/gate:ro', '--entrypoint', '/opt/venv/bin/python',
             image, '-u', '/gate/test_activation_trace_gpu.py'], 'gpu.log')
        if 'ACTIVATION-TRACE-GPU-PASS' not in (out / 'gpu.log').read_text().splitlines():
            raise RuntimeError('Missing exact GPU gate')
        if args.moe_capture:
            idle()
            run(['podman', 'run', '--rm', '--pull=never', '--network=none',
                 '--device', 'nvidia.com/gpu=all', '-e', 'PYTHONUNBUFFERED=1',
                 '--entrypoint', '/opt/venv/bin/python', image, '-u',
                 '/opt/ds41-moe-capture/claude_gate_moe_capture_cuda.py'], 'capture-gpu.log')
            if 'CLAUDE-MOE-CAPTURE-CUDA-GATE-PASS' not in (out / 'capture-gpu.log').read_text().splitlines():
                raise RuntimeError('Missing exact capture GPU gate')
        (out / 'BUILD-OK').write_text(image + '\n')
        (out / 'BUILD-STATUS').write_text('diagnostic gates passed; never model-qualified\n')
        (ROOT / ('receipts/' + prefix + '-build-receipt.json')).write_text(json.dumps({
            'directory': str(out), 'image_id': image, 'lock_sha256': digest}, indent=2) + '\n')
        print('ACTIVATION-TRACE-BUILD-OK', image, flush=True)
    except BaseException:
        (out / 'BUILD-STATUS').write_text('failed; protected diagnostic image retained\n')
        raise


if __name__ == '__main__':
    main()
