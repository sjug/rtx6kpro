"""Run the reviewed small synthetic MLA audit on idle dusty, with isolated caches.

The pool geometry is diagnostic only; this never changes serving KV sizing.
"""
import datetime
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from build_activation_trace import idle, ssh, REMOTE
from claude_decode_sparse_mla import decode, regimes

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true', help='Eight trials and 48 sampled rows per plan')
    parser.add_argument('--indexer', action='store_true', help='Run MXFP4 indexer conformance instead of MLA')
    args = parser.parse_args()
    build = json.loads((ROOT / 'receipts/decision-row-build-receipt.json').read_text())
    image = build['image_id']
    build_dir = Path(build['directory'])
    if (build_dir / 'BUILD-OK').read_text().strip() != image:
        raise RuntimeError('Capture image installation gate missing')
    idle()
    info = json.loads(ssh('dusty', 'podman image inspect ' + image))[0]
    if (info['Id'].removeprefix('sha256:') != image or
            info['Config']['Labels'].get('local-inference.ds41.diagnostic.lock.sha256') != build['lock_sha256']):
        raise RuntimeError('Image provenance differs')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    prefix = ('dsa' if args.indexer else 'mla') + '-conformance-' + ('full-' if args.full else 'smoke-')
    out = ROOT / 'receipts' / (prefix + stamp)
    out.mkdir(exist_ok=False)
    decoded = {}
    for node in ('dusty', 'toby'):
        path = ROOT / 'receipts/router-b1-tuning-20260925' / (node + '.json')
        records = json.loads(path.read_text())['records']
        found, _ = decode(records, [80927], (80927, 80927))
        decoded[node] = {'regimes': {str(b): {'configs': configs}
                                    for b, configs in regimes(records, found).items()}}
        (out / (node + '-selection.json')).write_bytes(path.read_bytes())
    (out / 'regime.json').write_text(json.dumps(decoded, indent=2) + '\n')
    names = ['claude_conformance_mla.py', 'claude_decode_sparse_mla.py',
             'claude-pinned-b12x-compressed_reference.py', 'run_conformance_smoke.py',
             'claude_conformance_dsa.py']
    hashes = {}
    for name in names:
        payload = (ROOT / name).read_bytes()
        hashes[name] = hashlib.sha256(payload).hexdigest()
        (out / name).write_bytes(payload)
    remote_out = REMOTE + '/receipts/' + out.name
    ssh('dusty', 'mkdir ' + shlex.quote(remote_out))
    subprocess.run(['scp', *[str(out / n) for n in names + ['regime.json', 'dusty-selection.json']],
                    'dusty:' + remote_out + '/'], check=True)
    command = ['podman', 'run', '--rm', '--pull=never', '--network=none', '--ipc=host',
               '--device', 'nvidia.com/gpu=all', '-e', 'PYTHONUNBUFFERED=1',
               '-e', 'B12X_PRINT_COMPILE_PROGRESS=1', '-e', 'B12X_STATE_COMPILE_WORKERS=16',
               '-v', remote_out + ':/gate:rw', '--entrypoint', '/opt/venv/bin/python', image,
               '-u', '/gate/claude_conformance_mla.py', '--regime', '/gate/regime.json',
               '--blocks', '80927', '--trials', '8' if args.full else '1',
               '--sample-rows', '48' if args.full else '4', '--out', '/gate/result.json']
    if args.indexer:
        command = command[:command.index('/gate/claude_conformance_mla.py')] + [
            '/gate/claude_conformance_dsa.py', '--selection', '/gate/dusty-selection.json',
            '--blocks', '80927', '--out', '/gate/result.json', '--distributions',
            'gaussian,codebook,outlier,low-magnitude' if args.full else 'gaussian']
    (out / 'invocation.json').write_text(json.dumps({'image_id': image, 'command': command,
                                                   'source_hashes': hashes}, indent=2) + '\n')
    (out / 'image-inspect.json').write_text(json.dumps(info, indent=2) + '\n')
    idle()
    print('CONFORMANCE-RECEIPTS', out, flush=True)
    with (out / 'gpu.log').open('x') as log:
        process = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', 'dusty', shlex.join(command)],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end='', flush=True)
        code = process.wait()
    (out / 'exit-code').write_text(str(code) + '\n')
    for name in (('result.json',) if args.indexer else ('result.json', 'result.envelope.json')):
        result = subprocess.run(['scp', 'dusty:' + remote_out + '/' + name, str(out / name)],
                                capture_output=True, text=True)
        (out / (name + '.copy.log')).write_text(result.stdout + result.stderr)
    if code:
        raise SystemExit(code)
    report = json.loads((out / 'result.json').read_text())
    if args.indexer:
        if len(report['summaries']) != (12 if args.full else 3) or report['failures']:
            raise RuntimeError('Indexer conformance did not complete all requested roles')
    elif len(report['plans']) != 8 or report['failures'] or not report['envelope']:
        raise RuntimeError('Conformance run did not complete all eight plans')
    print('CONFORMANCE-SMOKE-COMPLETE', flush=True)


if __name__ == '__main__':
    main()
