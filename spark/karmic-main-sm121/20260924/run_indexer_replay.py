"""Approved isolated replay orchestration; refuses any running job on the pair/fleet."""
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from build_activation_trace import ROOT, REMOTE, idle, ssh


def main():
    idle()
    image = '25c92dde8801c9601164dd9144ec5c9aa453581d670b212e60a91d4973b96248'
    inspect = ssh('dusty', 'podman image inspect ' + image)
    if json.loads(inspect)[0]['Id'].removeprefix('sha256:') != image:
        raise RuntimeError('Image identity mismatch')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = ROOT / 'receipts' / ('indexer-gpu-replay-' + stamp)
    out.mkdir(exist_ok=False)
    remote = REMOTE + '/receipts/' + out.name
    ssh('dusty', 'mkdir ' + shlex.quote(remote))
    files = ('replay_indexer_projection.py', 'compare_indexer_inputs.py',
             'reference_indexer_projection.py', 'ds41_indexer_capture.py')
    subprocess.run(['scp', *[str(ROOT / f) for f in files], 'dusty:' + remote + '/'], check=True)
    digests = {}
    for f in files:
        digest = hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
        if ssh('dusty', 'sha256sum ' + shlex.quote(remote + '/' + f)).split()[0] != digest:
            raise RuntimeError('Source mismatch')
        digests[f] = digest
        (out / f).write_bytes((ROOT / f).read_bytes())
    paths, sums = [], []
    for receipt in ('decision-row-matched8192-geometry-tree-simple-1ch-capture-20260926T211037Z',
                    'decision-row-matched4096-geometry-tree-simple-1ch-capture-20260926T212615Z'):
        c = json.loads((ROOT / 'receipts' / receipt / 'indexer-captures.json').read_text())['dusty']
        path = REMOTE + '/receipts/' + receipt + '/captures/' + Path(c['container_file']).name
        if ssh('dusty', 'sha256sum ' + shlex.quote(path)).split()[0] != c['sha256']:
            raise RuntimeError('Capture digest mismatch')
        paths.append('/receipts/' + receipt + '/captures/' + Path(path).name)
        sums.append(c['sha256'])
    cmd = ['podman', 'run', '--rm', '--pull=never', '--network=none',
           '--device', 'nvidia.com/gpu=all', '-e', 'PYTHONUNBUFFERED=1',
           '-e', 'CUBLASLT_LOG_LEVEL=5', '-e', 'CUBLASLT_LOG_FILE=/gate/cublaslt.log',
           '-e', 'CUBLAS_LOGINFO_DBG=1', '-e', 'CUBLAS_LOGDEST_DBG=/gate/cublas.log',
           '-v', remote + ':/gate:rw', '-v', REMOTE + '/receipts:/receipts:ro',
           '--entrypoint', '/opt/venv/bin/python', image, '-u',
           '/gate/replay_indexer_projection.py', *paths,
           '--left-sha256', sums[0], '--right-sha256', sums[1],
           '--test-no-reduced-precision']
    (out / 'image-inspect.json').write_text(inspect)
    (out / 'invocation.json').write_text(json.dumps({'command': cmd, 'sources': digests}, indent=2))
    print('RECEIPTS', out, flush=True)
    with (out / 'stdout.log').open('x') as log, (out / 'stderr.log').open('x') as err:
        code = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'dusty', shlex.join(cmd)],
                              stdout=log, stderr=err).returncode
    (out / 'exit-code').write_text(str(code))
    for f in ('cublaslt.log', 'cublas.log'):
        exists = ssh('dusty', 'if test -f ' + shlex.quote(remote + '/' + f) + '; then echo yes; fi')
        if exists.strip() == 'yes':
            subprocess.run(['scp', 'dusty:' + remote + '/' + f, str(out / f)], check=True)
    if code:
        raise RuntimeError('Replay failed, inspect retained logs')
    raw = (out / 'stdout.log').read_text()
    report = json.loads(raw[raw.index('{'):])
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    for name, v in report['variants'].items():
        print(name, 'cross-grid changed', v['cross_grid']['changed_values'],
              'serving/reference mismatches',
              {k: (a['serving_capture']['changed_values'], a['fp64_round_once']['changed_values'])
               for k, a in v['arms'].items()}, flush=True)
    idle()
    print('ISOLATED-REPLAY-COMPLETE', flush=True)


if __name__ == '__main__':
    main()
