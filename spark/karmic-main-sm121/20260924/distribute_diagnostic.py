"""Ship a gated diagnostic using the existing unchanged Docker-archive path."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent
REMOTE = '/home/jugs/git/ds41-r38/karmic-main-20260924'
KINDS = {'decision-row': 'claude-decision-row',
         'mhc-expanded': 'claude-mhc-expanded',
         'precision': 'ds41-precision',
         'precision-release': 'ds41-precision-release',
         'ratio1': 'ds41-ratio1',
         'indexer': 'ds41-indexer',
         'window': 'claude-window',
         'engram-fault': 'claude-engram-fault-inject'}


def resolve(kind, root=ROOT):
    stem = KINDS[kind]
    build = json.loads((root / f'receipts/{kind}-build-receipt.json').read_text())
    receipt = Path(build['directory']).resolve()
    if receipt.parent != (root / 'receipts').resolve():
        raise RuntimeError('Build receipt is outside this kit')
    image = build['image_id']
    if len(image) != 64 or any(c not in '0123456789abcdef' for c in image):
        raise RuntimeError('Invalid image identity')
    if (receipt / 'BUILD-OK').read_text().strip() != image:
        raise RuntimeError('Build gate identity differs')
    if hashlib.sha256((root / f'{stem}.lock.json').read_bytes()).hexdigest() != build['lock_sha256']:
        raise RuntimeError('Diagnostic lock changed since the build')
    return build, receipt


def image_tag(kind):
    if kind == 'ratio1':
        from build_ratio1 import TAG
        return TAG
    return 'localhost/voipmonitor/build-components:ds41-' + kind


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=KINDS)
    args = parser.parse_args()
    build, receipt = resolve(args.kind)
    prefix = args.kind + '-' + build['image_id'][:12]
    # Preserve the exact transfer program with this build's receipt.
    (receipt / 'distribution-program.py').write_bytes((ROOT / 'distribute_determinism.py').read_bytes())
    subprocess.run(['scp', str(ROOT / 'distribute_determinism.py'), 'dusty:' + REMOTE + '/'], check=True)
    remote_gate = prefix + '-build.ok'
    subprocess.run(['scp', str(receipt / 'BUILD-OK'), f'dusty:{REMOTE}/{remote_gate}'], check=True)
    command = ['python3', '-u', REMOTE + '/distribute_determinism.py', '--receipt', remote_gate,
               '--prefix', prefix, '--tag', image_tag(args.kind)]
    log_path = receipt / 'distribution.log'
    attempt = 1
    while log_path.exists():
        attempt += 1
        log_path = receipt / f'distribution-{attempt}.log'
    with log_path.open('x') as log:
        process = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', 'dusty', shlex.join(command)],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            log.write(line)
            log.flush()
            if any(word in line for word in ('ARCHIVE-SAVED', 'IMAGE-VERIFIED', 'DISTRIBUTED', 'Error', 'Traceback')):
                print(line, end='', flush=True)
        code = process.wait()
    log_path.with_suffix('.exit').write_text(str(code) + '\n')
    if code:
        raise SystemExit(code)
    for name in [prefix + '-archive.json', *[prefix + '-' + n + '-image.json' for n in ('toby', 'rusty', 'kirby')]]:
        subprocess.run(['scp', f'dusty:{REMOTE}/receipts/{name}', str(receipt / name)], check=True)
    print('DIAGNOSTIC-DISTRIBUTION-COMPLETE', args.kind, build['image_id'], flush=True)


if __name__ == '__main__':
    main()
