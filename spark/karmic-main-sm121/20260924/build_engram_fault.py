"""Build only the reviewed publication-skip diagnostic, on idle DS4.1 nodes.

Never selects or launches the result and never gives it a serving tag.
"""
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from build_activation_trace import idle, ssh, REMOTE

ROOT = Path(__file__).resolve().parent
STEM = 'claude-engram-fault-inject'


def frozen_inputs(root=ROOT):
    raw = (root / (STEM + '.lock.json')).read_bytes()
    lock = json.loads(raw)
    for name, expected in lock['inputs'].items():
        if Path(name).name != name or hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Diagnostic input drift: ' + name)
    expected_ignore = '**\n' + ''.join('!' + n + '\n' for n in
        sorted(lock['inputs']) + [STEM + '.lock.json'])
    if (root / (STEM + '.ignore')).read_text() != expected_ignore:
        raise RuntimeError('Diagnostic build context differs')
    base = json.loads((root / 'receipts/router-release-build-receipt.json').read_text())
    if base['image_id'] != lock['base_image_id'] or base['lock_sha256'] != lock['base_lock_sha256']:
        raise RuntimeError('Fault diagnostic is not based on the router candidate')
    if hashlib.sha256((root / lock['base_lock']).read_bytes()).hexdigest() != lock['base_lock_sha256']:
        raise RuntimeError('Local router lock differs from the declared base')
    return lock, hashlib.sha256(raw).hexdigest()


def main():
    lock, digest = frozen_inputs()
    idle()
    base = json.loads(ssh('dusty', 'podman image inspect ' + lock['base_image_id']))[0]
    labels = base['Config']['Labels']
    if (base['Id'].removeprefix('sha256:') != lock['base_image_id']
            or labels.get('local-inference.ds41.diagnostic.kind') != lock['base_kind']
            or labels.get('local-inference.ds41.diagnostic.lock.sha256') != lock['base_lock_sha256']
            or any(labels.get(c + '.source-tree') != t for c, t in lock['base_trees'].items())):
        raise RuntimeError('Installed diagnostic base provenance differs')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = ROOT / 'receipts' / ('engram-fault-build-' + stamp)
    (out / 'inputs').mkdir(parents=True, exist_ok=False)
    names = [*lock['inputs'], STEM + '.lock.json', STEM + '.ignore']
    for name in names + ['build_engram_fault.py']:
        (out / 'inputs' / name).write_bytes((ROOT / name).read_bytes())
    (out / 'BUILD-STATUS').write_text('building diagnostic; not qualified\n')
    subprocess.run(['scp', *[str(ROOT / n) for n in names], 'dusty:' + REMOTE + '/'], check=True)
    cache = 'ds41-engram-fault-' + digest[:20]
    command = ['podman', 'build', '--pull=never', '--network=none', '--format=docker', '--timestamp=0',
        '--tag', 'localhost/voipmonitor/build-components:ds41-engram-fault',
        '--iidfile', REMOTE + '/engram-fault-image.id', '--ignorefile', REMOTE + '/' + STEM + '.ignore',
        '-f', REMOTE + '/Dockerfile.' + STEM, '--build-arg', 'INJECT_LOCK=' + digest,
        '--build-arg', 'INJECT_CACHE=' + cache, REMOTE]
    try:
        with (out / 'build.log').open('x') as log:
            process = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', 'dusty', shlex.join(command)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
            code = process.wait()
        (out / 'build.exit-code').write_text(str(code) + '\n')
        if code:
            raise RuntimeError('Diagnostic build failed')
        image = ssh('dusty', 'cat ' + REMOTE + '/engram-fault-image.id').strip().removeprefix('sha256:')
        raw = ssh('dusty', 'podman image inspect ' + image)
        (out / 'image-inspect.json').write_text(raw)
        info = json.loads(raw)[0]
        inherited = base['RootFS']['Layers']
        # Podman Parent can identify the preceding build step rather than FROM.
        # Require the exact base layer prefix, plus the pinned recipe/install
        # inventory, rather than asserting a false direct-parent contract.
        if not inherited or info['RootFS']['Layers'][:len(inherited)] != inherited:
            raise RuntimeError('Diagnostic layer ancestry differs')
        labels = info['Config']['Labels']
        expected = {'local-inference.ds41.diagnostic.kind': 'engram-fault-inject',
            'local-inference.ds41.diagnostic.lock.sha256': digest,
            'local-inference.ds41.diagnostic.base-image': lock['base_image_id'],
            'local-inference.ds41.diagnostic.base-kind': lock['base_kind'],
            'local-inference.ds41.diagnostic.base-lock.sha256': lock['base_lock_sha256'],
            'local-inference.ds41.diagnostic.vllm-cache': cache,
            'local-inference.status': 'diagnostic-only-not-qualified',
            **{c + '.source-tree': t for c, t in lock['trees'].items()}}
        if any(labels.get(k) != v for k, v in expected.items()):
            raise RuntimeError('Diagnostic image labels differ')
        if 'ENGRAM-FAULT-INJECT-INSTALL-PASS' not in (out / 'build.log').read_text().splitlines():
            raise RuntimeError('Missing diagnostic installation gate')
        # No GPU work: this checks the newly installed helper's import and ABI
        # surface. A live TP4 fault test, not this smoke, establishes fail-closed.
        smoke = ['podman', 'run', '--rm', '--pull=never', '--network=none',
            '--entrypoint', '/opt/venv/bin/python', image, '-c',
            'import vllm.models.deepseek_v4_1.claude_engram_fault_inject as f; '
            'f.parse_trigger(\'{"token":"smoketest","rank":3,"node":"kirby","min_tokens":1024}\'); '
            'print("ENGRAM-FAULT-HELPER-IMPORT-PASS")']
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'dusty', shlex.join(smoke)],
            capture_output=True, text=True, timeout=120)
        (out / 'import.log').write_text(result.stdout + result.stderr)
        if result.returncode or 'ENGRAM-FAULT-HELPER-IMPORT-PASS' not in result.stdout.splitlines():
            raise RuntimeError('Diagnostic helper import failed')
        (out / 'BUILD-OK').write_text(image + '\n')
        (out / 'BUILD-STATUS').write_text('installation gates passed; live injected-fault gate pending\n')
        (ROOT / 'receipts/engram-fault-build-receipt.json').write_text(json.dumps({
            'directory': str(out), 'image_id': image, 'lock_sha256': digest,
            'scope': 'diagnostic installation only; no GPU or model qualification'}, indent=2) + '\n')
        print('ENGRAM-FAULT-DIAGNOSTIC-BUILT', image, flush=True)
    except BaseException:
        (out / 'BUILD-STATUS').write_text('failed; diagnostic artifact retained if created\n')
        raise


if __name__ == '__main__':
    main()
