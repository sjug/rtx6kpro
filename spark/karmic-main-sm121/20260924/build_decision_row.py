"""Build only the reviewed decision-row capture diagnostic, on idle DS4.1 nodes.

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
STEM = 'claude-decision-row'
SMOKE = '''
import hashlib, json
from pathlib import Path
import torch
import vllm.models.deepseek_v4_1.attention as attention
import vllm.models.deepseek_v4_1.claude_decision_row as helper
if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
    raise RuntimeError('A visible GB10 is required')
if attention._claude_row is not helper or not callable(helper.begin):
    raise RuntimeError('Attention does not use the capture helper')
if helper.SCHEMA != 'claude-decision-row-v3':
    raise RuntimeError('Unexpected capture schema')
root = Path('/opt/jovian-judgement')
lock = json.loads(Path('/opt/ds41-decision-row/claude-decision-row.lock.json').read_text())
for module, target in [(attention, 'vllm/vllm/models/deepseek_v4_1/attention.py'),
                       (helper, 'vllm/vllm/models/deepseek_v4_1/claude_decision_row.py')]:
    path = Path(module.__file__).resolve()
    if path != (root / target).resolve():
        raise RuntimeError('Wrong loaded module path: ' + str(path))
    if hashlib.sha256(path.read_bytes()).hexdigest() != lock['targets'][target]['output_sha256']:
        raise RuntimeError('Wrong loaded module bytes: ' + str(path))
print('DECISION-ROW-HELPER-IMPORT-PASS')
'''


def frozen_inputs(root=ROOT, stem=STEM):
    raw = (root / (stem + '.lock.json')).read_bytes()
    lock = json.loads(raw)
    for name, expected in lock['inputs'].items():
        if Path(name).name != name or hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Diagnostic input drift: ' + name)
    expected_ignore = '**\n' + ''.join('!' + n + '\n' for n in
        sorted(lock['inputs']) + [stem + '.lock.json'])
    if (root / (stem + '.ignore')).read_text() != expected_ignore:
        raise RuntimeError('Diagnostic build context differs')
    base_receipt = {'ds41-indexer': 'window', 'ds41-precision': 'indexer'}.get(stem, 'router-release')
    base = json.loads((root / f'receipts/{base_receipt}-build-receipt.json').read_text())
    if base['image_id'] != lock['base_image_id'] or base['lock_sha256'] != lock['base_lock_sha256']:
        raise RuntimeError('Capture diagnostic is not based on the router candidate')
    if hashlib.sha256((root / lock['base_lock']).read_bytes()).hexdigest() != lock['base_lock_sha256']:
        raise RuntimeError('Local router lock differs from the declared base')
    return lock, hashlib.sha256(raw).hexdigest()


def main(kind='decision-row', smoke=SMOKE, install_marker='DECISION-ROW-INSTALL-PASS',
         smoke_marker='DECISION-ROW-HELPER-IMPORT-PASS', wrapper=None):
    if kind not in ('decision-row', 'window', 'indexer', 'precision'):
        raise ValueError('Unsupported capture build kind')
    stem = 'ds41-' + kind if kind in ('indexer', 'precision') else 'claude-' + kind
    lock, digest = frozen_inputs(stem=stem)
    idle()
    base = json.loads(ssh('dusty', 'podman image inspect ' + lock['base_image_id']))[0]
    labels = base['Config']['Labels']
    if (base['Id'].removeprefix('sha256:') != lock['base_image_id']
            or labels.get('local-inference.ds41.diagnostic.kind') != lock['base_kind']
            or labels.get('local-inference.ds41.diagnostic.lock.sha256') != lock['base_lock_sha256']
            or any(labels.get(c + '.source-tree') != t for c, t in lock['base_trees'].items())):
        raise RuntimeError('Installed diagnostic base provenance differs')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = ROOT / 'receipts' / (kind + '-build-' + stamp)
    (out / 'inputs').mkdir(parents=True, exist_ok=False)
    names = [*lock['inputs'], stem + '.lock.json', stem + '.ignore']
    for name in names + ['build_decision_row.py'] + ([wrapper] if wrapper else []):
        (out / 'inputs' / name).write_bytes((ROOT / name).read_bytes())
    (out / 'BUILD-STATUS').write_text('building diagnostic; not qualified\n')
    subprocess.run(['scp', *[str(ROOT / n) for n in names], 'dusty:' + REMOTE + '/'], check=True)
    cache = 'ds41-' + kind + '-' + digest[:20]
    command = ['podman', 'build', '--pull=never', '--network=none', '--format=docker', '--timestamp=0',
        '--tag', 'localhost/voipmonitor/build-components:ds41-' + kind,
        '--iidfile', REMOTE + '/' + kind + '-image.id', '--ignorefile', REMOTE + '/' + stem + '.ignore',
        '-f', REMOTE + '/Dockerfile.' + stem, '--build-arg', 'DECISION_LOCK=' + digest,
        '--build-arg', 'DECISION_CACHE=' + cache, REMOTE]
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
        image = ssh('dusty', 'cat ' + REMOTE + '/' + kind + '-image.id').strip().removeprefix('sha256:')
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
        expected = {'local-inference.ds41.diagnostic.kind': kind + '-capture',
            'local-inference.ds41.diagnostic.lock.sha256': digest,
            'local-inference.ds41.diagnostic.base-image': lock['base_image_id'],
            'local-inference.ds41.diagnostic.base-kind': lock['base_kind'],
            'local-inference.ds41.diagnostic.base-lock.sha256': lock['base_lock_sha256'],
            'local-inference.ds41.diagnostic.vllm-cache': cache,
            'local-inference.status': 'diagnostic-only-not-qualified',
            **{c + '.source-tree': t for c, t in lock['trees'].items()}}
        if any(labels.get(k) != v for k, v in expected.items()):
            raise RuntimeError('Diagnostic image labels differ')
        if install_marker not in (out / 'build.log').read_text().splitlines():
            raise RuntimeError('Missing diagnostic installation gate')
        # Verify the actual GPU-platform attention import and its linked helper,
        # not just the standalone helper. Live capture remains a separate gate.
        idle()
        smoke_command = ['podman', 'run', '--rm', '--pull=never', '--network=none',
            '--device', 'nvidia.com/gpu=all',
            '--entrypoint', '/opt/venv/bin/python', image, '-c', smoke]
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'dusty', shlex.join(smoke_command)],
            capture_output=True, text=True, timeout=300)
        (out / 'import.log').write_text(result.stdout + result.stderr)
        if result.returncode or smoke_marker not in result.stdout.splitlines():
            raise RuntimeError('Diagnostic helper import failed')
        (out / 'BUILD-OK').write_text(image + '\n')
        (out / 'BUILD-STATUS').write_text('installation gates passed; live capture and reference audit pending\n')
        (ROOT / ('receipts/' + kind + '-build-receipt.json')).write_text(json.dumps({
            'directory': str(out), 'image_id': image, 'lock_sha256': digest,
            'scope': 'diagnostic installation and GPU-visible imports; no model qualification'}, indent=2) + '\n')
        print('DECISION-ROW-DIAGNOSTIC-BUILT', image, flush=True)
    except BaseException:
        (out / 'BUILD-STATUS').write_text('failed; diagnostic artifact retained if created\n')
        raise


if __name__ == '__main__':
    main()
