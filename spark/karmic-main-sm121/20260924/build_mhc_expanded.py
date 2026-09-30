"""Build only the real-input mHC diagnostic; never launch or promote it."""
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from build_activation_trace import idle, ssh, REMOTE

ROOT = Path(__file__).resolve().parent
STEM = 'claude-mhc-expanded'
KIND = 'mhc-expanded'
SMOKE = '''
import hashlib, importlib, json
from pathlib import Path
import torch
if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
    raise RuntimeError('A visible GB10 is required')
root = Path('/opt/jovian-judgement')
lock = json.loads(Path('/opt/ds41-mhc-expanded/claude-mhc-expanded.lock.json').read_text())
loaded = {}
for target, entry in lock['targets'].items():
    if not target.startswith('vllm/vllm/') or not target.endswith('.py'):
        raise RuntimeError('Unexpected diagnostic target')
    name = target.removeprefix('vllm/').removesuffix('.py').replace('/', '.')
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve()
    if path != (root / target).resolve():
        raise RuntimeError('Wrong loaded path: ' + name)
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry['output_sha256']:
        raise RuntimeError('Wrong loaded bytes: ' + name)
    loaded[name] = module
model = loaded['vllm.models.deepseek_v4_1.nvidia.model']
helper = loaded['vllm.models.deepseek_v4_1.claude_mhc_expanded']
if model._mhc_expanded is not helper or not callable(helper.install):
    raise RuntimeError('Model does not reference the capture helper')
from vllm.forward_context import get_forward_context
from b12x.preparation.types import Selection, Plan
for name in ('component_id', 'query', 'config', 'source'):
    if name not in Selection.__dataclass_fields__:
        raise RuntimeError('Missing Selection field: ' + name)
if not isinstance(Plan.selection, property):
    raise RuntimeError('Plan.selection contract changed')
runtime = torch.cuda.cudart()
storage = torch.empty(8192, dtype=torch.uint8)
offset = (-storage.data_ptr()) % 4096
address = storage.data_ptr() + offset
if int(runtime.cudaHostRegister(address, 4096, 0)) != 0:
    raise RuntimeError('Host registration failed')
try:
    device = torch.arange(1024, dtype=torch.int32, device='cuda')
    storage[offset:offset + 4096].copy_(device.view(torch.uint8), non_blocking=True)
    torch.cuda.synchronize()
    host = storage[offset:offset + 4096].view(torch.int32)
    if not torch.equal(host, device.cpu()) or not torch.equal(helper.digest(host), helper.digest(device).cpu()):
        raise RuntimeError('Registered copy or device digest failed')
finally:
    if int(runtime.cudaHostUnregister(address)) != 0:
        raise RuntimeError('Host unregistration failed')
print('MHC-EXPANDED-IMPORT-PASS')
'''


def frozen(root=ROOT):
    raw = (root / (STEM + '.lock.json')).read_bytes()
    lock = json.loads(raw)
    for name, expected in lock['inputs'].items():
        if Path(name).name != name or hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Diagnostic input drift: ' + name)
    base = json.loads((root / 'receipts/precision-release-build-receipt.json').read_text())
    if base['image_id'] != lock['base_image_id'] or base['lock_sha256'] != lock['base_lock_sha256']:
        raise RuntimeError('Base build receipt differs from the diagnostic lock')
    if hashlib.sha256((root / lock['base_lock']).read_bytes()).hexdigest() != lock['base_lock_sha256']:
        raise RuntimeError('Base lock differs')
    return lock, hashlib.sha256(raw).hexdigest()


def main():
    lock, digest = frozen()
    idle()
    base = json.loads(ssh('dusty', 'podman image inspect ' + lock['base_image_id']))[0]
    labels = base['Config']['Labels']
    if (base['Id'].removeprefix('sha256:') != lock['base_image_id']
            or labels.get('local-inference.ds41.diagnostic.lock.sha256') != lock['base_lock_sha256']
            or labels.get('local-inference.ds41.diagnostic.kind') != lock['base_kind']
            or any(labels.get(c + '.source-tree') != t for c, t in lock['base_trees'].items())):
        raise RuntimeError('Installed base identity differs')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = ROOT / 'receipts' / (KIND + '-build-' + stamp)
    (out / 'inputs').mkdir(parents=True, exist_ok=False)
    names = [*lock['inputs'], STEM + '.lock.json']
    for name in names + ['build_mhc_expanded.py', 'build_activation_trace.py']:
        (out / 'inputs' / name).write_bytes((ROOT / name).read_bytes())
    ignore = out / (STEM + '.ignore')
    ignore.write_text('**\n' + ''.join('!' + n + '\n' for n in sorted(names)))
    (out / 'BUILD-STATUS').write_text('building diagnostic; not qualified\n')
    try:
        subprocess.run(['scp', *[str(ROOT / n) for n in names], str(ignore), 'dusty:' + REMOTE + '/'], check=True)
        command = ['podman', 'build', '--pull=never', '--network=none', '--format=docker', '--timestamp=0',
                   '--tag', 'localhost/voipmonitor/build-components:ds41-' + KIND,
                   '--iidfile', REMOTE + '/' + KIND + '-image.id',
                   '--ignorefile', REMOTE + '/' + STEM + '.ignore',
                   '-f', REMOTE + '/Dockerfile.' + STEM, '--build-arg', 'MHC_EXPANDED_LOCK=' + digest, REMOTE]
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
        image = ssh('dusty', 'cat ' + REMOTE + '/' + KIND + '-image.id').strip().removeprefix('sha256:')
        raw = ssh('dusty', 'podman image inspect ' + image)
        (out / 'image-inspect.json').write_text(raw)
        info = json.loads(raw)[0]
        layers = base['RootFS']['Layers']
        if not layers or info['RootFS']['Layers'][:len(layers)] != layers:
            raise RuntimeError('Base layer ancestry differs')
        expected = {'local-inference.ds41.diagnostic.kind': lock['kind'],
                    'local-inference.ds41.diagnostic.lock.sha256': digest,
                    'local-inference.ds41.diagnostic.base-image': lock['base_image_id'],
                    'local-inference.status': 'diagnostic-only-not-qualified',
                    'b12x.source-tree': lock['base_trees']['b12x']}
        if any(info['Config']['Labels'].get(k) != v for k, v in expected.items()):
            raise RuntimeError('Diagnostic label mismatch')
        if info['Config']['Env'] != base['Config']['Env']:
            raise RuntimeError('Diagnostic changed the base environment')
        if lock['install_pass_marker'] not in (out / 'build.log').read_text().splitlines():
            raise RuntimeError('Missing installation gate')
        idle()
        command = ['podman', 'run', '--rm', '--pull=never', '--network=none', '--device', 'nvidia.com/gpu=all',
                   '--entrypoint', '/opt/venv/bin/python', image, '-c', SMOKE]
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'dusty', shlex.join(command)],
                                capture_output=True, text=True, timeout=300)
        (out / 'import.log').write_text(result.stdout + result.stderr)
        if result.returncode or 'MHC-EXPANDED-IMPORT-PASS' not in result.stdout.splitlines():
            raise RuntimeError('GPU-visible import gate failed')
        (out / 'BUILD-OK').write_text(image + '\n')
        (out / 'BUILD-STATUS').write_text('diagnostic installation passed; live capture not yet verified\n')
        (ROOT / 'receipts/mhc-expanded-build-receipt.json').write_text(json.dumps({
            'directory': str(out), 'image_id': image, 'lock_sha256': digest}, indent=2) + '\n')
        print('MHC-EXPANDED-BUILD-OK', image, flush=True)
    except BaseException:
        (out / 'BUILD-STATUS').write_text('failed; diagnostic artifacts retained\n')
        raise


if __name__ == '__main__':
    main()
