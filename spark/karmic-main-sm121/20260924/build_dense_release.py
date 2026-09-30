"""Build on idle dusty, retain rejected candidates, publish only after gates."""
import datetime
import hashlib
import json
from pathlib import Path
import socket
import shutil
import subprocess
import sys

root = Path(__file__).resolve().parent
if socket.gethostname().split('.')[0] != 'dusty':
    raise RuntimeError('Build requires dusty')
lock_bytes = (root / 'dense-release.lock.json').read_bytes()
lock = json.loads(lock_bytes)
lock_sha = hashlib.sha256(lock_bytes).hexdigest()
cache = Path.home() / '.cache/vllm-jj-ds41-tp4'
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
out = root / 'receipts' / ('dense-release-build-' + stamp)


def failed(kind, value, traceback):
    if out.exists():
        (out / 'BUILD-STATUS').write_text(f'failed; not qualified: {kind.__name__}: {value}\n')
    sys.__excepthook__(kind, value, traceback)


sys.excepthook = failed


def idle():
    for node in ('dusty', 'toby', 'rusty', 'kirby'):
        for probe in (['podman', 'ps', '-q'], ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits']):
            command = probe if node == 'dusty' else ['ssh', '-o', 'BatchMode=yes', node, *probe]
            result = subprocess.run(command, text=True, capture_output=True, timeout=30)
            if result.returncode or result.stdout.strip():
                raise RuntimeError(f'{node}: idle check failed: {result.returncode} {result.stdout} {result.stderr}')


def run(command, name):
    with (out / name).open('x') as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end='', flush=True)
        if process.wait():
            raise RuntimeError(f'{name} failed; candidate and receipt retained at {out}')


for section in ('inputs', 'build_inputs'):
    for name, digest in lock[section].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Recipe drift: ' + name)
for node in ('dusty', 'kirby'):
    for length in (129, 160, 192):
        label = 'fence-before-release-confirm' if (node, length) == ('kirby', 129) else f'release-matrix-m{length}'
        receipt = root / 'receipts' / f'engram-dense-isolated-{node}-{label}.json'
        data = json.loads(receipt.read_text())
        reference = {
            129: '97399c9db1e5c91dbaa589aeee5f4147b852c681b66bb3e5d7ed70b9527f1846',
            160: '916beab78ce52fd6487b119ca7364049b1e597db8cfddba002ddea2d8487d3cb',
            192: '29b5a61994edf5da3c6c7ba4c70a22b444fb42caf6a3a87fdad7bf1856dbaabb',
        }[length]
        reference_key = 'dense_replay_synchronized' if length == 192 else 'projected_kv'
        # Kirby's original 129-row confirmation predates the explicit selector
        # field. Its recorded reference digest already pins projected_kv.
        actual_reference_key = data.get('reference_key')
        if actual_reference_key is None and (node, length) == ('kirby', 129):
            actual_reference_key = 'projected_kv'
        if (data['trace'] != f'/cache/ds41-attention-trace/replay-engram-{length}-0-{node}.pt'
                or actual_reference_key != reference_key or data['reference_sha256'] != reference
                or data['dense_source_sha256'] != lock['output_sha256'] or len(data['results']) != 4096
                or not all(r['same_as_serving'] and r['same_as_first'] and r['nan_elements'] == 0 for r in data['results'])):
            raise RuntimeError('Incomplete isolated gate: ' + str(receipt))
idle()
base = json.loads(subprocess.check_output(['podman', 'image', 'inspect', lock['base_image_id']]))[0]
if base['Id'].removeprefix('sha256:') != lock['base_image_id'] or base['Config']['Labels']['b12x.source-tree'] != lock['base_b12x_tree']:
    raise RuntimeError('Base image or source tree mismatch')
out.mkdir(exist_ok=False)
(out / 'inputs').mkdir()
for name in set(lock['inputs']) | set(lock['build_inputs']):
    shutil.copyfile(root / name, out / 'inputs' / name)
(out / 'source.lock.json').write_bytes(lock_bytes)
(out / 'BUILD-STATUS').write_text('building; not qualified\n')
component = 'localhost/voipmonitor/build-components:ds41-dense-release'
run(['podman', 'build', '--pull=never', '--network=none', '--format=docker',
     '--tag', component, '--iidfile', str(out / 'image.id'),
     '--ignorefile', str(root / 'dense-release.ignore'), '-f', str(root / 'Dockerfile.dense-release'),
     '--build-arg', 'DENSE_RELEASE_TREE=' + lock['b12x_tree'],
     '--build-arg', 'DENSE_RELEASE_LOCK=' + lock_sha,
     '--build-arg', 'DENSE_RELEASE_CACHE=ds41-release-' + lock_sha[:20], str(root)], 'build.log')
image = (out / 'image.id').read_text().strip().removeprefix('sha256:')
inspection = subprocess.check_output(['podman', 'image', 'inspect', image])
(out / 'image-inspect.json').write_bytes(inspection)
labels = json.loads(inspection)[0]['Config']['Labels']
if labels['b12x.source-tree'] != lock['b12x_tree'] or labels['local-inference.ds41.diagnostic.lock.sha256'] != lock_sha:
    raise RuntimeError('Built labels mismatch')
idle()
common = ['podman', 'run', '--rm', '--pull=never', '--network=none', '--device', 'nvidia.com/gpu=all',
          '-v', f'{root}:/diag:ro', '-e', 'PYTHONUNBUFFERED=1', '-e', 'B12X_PRINT_COMPILE_PROGRESS=1']
for script, marker in [('test_moe_dependency.py', 'DETERMINISTIC-MOE-DEPENDENCY-PASS'),
                       ('claude_test_topk_tiebreak.py', 'CLAUDE-TOPK-TIEBREAK-PASS checks=10598')]:
    run(common + ['--entrypoint', '/opt/venv/bin/python', image, '-u', '/diag/' + script], script + '.log')
    if marker not in (out / (script + '.log')).read_text().splitlines():
        raise RuntimeError('Missing exact gate marker: ' + marker)
key = '68f67353dbff5c4b5760c3c24ce3a50624366e2e4735b23440b6c751a1b82069'
relative_manifest = f'jit/ds41-trace-9e561cc431981197a101/b12x/68/{key}.json'
report = f'/cache/ds41-attention-trace/dense-release-built-{stamp}.json'
run(common + ['-v', f'{cache}:/cache',
    '-v', f'{Path.home()}/.cache/huggingface:/root/.cache/huggingface:ro',
    '--entrypoint', '/opt/venv/bin/python', image, '-u', '/diag/probe_engram_dense_replay.py',
    '--trace', '/cache/ds41-attention-trace/replay-engram-129-0-dusty.pt',
    '--manifest', '/cache/' + relative_manifest, '--report', report,
    '--reference-sha256', '97399c9db1e5c91dbaa589aeee5f4147b852c681b66bb3e5d7ed70b9527f1846',
    '--source-sha256', lock['output_sha256'], '--repeats', '4096', '--delay-ms', '50',
    '--checkpoint', '/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/fb2764a5cf321eaa5070ca8f9e892818f477c16d/model-00047-of-00048.safetensors'], 'dense-replay.log')
data = json.loads((cache / report.removeprefix('/cache/')).read_text())
(out / 'dense-replay.json').write_text(json.dumps(data, indent=2) + '\n')
compiled_key = data['programs'][0]['key']
object_path = (cache / relative_manifest).parent.parent / compiled_key[:2] / (compiled_key + '.o')
run([sys.executable, str(root / 'extract_dense_cubin.py'), str(object_path), str(out / 'dense.cubin')], 'extract.log')
sass = subprocess.check_output(['podman', 'run', '--rm', '--pull=never', '--network=none',
    '-v', f'{out}:/receipt:ro', '--entrypoint', '/usr/local/cuda/bin/cuobjdump',
    image, '--dump-sass', '/receipt/dense.cubin'])
(out / 'dense.sass').write_bytes(sass)
run([sys.executable, str(root / 'verify_dense_release_sass.py'), str(out / 'dense.sass')], 'sass-gate.log')
tag = 'localhost/voipmonitor/vllm:karmic-main-ds41-deterministic-spark-sm121'
subprocess.run(['podman', 'tag', image, tag], check=True)
(out / 'BUILD-OK').write_text(image + '\n')
(out / 'BUILD-STATUS').write_text('build gates passed; model qualification pending\n')
(root / 'receipts/dense-release-build-receipt.json').write_text(json.dumps({'directory': str(out), 'image_id': image}) + '\n')
print('DENSE-RELEASE-BUILD-OK', image, out, flush=True)
