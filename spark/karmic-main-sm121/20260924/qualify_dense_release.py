"""Correctness and exact repeatability gates for the uninstrumented repair."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from runtime import kit_digest

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
pin = json.loads((root / 'candidate.json').read_text())
digest = kit_digest()
if pin.get('diagnostic', {}).get('kind') not in ('dense-release-fence', 'compressor-ring-mapping-fix', 'engram-prequeued-native-io-fix', 'router-stage-release-candidate', 'precision-release-candidate') or 'diagnostic_overlay' in pin:
    raise RuntimeError('Not the uninstrumented dense release candidate')
def inspect_node(node, phase):
    raw = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', node,
        'podman inspect ds41-flash-karmic-main-tp4'], text=True)
    info = json.loads(raw)[0]
    (args.out / f'{node}-identity-{phase}.json').write_text(raw)
    if (not info['State']['Running'] or info['Image'].removeprefix('sha256:') != pin['image_id']
            or info['Config']['Labels'].get('local-inference.ds41.kit.sha256') != digest):
        raise RuntimeError('Serving identity mismatch: ' + node)
    env = dict(item.split('=', 1) for item in info['Config']['Env'])
    if 'DS41_ROUTER_CONTROL_BLOCKS' in env:
        raise RuntimeError('Fixed-capacity diagnostic is not a serving qualification: ' + node)
    if env.get('B12X_DYNAMIC_DETERMINISTIC_OUTPUT') != '1':
        raise RuntimeError('Deterministic MoE is not enabled: ' + node)
    if env.get('B12X_DENSE_SPLITK_TURBO') != '0' or 'VLLM_DS41_L2_PREFETCH' in env:
        raise RuntimeError('Dense reduction/prefetch profile mismatch: ' + node)
    return {key: info[key] for key in ('Id', 'Image', 'RestartCount')} | {
        'StartedAt': info['State']['StartedAt']}

identities = {node: inspect_node(node, 'before') for node in ('dusty', 'toby', 'rusty', 'kirby')}


def run(script, options, marker, label):
    command = [sys.executable, '-u', str(root / script), *map(str, options)]
    (args.out / f'{label}-command.json').write_text(json.dumps({
        'command': command, 'script_sha256': hashlib.sha256((root / script).read_bytes()).hexdigest(),
        'cache_metrics_sha256': hashlib.sha256((root / 'cache_metrics.py').read_bytes()).hexdigest()}, indent=2) + '\n')
    print('QUALIFICATION-START', label, flush=True)
    with (args.out / f'{label}.log').open('x') as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end='', flush=True)
        code = process.wait()
    if code or marker not in (args.out / f'{label}.log').read_text().splitlines():
        raise RuntimeError('Qualification failed: ' + label)
    print('QUALIFICATION-PASS', label, flush=True)


run('probe_repeatability.py', ['--corpus', root / 'receipts/determinism-corpus.json',
    '--out', args.out / 'repeatability', '--lengths',
    '128,129,160,192,255,256,257,258,259,260,261,262,263,264,265,272,288,320,384,385,400,448,512,513,514,515,1024,16384',
    '--repeats', '6'], 'REPEATABILITY-PASS', 'repeatability')
run('qualify_upstream.py', ['--out', args.out / 'semantic'], 'DS41-QUALIFICATION-PASS', 'semantic')
run('qualify_original_needle.py', ['--input', root.parents[1] / 'ds41/r38/receipts/20260916/admission-k7-1m-u80/needle-524288-input.json',
    '--out', args.out / 'original-524k', '--repeats', '3'], 'ORIGINAL-NEEDLE-PASS', 'original-524k')
# Longer forwards exercise repeated 8192-row projections, beyond the short
# router reproducer. Keep each length's cold repetitions independently recorded.
for length, repeats in ((131072, 8), (524288, 4)):
    label = f'long-repeatability-{length}'
    run('probe_repeatability.py', ['--corpus', root / 'receipts/determinism-corpus.json',
        '--out', args.out / label, '--lengths', str(length), '--repeats', str(repeats)],
        'REPEATABILITY-PASS', label)
run('qualify_upstream.py', ['--out', args.out / 'context', '--long', '--dual-needle',
    '--needle-lengths', '262000,500000,599936'], 'DS41-QUALIFICATION-PASS', 'context')
for node, before in identities.items():
    if inspect_node(node, 'after') != before:
        raise RuntimeError('Container restarted or changed during qualification: ' + node)
print('DENSE-RELEASE-CORRECTNESS-PASS: concurrency/cache and benchmark remain separate gates', flush=True)
