"""Run the reviewed broader kernel gate on the build-gated, idle candidate."""
import hashlib
import argparse
import json
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
arm = parser.add_mutually_exclusive_group()
arm.add_argument('--nonatomic-control', action='store_true')
arm.add_argument('--live-winners', action='store_true',
                 help='Test the retained fresh Turbo=0 serving selections, not mapped controls')
parser.add_argument('--dry-run', action='store_true')
args = parser.parse_args()
remote = '/home/jugs/git/ds41-r38/karmic-main-20260924'
receipt = json.loads((root / 'receipts/dense-release-build-receipt.json').read_text())
image = receipt['image_id']
suffix = '-live-winners' if args.live_winners else ('-nonatomic' if args.nonatomic_control else '')
stem = 'dense-release-shared-complete-' + image[:12] + suffix
tuning_name = ('dense-release-live-tuning-initial.json' if args.live_winners else
               ('non-atomic-complete-control-tuning.json' if args.nonatomic_control else 'combined-tuning.json'))
turbo = not (args.nonatomic_control or args.live_winners)
from claude_dense_regression import enumerate_winners
_, winners = enumerate_winners(root / 'receipts' / tuning_name, turbo=turbo)
expected_programs = 64 if args.nonatomic_control else 152
expected_cases = 120 if args.nonatomic_control else 296
if len(winners) != expected_programs:
    raise RuntimeError('Incomplete dense tuning inventory')
if args.dry_run:
    print(json.dumps({'image': image, 'receipt': tuning_name,
                      'receipt_sha256': hashlib.sha256((root / 'receipts' / tuning_name).read_bytes()).hexdigest(),
                      'turbo': turbo, 'programs': len(winners),
                      'expected_cases': expected_cases, 'output_stem': stem}, indent=2))
    raise SystemExit(0)
log = root / 'receipts' / (stem + '.log')
if log.exists():
    raise RuntimeError('Gate receipt exists; do not overwrite')
for node in ('dusty', 'toby', 'rusty', 'kirby'):
    for probe in ('podman ps -q', 'nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits'):
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', node, probe], capture_output=True, text=True, timeout=30)
        if result.returncode or result.stdout.strip():
            raise RuntimeError('Host not idle or probe failed: ' + node)
files = ['claude_dense_regression.py', 'claude_tuning_key_recon.py', 'launch_contract.py', 'upstream-launch.json']
subprocess.run(['scp', *[str(root / file) for file in files], f'dusty:{remote}/'], check=True)
subprocess.run(['scp', str(root / 'receipts' / tuning_name), f'dusty:{remote}/receipts/'], check=True)
command = ['podman', 'run', '--rm', '--pull=never', '--network=none', '--device', 'nvidia.com/gpu=all',
    '--name', 'ds41-dense-shared-regression', '-v', f'{remote}:/diag:ro',
    '-v', f'{remote}/receipts:/diag/receipts:rw', '-e', 'PYTHONUNBUFFERED=1',
    '-e', 'B12X_PRINT_COMPILE_PROGRESS=1', '--entrypoint', '/opt/venv/bin/python', image,
    '-u', '/diag/claude_dense_regression.py', '--receipt', '/diag/receipts/' + tuning_name,
    '--launch-env', 'dusty', '--repeats', '64', '--rows-per-capacity', '2', '--cpasync-attempt',
    '--report', f'/diag/receipts/{stem}.json']
if not turbo:
    command += ['--env', 'B12X_DENSE_SPLITK_TURBO=0']
(root / 'receipts' / (stem + '-command.json')).write_text(json.dumps({
    'command': command, 'inputs': {file: hashlib.sha256((root / file).read_bytes()).hexdigest()
                                for file in files + ['receipts/' + tuning_name]}}, indent=2) + '\n')
with log.open('x') as stream:
    process = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', 'dusty', shlex.join(command)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        stream.write(line)
        stream.flush()
        print(line, end='', flush=True)
    code = process.wait()
subprocess.run(['scp', f'dusty:{remote}/receipts/{stem}.json', str(root / 'receipts')], check=True)
if code:
    raise SystemExit(code)
report = json.loads((root / 'receipts' / (stem + '.json')).read_text())
if (report['dense_source_sha256'] != json.loads((root / 'dense-release.lock.json').read_text())['output_sha256']
        or report['repeats'] != 64 or len(report['cases']) != expected_cases
        or report['split_k_turbo'] != turbo
        or any(case['status'] != 'PASS' for case in report['cases'])):
    raise RuntimeError('Shared dense regression incomplete or failed')
print('DENSE-RELEASE-SHARED-PASS', flush=True)
