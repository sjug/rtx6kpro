"""Run the isolated MoE probe with one retained boot's image and kernel environment.

All four DS4.1 hosts must be idle. This is diagnostic, never a serving gate.
The probe itself owns argument validation and frozen-plan engagement checks.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from build_activation_trace import idle, ssh, REMOTE

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--node', choices=('dusty', 'toby', 'rusty', 'kirby'), required=True)
    parser.add_argument('--state', type=Path, required=True, help='Retained pre-stop podman inspect JSON')
    parser.add_argument('--tuning', required=True, help='Absolute tuning JSON path inside /cache')
    parser.add_argument('probe_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    extra = args.probe_args[1:] if args.probe_args[:1] == ['--'] else args.probe_args
    tuning = Path(args.tuning)
    if '..' in tuning.parts or not tuning.is_relative_to('/cache') or tuning.suffix != '.json':
        parser.error('Tuning path must be a JSON file under /cache')
    info = json.loads(args.state.read_text())
    if isinstance(info, list):
        info = info[0]
    pin = json.loads((ROOT / 'candidate.json').read_text())
    image = info['Image'].removeprefix('sha256:')
    if image != pin['image_id']:
        raise RuntimeError('Retained boot is not the pinned diagnostic image')
    if 'DS41_NODE=' + args.node not in info['Config']['Env']:
        raise RuntimeError('Retained boot belongs to a different node')
    mounts = [m for m in info['Mounts'] if m['Destination'] == '/cache']
    if len(mounts) != 1 or mounts[0]['Type'] != 'bind':
        raise RuntimeError('Expected exactly one cache bind')
    env = dict(e.split('=', 1) for e in info['Config']['Env'] if e.startswith((
        'B12X_', 'CUTE_', 'CUTLASS_', 'CUDA_', 'NVCC_', 'CC=', 'CXX=',
        'VLLM_CACHE_ROOT=', 'TORCHINDUCTOR_CACHE_DIR=')))
    for key, value in {'B12X_DYNAMIC_DETERMINISTIC_OUTPUT': '1',
                       'B12X_MOE_FORCE_A8': '1', 'B12X_DENSE_SPLITK_TURBO': '0'}.items():
        if env.get(key) != value:
            raise RuntimeError('Wrong retained diagnostic environment: ' + key)
    idle()
    now = datetime.datetime.now(datetime.timezone.utc)
    stamp = now.strftime('%Y%m%dT%H%M%SZ')
    since = now.strftime('%Y-%m-%d %H:%M:%S UTC')
    out = ROOT / 'receipts' / ('routed-isolation-' + args.node + '-' + stamp)
    out.mkdir(exist_ok=False)
    (out / 'retained-state.json').write_bytes(args.state.read_bytes())
    probe = ROOT / 'claude_probe_routed_moe.py'
    (out / probe.name).write_bytes(probe.read_bytes())
    (out / 'probe.sha256').write_text(hashlib.sha256(probe.read_bytes()).hexdigest() + '\n')
    host_tuning = str(Path(mounts[0]['Source']) / tuning.relative_to('/cache'))
    subprocess.run(['scp', args.node + ':' + host_tuning, str(out / 'tuning-before.json')], check=True)
    subprocess.run(['scp', str(probe), args.node + ':' + REMOTE + '/'], check=True)
    command = ['podman', 'run', '--rm', '--pull=never', '--network=none',
               '--name', 'ds41-routed-isolation-' + stamp.lower(),
               '--device', 'nvidia.com/gpu=all', '--ipc=host', '--ulimit', 'memlock=-1:-1',
               '-v', mounts[0]['Source'] + ':/cache', '-v', REMOTE + ':/diag', '-w', '/diag']
    for key, value in sorted(env.items()):
        command += ['-e', key + '=' + value]
    command += ['-e', 'PYTHONUNBUFFERED=1', '-e', 'B12X_PRINT_COMPILE_PROGRESS=1',
                '--entrypoint', '/opt/venv/bin/python', image, '-u',
                '/diag/claude_probe_routed_moe.py', '--tuning-receipt', str(tuning), *extra]
    (out / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    idle()
    with (out / 'run.log').open('x') as log:
        process = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', args.node, shlex.join(command)],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end='', flush=True)
        code = process.wait()
    (out / 'exit-code').write_text(str(code) + '\n')
    (out / 'kernel.log').write_text(ssh(args.node, shlex.join(
        ['journalctl', '-k', '--since', since, '--no-pager'])))
    subprocess.run(['scp', args.node + ':' + host_tuning, str(out / 'tuning-after.json')], check=True)
    print('ROUTED-ISOLATION-RECEIPTS', out, flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
