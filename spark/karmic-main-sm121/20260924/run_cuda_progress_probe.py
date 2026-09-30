"""Run the bounded isolated probe on idle dusty and preserve each process log."""
import datetime
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument('--repeats', type=int, default=1)
p.add_argument('--arms', default='no-mm,cold-mm,warm-mm,prequeued-cold-mm')
args = p.parse_args()
if not 1 <= args.repeats <= 10:
    p.error('repeats must be in 1..10')
arms = args.arms.split(',')
if not arms or any(a not in {'no-mm', 'cold-mm', 'warm-mm', 'prequeued-cold-mm',
                             'host-callback-cold-mm'} for a in arms):
    p.error('Unknown arm')
remote = '/home/jugs/git/ds41-r38/karmic-main-20260924/probe_cuda_producer_progress.py'
image = '32f93547a3c50a8b72ecfcbd8f0f2f0685bff34cfb4213b77748db80b5a6bebb'
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
out = root / 'receipts' / ('cuda-producer-progress-' + stamp)
out.mkdir(exist_ok=False)
(out / 'source.sha256').write_text(hashlib.sha256((root / 'probe_cuda_producer_progress.py').read_bytes()).hexdigest())


def ssh(command):
    return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', 'dusty', command], text=True)


if ssh('podman ps -q').strip():
    raise RuntimeError('dusty has running containers')
gpu_processes = ssh('nvidia-smi --query-compute-apps=pid --format=csv,noheader')
if gpu_processes.strip():
    raise RuntimeError('dusty has active GPU processes')
info = json.loads(ssh('podman image inspect ' + image))
if info[0]['Id'].removeprefix('sha256:') != image:
    raise RuntimeError('Wrong image identity')
(out / 'image.json').write_text(json.dumps(info, indent=2))
subprocess.run(['scp', str(root / 'probe_cuda_producer_progress.py'), 'dusty:' + remote], check=True)
native = str(Path(remote).with_name('progress_host_callback.so'))
if 'host-callback-cold-mm' in arms:
    native_source = str(Path(remote).with_name('progress_host_callback.c'))
    subprocess.run(['scp', str(root / 'progress_host_callback.c'), 'dusty:' + native_source], check=True)
    (out / 'native-source.sha256').write_text(hashlib.sha256((root / 'progress_host_callback.c').read_bytes()).hexdigest())
    (out / 'compiler.txt').write_text(ssh('cc --version'))
    build_command = shlex.join(['cc', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
                               '-shared', '-fPIC', native_source, '-o', native])
    (out / 'native-build-command.txt').write_text(build_command)
    (out / 'native-build.log').write_text(ssh(build_command))
    (out / 'native.sha256').write_text(ssh(shlex.join(['sha256sum', native])))
for repetition, arm in ((r, a) for r in range(args.repeats)
                        for a in arms):
    if ssh('podman ps -q').strip():
        raise RuntimeError('dusty became busy')
    command = ['podman', 'run', '--rm', '--pull=never', '--network=none',
               '--device', 'nvidia.com/gpu=all', '-e', 'CUDA_MODULE_LOADING=LAZY',
               '-e', 'PYTHONUNBUFFERED=1', '-v', remote + ':/probe.py:ro']
    if arm == 'host-callback-cold-mm':
        command += ['-v', native + ':/progress_host_callback.so:ro']
    command += ['--entrypoint', '/opt/venv/bin/python', image, '/probe.py', '--arm', arm]
    name = f'{repetition}-{arm}'
    (out / (name + '-command.json')).write_text(json.dumps(command, indent=2))
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'dusty', shlex.join(command)],
                            text=True, capture_output=True)
    (out / (name + '.log')).write_text(result.stdout + result.stderr)
    (out / (name + '.exit')).write_text(str(result.returncode))
    print(name, result.returncode, result.stdout, result.stderr, flush=True)
    result.check_returncode()
print('CUDA-PROGRESS-RECEIPTS', out, flush=True)
