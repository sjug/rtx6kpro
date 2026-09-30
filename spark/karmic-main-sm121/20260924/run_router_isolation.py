"""Run a router replay on an idle Spark with the retained capture boot environment."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from build_activation_trace import idle, ssh, REMOTE

ROOT = Path(__file__).resolve().parent
TUNING = '/cache/jit/ds41-engram-b3e4f0ada602fab6f3ce/b12x/preparation/6115b03c7a814701d5610b3e6aecf82e30fdbb46a122d70441228538b9bd1e5e.json'
CAPTURE = '/cache/ds41-act-trace/dusty-524-g1-capture.pt'
SNAPSHOT = '/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/fb2764a5cf321eaa5070ca8f9e892818f477c16d'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--fenced', action='store_true')
    parser.add_argument('--repeats', type=int, default=10000)
    parser.add_argument('--side-iters', type=int, default=16)
    parser.add_argument('--stop-after', type=int, default=0)
    args = parser.parse_args()
    info = json.loads(args.state.read_text())
    if isinstance(info, list):
        info = info[0]
    pin = json.loads((ROOT / 'candidate.json').read_text())
    image = info['Image'].removeprefix('sha256:')
    if image != pin['image_id'] or 'DS41_NODE=dusty' not in info['Config']['Env']:
        raise RuntimeError('Expected the pinned dusty capture boot')
    mounts = {m['Destination']: m for m in info['Mounts']}
    for dest in ('/cache', '/root/.cache/huggingface'):
        if mounts[dest]['Type'] != 'bind':
            raise RuntimeError('Unexpected mount: ' + dest)
    env = dict(e.split('=', 1) for e in info['Config']['Env'] if e.startswith((
        'B12X_', 'CUTE_', 'CUTLASS_', 'CUDA_', 'NVCC_', 'CC=', 'CXX=',
        'VLLM_CACHE_ROOT=', 'TORCHINDUCTOR_CACHE_DIR=')))
    idle()
    now = datetime.datetime.now(datetime.timezone.utc)
    stamp = now.strftime('%Y%m%dT%H%M%SZ')
    since = now.strftime('%Y-%m-%d %H:%M:%S UTC')
    arm = 'fenced' if args.fenced else 'baseline'
    out = ROOT / 'receipts' / f'router-isolation-{arm}-{stamp}'
    out.mkdir(exist_ok=False)
    (out / 'retained-state.json').write_bytes(args.state.read_bytes())
    downloads = json.loads((ROOT / 'receipts/moe-capture-localization-20260925/probe/'
                           'traces-00/capture-downloads.json').read_text())
    expected_capture = [r for r in downloads if r['node'] == 'dusty'
                        and Path(r['file']).name == Path(CAPTURE).name]
    if len(expected_capture) != 1:
        raise RuntimeError('Missing unique verified capture identity')
    host_capture = str(Path(mounts['/cache']['Source']) / Path(CAPTURE).relative_to('/cache'))
    actual_capture = ssh('dusty', shlex.join(['sha256sum', host_capture])).split()[0]
    if actual_capture != expected_capture[0]['sha256']:
        raise RuntimeError('Remote capture differs from the verified receipt')
    (out / 'capture-identity.json').write_text(json.dumps(expected_capture[0], indent=2) + '\n')
    inputs = ['claude_replay_gate_prefill.py', 'claude_gate_capture_forensics.py',
              'claude-pinned-bf16_gemv-_prefill.py']
    if args.fenced:
        inputs += ['router-prefill-release-before.py']
    hashes = {}
    for name in inputs:
        data = (ROOT / name).read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
        (out / name).write_bytes(data)
        subprocess.run(['scp', str(ROOT / name), 'dusty:' + REMOTE + '/'], check=True)
    (out / 'inputs.json').write_text(json.dumps(hashes, indent=2) + '\n')
    host_tuning = str(Path(mounts['/cache']['Source']) / Path(TUNING).relative_to('/cache'))
    subprocess.run(['scp', 'dusty:' + host_tuning, str(out / 'tuning-before.json')], check=True)
    result_name = f'router-replay-{arm}-{stamp}.json'
    command = ['podman', 'run', '--rm', '--pull=never', '--network=none',
               '--name', f'ds41-router-{arm}-{stamp.lower()}', '--device', 'nvidia.com/gpu=all',
               '--ipc=host', '--ulimit', 'memlock=-1:-1', '-v', REMOTE + ':/diag', '-w', '/diag']
    for dest in ('/cache', '/root/.cache/huggingface'):
        command += ['-v', mounts[dest]['Source'] + ':' + dest + ':ro']
    if args.fenced:
        env['CUTE_DSL_CACHE_DIR'] = '/diag/router-compile-' + stamp + '/cute-dsl'
    for key, value in sorted(env.items()):
        command += ['-e', key + '=' + value]
    if args.fenced:
        command += ['-v', REMOTE + '/router-prefill-release-before.py:'
                    '/opt/jovian-judgement/b12x/b12x/gemm/bf16_gemv/_prefill.py:ro']
    command += ['-e', 'PYTHONUNBUFFERED=1', '-e', 'B12X_PRINT_COMPILE_PROGRESS=1',
                '--entrypoint', '/opt/venv/bin/python', image, '-u', '/diag/claude_replay_gate_prefill.py',
                '--capture', CAPTURE, '--checkpoint-dir', SNAPSHOT,
                '--weight-sha256', 'b98a45639ac40ddb1aba726b58fdc94300011e3cd84eaea210795095669294b0',
                '--repeats', str(args.repeats), '--side-iters', str(args.side_iters),
                '--stop-after', str(args.stop_after), '--out', '/diag/' + result_name]
    if args.fenced:
        command += ['--fenced', '--compile-cache-dir', '/diag/router-compile-' + stamp]
    else:
        command += ['--tuning-receipt', TUNING]
    (out / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    idle()
    with (out / 'run.log').open('x') as log:
        process = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', 'dusty', shlex.join(command)],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end='', flush=True)
        code = process.wait()
    (out / 'exit-code').write_text(str(code) + '\n')
    fetched = subprocess.run(['scp', 'dusty:' + REMOTE + '/' + result_name, str(out / 'result.json')])
    (out / 'kernel.log').write_text(ssh('dusty', shlex.join(['journalctl', '-k', '--since', since, '--no-pager'])))
    subprocess.run(['scp', 'dusty:' + host_tuning, str(out / 'tuning-after.json')], check=True)
    if (out / 'tuning-before.json').read_bytes() != (out / 'tuning-after.json').read_bytes():
        raise RuntimeError('Production selection cache changed')
    print('ROUTER-ISOLATION-RECEIPTS', out, flush=True)
    if fetched.returncode and code == 0:
        raise RuntimeError('No complete result receipt')
    raise SystemExit(code)


if __name__ == '__main__':
    main()
