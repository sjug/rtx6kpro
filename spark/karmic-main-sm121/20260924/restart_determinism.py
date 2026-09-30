"""Explicit four-node diagnostic restart with preserved containers and receipts."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
from cache_metrics import idle_snapshot
from runtime import kit_digest

ROOT = Path(__file__).resolve().parent
NAME = 'ds41-flash-karmic-main-tp4'
NODES = ('toby', 'rusty', 'kirby', 'dusty')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['moe-control'], required=True)
    args = parser.parse_args()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = ROOT / 'receipts' / f'{args.arm}-{stamp}'
    out.mkdir(exist_ok=False)
    print('RECEIPTS', out, flush=True)
    kit_digest()
    state = idle_snapshot('http://dusty:8000')
    (out / 'idle.json').write_text(json.dumps(state, indent=2) + '\n')
    pin = json.loads((ROOT / 'candidate.json').read_text())['image_id']
    def ssh(node, command, timeout=120):
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', node, command],
                                text=True, capture_output=True, timeout=timeout)
        with (out / 'commands.log').open('a') as stream:
            stream.write(f'{node}: {command}\n{result.stdout}\n{result.stderr}\nrc={result.returncode}\n')
        if result.returncode:
            raise RuntimeError(f'{node} failed; no automatic rollback. Inspect {out}/commands.log')
        return result.stdout
    # Verify every intended target before changing any rank.
    for node in NODES:
        info = json.loads(ssh(node, f'podman inspect {NAME}'))[0]
        (out / f'{node}-before.json').write_text(json.dumps(info, indent=2) + '\n')
        if info['Image'].removeprefix('sha256:') != pin or not info['State']['Running']:
            raise RuntimeError(f'{node}: unexpected image or serving state')
        (out / f'{node}-prior.log').write_text(ssh(node, f'podman logs {NAME} 2>&1'))
    for node in NODES:
        print('STOP', node, flush=True)
        ssh(node, f'podman stop -t 60 {NAME}', timeout=100)
        info = json.loads(ssh(node, f'podman inspect {NAME}'))[0]
        (out / f'{node}-stopped.json').write_text(json.dumps(info, indent=2) + '\n')
        if info['State']['Running'] or info['State']['OOMKilled'] or info['State']['ExitCode'] != 0:
            raise RuntimeError(f'{node}: unclean stop; inspect retained container')
        ssh(node, f'podman rename {NAME} {NAME}-baseline-{stamp.lower()}')
    paths = ['launch_contract.py', 'runtime-files.json', 'determinism-profile-amendment.json',
             'test_launch_contract.py']
    remote = '/home/jugs/git/ds41-r38/karmic-main-20260924'
    for node in NODES:
        subprocess.run(['scp', *[str(ROOT / p) for p in paths], f'{node}:{remote}/'], check=True)
        ssh(node, f'cd {remote} && python3 -c "from runtime import kit_digest; print(kit_digest())"')
    for node in NODES:
        print('START', node, flush=True)
        ssh(node, f'cd {remote} && B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1 python3 run_node.py --node {node}')
    subprocess.run([sys.executable, str(ROOT / 'watch-startup.py'), str(out)], check=True)
    subprocess.run([sys.executable, str(ROOT / 'probe_repeatability.py'),
                    '--corpus', str(ROOT / 'receipts/determinism-corpus.json'),
                    '--out', str(out / 'repeatability'), '--lengths', '256,513,514,1024,16384',
                    '--repeats', '6'], check=True)


if __name__ == '__main__':
    main()
