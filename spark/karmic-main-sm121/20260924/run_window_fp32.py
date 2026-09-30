"""Run offline operand analysis on idle dusty, preserving source hashes and output."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from build_activation_trace import idle, ssh, REMOTE

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    receipt = args.receipt.resolve()
    if receipt.parent != ROOT / 'receipts':
        raise RuntimeError('Receipt outside this kit')
    summary = json.loads((receipt / 'summary.json').read_text())
    captures = json.loads((receipt / 'window-captures.json').read_text())
    remote = REMOTE + '/receipts/' + receipt.name
    if summary['remote_dir'] != remote:
        raise RuntimeError('Remote receipt mismatch')
    nodes = ('dusty', 'toby', 'rusty', 'kirby')
    if set(captures) != set(nodes) or any(c['problems'] for c in captures.values()):
        raise RuntimeError('Incomplete captures')
    idle()
    source = ROOT / 'analyze_window_fp32.py'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    subprocess.run(['scp', str(source), 'dusty:' + remote + '/' + source.name], check=True)
    if ssh('dusty', 'sha256sum ' + shlex.quote(remote + '/' + source.name)).split()[0] != digest:
        raise RuntimeError('Analysis transfer mismatch')
    paths = []
    for node in nodes:
        name = Path(captures[node]['container_file']).name
        if ssh('dusty', 'sha256sum ' + shlex.quote(remote + '/captures/' + name)).split()[0] != captures[node]['sha256']:
            raise RuntimeError('Capture checksum mismatch: ' + node)
        paths.append('/gate/captures/' + name)
    command = ['podman', 'run', '--rm', '--pull=never', '--network=none', '--memory=24g',
               '-v', remote + ':/gate:ro', '--entrypoint', '/opt/venv/bin/python',
               summary['image_id'], '/gate/' + source.name, '--captures', *paths]
    with (receipt / 'window-fp32.json').open('x') as out, (receipt / 'window-fp32.stderr').open('x') as err:
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'dusty', shlex.join(command)], stdout=out, stderr=err)
    (receipt / 'window-fp32-invocation.json').write_text(json.dumps(
        {'command': command, 'source_sha256': digest, 'captures': captures, 'exit_code': result.returncode}, indent=2) + '\n')
    result.check_returncode()
    print('FP32-OPERAND-ANALYSIS-COMPLETE', receipt, flush=True)


if __name__ == '__main__':
    main()
