"""CPU-only window checks on idle dusty; source captures stay on the fabric.

Requires completed, digest-verified capture collection and decision-row audit.
No serving, image selection, GPU device, network inside container, or deletion.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from build_activation_trace import idle, ssh, REMOTE

ROOT = Path(__file__).resolve().parent
FILES = ('claude-window-compare.py', 'compare_window_baseline.py', 'compare_decision_grids.py')
NODES = ('dusty', 'toby', 'rusty', 'kirby')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    a = p.parse_args()
    current, prior = a.receipt.resolve(), a.baseline.resolve()
    if any(path.parent != ROOT / 'receipts' for path in (current, prior)):
        raise RuntimeError('Receipt outside this kit')
    summary, old = [json.loads((path / 'summary.json').read_text()) for path in (current, prior)]
    windows = json.loads((current / 'window-captures.json').read_text())
    if set(windows) != set(NODES) or any(v['problems'] for v in windows.values()):
        raise RuntimeError('Incomplete or invalid window captures')
    if json.loads((current / 'audit.json').read_text())['verdict'] != 'within-conformance-envelope':
        raise RuntimeError('Decision-row audit did not pass')
    build = json.loads((ROOT / 'receipts/window-build-receipt.json').read_text())
    if summary['image_id'] != build['image_id'] or summary['chunk_rows'] != old['chunk_rows']:
        raise RuntimeError('Wrong image or baseline grid')
    for data, path in ((summary, current), (old, prior)):
        if data['remote_dir'] != REMOTE + '/receipts/' + path.name:
            raise RuntimeError('Unexpected remote receipt path')
        if not data['signature_equals_control'] or not data['regime_equals_reference']:
            raise RuntimeError('Unmatched capture controls or plans')
    idle()
    remote, before = summary['remote_dir'], old['remote_dir']
    hashes = {}
    for name in FILES:
        source = ROOT / name
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        subprocess.run(['scp', str(source), 'dusty:' + remote + '/' + name], check=True)
        if ssh('dusty', 'sha256sum ' + shlex.quote(remote + '/' + name)).split()[0] != hashes[name]:
            raise RuntimeError('Analysis source transfer differs')
    commands = {}
    window_paths = []
    for node in NODES:
        window = Path(windows[node]['container_file']).name
        if not window.endswith('-window.pt'):
            raise RuntimeError('Invalid window filename')
        window_path = '/gate/captures/' + window
        window_paths.append(window_path)
        decision = '/gate/captures/' + summary['captures'][node]
        commands['window-self-' + node] = ['/gate/claude-window-compare.py', 'compare', window_path, window_path,
            '--decision-left', decision, '--decision-right', decision]
        commands['window-baseline-' + node] = ['/gate/compare_window_baseline.py',
            '/baseline/captures/' + old['captures'][node], decision]
    commands['window-ranks'] = ['/gate/claude-window-compare.py', 'ranks', *window_paths]
    invocations = {}
    for name, args in commands.items():
        command = ['podman', 'run', '--rm', '--pull=never', '--network=none', '--memory=24g',
            '-v', remote + ':/gate:ro', '-v', before + ':/baseline:ro',
            '--entrypoint', '/opt/venv/bin/python', summary['image_id'], *args]
        invocations[name] = command
        with (current / (name + '.json')).open('x') as out, (current / (name + '.stderr')).open('x') as err:
            code = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'dusty', shlex.join(command)],
                                  stdout=out, stderr=err).returncode
        print(name, 'exit', code, flush=True)
        if code:
            raise RuntimeError(name + ' failed; retain and inspect the result')
    with (current / 'window-analysis-invocation.json').open('x') as out:
        json.dump({'image_id': summary['image_id'], 'files': hashes, 'commands': invocations}, out, indent=2)
    print('WINDOW-CPU-ANALYSIS-COMPLETE', current, flush=True)


if __name__ == '__main__':
    main()
