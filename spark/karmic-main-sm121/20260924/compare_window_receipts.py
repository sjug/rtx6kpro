"""Descriptive cross-grid comparison of two audited window captures on idle dusty."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from build_activation_trace import idle, ssh, REMOTE

ROOT = Path(__file__).resolve().parent
NODES = ('dusty', 'toby', 'rusty', 'kirby')


def validate_completed_audits(baselines, reports, ranks):
    """Require observed consistency, not merely successful analysis processes."""
    if set(baselines) != set(NODES) or set(reports) != set(NODES):
        raise ValueError('Incomplete rank audits')
    if ranks.get('replicated_all_equal') is not True:
        raise ValueError('Replicated window tensors differ')
    for node in NODES:
        if baselines[node].get('equal_observations') is not True:
            raise ValueError('Baseline observations differ: ' + node)
        for side in ('left', 'right'):
            report = reports[node]
            for layer in ('0', '1'):
                if report[f'decision_consistency_{side}'][layer]['all_equal'] is not True:
                    raise ValueError('Sibling capture differs: ' + node)
                within = report[f'within_{side}'][layer]
                if (within['packed']['rows_bit_equal'] != 128
                        or within['packed']['rows_not_in_gathered_records']):
                    raise ValueError('Packed rows fail reference: ' + node)
                for field in ('write_slots_equal_last_row_window', 'write_offsets_equal_position_offsets'):
                    if within['slots'][field] is not True:
                        raise ValueError('Window slot check failed: ' + node)


def validate_pair(left, right):
    if (left['chunk_rows'], right['chunk_rows']) != (8192, 4096):
        raise ValueError('Requires the 8192 then 4096 matched grids')
    for key in ('image_id', 'blocks', 'mode', 'regime_source'):
        if left[key] != right[key]:
            raise ValueError('Unmatched ' + key)
    if left['mode'] != 'matched' or left['blocks'] != 81389:
        raise ValueError('Unexpected experiment envelope')
    # The NCCL geometry arm is a separate numerical arm (compare_window_geometry.py); its
    # receipts pair only with each other, never with a baseline arm through this gate.
    if left.get('nccl_geometry') != right.get('nccl_geometry'):
        raise ValueError('NCCL geometry differs between the receipts')
    for data in (left, right):
        if not data['signature_equals_control'] or not data['regime_equals_reference']:
            raise ValueError('Controls or numerical plans differ')
        if set(data['capture_problems']) != set(NODES) or any(data['capture_problems'].values()):
            raise ValueError('Capture failures')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left', type=Path, required=True)
    parser.add_argument('--right', type=Path, required=True)
    args = parser.parse_args()
    paths = [args.left.resolve(), args.right.resolve()]
    summaries, windows = [], []
    for path in paths:
        if path.parent != ROOT / 'receipts':
            raise ValueError('Receipt outside this kit')
        data = json.loads((path / 'summary.json').read_text())
        if data['remote_dir'] != REMOTE + '/receipts/' + path.name:
            raise ValueError('Unexpected remote path')
        if json.loads((path / 'audit.json').read_text())['verdict'] != 'within-conformance-envelope':
            raise ValueError('Decision-row conformance gate failed')
        if not (path / 'window-analysis-invocation.json').is_file():
            raise ValueError('Window self/baseline audits incomplete')
        validate_completed_audits(
            {n: json.loads((path / f'window-baseline-{n}.json').read_text()) for n in NODES},
            {n: json.loads((path / f'window-self-{n}.json').read_text()) for n in NODES},
            json.loads((path / 'window-ranks.json').read_text()),
        )
        window = json.loads((path / 'window-captures.json').read_text())
        if set(window) != set(NODES) or any(v['problems'] for v in window.values()):
            raise ValueError('Window capture failures')
        summaries.append(data)
        windows.append(window)
    validate_pair(*summaries)
    idle()
    left, right = summaries
    remote = right['remote_dir']
    source_hash = hashlib.sha256((ROOT / 'claude-window-compare.py').read_bytes()).hexdigest()
    if ssh('dusty', 'sha256sum ' + shlex.quote(remote + '/claude-window-compare.py')).split()[0] != source_hash:
        raise ValueError('Remote comparator differs')
    commands = {}
    for node in NODES:
        files = [mount + '/captures/' + Path(w[node]['container_file']).name
                 for mount, w in zip(('/left', '/gate'), windows)]
        commands[node] = ['podman', 'run', '--rm', '--pull=never', '--network=none', '--memory=24g',
            '-v', left['remote_dir'] + ':/left:ro', '-v', remote + ':/gate:ro',
            '--entrypoint', '/opt/venv/bin/python', right['image_id'], '/gate/claude-window-compare.py',
            'compare', *files, '--decision-left', '/left/captures/' + left['captures'][node],
            '--decision-right', '/gate/captures/' + right['captures'][node]]
    out = paths[1]
    with (out / 'window-cross-invocation.json').open('x') as handle:
        json.dump({'comparator_sha256': source_hash, 'commands': commands,
                   'left': str(paths[0]), 'right': str(paths[1])}, handle, indent=2)
    for node, command in commands.items():
        with (out / f'window-cross-{node}.json').open('x') as stdout, (out / f'window-cross-{node}.stderr').open('x') as stderr:
            code = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'dusty', shlex.join(command)],
                                  stdout=stdout, stderr=stderr).returncode
        print(node, 'comparison exit', code, flush=True)
        if code:
            raise RuntimeError('Comparison failed; preserve and inspect ' + node)
    print('WINDOW-CROSS-COMPLETE', out, flush=True)


if __name__ == '__main__':
    main()
