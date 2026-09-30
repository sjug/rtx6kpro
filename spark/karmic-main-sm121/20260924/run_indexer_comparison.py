"""Run the layer-2 comparison on idle dusty using verified, read-only captures."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from audit_window_geometry import (load_receipt, validate_geometry_grids, container, NODES, ROOT, ssh,
                                   STAGED, window_file, validate_within)
from build_activation_trace import idle
from precision_default_nccl import ARMS, GEOMETRY, STANDARD_NCCL_ARM, load_grids

FILES = ('compare_indexer_inputs.py', 'ds41_indexer_capture.py', *STAGED)


def capture_files(receipt):
    captures = json.loads((receipt['path'] / 'indexer-captures.json').read_text())
    if set(captures) != set(NODES) or any(v['problems'] for v in captures.values()):
        raise ValueError('Incomplete or invalid indexer captures')
    return captures


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--left', type=Path, required=True)
    p.add_argument('--right', type=Path, required=True)
    p.add_argument('--kind', choices=('indexer', 'precision'), default='indexer')
    p.add_argument('--nccl', choices=ARMS, default=GEOMETRY, help='explicit NCCL arm of both receipts')
    a = p.parse_args()
    if a.nccl == STANDARD_NCCL_ARM and a.kind != 'precision':
        p.error('The standard NCCL arm exists only for the precision kind')
    left, right = load_grids(a.left, a.right, a.nccl)
    build = json.loads((ROOT / f'receipts/{a.kind}-build-receipt.json').read_text())
    lock_raw = (ROOT / f'ds41-{a.kind}.lock.json').read_bytes()
    lock = json.loads(lock_raw)
    helper_sha = (lock['inputs']['ds41_indexer_capture.py'] if a.kind == 'indexer' else
                  lock['preserved']['vllm/vllm/models/deepseek_v4_1/ds41_indexer_capture.py'])
    if (build['lock_sha256'] != hashlib.sha256(lock_raw).hexdigest()
            or left['summary']['image_id'] != build['image_id']
            or (Path(build['directory']) / 'BUILD-OK').read_text().strip() != build['image_id']
            or hashlib.sha256((ROOT / 'ds41_indexer_capture.py').read_bytes()).hexdigest()
                != helper_sha):
        raise ValueError('Capture image or helper does not match the frozen build')
    idle()
    captures = [capture_files(r) for r in (left, right)]
    for receipt, files in zip((left, right), captures):
        decision = json.loads((receipt['path'] / 'captures.json').read_text())
        for group in (files, receipt['windows'], decision):
            if set(group) != set(NODES) or any(v['problems'] for v in group.values()):
                raise ValueError('Incomplete sibling captures')
            for node, value in group.items():
                path = receipt['summary']['remote_dir'] + '/captures/' + Path(value['container_file']).name
                if ssh('dusty', 'sha256sum ' + shlex.quote(path)).split()[0] != value['sha256']:
                    raise RuntimeError('Captured bytes changed: ' + node)
    remote = right['summary']['remote_dir']
    sources = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES}
    subprocess.run(['scp', *[str(ROOT / name) for name in FILES], 'dusty:' + remote + '/'], check=True)
    for name, expected in sources.items():
        if ssh('dusty', 'sha256sum ' + shlex.quote(remote + '/' + name)).split()[0] != expected:
            raise RuntimeError('Comparator source changed: ' + name)
    commands = {}
    for node in NODES:
        commands[node] = container(right['summary']['image_id'],
            {'/gate': remote, '/left': left['summary']['remote_dir']},
            ['/gate/compare_indexer_inputs.py',
             '/left/captures/' + Path(captures[0][node]['container_file']).name,
             '/gate/captures/' + Path(captures[1][node]['container_file']).name,
             '--left-token', left['summary']['token'], '--right-token', right['summary']['token']], '24g')
    mounts = {'/gate': remote, '/left': left['summary']['remote_dir']}
    for label, receipt, prefix in (('left', left, '/left'), ('right', right, '/gate')):
        for node in NODES:
            window = prefix + '/captures/' + window_file(receipt, node)
            decision = prefix + '/captures/' + receipt['summary']['captures'][node]
            commands[f'window-{label}-self-{node}'] = container(build['image_id'], mounts,
                ['/gate/claude-window-compare.py', 'compare', window, window,
                 '--decision-left', decision, '--decision-right', decision], '24g')
        commands[f'window-{label}-ranks'] = container(build['image_id'], mounts,
            ['/gate/claude-window-compare.py', 'ranks',
             *[prefix + '/captures/' + window_file(receipt, n) for n in NODES]], '24g')
    for node in NODES:
        commands[f'window-grids-{node}'] = container(build['image_id'], mounts,
            ['/gate/claude-window-compare.py', 'compare',
             '/left/captures/' + window_file(left, node), '/gate/captures/' + window_file(right, node)], '24g')
    (right['path'] / 'indexer-comparison-invocation.json').write_text(json.dumps(
        {'sources': sources, 'commands': commands, 'captures': captures, 'nccl': a.nccl}, indent=2) + '\n')
    for node, command in commands.items():
        with (right['path'] / f'indexer-comparison-{node}.json').open('x') as out, \
                (right['path'] / f'indexer-comparison-{node}.stderr').open('x') as err:
            subprocess.run(['ssh', '-o', 'BatchMode=yes', 'dusty', shlex.join(command)],
                           stdout=out, stderr=err, check=True)
        print('INDEXER-COMPARISON', node, flush=True)
    for label in ('left', 'right'):
        reports = {node: json.loads((right['path'] / f'indexer-comparison-window-{label}-self-{node}.json').read_text())
                   for node in NODES}
        ranks = json.loads((right['path'] / f'indexer-comparison-window-{label}-ranks.json').read_text())
        validate_within(reports, ranks)
    print('INDEXER-WINDOW-STRUCTURAL-PASS', flush=True)


if __name__ == '__main__':
    main()
