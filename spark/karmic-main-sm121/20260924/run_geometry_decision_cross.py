"""Compare existing full-depth decision captures, CPU-only while nodes are idle."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from audit_window_geometry import load_receipt, validate_geometry_grids, container, NODES, ROOT, ssh
from build_activation_trace import idle
from precision_default_nccl import ARMS, GEOMETRY, load_grids


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--left', type=Path, required=True)
    p.add_argument('--right', type=Path, required=True)
    p.add_argument('--nccl', choices=ARMS, default=GEOMETRY, help='explicit NCCL arm of both receipts')
    a = p.parse_args()
    left, right = load_grids(a.left, a.right, a.nccl)
    stem = 'geometry-decision-cross' if a.nccl == GEOMETRY else 'standard-decision-cross'
    idle()
    remote = right['summary']['remote_dir']
    source = ROOT / 'compare_decision_grids.py'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if ssh('dusty', 'sha256sum ' + shlex.quote(remote + '/' + source.name)).split()[0] != digest:
        raise RuntimeError('Staged comparator mismatch')
    commands = {}
    for node in NODES:
        commands[node] = container(right['summary']['image_id'],
            {'/gate': remote, '/left': left['summary']['remote_dir']},
            ['/gate/' + source.name, '/left/captures/' + left['summary']['captures'][node],
             '/gate/captures/' + right['summary']['captures'][node]], '24g')
    (right['path'] / f'{stem}-invocation.json').write_text(json.dumps(
        {'comparator_sha256': digest, 'commands': commands, 'nccl': a.nccl}, indent=2) + '\n')
    for node, command in commands.items():
        with (right['path'] / f'{stem}-{node}.json').open('x') as out, \
                (right['path'] / f'{stem}-{node}.stderr').open('x') as err:
            subprocess.run(['ssh', '-o', 'BatchMode=yes', 'dusty', shlex.join(command)],
                           stdout=out, stderr=err, check=True)
        print(stem.upper(), node, flush=True)


if __name__ == '__main__':
    main()
