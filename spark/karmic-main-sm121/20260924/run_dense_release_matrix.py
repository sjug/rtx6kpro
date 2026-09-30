"""Sequential, fail-closed confirmation of the two-line dense release fence.

The 192-row original projection is corrupt on dusty. Only its synchronized
replay has unanimous four-rank reference identity, required by the runner.
The already completed Kirby 129-row 4096-call arm is not repeated here.
"""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
for node, length in [('dusty', 129), ('dusty', 160), ('dusty', 192),
                     ('kirby', 160), ('kirby', 192)]:
    print('RELEASE-MATRIX-START', node, length, flush=True)
    command = [sys.executable, '-u', str(root / 'run_engram_dense_replay.py'),
               '--node', node, '--length', str(length),
               '--label', f'release-matrix-m{length}', '--repeats', '4096',
               '--delay-ms', '50', '--source-variant', 'fence-before-release']
    if length == 192:
        command += ['--reference-key', 'dense_replay_synchronized']
    subprocess.run(command, cwd=root, check=True)
    print('RELEASE-MATRIX-PASS', node, length, flush=True)
print('RELEASE-MATRIX-COMPLETE', flush=True)
