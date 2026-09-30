#!/usr/bin/env python3
"""Require all nine pinned upstream Engram cases, including disk/capture parity."""
import os
import subprocess
import sys
from pathlib import Path

if not __debug__:
    raise RuntimeError('Assertions required')

if len(sys.argv) == 1:
    for mode in ('collect', 'execute'):
        subprocess.run([sys.executable, __file__, mode], check=True)
else:
    sys.path.insert(0, '/gate/inherited/tests')
    from run_r38_regressions import RequiredCases
    import pytest
    root = '/opt/jovian-judgement/b12x'
    os.chdir(root)
    sys.path.insert(0, root)
    gate = RequiredCases(9)
    args = ['-s', '-vv', '-p', 'no:cacheprovider', '--confcutdir=tests', 'tests/sequence/test_engram.py']
    if sys.argv[1] == 'collect':
        args += ['--collect-only']
    status = pytest.main(args, plugins=[gate])
    if sys.argv[1] == 'collect':
        gate.verify_collection(status, 9)
    else:
        gate.verify(status)
    print('DS41-UPSTREAM-ENGRAM-PASS', sys.argv[1], 'count=9', flush=True)
