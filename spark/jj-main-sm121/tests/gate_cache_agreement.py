#!/usr/bin/env python3
"""Require the complete adapted upstream startup matrix, including TP cache exchange."""
if not __debug__:
    raise RuntimeError('Assertions must be enabled')
import os
import sys
import pytest
from run_regressions import RequiredCases

os.chdir('/opt/jovian-judgement/vllm')
gate = RequiredCases(25)
args = ['-s', '-vv', '--confcutdir=tests/v1/executor',
        'tests/v1/executor/test_b12x_startup.py']
collect = '--collect-only' in sys.argv
if collect:
    args.append('--collect-only')
status = pytest.main(args, plugins=[gate])
if collect:
    gate.verify_collection(status, 25)
else:
    gate.verify(status)
print(f'CACHE-AGREEMENT-GATE-PASS collect_only={collect} cases=25', flush=True)
