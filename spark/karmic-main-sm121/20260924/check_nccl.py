#!/usr/bin/env python3
"""Fail before a build window on source/recipe drift."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / '20260922'))
from contracts import file_sha, require


def check():
    require(__debug__, 'Optimized Python is not admitted')
    row = json.loads((ROOT / 'nccl.lock.json').read_text())
    require(row['commit'] == '73cf112295c33aee2b895f329f592f2a9b4b0f97', 'Wrong NCCL commit')
    require(row['tree'] == '3e7de6f92f0190d1afe9f05642e634cbf43ae4c9', 'Wrong NCCL source tree')
    require(row['payload_sha256'] == file_sha(ROOT / 'nccl-source.tar'), 'NCCL payload drift')
    require(row['dockerfile_sha256'] == file_sha(ROOT / 'Dockerfile.nccl'), 'NCCL recipe drift')
    require(row['max_jobs'] == 20 and row['version_code'] == 23007 and row['target'] == 'sm_121',
            'NCCL compile policy drift')
    return row


if __name__ == '__main__':
    row = check()
    if sys.argv[1:] == ['--values']:
        print(row['payload_sha256'])
        print(row['dockerfile_sha256'])
    else:
        require(not sys.argv[1:], 'Unknown preflight argument')
        print('NCCL-SOURCE-PREFLIGHT-PASS')
