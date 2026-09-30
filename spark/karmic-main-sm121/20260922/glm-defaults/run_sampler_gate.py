#!/usr/bin/env python3
"""Run pinned upstream block-verification gates on an otherwise idle GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path

if not __debug__:
    raise RuntimeError('Upstream assertion-based tests require assertions enabled')

CASES = {
    'distribution': ('tests/v1/spec_decode/test_rejection_sampler_utils.py',
                     'test_block_verification_rejection_sample or test_block_verification_accepts_at_least_as_many', 12),
    'glm-indexing': ('tests/v1/worker/test_gpu_rejection_sampler_i64.py',
                     'test_block_verification_i64_indexing', 2),
    'placeholders': ('tests/v1/spec_decode/test_rejection_sampler_utils.py',
                     'test_block_verification_placeholder_truncates_block or test_placeholder_blocks_later_draft_tokens', 6),
}
SOURCE_SHA256 = {
    'tests/v1/spec_decode/test_rejection_sampler_utils.py': '8f7de8165140e33e005385307413813ad3b9e33ed75e66fefeebe026f63f193f',
    'tests/v1/worker/test_gpu_rejection_sampler_i64.py': 'b628227c559ee28256fbd372ebe85eb3b11deb595b6b134a8a9d3cb7ab5668a7',
    'vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py': '214b61fa0abe88573beb779872c035b3954e5651a85bd438d99ee5905d3a62ec',
}


class RequiredCases:
    def __init__(self):
        self.collected = 0
        self.passed = 0
        self.invalid = []

    def pytest_collection_finish(self, session):
        self.collected = len(session.items)

    def pytest_runtest_logreport(self, report):
        if report.failed or report.skipped or hasattr(report, 'wasxfail'):
            self.invalid.append(report.nodeid + ':' + report.outcome)
        elif report.when == 'call' and report.passed:
            self.passed += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('case', choices=CASES)
    parser.add_argument('--collect-only', action='store_true')
    args = parser.parse_args()
    root = Path('/opt/jovian-judgement/vllm')
    for file, digest in SOURCE_SHA256.items():
        if hashlib.sha256((root / file).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Pinned source changed: ' + file)
    import pytest
    import torch
    import vllm
    if not Path(vllm.__file__).resolve().is_relative_to(root / 'vllm'):
        raise RuntimeError('Wrong runtime source tree')
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError('This gate requires a visible GB10 GPU')
    print(json.dumps({'case': args.case, 'source_sha256': SOURCE_SHA256,
                      'vllm_file': vllm.__file__, 'torch': torch.__version__,
                      'cuda': torch.version.cuda, 'capability': [12, 1]}), flush=True)
    os.chdir(root)
    path, selection, expected = CASES[args.case]
    plugin = RequiredCases()
    argv = ['-s', '-vv', '--confcutdir=' + path.rsplit('/', 1)[0], path, '-k', selection]
    if args.collect_only:
        argv.append('--collect-only')
    status = pytest.main(argv, plugins=[plugin])
    if status != 0 or plugin.collected != expected or plugin.invalid or (
        not args.collect_only and plugin.passed != expected
    ):
        raise RuntimeError(f'Incomplete sampler gate: status={status}, collected={plugin.collected}, '
                           f'passed={plugin.passed}, expected={expected}, invalid={plugin.invalid}')
    print(f'SAMPLER-{args.case}-' + ('COLLECTION' if args.collect_only else 'PASS') +
          f': {expected}', flush=True)


if __name__ == '__main__':
    main()
