"""Require actual execution of every circular metadata regression."""
import sys
from pathlib import Path
import pytest
import torch

if not __debug__ or not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
    raise RuntimeError('Assertions and a real GB10 GPU are required')

class Counts:
    collected = 0
    passed = 0
    skipped = 0
    def pytest_collection_finish(self, session):
        self.collected = len(session.items)
    def pytest_runtest_logreport(self, report):
        self.skipped += report.skipped
        self.passed += report.when == 'call' and report.passed

counts = Counts()
code = pytest.main(['-s', '-vv', '-p', 'no:cacheprovider', str(Path(__file__).with_name('test_compressor_ring_mapping.py'))], plugins=[counts])
if code or counts.collected != 29 or counts.passed != 29 or counts.skipped:
    raise RuntimeError(f'Incomplete ring regression: code={code}, collected={counts.collected}, passed={counts.passed}, skipped={counts.skipped}')
print('RING-REGRESSION-PASS cases=29', flush=True)
