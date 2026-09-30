#!/usr/bin/env python3
"""Summarize retained qualification health evidence without querying serving."""
import csv
from datetime import datetime
import json
from pathlib import Path
import re
from statistics import mean

root = Path(__file__).resolve().parent / 'receipts/qualification'
driver = (root / 'benchmark-driver.log').read_text()
start = datetime.fromisoformat(re.search(r'BENCHMARK-START (\S+)', driver)[1]).replace(tzinfo=None)
end = datetime.fromisoformat(re.search(r'BENCHMARK-END (\S+)', driver)[1]).replace(tzinfo=None)
report = {'benchmark_window': {'start': start.isoformat(), 'end': end.isoformat()}, 'nodes': {}}
for node in ('rusty', 'toby'):
    with (root / f'{node}-gpu.csv').open() as handle:
        rows = list(csv.DictReader(handle, skipinitialspace=True))
    selected = [row for row in rows if start <= datetime.strptime(
        row['timestamp'], '%Y/%m/%d %H:%M:%S.%f') <= end]
    memory = (root / f'{node}-memory.log').read_text()
    final_log = root / f'{node}-after-grid.log'
    log = (final_log if final_log.exists() else root / f'{node}-live.log').read_text()
    kernel = (root / f'{node}-full-window-kernel.log').read_text()
    clock_key = next(key for key in rows[0] if key.startswith('clocks.current.sm'))
    flag_key = next(key for key in rows[0] if key.startswith('clocks_event_reasons.active'))
    report['nodes'][node] = {
        'post_grid_receipt_present': final_log.exists(),
        'gpu_samples_in_benchmark_window': len(selected),
        'mean_sm_clock_mhz': mean(float(row[clock_key].split()[0]) for row in selected),
        'nonzero_clock_event_samples': sum(int(row[flag_key], 16) != 0 for row in selected),
        'min_available_gib_full_window': min(map(int, re.findall(r'^MemAvailable:\s+(\d+)', memory, re.M))) / 1048576,
        'kernel_allocation_warnings': kernel.count('NV_ERR_NO_MEMORY'),
        'kernel_xid_lines': [line for line in kernel.splitlines() if 'NVRM: Xid' in line],
        'engine_fault_lines': [line for line in log.splitlines() if any(word in line for word in ('EngineDeadError', 'AssertionError', 'Traceback (most recent'))],
        'jit_warning_lines': [line for line in log.splitlines() if '[jit_monitor.py:' in line],
        'inference_clients': sorted(set(re.findall(r'INFO:\s+([\d.]+):\d+ - "POST /v1/(?:chat/)?completions', log))),
    }
print(json.dumps(report, indent=2))
