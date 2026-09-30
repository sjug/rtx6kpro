#!/usr/bin/env python3
"""Run the approved same-boot repeat, report cell completion, preserve health receipts."""
import datetime
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'qualification/qwen-mtp3-repeat'
RUNS = ROOT.parents[1] / 'runs/qwen3.8-flash-next/nvfp4-4p89/2026-09-karmic-sm121-qualification/throughput'
OUT.mkdir(parents=True, exist_ok=True)
if (OUT / 'watcher.jsonl').exists():
    raise RuntimeError('Watcher receipt already exists')
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
def emit(record):
    record['observed_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (OUT / 'watcher.jsonl').open('a') as stream:
        stream.write(json.dumps(record) + '\n')
    print(json.dumps(record), flush=True)
def ssh(node, command):
    return subprocess.check_output(['ssh', '-n', '-o', 'BatchMode=yes', '-o',
        'ConnectTimeout=10', node, command], text=True, timeout=45)
for node in ('dusty', 'kirby'):
    (OUT / f'{node}-before-memory.txt').write_text(ssh(node, 'cat /proc/meminfo'))
telemetry = subprocess.Popen(['bash', str(ROOT.parent / 'glm53/r38-spark/qualification/sample-qwen.sh'),
                              str(OUT / 'telemetry')], start_new_session=True)
process = subprocess.Popen(['bash', str(ROOT / 'benchmark-qwen.sh'), '2'])
emit({'event': 'started', 'pid': process.pid, 'same_boot': True})
seen = set()
try:
    while process.poll() is None:
        files = list(RUNS.glob('*__r02.json.resume.json')) + list(RUNS.glob('*__r02.json'))
        if files:
            try:
                record = json.loads(max(files, key=lambda p: p.stat().st_mtime).read_text())
                for row in record.get('results', []):
                    key = (row['concurrency'], row['context_tokens'])
                    if key not in seen:
                        seen.add(key)
                        emit({'event': 'cell_complete', 'cell': key, 'completed': len(seen),
                              'aggregate_tps': row['aggregate_tps'],
                              'steps_per_s': row['server_steps_per_s'], 'errors': row['num_errors']})
            except json.JSONDecodeError:
                pass  # Receipt replacement may overlap this read.
        time.sleep(2)
    code = process.wait()
    emit({'event': 'benchmark_exit', 'exit_code': code})
    for node in ('dusty', 'kirby'):
        journal = ssh(node, f'journalctl -k --since "{started}" --no-pager')
        (OUT / f'{node}-kernel.log').write_text(journal)
        (OUT / f'{node}-after-memory.txt').write_text(ssh(node, 'cat /proc/meminfo'))
        emit({'event': 'kernel_summary', 'node': node,
              'allocation_warning_count': journal.count('NV_ERR_NO_MEMORY'),
              'xid_lines': [line for line in journal.splitlines() if 'Xid' in line]})
    if code:
        raise RuntimeError(f'Benchmark failed: {code}')
finally:
    # Stop only this watcher's own read-only telemetry children.
    import os
    import signal
    os.killpg(telemetry.pid, signal.SIGTERM)
    telemetry.wait(timeout=15)
