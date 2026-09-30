"""Summarize host timing intervals without claiming GPU completion or causality."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('directory', type=Path)
p.add_argument('--min-ms', type=float, default=100)
a = p.parse_args()
rows = []
for path in sorted(a.directory.rglob('*.log')):
    previous = {}
    for line in path.read_text(errors='replace').splitlines():
        fields = line.split()
        if not fields or not fields[0].isdigit():
            continue
        meta = dict(x.split('=', 1) for x in fields[1:] if '=' in x)
        if not {'node', 'pid', 'tid', 'job'} <= meta.keys():
            continue
        # '-' spans idle periods and different requests; it is not a job key.
        if meta['job'] == '-':
            continue
        names = [x for x in fields[1:] if '=' not in x]
        if len(names) != 1:
            raise RuntimeError('Ambiguous event line in ' + str(path))
        key = (meta['pid'], meta['tid'], meta['job'])
        now = int(fields[0])
        if key in previous:
            then, event = previous[key]
            ms = (now - then) / 1_000_000
            if ms < 0:
                raise RuntimeError('Nonmonotonic thread timestamps in ' + str(path))
            if ms >= a.min_ms:
                rows.append({'file': str(path), 'node': meta['node'], 'job': meta['job'],
                    'tid': meta['tid'], 'from': event, 'to': names[0], 'host_ms': ms,
                    'start_monotonic_ns': then})
        previous[key] = (now, names[0])
print(json.dumps({'scope': 'adjacent events within the same thread and job; host time only',
                  'intervals': sorted(rows, key=lambda r: -r['host_ms'])}, indent=2))
