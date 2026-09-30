"""Validate the separate replacement measurement without altering the first run."""
import hashlib
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
if len(data['results']) != 1 or data['metadata']['concurrency_levels'] != [4]:
    raise ValueError('Expected one C4 measurement')
row = data['results'][0]
if (row['concurrency'], row['context_tokens']) != (4, 16384):
    raise ValueError('Wrong replacement cell')
for flag in ('num_errors', 'warmup_timed_out', 'capacity_limited', 'underfilled'):
    if row[flag]:
        raise ValueError(f'Invalid cell: {flag}={row[flag]}')
for key in ('aggregate_tps', 'server_steps_per_s', 'server_accept_len_effective'):
    if not math.isfinite(row[key]) or row[key] <= 0:
        raise ValueError(f'Invalid {key}')
for key in ('avg_running_reqs', 'max_running_reqs'):
    if not math.isfinite(row[key]) or not 0 <= row[key] <= 4:
        raise ValueError(f'Invalid {key}')
print(json.dumps({'valid': True, 'path': str(path),
                  'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'cell': row}, indent=2))
