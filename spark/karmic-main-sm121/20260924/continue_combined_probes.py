"""Continue the matched short diagnosis as soon as the long arm completes."""
import json
from pathlib import Path
import subprocess
import sys
import time
from cache_metrics import idle_snapshot

root = Path(__file__).resolve().parent
report = root / 'receipts/combined-long-repeatability/report.json'
deadline = time.monotonic() + 3600
while True:
    rows = json.loads(report.read_text()) if report.exists() else []
    if any(row['length'] == 524288 and row['repeats'] == 3 for row in rows):
        break
    if time.monotonic() > deadline:
        raise RuntimeError('Long-arm receipt did not finish within continuation window')
    time.sleep(5)
idle_snapshot('http://dusty:8000')
for name, lengths, repeats, budget in [
    ('combined-upper-edge', '385,400,416,447', 4, 8),
    ('combined-prefill-only', '300', 10, 1),
    ('combined-cross-middle', '256', 1, 1),
    ('combined-cross-after', '300', 10, 1),
]:
    print('RUNNING', name, flush=True)
    command = [sys.executable, str(root / 'probe_repeatability.py'),
               '--corpus', str(root / 'receipts/determinism-corpus.json'),
               '--out', str(root / 'receipts' / name), '--lengths', lengths,
               '--repeats', str(repeats), '--max-tokens', str(budget)]
    result = subprocess.run(command, check=False)
    # A repeatability failure is the measured outcome, not a reason to omit
    # the next independent discriminator. Incomplete receipts do stop us.
    evidence = root / 'receipts' / name / 'report.json'
    if not evidence.exists() or len(json.loads(evidence.read_text())) != len(lengths.split(',')):
        raise RuntimeError(f'Incomplete probe: {name}, exit {result.returncode}')
print('COMBINED-FOLLOWUP-COMPLETE', flush=True)
