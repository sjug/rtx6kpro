"""Advance immediately from the diagnostic boot's completed short matrix."""
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
log = root / 'receipts/attention-trace-boot-driver.log'
deadline = time.monotonic() + 3600
while time.monotonic() < deadline:
    lines = log.read_text().splitlines() if log.exists() else []
    receipts = next((line.removeprefix('RECEIPTS ') for line in lines if line.startswith('RECEIPTS ')), None)
    if receipts:
        report = Path(receipts) / 'repeatability/report.json'
        if report.exists():
            rows = json.loads(report.read_text())
            if len(rows) == 3:
                print('TRACE-BOOT-REPEATABILITY', json.dumps(rows), flush=True)
                bad = next(row for row in rows if row['length'] == 300)
                if bad['identical']:
                    raise SystemExit('Trace-off boot does not reproduce; investigate before tracing')
                subprocess.run([sys.executable, str(root / 'run_attention_traces.py')], check=True)
                break
    if any('Traceback (most recent call last)' in line for line in lines) and not any('REPEATABILITY-FAIL' in line for line in lines):
        raise SystemExit('Diagnostic boot failed before its repeatability gate')
    time.sleep(5)
else:
    raise SystemExit('Diagnostic boot deadline exceeded')
