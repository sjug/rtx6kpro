"""Continue diagnostic coverage after a stable wrong historical answer.

Does not amend or pass the failed qualification, and does not run benchmarks.
Run under run_observed.py after the qualification process has fully exited.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from qualify_engram_repair import ROOT, snapshot


def stable_wrong(rows):
    if (len(rows) != 3 or [r['repeat'] for r in rows] != [0, 1, 2]
            or any(not r['identical'] or r['cached_tokens'] != 0 for r in rows)
            or any(r['correct'] for r in rows)):
        raise RuntimeError('This diagnostic requires three cold, identical, wrong historical responses')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--failed-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if (args.failed_run / 'exit-code').read_text().strip() != '1':
        raise RuntimeError('Original qualification must have completed and failed')
    report = args.failed_run / 'gates/correctness/original-524k/report.json'
    stable_wrong(json.loads(report.read_text()))
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'status.json').write_text(json.dumps({
        'scope': 'diagnostic only; original qualification remains failed',
        'failed_report': str(report.resolve()),
        'failed_report_sha256': hashlib.sha256(report.read_bytes()).hexdigest(),
    }, indent=2) + '\n')
    before = snapshot(args.out, 'before', 'router-stage-release-candidate')
    for node, current in before.items():
        old = json.loads((args.failed_run / 'gates' / f'{node}-identity-after.json').read_text())[0]
        if current['Id'] != old['Id'] or current['StartedAt'] != old['State']['StartedAt']:
            raise RuntimeError('Diagnostic is not on the failed qualification boot')
    try:
        for length, repeats in ((131072, 8), (524288, 4)):
            command = [sys.executable, '-u', str(ROOT / 'probe_repeatability.py'),
                '--corpus', str(ROOT / 'receipts/determinism-corpus.json'),
                '--out', str(args.out / str(length)), '--lengths', str(length),
                '--repeats', str(repeats)]
            print('DIAGNOSTIC-LONG-REPEATABILITY', length, repeats, flush=True)
            with (args.out / f'{length}.log').open('x') as stream:
                process = subprocess.Popen(command, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True)
                for line in process.stdout:
                    stream.write(line)
                    stream.flush()
                    print(line, end='', flush=True)
                code = process.wait()
            (args.out / f'{length}.exit-code').write_text(str(code) + '\n')
            # A shorter failure is the better localization target. Do not send
            # larger work after a transport/engine error or a new divergence.
            if code:
                raise RuntimeError(f'Long diagnostic failed at {length}; inspect before advancing')
    finally:
        after = snapshot(args.out, 'after', 'router-stage-release-candidate')
        if before != after:
            raise RuntimeError('Serving identity changed during diagnostics')
    print('ROUTER-LONG-DIAGNOSTICS-COMPLETE; original qualification remains failed', flush=True)


if __name__ == '__main__':
    main()
