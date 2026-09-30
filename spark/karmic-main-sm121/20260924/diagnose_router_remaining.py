"""Run skipped coverage, without changing the failed qualification verdict."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from diagnose_router_long import stable_wrong
from qualify_engram_repair import ROOT, snapshot


def commands(out):
    return [
        ('context', 'qualify_upstream.py', ['--out', str(out / 'context'),
         '--long', '--dual-needle', '--needle-lengths', '262000,500000,599936'],
         'DS41-QUALIFICATION-PASS'),
        ('conversations', 'claude_gate_conversations.py',
         ['--out', str(out / 'conversations'), '--mixed-needle-input',
          str(ROOT.parents[1] / 'ds41/r38/receipts/20260916/admission-k7-1m-u80/needle-524288-input.json')],
         'CLAUDE-CONVERSATIONS-PASS'),
        ('final', 'probe_repeatability.py', ['--out', str(out / 'final'),
         '--corpus', str(ROOT / 'receipts/determinism-corpus.json'),
         '--lengths', '385', '--repeats', '3'], 'REPEATABILITY-PASS'),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--failed-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if (args.failed_run / 'exit-code').read_text().strip() != '1':
        raise RuntimeError('Requires completed failed qualification')
    report = args.failed_run / 'gates/correctness/original-524k/report.json'
    stable_wrong(json.loads(report.read_text()))
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'scope.json').write_text(json.dumps({
        'scope': 'Remaining coverage only; original qualification remains failed',
        'failed_report_sha256': hashlib.sha256(report.read_bytes()).hexdigest(),
    }, indent=2) + '\n')
    before = snapshot(args.out, 'before', 'router-stage-release-candidate')
    for node, current in before.items():
        old = json.loads((args.failed_run / 'gates' / f'{node}-identity-after.json').read_text())[0]
        if current['Id'] != old['Id'] or current['StartedAt'] != old['State']['StartedAt']:
            raise RuntimeError('Not the failed qualification boot: ' + node)
    try:
        for label, script, options, marker in commands(args.out):
            command = [sys.executable, '-u', str(ROOT / script), *options]
            (args.out / f'{label}-command.json').write_text(json.dumps({
                'argv': command,
                'script_sha256': hashlib.sha256((ROOT / script).read_bytes()).hexdigest(),
            }, indent=2) + '\n')
            print('REMAINING-DIAGNOSTIC', label, flush=True)
            with (args.out / f'{label}.log').open('x') as log:
                child = subprocess.Popen(command, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True)
                for line in child.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end='', flush=True)
                code = child.wait()
            (args.out / f'{label}.exit-code').write_text(str(code) + '\n')
            if code or marker not in (args.out / f'{label}.log').read_text().splitlines():
                raise RuntimeError('Remaining diagnostic failed: ' + label)
    finally:
        if before != snapshot(args.out, 'after', 'router-stage-release-candidate'):
            raise RuntimeError('Serving identity changed')
    print('REMAINING-DIAGNOSTICS-COMPLETE; original qualification remains failed', flush=True)


if __name__ == '__main__':
    main()
