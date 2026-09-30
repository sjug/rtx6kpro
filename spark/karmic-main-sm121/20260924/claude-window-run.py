"""Window-capture transport beside the decision-row driver (claude_run_decision_capture.py --kind window).

The driver arms the shared trigger, sends the request, waits for the
decision-row receipts and gathers the decision-row files to dusty. The window
helper arms from that same trigger (its optional "window" field defaults to
armed) and writes a sibling file in the same output directory with its own
receipt, which the unchanged driver does not know. This module adds only that
last step and the payload helper:

  python3 claude-window-run.py payload TOKEN [--chunk-rows 8192|4096]
      the driver's trigger payload plus "window": {"rows": 128, "layers": [0, 1]} (explicit form)
  python3 claude-window-run.py collect --receipt RECEIPT_DIR [--timeout 600] [--capture window|indexer]
      read the driver's summary.json (token, remote_dir), wait for the four consumed receipts of that
      helper (window-consumed for the layers 0-1 window helper, indexer-consumed for the layer-2
      indexer helper), write <kind>-captures.json and <kind>-gather.{sh,log}, and pull the four files
      to <remote_dir>/captures/ on dusty over the switched fabric, verifying every sha256 (the driver's
      own gather script generator is reused unchanged). The default is the window helper, unchanged.

Node access is ssh read/copy only: cat of receipts, podman logs grep, rsync of
capture files. No arming, requests, restarts or container changes happen here.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
WINDOW_FIELD = {'rows': 128, 'layers': [0, 1]}
NODES = ('dusty', 'toby', 'rusty', 'kirby')


def driver():
    import claude_run_decision_capture as module
    return module


def trigger_payload(token, chunk_rows=8192, window=True):
    """The decision-row payload with the window field; window=False disarms the window helper only."""
    payload = driver().trigger_payload(token, chunk_rows)
    payload['window'] = dict(WINDOW_FIELD) if window else False
    return payload


KINDS = {
    'window': {'suffix': '-window.pt', 'consumed': 'window-consumed', 'log_prefix': 'CLAUDE-WINDOW',
               'stem': 'window', 'label': 'WINDOW'},
    'indexer': {'suffix': '-indexer.pt', 'consumed': 'indexer-consumed', 'log_prefix': 'DS41-INDEXER',
                'stem': 'indexer', 'label': 'INDEXER'},
}


def window_name(node, rank, token):
    return f'{node}-rank{rank}-{token}-window.pt'


def capture_name(kind, node, rank, token):
    return f'{node}-rank{rank}-{token}' + KINDS[kind]['suffix']


def parse_consumed(text, token, rank, kind='window', host_path=None):
    """A helper's consumed receipt for this token and rank: container path, host path, sha256, problems."""
    record = json.loads(text)
    expected_name = f'-rank{rank}-{token}' + KINDS[kind]['suffix']
    if (not record['file'].endswith(expected_name)
            or not re.fullmatch('[0-9a-f]{64}', record['sha256'])):
        raise ValueError(f'rank {rank}: {kind} receipt names another capture')
    host_path = host_path or driver().host_path
    return {'container_file': record['file'], 'host_file': host_path(record['file']),
            'sha256': record['sha256'], 'problems': record['problems']}


def parse_window_consumed(text, token, rank, host_path=None):
    return parse_consumed(text, token, rank, 'window', host_path)


def wait_for_receipts(token, out, *, timeout, ssh, out_dir, name, kind='window', sleep=time.sleep):
    """Poll the four ranks for a helper's consumed receipts; abort on its abort line (same discipline as the driver)."""
    spec = KINDS[kind]
    captures, deadline = {}, time.monotonic() + timeout
    while len(captures) < len(NODES):
        for rank, node in enumerate(NODES):
            if node in captures:
                continue
            lines = ssh(node, f'podman logs {name} 2>&1 | grep -F {spec["log_prefix"]} || true')
            (out / f'{node}-{spec["stem"]}-helper.log').write_text(lines)
            if any(f'{kind_} token={token} ' in lines for kind_ in ('aborted', 'incomplete decision forward', 'error')):
                raise RuntimeError(f'{node}: {kind} helper aborted; no receipt will arrive')
            path = f'{out_dir}/{token}-rank{rank}.{spec["consumed"]}'
            text = ssh(node, f'cat {shlex.quote(path)} 2>/dev/null || true')
            if text.strip():
                captures[node] = parse_consumed(text, token, rank, kind)
                (out / f'{spec["stem"]}-partial-captures.json').write_text(json.dumps(captures, indent=2) + '\n')
                print(f'{spec["label"]}-CAPTURED', node, json.dumps(captures[node]), flush=True)
        if len(captures) < len(NODES):
            if time.monotonic() > deadline:
                raise RuntimeError(f'{kind} receipts missing on: ' + ', '.join(n for n in NODES if n not in captures))
            sleep(5)
    return captures


def wait_for_window_receipts(token, out, *, timeout, ssh, out_dir, name):
    return wait_for_receipts(token, out, timeout=timeout, ssh=ssh, out_dir=out_dir, name=name, kind='window')


def collect(receipt, timeout, kind='window', run=subprocess.check_output):
    """Wait for one helper's four receipts, record them and pull its files to dusty; window behavior unchanged."""
    d = driver()
    spec = KINDS[kind]
    summary = json.loads((receipt / 'summary.json').read_text())
    token, remote_dir = summary['token'], summary['remote_dir']
    captures = wait_for_receipts(token, receipt, timeout=timeout, ssh=d.ssh, out_dir=d.OUT_DIR, name=d.NAME, kind=kind)
    if any(c['problems'] for c in captures.values()):
        raise RuntimeError(f'{kind} helper reported problems: ' + json.dumps(captures))
    (receipt / f'{spec["stem"]}-captures.json').write_text(json.dumps(captures, indent=2, sort_keys=True) + '\n')
    script = d.gather_script(remote_dir, captures)
    (receipt / f'{spec["stem"]}-gather.sh').write_text(script)
    gathered = run(['ssh', '-o', 'BatchMode=yes', 'dusty', 'bash -s'], input=script, text=True)
    (receipt / f'{spec["stem"]}-gather.log').write_text(gathered)
    if gathered.count('GATHERED ') != len(NODES):
        raise RuntimeError(f'gather did not confirm all four {kind} files on dusty')
    print(f'{spec["label"]}-COLLECTED', remote_dir + '/captures', flush=True)
    return captures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('payload')
    p.add_argument('token')
    p.add_argument('--chunk-rows', type=int, choices=(8192, 4096), default=8192)
    c = sub.add_parser('collect')
    c.add_argument('--receipt', type=Path, required=True)
    c.add_argument('--timeout', type=int, default=600)
    c.add_argument('--capture', choices=tuple(KINDS), default='window',
                   help='which sibling capture to collect: window (layers 0-1) or indexer (layer-2 observation)')
    a = parser.parse_args(argv)
    if a.command == 'payload':
        print(json.dumps(trigger_payload(a.token, a.chunk_rows)))
    else:
        collect(a.receipt, a.timeout, a.capture)


if __name__ == '__main__':
    main()
