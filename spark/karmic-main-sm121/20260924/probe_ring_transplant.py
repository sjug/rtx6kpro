"""One-shot compressor-ring causal transplant on the pinned diagnostic server."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import sys

import cache_metrics
import claude_stale_capture as cap
from runtime import kit_digest

parser = argparse.ArgumentParser()
parser.add_argument('--capture', type=Path, required=True)
parser.add_argument('--out', type=Path)
parser.add_argument('--source-arm', choices=('first', 'repeat'), default='repeat')
parser.add_argument('--layer-regex', default=r'\.compressor\.state_cache$')
parser.add_argument('--positions', type=int, nargs=2, default=[0, 16])
parser.add_argument('--expected-slots', type=int, default=48)
parser.add_argument('--cycles', type=int, default=2)
parser.add_argument('--dry-run', action='store_true')
args = parser.parse_args()
if args.positions[0] < 0 or args.positions[1] <= args.positions[0] or args.expected_slots < 1:
    parser.error('Require a nonempty nonnegative position interval and positive slot count')
helper, pin, lock_sha, original, prime, original_sha, _ = cap.local_identity()
receipt = json.loads((args.capture / 'report.json').read_text())
if receipt['status'] != 'captured' or receipt['image_id'] != pin['image_id']:
    raise RuntimeError('Source capture identity mismatch')
source_index = 0 if args.source_arm == 'first' else 1
source = receipt['runs'][source_index]
expected = [r['signature'] for r in receipt['requests'] if r['label'].startswith('original-')][source_index]
digests = json.loads((args.capture / 'snapshots.sha256.json').read_text())
for node in cap.NODES:
    analysis = json.loads((args.capture / f'diff-{node}.json').read_text())
    if analysis['seq_len'] != [393, 393] or analysis['step_inputs'] or analysis['extra']:
        raise RuntimeError('Source capture is not a matched verification pair')
    selected = [name for name in analysis['layers'] if re.search(args.layer_regex, name)]
    if not selected or any(not name.endswith('.compressor.state_cache') for name in selected):
        raise RuntimeError('Offline selector must match captured compressor rings: ' + args.layer_regex)
    if args.positions[1] > min(analysis['seq_len']):
        raise RuntimeError('Selection extends beyond the captured position domain')
    expected_slots = len(selected) * min(16, args.positions[1] - args.positions[0])
    if expected_slots != args.expected_slots:
        raise RuntimeError(f'Offline ring slot count {expected_slots} differs from expected {args.expected_slots}')
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
controls = []
for cycle in range(args.cycles):
    control, _ = cap.build_control(original_sha, stamp)
    control['actions'] = [{'run': f'{stamp}-ring-{args.source_arm}-{cycle}',
        'mode': 'transplant', 'source': source,
        'select': [{'layer_regex': args.layer_regex, 'positions': args.positions}]}]
    helper.parse_control(control)
    controls.append(control)
if args.dry_run:
    print(json.dumps(controls, indent=2))
    sys.exit(0)
if args.out is None or args.cycles < 1:
    parser.error('--out and positive cycles required')
args.out.mkdir(parents=True, exist_ok=False)
kit = kit_digest()
identities = {n: cap.container_identity(n, pin, kit) for n in cap.NODES}
for node in cap.NODES:
    if cap.ssh(node, f'test ! -e {cap.CONTROL} && echo DISARMED').strip() != 'DISARMED':
        raise RuntimeError('A prior control remains armed')
    name = f'{source}-{node}.pt'
    if cap.ssh(node, f'sha256sum {cap.SNAPSHOTS}/{name}').split()[0] != digests[name]:
        raise RuntimeError('Remote source snapshot digest mismatch: ' + node)
report = {'image_id': pin['image_id'], 'lock_sha256': lock_sha, 'kit_sha256': kit,
          'source': source, 'expected_signature': expected,
          'layer_regex': args.layer_regex, 'positions': args.positions,
          'expected_slots': args.expected_slots, 'requests': [], 'status': 'started',
          'identities_before': identities}
def save():
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
save()
try:
    for cycle, control in enumerate(controls):
        # Stable source into first-after-meadow; inverse source into an already
        # stable original sequence. Controls are armed only after both primes.
        predecessor = prime if args.source_arm == 'repeat' else original
        for repeat in range(2):
            report['requests'].append(cap.one_request(f'{cycle}-prime-{repeat}', predecessor, args.out, cache_metrics))
        cache_metrics.idle_snapshot(cap.BASE_URL)
        since = {n: cap.ssh(n, 'date -u +%Y-%m-%dT%H:%M:%SZ').strip() for n in cap.NODES}
        payload = (json.dumps(control, indent=1, sort_keys=True) + '\n').encode()
        (args.out / f'control-{cycle}.json').write_bytes(payload)
        sha = hashlib.sha256(payload).hexdigest()
        for node in cap.NODES:
            tmp = f'{cap.HOST_CACHE}/.ds41-ring-{stamp}-{cycle}.tmp'
            cap.ssh(node, f'umask 077; cat > {tmp} && mv -T {tmp} {cap.CONTROL}', stdin=payload)
            if cap.ssh(node, f'sha256sum {cap.CONTROL}').split()[0] != sha:
                raise RuntimeError('Control digest mismatch: ' + node)
        result = cap.one_request(f'{cycle}-transplant', original, args.out, cache_metrics)
        report['requests'].append(result)
        for node in cap.NODES:
            logs = cap.logs_since(node, since[node])
            (args.out / f'{cycle}-{node}-logs.txt').write_text(logs)
            loaded, _, failures = cap.markers(logs)
            run_name = re.escape(control['actions'][0]['run'])
            matches = re.findall(r'\[DS41-STALE-PROBE\] transplant ' + run_name + r' slots=(\d+)', logs)
            if len(loaded) != 1 or failures or len(matches) != 1 or int(matches[0]) != args.expected_slots:
                raise RuntimeError(f'{node}: missing or invalid transplant engagement')
        disarmed = cap.disarm(cap.NODES, f'{stamp}-{cycle}', args.out)
        if any(v != 'disarmed' for v in disarmed.values()):
            raise RuntimeError('Failed to disarm')
        print('RING-TRANSPLANT', cycle, result['signature'], 'expected', expected, flush=True)
        save()
    observed = [r['signature'] for r in report['requests'] if r['label'].endswith('transplant')]
    report['matched_source_signature'] = all(s == expected for s in observed)
    report['status'] = 'complete'
finally:
    report['disarm'] = cap.disarm(cap.NODES, stamp + '-final', args.out)
    report['identities_after'] = {n: cap.container_identity(n, pin, kit) for n in cap.NODES}
    if report['identities_after'] != identities:
        report['status'] = 'failed-identity'
    save()
print('RING-TRANSPLANT-COMPLETE', report.get('matched_source_signature'), flush=True)
