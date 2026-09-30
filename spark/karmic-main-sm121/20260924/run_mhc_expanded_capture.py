"""Record one expanded-mHC capture boot: identity, one armed cold trial, per-rank capture receipts.

pin     (local) print the capture pin for a built image: the release pin with image_id replaced and
        diagnostic {kind mhc-expanded-capture, lock_sha256, base_image_id}. Installing it as
        candidate.json is the user's step; restore the release pin afterwards.
record  (serving capture boot, all four ranks, started with
        start_moe_repaired.py --arm mhc-expanded-capture --capture-selections passing|mhc-expanded8192)
        1. the pin is the capture lock over the release image; the lock is the prepared one
        2. before: the unchanged replay or transplant identity gate for that selection root (image,
           kit label, environment, precision marker, zero B12X measurements, selections == seed)
        3. memory: every rank's MemAvailable must exceed its planned need (lock memory_plan) plus the
           required --reserve-mib; otherwise nothing is armed and the recorder stops
        4. arm: the trigger (layer, reserve) written into the mounted seeded root on every rank
        5. three cold trials using unchanged qualify_original_needle.py; capture fires only on
           the first, followed by two uninstrumented repeats of the frozen 524288-token input
        6. wait for four .mhc-consumed receipts (any .mhc-aborted fails); copy the four small per-rank
           records into the receipt; gather the four row-shard staging files, records and receipts onto
           dusty inside the remote task tree over the switched fabric (route-checked, sha256-verified;
           originals stay on their nodes), the pattern of claude_run_decision_capture.gather_script
        7. after: the same identity gate, equal to before
        8. token 0 of the trial compared bit for bit with the three receipt references
        It prints the comparator command. It never builds, launches, stops or deletes anything.

The served side is fixed by the selection root: passing selections serve the passing expanded
configuration; the mhc-expanded8192 root serves the release one with every other record passing.
Operator evidence only; the capture's copies change timing, and token-0 equality with the matching
reference is the only output check.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

KIND = 'mhc-expanded-capture'
LOCK = 'claude-mhc-expanded.lock.json'
RELEASE_IMAGE = '1989e16daf38d2966b03a2e2abcb878263fb7d17183c8d6c5b084a51fbed7e8f'
NODES = ('dusty', 'toby', 'rusty', 'kirby')
SIDES = {'passing': 'passing', 'mhc-expanded8192': 'release'}
EXPECTED_TOKEN0 = {'passing': 'passing-anchor', 'mhc-expanded8192': None}
REMOTE = '/home/jugs/git/ds41-r38/karmic-main-20260924'
INPUT = ROOT.parents[1] / 'ds41/r38/receipts/20260916/admission-k7-1m-u80/needle-524288-input.json'
WAIT_SECONDS = 900


def sha(data):
    return hashlib.sha256(data).hexdigest()


def capture_pin(release_pin, image_id, root=ROOT):
    if len(image_id) != 64 or any(c not in '0123456789abcdef' for c in image_id):
        raise ValueError('image_id must be a full sha256 image id')
    if release_pin['image_id'] != RELEASE_IMAGE or release_pin['diagnostic']['kind'] != 'precision-release-candidate':
        raise RuntimeError('The capture pin derives from the release pin only')
    pin = json.loads(json.dumps(release_pin))
    pin['image_id'] = image_id
    pin['diagnostic'] = {'kind': KIND, 'lock_sha256': sha((root / LOCK).read_bytes()), 'base_image_id': RELEASE_IMAGE,
                         'b12x_tree': release_pin['diagnostic']['b12x_tree']}
    return pin


def pin_problems(pin, root=ROOT):
    import launch_contract
    problems = []
    lock_sha = sha((root / LOCK).read_bytes())
    diagnostic = pin.get('diagnostic', {})
    if diagnostic != {'kind': KIND, 'lock_sha256': lock_sha, 'base_image_id': RELEASE_IMAGE,
                      'b12x_tree': json.loads((root / LOCK).read_text())['base_trees']['b12x']}:
        problems.append('candidate.json is not the capture pin of the prepared lock')
    if lock_sha != launch_contract.MHC_EXPANDED_CAPTURE_LOCK:
        problems.append('launch contract admits a different capture lock')
    if 'diagnostic_overlay' in pin or pin.get('image_id') == RELEASE_IMAGE:
        problems.append('capture pin must name the built derivative, without overlay')
    lock = json.loads((root / LOCK).read_text())
    if lock['base_image_id'] != RELEASE_IMAGE or lock['kind'] != KIND:
        problems.append('capture lock is not over the release image')
    return problems


def seeded_bytes(selections, root=ROOT):
    if selections == 'passing':
        import seed_selection_replay
        _, files = seed_selection_replay.check(root)
        return seed_selection_replay.HOST_ROOT, {n: f['bytes'].decode() for n, f in files.items()}
    import seed_selection_transplant
    _, files = seed_selection_transplant.check(root, selections)
    return seed_selection_transplant.SETS[selections]['host_root'], {n: f['bytes'].decode() for n, f in files.items()}


def identity(selections, ssh, pin, out, phase, seeded, kit):
    if selections == 'passing':
        import run_selection_replay
        return run_selection_replay.identity(ssh, pin, out, phase, seeded, kit)
    import run_selection_transplant
    return run_selection_transplant.identity(ssh, pin, out, phase, seeded, kit, selections)


LAYERS = (1, 14)
RESERVE_MIB = (256, 16384)


def trigger(token, layer, reserve_mib):
    if layer not in LAYERS or type(reserve_mib) is not int or not RESERVE_MIB[0] <= reserve_mib <= RESERVE_MIB[1]:
        raise ValueError('layer must be 1 or 14 and reserve_mib within [256, 16384]')
    return {'token': token, 'prompt_tokens': 524288, 'chunk_rows': 8192, 'layer': layer, 'reserve_mib': reserve_mib}


def memory_preflight(ssh, reserve_mib, root=ROOT):
    """{node: {available, need, reserve, ok}} from each node's /proc/meminfo against the lock's plan."""
    plan = json.loads((root / LOCK).read_text())['memory_plan']
    rows = {}
    for rank, node in enumerate(NODES):
        text = ssh(node, 'grep -m1 ^MemAvailable: /proc/meminfo')
        fields = text.split()
        if len(fields) != 3 or fields[0] != 'MemAvailable:' or fields[2] != 'kB':
            raise RuntimeError(f'Unexpected meminfo from {node}: {text!r}')
        available = int(fields[1]) * 1024
        if plan['world'] != len(NODES) or plan['ranks'][rank]['rank'] != rank:
            raise RuntimeError('Lock memory plan is not the four-rank plan')
        need = plan['ranks'][rank]['peak_bytes']
        rows[node] = {'available': available, 'need': need, 'reserve': reserve_mib << 20,
                      'ok': available - need >= reserve_mib << 20}
    return rows


def arm_command(host_root, payload):
    path = host_root + '/claude-mhc-expanded.json'
    return (f'set -eu; test -d {shlex.quote(host_root)}; printf %s {shlex.quote(json.dumps(payload))} > '
            f'{shlex.quote(path + ".tmp")}; mv {shlex.quote(path + ".tmp")} {shlex.quote(path)}; echo ARMED')


def receipt_state(listing, token, rank):
    names = set(listing.split())
    if f'{token}-rank{rank}.mhc-aborted' in names:
        return 'aborted'
    return 'consumed' if f'{token}-rank{rank}.mhc-consumed' in names else 'pending'


def gather_script(remote_dir, files):
    """bash for dusty: files = {node: [(host path, sha256)]}; pull over the fabric, verify each sha256."""
    from claude_run_decision_capture import CONTRACT_NODES, FABRIC_DEV, FABRIC_SELF
    dest_dir = remote_dir + '/captures'
    lines = ['set -euo pipefail', f'mkdir -p {shlex.quote(dest_dir)}']
    for node in NODES:
        for source, digest in files[node]:
            dest = dest_dir + '/' + Path(source).name
            if node == 'dusty':
                lines.append(f'cp --reflink=auto {shlex.quote(source)} {shlex.quote(dest)}')
            else:
                address = CONTRACT_NODES[node][1]
                lines.append(f'ip -j route get {address} | python3 -c "import json,sys; r=json.load(sys.stdin)[0]; '
                             f'assert r.get(\'dev\')==\'{FABRIC_DEV}\' and r.get(\'prefsrc\')==\'{FABRIC_SELF}\', r"')
                lines.append('rsync --whole-file -e ' + shlex.quote('ssh -o BatchMode=yes -o Compression=no -c aes128-gcm@openssh.com')
                             + f' {shlex.quote(address + ":" + source)} {shlex.quote(dest)}')
            lines.append(f'echo "{digest}  {dest}" | sha256sum -c -')
        lines.append(f'echo GATHERED {node}')
    return '\n'.join(lines) + '\n'


def token0_verdict(trial, references):
    first = trial['response']['choices'][0]['logprobs']['content'][0]
    hits = [name for name, ref in references.items() if ref == first]
    return {'equals': hits[0] if len(hits) == 1 else None, 'matches': hits, 'token0': first}


def compare_command(captures_dir, out, token, served):
    manifest = ROOT / 'ds41-mhc-layer0-inputs.json'
    return {
        'image': RELEASE_IMAGE, 'served': served, 'token': token,
        'argv': ['python3', '-P', '/gate/ds41_mhc_expanded_compare.py', '--manifest', '/gate/ds41-mhc-layer0-inputs.json',
                 '--manifest-sha256', sha(manifest.read_bytes()), '--captures-dir', '/captures', '--token', token,
                 '--served', served, '--out', f'/receipts/mhc-expanded-compare-{token}'],
        'mounts': {'/captures': captures_dir + ' (ro)', '/receipts': REMOTE + '/receipts (rw)',
                   '/gate': 'node copy of this kit (ro)', '/cache': 'fresh task-tree root (rw)'},
        'env': {'PYTHONPATH': '/opt/jovian-judgement/vllm:/opt/jovian-judgement/b12x', 'HF_HUB_OFFLINE': '1'},
    }


def record(selections, out, ssh, layer, reserve_mib, root=ROOT, run=subprocess.Popen, sleep=time.sleep,
           now=time.monotonic):
    from runtime import kit_digest
    import seed_selection_transplant
    pin = json.loads((root / 'candidate.json').read_text())
    problems = pin_problems(pin, root)
    if problems:
        raise RuntimeError('; '.join(problems))
    host_root, seeded = seeded_bytes(selections, root)
    references = seed_selection_transplant.references(root)
    out.mkdir(parents=True, exist_ok=False)
    kit = kit_digest()
    before = identity(selections, ssh, pin, out, 'before', seeded, kit)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dt%H%M%Sz')
    token = f'mhc-expanded-{selections}-l{layer}-{stamp}'
    payload = trigger(token, layer, reserve_mib)
    (out / 'trigger.json').write_text(json.dumps(payload) + '\n')
    memory = memory_preflight(ssh, reserve_mib, root)
    (out / 'memory-preflight.json').write_text(json.dumps(memory, indent=2) + '\n')
    if not all(row['ok'] for row in memory.values()):
        raise RuntimeError(f'Not armed: planned capture memory does not fit with the reserve: {memory}')
    for node in NODES:
        if 'ARMED' not in ssh(node, arm_command(host_root, payload)).split():
            raise RuntimeError('Trigger not written on ' + node)
    command = [sys.executable, '-u', str(root / 'qualify_original_needle.py'), '--input', str(INPUT),
               '--out', str(out / 'original-524k'), '--repeats', '3']
    (out / 'command.json').write_text(json.dumps(command) + '\n')
    with (out / 'original-524k.log').open('x') as stream:
        process = run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(line, end='', flush=True)
        code = process.wait()
    report_path = out / 'original-524k/report.json'
    if not report_path.is_file():
        raise RuntimeError(f'Probe failed without a report (exit {code}); no capture wait is appropriate')
    report = json.loads(report_path.read_text())
    if (len(report) != 3 or [r['repeat'] for r in report] != [0, 1, 2]
            or any(r['cached_tokens'] != 0 for r in report)):
        raise RuntimeError('Capture sequence lacks three complete cold trials')
    captures = out / 'captures'
    captures.mkdir()
    directory = host_root + '/claude-mhc-expanded'
    deadline, states = now() + WAIT_SECONDS, {}
    while True:
        for rank, node in enumerate(NODES):
            listing = ssh(node, f'ls -1 {shlex.quote(directory)} 2>/dev/null || true')
            states[node] = receipt_state(listing, token, rank)
        if 'aborted' in states.values():
            raise RuntimeError(f'Capture aborted: {states}')
        if all(s == 'consumed' for s in states.values()) or now() > deadline:
            break
        sleep(10)
    if not all(s == 'consumed' for s in states.values()):
        raise RuntimeError(f'Capture receipts incomplete after {WAIT_SECONDS}s: {states}')
    receipts, gather = {}, {}
    for rank, node in enumerate(NODES):
        receipt_text = ssh(node, f'cat {shlex.quote(directory)}/{token}-rank{rank}.mhc-consumed')
        (captures / f'{token}-rank{rank}.mhc-consumed').write_text(receipt_text)
        receipts[node] = json.loads(receipt_text)
        record_path = receipts[node]['json'].replace('/cache', host_root, 1)
        record_text = ssh(node, 'cat ' + shlex.quote(record_path))
        (captures / Path(record_path).name).write_text(record_text)
        if 'bin_sha256' not in receipts[node]:
            raise RuntimeError(f'{node} kept no staging shard')
        gather[node] = [(receipts[node]['bin'].replace('/cache', host_root, 1), receipts[node]['bin_sha256']),
                        (record_path, sha(record_text.encode())),
                        (f'{directory}/{token}-rank{rank}.mhc-consumed', sha(receipt_text.encode()))]
    remote_dir = REMOTE + '/receipts/' + out.name
    script = gather_script(remote_dir, gather)
    (out / 'gather.sh').write_text(script)
    gathered = ssh('dusty', 'bash -c ' + shlex.quote(script))
    (out / 'gather.log').write_text(gathered)
    if [l for l in gathered.splitlines() if l.startswith('GATHERED')] != [f'GATHERED {n}' for n in NODES]:
        raise RuntimeError('Gather to dusty did not confirm every rank')
    after = identity(selections, ssh, pin, out, 'after', seeded, kit)
    strip = lambda side: {n: {k: v for k, v in side[n].items() if k != 'ready_lines'} for n in side}
    (out / 'identity.json').write_text(json.dumps({'before': before, 'after': after}, indent=2) + '\n')
    if strip(before) != strip(after):
        raise RuntimeError('Container identity or selections changed during the capture')
    trial = json.loads((out / f'original-524k/{report[0]["repeat"]}.json').read_text())
    verdict = token0_verdict(trial, references)
    expected = EXPECTED_TOKEN0[selections]
    summary = {'selections': selections, 'served': SIDES[selections], 'layer': layer, 'token': token,
               'needle_exit': code, 'memory_preflight': memory,
               'correct': report[0]['correct'], 'margin_nats': report[0]['first_token']['margin_nats'],
               'cold_trials': report,
               'token0': verdict, 'token0_as_expected': expected is None or verdict['equals'] == expected,
               'capture_host_dir': directory, 'receipts': receipts, 'gathered_dir': remote_dir + '/captures',
               'compare': compare_command(remote_dir + '/captures', out, token, SIDES[selections]),
               'scope': 'operator input capture; copies change timing; token-0 equality is the only output check'}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('MHC-EXPANDED-CAPTURE', token, verdict['equals'], json.dumps(summary['compare']['argv']), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='action', required=True)
    pin = sub.add_parser('pin')
    pin.add_argument('--image-id', required=True)
    rec = sub.add_parser('record')
    rec.add_argument('--selections', choices=tuple(SIDES), required=True)
    rec.add_argument('--out', type=Path, required=True)
    rec.add_argument('--layer', type=int, choices=LAYERS, default=1,
                     help='one engram layer per capture; layer 1 inputs are identical across the two roots')
    rec.add_argument('--reserve-mib', type=int, required=True,
                     help='MemAvailable that must remain after the planned capture memory, on every rank')
    a = p.parse_args()
    if a.action == 'pin':
        release = json.loads((ROOT / 'candidate.json').read_text())
        print(json.dumps(capture_pin(release, a.image_id), indent=1))
        return
    from build_activation_trace import ssh
    record(a.selections, a.out, ssh, a.layer, a.reserve_mib)


if __name__ == '__main__':
    main()
