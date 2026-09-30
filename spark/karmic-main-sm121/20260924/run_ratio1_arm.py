"""Record one boot of the ratio-1 compute-mode diagnostic, and compare control, variant and return.

record --arm control|variant|return (serving head, one boot, no node actions):
  identity before: the pinned image and kind for the arm, kit label, KV pinned at the lock's
  count, the normal profile otherwise, one precision marker per rank, and the ratio-1 pin
  marker present on every rank for the variant only; every rank's selection file keeps every
  release-snapshot record unchanged and its eight-plan regime at the pinned count is recorded.
  Then the unchanged qualify_original_needle.py (three cold trials, margins), then identity and
  selections again. Selection additions are listed, never hidden.
compare --control DIR --variant DIR --return DIR: the only variable is ratio1.extend compute
  mode. Validity requires control and return to reproduce one signature (same image, same
  selections, same KV); the variant is then reported against them. Correctness and determinism
  are reported separately. No verdict names a precision policy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
NODES = ('dusty', 'toby', 'rusty', 'kirby')
NAME = 'ds41-flash-karmic-main-tp4'
INPUT = ROOT.parents[1] / 'ds41/r38/receipts/20260916/admission-k7-1m-u80/needle-524288-input.json'
ARMS = {'control': 'precision-release-candidate', 'variant': 'ratio1-bf16-diagnostic',
        'return': 'precision-release-candidate'}
PIN_MARKER = 'DS41-RATIO1-EXTEND-PIN'


def lock():
    return json.loads((ROOT / 'ds41-ratio1.lock.json').read_text())


def pin_marker_problems(log, expect):
    lines = [line for line in log.splitlines() if PIN_MARKER in line]
    if not expect:
        return ['ratio-1 pin marker present on a release arm'] if lines else []
    modes = {line.split('v41_compute_mode=')[-1].strip() for line in lines}
    if not lines or modes != {'bf16'} or any('mode=extend' not in line for line in lines):
        return ['missing or incorrect ratio-1 pin marker']
    return []


def environment_problems(env, blocks):
    import precision_release
    problems = precision_release.environment_problems({k: v for k, v in env.items() if k != 'DS41_DECISION_ROW_BLOCKS'})
    if env.get('DS41_DECISION_ROW_BLOCKS') != str(blocks):
        problems.append(f'KV is not pinned at {blocks} blocks')
    return problems


def selection_check(records, snapshot_records, blocks):
    """Every snapshot record unchanged; additions listed; regime at the pinned count."""
    from claude_decode_sparse_mla import decode, regimes
    changed = sorted(k for k, v in snapshot_records.items() if records.get(k) != v)
    if changed:
        raise RuntimeError(f'{len(changed)} release-snapshot selection records changed or vanished')
    found, _ = decode(records, [blocks], None)
    return {'records': len(records), 'added': sorted(set(records) - set(snapshot_records)),
            'regime': regimes(records, found).get(blocks)}


def identity(arm, pin, ssh, out, phase, snapshot_dir, cache_dir):
    from claude_run_decision_capture import check_precision_marker
    from runtime import kit_digest
    blocks = lock()['kv_blocks']
    expected_kit = kit_digest()
    result = {}
    for rank, node in enumerate(NODES):
        info = json.loads(ssh(node, 'podman inspect ' + NAME))[0]
        (out / f'{node}-identity-{phase}.json').write_text(json.dumps(info, indent=1) + '\n')
        env = dict(item.split('=', 1) for item in info['Config']['Env'])
        problems = environment_problems(env, blocks)
        if not info['State']['Running'] or info['Image'].removeprefix('sha256:') != pin['image_id']:
            problems.append('container is not the pinned image')
        if info['Config']['Labels'].get('local-inference.ds41.kit.sha256') != expected_kit:
            problems.append('container kit differs from the validated local manifest')
        log = ssh(node, f'podman logs {NAME} 2>&1 | grep -F -e DS41-PRECISION-APPLIED -e {PIN_MARKER} '
                       f'-e "allocating {blocks} serving blocks" || true')
        (out / f'{node}-markers-{phase}.log').write_text(log)
        problems += check_precision_marker(log, rank)
        problems += pin_marker_problems(log, arm == 'variant')
        if f'allocating {blocks} serving blocks' not in log:
            problems.append('boot did not report the pinned serving-block allocation')
        text = ssh(node, 'cat ' + shlex.quote(cache_dir(env)))
        (out / f'{node}-selection-{phase}.json').write_text(text)
        snapshot = json.loads((snapshot_dir / f'{node}.json').read_text())['records']
        selection = selection_check(json.loads(text)['records'], snapshot, blocks)
        if problems:
            raise RuntimeError(f'{node}: ' + '; '.join(problems))
        result[node] = {'Id': info['Id'], 'StartedAt': info['State']['StartedAt'], 'Image': info['Image'],
                        'kit': info['Config']['Labels'].get('local-inference.ds41.kit.sha256'),
                        'selection_sha256': hashlib.sha256(text.encode()).hexdigest(), **selection}
    return result


def host_selection_path(env):
    cache = env['B12X_COMPILE_CACHE_DIR']
    if not cache.startswith('/cache/jit/') or '..' in cache:
        raise RuntimeError('Unexpected B12X cache path')
    return '/home/jugs/.cache/vllm-jj-ds41-tp4' + cache.removeprefix('/cache') + '/preparation/' + \
        '6115b03c7a814701d5610b3e6aecf82e30fdbb46a122d70441228538b9bd1e5e.json'


def record(arm, out, snapshot_dir, ssh):
    pin = json.loads((ROOT / 'candidate.json').read_text())
    if pin.get('diagnostic', {}).get('kind') != ARMS[arm] or 'diagnostic_overlay' in pin:
        raise RuntimeError(f'Pinned candidate is not the {arm} image')
    out.mkdir(parents=True, exist_ok=False)
    before = identity(arm, pin, ssh, out, 'before', snapshot_dir, host_selection_path)
    command = [sys.executable, '-u', str(ROOT / 'qualify_original_needle.py'), '--input', str(INPUT),
               '--out', str(out / 'original-524k'), '--repeats', '3']
    (out / 'arm.json').write_text(json.dumps({'arm': arm, 'kind': ARMS[arm], 'image_id': pin['image_id'],
                                              'command': command}, indent=2) + '\n')
    with (out / 'original-524k.log').open('x') as stream:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(line, end='', flush=True)
        code = process.wait()
    after = identity(arm, pin, ssh, out, 'after', snapshot_dir, host_selection_path)
    stable = {n: {k: v for k, v in before[n].items() if k not in ('selection_sha256', 'records', 'added')} for n in NODES}
    if stable != {n: {k: v for k, v in after[n].items() if k not in ('selection_sha256', 'records', 'added')} for n in NODES}:
        raise RuntimeError('Container identity or regime changed during the arm')
    (out / 'identity.json').write_text(json.dumps({'before': before, 'after': after}, indent=2) + '\n')
    print('RATIO1-ARM-RECORDED', arm, 'needle_exit', code, flush=True)


def summarize(directory):
    directory = Path(directory)
    arm = json.loads((directory / 'arm.json').read_text())
    ident = json.loads((directory / 'identity.json').read_text())
    report = json.loads((directory / 'original-524k/report.json').read_text())
    trials = [json.loads((directory / f'original-524k/{r["repeat"]}.json').read_text()) for r in report]
    signatures = []
    for t in trials:
        choice = t['response']['choices'][0]
        payload = {k: choice[k] for k in ('message', 'finish_reason', 'logprobs')}
        signatures.append(hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest())
    if len(report) < 3 or any(r['cached_tokens'] != 0 for r in report):
        raise RuntimeError('Arm lacks three cold trials: ' + str(directory))
    return {'arm': arm['arm'], 'image_id': arm['image_id'],
            'regime': ident['before']['dusty']['regime'],
            'regimes_agree': len({json.dumps(ident['before'][n]['regime'], sort_keys=True) for n in NODES}) == 1,
            'kits': sorted({ident['before'][n]['kit'] for n in NODES}),
            'correct': [r['correct'] for r in report], 'margins': [r['first_token']['margin_nats'] for r in report],
            'answers': [t['response']['choices'][0]['message']['content'] for t in trials],
            'signatures': signatures, 'elapsed_s': [r['elapsed_s'] for r in report],
            'selection_additions': {n: ident['after'][n]['added'] for n in NODES}}


def compare(control, variant, back):
    rows = {name: summarize(path) for name, path in (('control', control), ('variant', variant), ('return', back))}
    for name, row in rows.items():
        if row['arm'] != name or not row['regimes_agree'] or len(row['kits']) != 1:
            raise RuntimeError(f'{name}: wrong arm, rank-inconsistent regime, or mixed kits')
    c, v, r = rows['control'], rows['variant'], rows['return']
    if c['image_id'] != r['image_id'] or v['image_id'] == c['image_id']:
        raise RuntimeError('Control and return must share the release image; the variant must differ')
    diff = {slot for slot in c['regime'] if c['regime'][slot] != v['regime'][slot]}
    if c['regime'] != r['regime'] or diff:
        raise RuntimeError('Recorded tuned regimes differ between arms; the pin is not visible in tuned records, '
                           'so all three must record the same tuned regime')
    valid = len(set(c['signatures'])) == 1 and set(c['signatures']) == set(r['signatures'])
    return {'scope': 'one-variable diagnostic: ratio1.extend compute mode pinned bf16 in the variant only; '
                     'not a precision policy decision and not a qualification',
            'valid': valid, 'validity_rule': 'control and return reproduce one full response signature',
            'correctness': {k: {'correct': row['correct'], 'answers': row['answers']} for k, row in rows.items()},
            'determinism': {k: {'signatures': row['signatures'], 'margins_nats': row['margins'],
                                'within_arm_identical': len(set(row['signatures'])) == 1} for k, row in rows.items()},
            'variant_differs_from_control': set(v['signatures']) != set(c['signatures']),
            'selection_additions': {k: row['selection_additions'] for k, row in rows.items()},
            'elapsed_s': {k: row['elapsed_s'] for k, row in rows.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('record')
    r.add_argument('--arm', choices=tuple(ARMS), required=True)
    r.add_argument('--snapshot', type=Path, required=True, help='ratio1-namespace-snapshot receipt directory')
    r.add_argument('--out', type=Path, required=True)
    c = sub.add_parser('compare')
    c.add_argument('--control', type=Path, required=True)
    c.add_argument('--variant', type=Path, required=True)
    c.add_argument('--return', dest='back', type=Path, required=True)
    c.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.command == 'record':
        from build_activation_trace import ssh
        record(a.arm, a.out, a.snapshot, ssh)
        return
    result = compare(a.control, a.variant, a.back)
    with a.out.open('x') as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print('RATIO1-COMPARISON', json.dumps({'valid': result['valid'],
                                           'variant_differs_from_control': result['variant_differs_from_control']}),
          flush=True)


if __name__ == '__main__':
    main()
