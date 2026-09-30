"""Record one fixed-verification boot, and compare two such boots with the adaptive control and return.

record --boot a|b --snapshot DIR --out DIR (serving head; one boot; no node actions):
  before and after the unchanged qualify_original_needle.py (three cold trials, margins), on every
  rank: the gated precision release image with the validated kit; the normal profile plus exactly
  DS41_DECISION_ROW_BLOCKS=80022 and DS41_FIXED_VERIFICATION=depth7-adaptive-off; the 80022-block
  allocation line; one precision marker and no ratio-1 pin marker; zero B12X measurements where the
  logs report counts; every release-snapshot selection record unchanged (additions listed). On the
  head, the engine's own non-default args must show enable_adaptive_verification False with 7
  speculative tokens and cost scale 1.0; SpecDecoding lines are kept as descriptive evidence.
compare --a DIR --b DIR --control DIR --return DIR: reports, never gates.

What the comparison can support: if the two fixed-depth boots are bitwise equal while control and
return (adaptive) differed, that is consistent with the adaptive path, as a whole, being needed for
the cross-boot divergence. It does not prove it is the only mechanism, and it cannot separate the
timed cost tables from confidence differences or from the adaptive CUDA-graph route and startup
profiling that disabling the feature also removes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
NODES = ('dusty', 'toby', 'rusty', 'kirby')
NAME = 'ds41-flash-karmic-main-tp4'
HOST_CACHE = '/home/jugs/.cache/vllm-jj-ds41-tp4'
SELECTION_NAME = '6115b03c7a814701d5610b3e6aecf82e30fdbb46a122d70441228538b9bd1e5e.json'
FIXED_ENV = {'DS41_DECISION_ROW_BLOCKS': '80022', 'DS41_FIXED_VERIFICATION': 'depth7-adaptive-off'}
INPUT = ROOT.parents[1] / 'ds41/r38/receipts/20260916/admission-k7-1m-u80/needle-524288-input.json'
SPEC_REQUIRED = ("'enable_adaptive_verification': False", "'num_speculative_tokens': 7",
                 "'adaptive_verification_cost_scale': 1.0", "'method': 'dspark'")
SPEC_LINE = re.compile(r"'speculative_config': \{[^}]*\}")


def environment_problems(env):
    import precision_release
    problems = precision_release.environment_problems({k: v for k, v in env.items() if k not in FIXED_ENV})
    return problems + [f'{k} is not {v}' for k, v in FIXED_ENV.items() if env.get(k) != v]


def speculative_problems(log):
    match = SPEC_LINE.search(log)
    if match is None:
        return ['head log lacks the engine speculative_config']
    return [f'effective speculative config lacks {item}' for item in SPEC_REQUIRED if item not in match.group(0)]


def selection_path(info, env):
    """Host path of the selection file, derived from the container's actual /cache bind mount."""
    mounts = {m['Destination']: m['Source'] for m in info['Mounts']}
    if mounts.get('/cache') != HOST_CACHE:
        raise RuntimeError(f'/cache is mounted from {mounts.get("/cache")!r}, not the release cache root')
    cache = env.get('B12X_COMPILE_CACHE_DIR', '')
    if not cache.startswith('/cache/jit/') or '..' in cache.split('/'):
        raise RuntimeError('Unexpected B12X cache path')
    return mounts['/cache'] + cache.removeprefix('/cache') + '/preparation/' + SELECTION_NAME


def identity(ssh, pin, out, phase, snapshot_dir, expected_kit):
    from claude_run_decision_capture import check_precision_marker
    from run_ratio1_arm import pin_marker_problems, selection_check
    from run_selection_replay import measured_problems
    result = {}
    for rank, node in enumerate(NODES):
        info = json.loads(ssh(node, 'podman inspect ' + NAME))[0]
        (out / f'{node}-identity-{phase}.json').write_text(json.dumps(info, indent=1) + '\n')
        env = dict(item.split('=', 1) for item in info['Config']['Env'])
        problems = environment_problems(env)
        if not info['State']['Running'] or info['Image'].removeprefix('sha256:') != pin['image_id']:
            problems.append('container is not the pinned release image')
        if info['Config']['Labels'].get('local-inference.ds41.kit.sha256') != expected_kit:
            problems.append('container kit differs from the validated local manifest')
        log = ssh(node, f'podman logs {NAME} 2>&1 | grep -F -e DS41-PRECISION-APPLIED -e DS41-RATIO1-EXTEND-PIN '
                        f'-e "b12x " -e "allocating 80022 serving blocks" -e "speculative_config" '
                        f'-e "SpecDecoding metrics" || true')
        (out / f'{node}-markers-{phase}.log').write_text(log)
        problems += check_precision_marker(log, rank)
        problems += pin_marker_problems(log, False)
        measured, ready_lines = measured_problems(log)
        problems += measured
        if 'allocating 80022 serving blocks' not in log:
            problems.append('boot did not report the pinned 80022-block allocation')
        if rank == 0:
            problems += speculative_problems(log)
            if not ready_lines:
                problems.append('head has no B12X measurement-count evidence')
        try:
            path = selection_path(info, env)
        except RuntimeError as error:
            raise RuntimeError(f'{node}: {error}') from None
        text = ssh(node, 'cat ' + path)
        (out / f'{node}-selection-{phase}.json').write_text(text)
        snapshot = json.loads((snapshot_dir / f'{node}.json').read_text())['records']
        selection = selection_check(json.loads(text)['records'], snapshot, int(FIXED_ENV['DS41_DECISION_ROW_BLOCKS']))
        if problems:
            raise RuntimeError(f'{node}: ' + '; '.join(problems))
        result[node] = {'Id': info['Id'], 'StartedAt': info['State']['StartedAt'], 'Image': info['Image'],
                        'kit': info['Config']['Labels'].get('local-inference.ds41.kit.sha256'),
                        'env': {k: v for k, v in sorted(env.items()) if k != 'HOSTNAME'},
                        'cache_source': HOST_CACHE, 'selection_path': path,
                        'selection_sha256': hashlib.sha256(text.encode()).hexdigest(), **selection,
                        'spec_decoding_lines': [l for l in log.splitlines() if 'SpecDecoding metrics' in l]}
    return result


def record(boot, out, snapshot_dir, ssh):
    import precision_release
    from runtime import kit_digest
    pin = json.loads((ROOT / 'candidate.json').read_text())
    precision_release.validate_build(pin, ROOT)
    out.mkdir(parents=True, exist_ok=False)
    expected_kit = kit_digest()
    before = identity(ssh, pin, out, 'before', snapshot_dir, expected_kit)
    command = [sys.executable, '-u', str(ROOT / 'qualify_original_needle.py'), '--input', str(INPUT),
               '--out', str(out / 'original-524k'), '--repeats', '3']
    (out / 'boot.json').write_text(json.dumps({'boot': boot, 'image_id': pin['image_id'], 'command': command,
                                               'fixed_env': FIXED_ENV}, indent=2) + '\n')
    with (out / 'original-524k.log').open('x') as stream:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(line, end='', flush=True)
        code = process.wait()
    after = identity(ssh, pin, out, 'after', snapshot_dir, expected_kit)
    (out / 'identity.json').write_text(json.dumps({'before': before, 'after': after}, indent=2) + '\n')
    keep = ('Id', 'StartedAt', 'Image', 'kit', 'env', 'cache_source', 'regime')
    if {n: {k: before[n][k] for k in keep} for n in NODES} != {n: {k: after[n][k] for k in keep} for n in NODES}:
        raise RuntimeError('Container identity, profile or regime changed during the boot record')
    stable_selections(out)
    print('FIXED-VERIFICATION-BOOT-RECORDED', boot, 'needle_exit', code, flush=True)


def stable_selections(directory):
    """Every rank's full selection records identical before and after the requests."""
    directory = Path(directory)
    for node in NODES:
        before = json.loads((directory / f'{node}-selection-before.json').read_text())
        after = json.loads((directory / f'{node}-selection-after.json').read_text())
        if before != after:
            added = sorted(set(after['records']) - set(before['records']))
            changed = sorted(k for k in before['records'] if after['records'].get(k) != before['records'][k])
            raise RuntimeError(f'{node}: selections changed during requests: {len(added)} added, '
                               f'{len(changed)} changed or lost')


def boot_identity(directory):
    """Per-rank identity and full selection records of one recorded fixed boot."""
    directory = Path(directory)
    stable_selections(directory)
    before = json.loads((directory / 'identity.json').read_text())['before']
    selections = {n: json.loads((directory / f'{n}-selection-before.json').read_text()) for n in NODES}
    return before, selections


def check_fixed_pair(a, b):
    """Two independent boots of one image, kit and per-rank profile with identical selection records."""
    (ia, sa), (ib, sb) = boot_identity(a), boot_identity(b)
    problems = []
    for node in NODES:
        x, y = ia[node], ib[node]
        if x['Id'] == y['Id'] or x['StartedAt'] == y['StartedAt']:
            problems.append(f'{node}: not independent containers')
        for key in ('Image', 'kit', 'cache_source', 'regime'):
            if x[key] != y[key]:
                problems.append(f'{node}: {key} differs between fixed boots')
        env_x = {k: v for k, v in x['env'].items() if k != 'DS41_KIT_SHA256'}
        env_y = {k: v for k, v in y['env'].items() if k != 'DS41_KIT_SHA256'}
        if env_x != env_y:
            problems.append(f'{node}: profile environment differs: {sorted(k for k in set(env_x) | set(env_y) if env_x.get(k) != env_y.get(k))}')
        if sa[node] != sb[node]:
            ra, rb = sa[node]['records'], sb[node]['records']
            problems.append(f'{node}: selection records differ between fixed boots: '
                            f'{len(set(ra) ^ set(rb))} keys in one only, '
                            f'{sum(1 for k in set(ra) & set(rb) if ra[k] != rb[k])} changed')
    if problems:
        raise RuntimeError('Fixed boots are not a matched pair: ' + '; '.join(problems))
    return {node: {'boot_a_added': ia[node]['added'], 'boot_b_added': ib[node]['added'],
                   'records': ia[node]['records']} for node in NODES}


def trials(directory):
    directory = Path(directory)
    report = json.loads((directory / 'original-524k/report.json').read_text())
    if len(report) != 3 or any(r['cached_tokens'] != 0 for r in report):
        raise RuntimeError('Need exactly three cold trials: ' + str(directory))
    return report, [json.loads((directory / f'original-524k/{r["repeat"]}.json').read_text())['response']['choices'][0]
                    for r in report]


def signature(choice):
    payload = {k: choice[k] for k in ('message', 'finish_reason', 'logprobs')}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def first_divergence(left, right):
    for index, (x, y) in enumerate(zip(left['logprobs']['content'], right['logprobs']['content'])):
        if x != y:
            return index
    return None if len(left['logprobs']['content']) == len(right['logprobs']['content']) else \
        min(len(left['logprobs']['content']), len(right['logprobs']['content']))


def compare(a, b, control, back):
    rows = {name: trials(path) for name, path in (('a', a), ('b', b), ('control', control), ('return', back))}
    selections = check_fixed_pair(a, b)
    signatures = {name: [signature(c) for c in choices] for name, (_, choices) in rows.items()}
    first = {name: choices[0] for name, (_, choices) in rows.items()}
    return {
        'scope': 'descriptive; supports or weakens the adaptive-path mechanism, never proves it the only cause',
        'within_boot_identical': {n: len(set(s)) == 1 for n, s in signatures.items()},
        'fixed_boots_equal': set(signatures['a']) == set(signatures['b']) and len(set(signatures['a'])) == 1,
        'adaptive_boots_equal': set(signatures['control']) == set(signatures['return']),
        'first_divergence_fixed_a_b': first_divergence(first['a'], first['b']),
        'first_divergence_fixed_vs_control': first_divergence(first['a'], first['control']),
        'first_divergence_control_return': first_divergence(first['control'], first['return']),
        'token0_equal_to_control': {n: first[n]['logprobs']['content'][0] == first['control']['logprobs']['content'][0]
                                    for n in ('a', 'b', 'return')},
        'answers': {n: [c['message']['content'] for c in choices] for n, (_, choices) in rows.items()},
        'correct': {n: [r['correct'] for r in report] for n, (report, _) in rows.items()},
        'margins_nats': {n: [r['first_token']['margin_nats'] for r in report] for n, (report, _) in rows.items()},
        'signatures': signatures,
        'fixed_pair_selections': selections,
        'selection_additions_note': 'records added relative to the release snapshot are listed per boot and '
                                    'must be identical between the two fixed boots; they are not assumed absent'}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('record')
    r.add_argument('--boot', choices=('a', 'b'), required=True)
    r.add_argument('--snapshot', type=Path, required=True, help='ratio1-namespace-snapshot receipt (release records)')
    r.add_argument('--out', type=Path, required=True)
    c = sub.add_parser('compare')
    for name in ('a', 'b', 'control'):
        c.add_argument('--' + name, type=Path, required=True)
    c.add_argument('--return', dest='back', type=Path, required=True)
    c.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.command == 'record':
        from build_activation_trace import ssh
        record(args.boot, args.out, args.snapshot, ssh)
        return
    result = compare(args.a, args.b, args.control, args.back)
    with args.out.open('x') as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print('FIXED-VERIFICATION-COMPARISON', json.dumps({k: result[k] for k in ('fixed_boots_equal', 'adaptive_boots_equal')}),
          flush=True)


if __name__ == '__main__':
    main()
