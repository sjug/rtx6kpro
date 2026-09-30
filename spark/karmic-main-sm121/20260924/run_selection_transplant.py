"""Record the selection-transplant boot and compare its first token with three receipt references.

Before and after the unchanged qualify_original_needle.py (exactly three cold trials), on every rank:
the gated precision release image with the validated kit; the normal profile plus exactly
DS41_DECISION_ROW_BLOCKS=81389, DS41_PREFILL_THRESHOLD=8192 and
DS41_SELECTION_REPLAY=transplant275-release-into-233036Z; /cache mounted from the transplant root; the
81389-block allocation line; one precision marker and no ratio-1 pin marker; zero B12X measurements on
every count line (head must have one); and selection records exactly equal to that rank's seeded
transplant file, before and after.

Report (root-cause discrimination only, never a profile or numerical-policy choice):
  token0 compared bit for bit with the full top-20 token-0 records of
    passing-anchor       fba61148 standard 8192 arm        739 by 0.125
    bf16-ratio1-variant  release selections, ratio1 BF16   510 by 4.125  (same ratio1 mode as this arm)
    release-control      release selections, ratio1 FP8   510 by 3.375
  An exact match supports sufficiency of that selection set for this input at this geometry; it does
  not establish geometry irrelevance in general. Matching none does not prove both sides contribute.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import seed_selection_transplant as transplant  # noqa: E402
from run_selection_replay import measured_problems, signature  # noqa: E402

NAME = 'ds41-flash-karmic-main-tp4'
ARM_ENV = {'DS41_DECISION_ROW_BLOCKS': '81389', 'DS41_PREFILL_THRESHOLD': '8192',
           'DS41_SELECTION_REPLAY': transplant.SELECTOR}
INPUT = ROOT.parents[1] / 'ds41/r38/receipts/20260916/admission-k7-1m-u80/needle-524288-input.json'


def arm_env(name='transplant275'):
    return dict(ARM_ENV, DS41_SELECTION_REPLAY=transplant.SETS[name]['selector'])


def environment_problems(env, name='transplant275'):
    import precision_release
    expected = arm_env(name)
    problems = precision_release.environment_problems({k: v for k, v in env.items() if k not in expected})
    return problems + [f'{k} is not {v}' for k, v in expected.items() if env.get(k) != v]


def identity(ssh, pin, out, phase, seeded, expected_kit, name='transplant275'):
    from claude_run_decision_capture import check_precision_marker
    from run_ratio1_arm import pin_marker_problems
    result = {}
    for rank, node in enumerate(transplant.NODES):
        info = json.loads(ssh(node, 'podman inspect ' + NAME))[0]
        (out / f'{node}-identity-{phase}.json').write_text(json.dumps(info, indent=1) + '\n')
        env = dict(item.split('=', 1) for item in info['Config']['Env'])
        host_root = transplant.SETS[name]['host_root']
        problems = environment_problems(env, name)
        if not info['State']['Running'] or info['Image'].removeprefix('sha256:') != pin['image_id']:
            problems.append('container is not the pinned release image')
        if info['Config']['Labels'].get('local-inference.ds41.kit.sha256') != expected_kit:
            problems.append('container kit differs from the validated local manifest')
        mounts = {m['Destination']: m['Source'] for m in info['Mounts']}
        if mounts.get('/cache') != host_root:
            problems.append(f'/cache is mounted from {mounts.get("/cache")!r}, not the transplant root')
        log = ssh(node, f'podman logs {NAME} 2>&1 | grep -F -e DS41-PRECISION-APPLIED -e DS41-RATIO1-EXTEND-PIN '
                        f'-e "b12x " -e "allocating 81389 serving blocks" || true')
        (out / f'{node}-markers-{phase}.log').write_text(log)
        problems += check_precision_marker(log, rank)
        problems += pin_marker_problems(log, False)
        measured, ready_lines = measured_problems(log)
        problems += measured
        if rank == 0 and not ready_lines:
            problems.append('head has no B12X measurement-count evidence')
        if 'allocating 81389 serving blocks' not in log:
            problems.append('boot did not report the pinned serving-block allocation')
        cache = env.get('B12X_COMPILE_CACHE_DIR', '')
        if not cache.startswith('/cache/jit/') or '..' in cache.split('/'):
            problems.append('unexpected B12X cache path')
        text = ssh(node, 'cat ' + host_root + cache.removeprefix('/cache') + '/preparation/' +
                   Path(transplant.replay.SELECTION).name)
        (out / f'{node}-selection-{phase}.json').write_text(text)
        records, expected = json.loads(text)['records'], json.loads(seeded[node])['records']
        if records != expected:
            added = sorted(set(records) - set(expected))
            changed = sorted(k for k in expected if records.get(k) != expected[k])
            problems.append(f'selection records differ from the seed: {len(added)} added, {len(changed)} changed or lost')
        if problems:
            raise RuntimeError(f'{node}: ' + '; '.join(problems))
        result[node] = {'Id': info['Id'], 'StartedAt': info['State']['StartedAt'], 'Image': info['Image'],
                        'kit': info['Config']['Labels'].get('local-inference.ds41.kit.sha256'),
                        'selection_sha256': hashlib.sha256(text.encode()).hexdigest(), 'ready_lines': ready_lines}
    return result


def classify(report, trials, refs):
    if len(report) != 3 or len(trials) != 3 or any(r['cached_tokens'] != 0 for r in report):
        raise RuntimeError('Transplant lacks exactly three cold trials')
    token0 = [t['response']['choices'][0]['logprobs']['content'][0] for t in trials]
    signatures = [signature(t['response']['choices'][0]) for t in trials]
    matches = {name: [record == ref for record in token0] for name, ref in refs.items()}
    if len({json.dumps(r, sort_keys=True) for r in token0}) != 1:
        verdict = 'token0-nonrepeatable'
    else:
        hits = [name for name, m in matches.items() if all(m)]
        verdict = 'token0-equals-' + hits[0] if len(hits) == 1 else ('token0-matches-none' if not hits else 'ambiguous')
    return {'verdict': verdict, 'token0_matches': matches, 'token0': token0[0],
            'signatures': signatures, 'within_boot_identical': len(set(signatures)) == 1,
            'correct': [r['correct'] for r in report], 'margins_nats': [r['first_token']['margin_nats'] for r in report],
            'scope': ('root-cause discrimination on one input at one geometry; an exact match supports sufficiency, '
                      'not general geometry irrelevance; matching none does not prove both sides contribute; '
                      'no numerical policy follows from this needle')}


def record(out, ssh, name='transplant275'):
    import precision_release
    from runtime import kit_digest
    pin = json.loads((ROOT / 'candidate.json').read_text())
    precision_release.validate_build(pin, ROOT)
    _, files = transplant.check(ROOT, name)
    refs = transplant.references(ROOT)
    seeded = {n: f['bytes'].decode() for n, f in files.items()}
    out.mkdir(parents=True, exist_ok=False)
    expected_kit = kit_digest()
    before = identity(ssh, pin, out, 'before', seeded, expected_kit, name)
    command = [sys.executable, '-u', str(ROOT / 'qualify_original_needle.py'), '--input', str(INPUT),
               '--out', str(out / 'original-524k'), '--repeats', '3']
    (out / 'command.json').write_text(json.dumps(command) + '\n')
    with (out / 'original-524k.log').open('x') as stream:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(line, end='', flush=True)
        code = process.wait()
    after = identity(ssh, pin, out, 'after', seeded, expected_kit, name)
    strip = lambda side: {n: {k: v for k, v in side[n].items() if k != 'ready_lines'} for n in side}
    (out / 'identity.json').write_text(json.dumps({'before': before, 'after': after}, indent=2) + '\n')
    if strip(before) != strip(after):
        raise RuntimeError('Container identity or selections changed during the transplant')
    report = json.loads((out / 'original-524k/report.json').read_text())
    trials = [json.loads((out / f'original-524k/{r["repeat"]}.json').read_text()) for r in report]
    result = classify(report, trials, refs) | {'needle_exit': code, 'set': name}
    (out / 'verdict.json').write_text(json.dumps(result, indent=2) + '\n')
    print('SELECTION-TRANSPLANT-VERDICT', result['verdict'], json.dumps(result['margins_nats']), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--set', choices=tuple(transplant.SETS), default='transplant275')
    a = p.parse_args()
    from build_activation_trace import ssh
    record(a.out, ssh, a.set)


if __name__ == '__main__':
    main()
