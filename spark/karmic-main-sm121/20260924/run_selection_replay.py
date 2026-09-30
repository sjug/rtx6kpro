"""Record the selection-replay boot and classify it against the passing arm's known signature.

Before and after the unchanged qualify_original_needle.py (three cold trials, margins), on every
rank: the pinned image is the gated precision release; the environment is the normal profile
plus exactly DS41_DECISION_ROW_BLOCKS=81389, DS41_PREFILL_THRESHOLD=8192 and
DS41_SELECTION_REPLAY=standard8192-233036Z; /cache is the seeded replay root; one precision
marker per rank and no ratio-1 pin marker; the selection file's records equal the seeded
passing file exactly (no addition, change or loss, which is how a B12X measurement would show);
and every "b12x ready" line the logs carry reports 0 measured.

Verdict (reported, not a qualification):
  identical-to-passing   all trials equal the passing signature fba61148...
  correct-different      correct, repeatable, but a different signature
  correct-nonrepeatable  correct answers with differing recorded signatures
  wrong                  any trial wrong
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
import seed_selection_replay as seeding  # noqa: E402

NAME = 'ds41-flash-karmic-main-tp4'
REPLAY_ENV = {'DS41_DECISION_ROW_BLOCKS': '81389', 'DS41_PREFILL_THRESHOLD': '8192',
              'DS41_SELECTION_REPLAY': 'standard8192-233036Z'}
READY = re.compile(r'b12x \S+ (\S+): .*?(\d+) measured, (\d+) cached')
INPUT = ROOT.parents[1] / 'ds41/r38/receipts/20260916/admission-k7-1m-u80/needle-524288-input.json'


def environment_problems(env):
    import precision_release
    problems = precision_release.environment_problems({k: v for k, v in env.items() if k not in REPLAY_ENV})
    problems += [f'{k} is not {v}' for k, v in REPLAY_ENV.items() if env.get(k) != v]
    return problems


def measured_problems(log):
    counts = [(m.group(1), int(m.group(2))) for m in READY.finditer(log)]
    return [f'B12X measured {n} candidates for {c}' for c, n in counts if n], len(counts)


def signature(choice):
    payload = {k: choice[k] for k in ('message', 'finish_reason', 'logprobs')}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def identity(ssh, pin, out, phase, seeded, expected_kit):
    from claude_run_decision_capture import check_precision_marker
    from run_ratio1_arm import pin_marker_problems
    result = {}
    for rank, node in enumerate(seeding.NODES):
        info = json.loads(ssh(node, 'podman inspect ' + NAME))[0]
        (out / f'{node}-identity-{phase}.json').write_text(json.dumps(info, indent=1) + '\n')
        env = dict(item.split('=', 1) for item in info['Config']['Env'])
        problems = environment_problems(env)
        if not info['State']['Running'] or info['Image'].removeprefix('sha256:') != pin['image_id']:
            problems.append('container is not the pinned release image')
        if info['Config']['Labels'].get('local-inference.ds41.kit.sha256') != expected_kit:
            problems.append('container kit differs from the validated local manifest')
        mounts = {m['Destination']: m['Source'] for m in info['Mounts']}
        if mounts.get('/cache') != seeding.HOST_ROOT:
            problems.append('/cache is not the seeded replay root')
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
        if not cache.startswith('/cache/jit/') or '..' in cache:
            problems.append('unexpected B12X cache path')
        text = ssh(node, 'cat ' + seeding.HOST_ROOT + cache.removeprefix('/cache') + '/preparation/' +
                   Path(seeding.SELECTION).name)
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
                        'selection_sha256': hashlib.sha256(text.encode()).hexdigest(),
                        'ready_lines': ready_lines}
    return result


def classify(report, trials, passing=seeding.PASSING_SIGNATURE, anchor_first=None):
    signatures = [signature(t['response']['choices'][0]) for t in trials]
    if len(report) != 3 or len(trials) != 3 or any(r['cached_tokens'] != 0 for r in report):
        raise RuntimeError('Replay lacks three cold trials')
    if not all(r['correct'] for r in report):
        verdict = 'wrong'
    elif len(set(signatures)) != 1:
        verdict = 'correct-nonrepeatable'
    elif set(signatures) == {passing}:
        verdict = 'identical-to-passing'
    else:
        verdict = 'correct-different'
    return {'verdict': verdict, 'signatures': signatures, 'within_boot_identical': len(set(signatures)) == 1,
            'first_token_matches_anchor': [t['response']['choices'][0]['logprobs']['content'][0] == anchor_first
                                           for t in trials] if anchor_first is not None else None,
            'passing_signature': passing, 'correct': [r['correct'] for r in report],
            'margins_nats': [r['first_token']['margin_nats'] for r in report],
            'scope': 'selection replay on the exact release image; not a qualification and not a model-correctness claim'}


def record(out, ssh):
    import precision_release
    from runtime import kit_digest
    pin = json.loads((ROOT / 'candidate.json').read_text())
    precision_release.validate_build(pin, ROOT)
    _, files = seeding.check(ROOT)
    seeded = {n: f['bytes'].decode() for n, f in files.items()}
    out.mkdir(parents=True, exist_ok=False)
    expected_kit = kit_digest()
    before = identity(ssh, pin, out, 'before', seeded, expected_kit)
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
    after = identity(ssh, pin, out, 'after', seeded, expected_kit)
    if {n: {k: v for k, v in before[n].items() if k != 'ready_lines'} for n in before} != \
            {n: {k: v for k, v in after[n].items() if k != 'ready_lines'} for n in after}:
        raise RuntimeError('Container identity or selections changed during the replay')
    report = json.loads((out / 'original-524k/report.json').read_text())
    trials = [json.loads((out / f'original-524k/{r["repeat"]}.json').read_text()) for r in report]
    anchor = json.loads((ROOT / seeding.SOURCE / 'unarmed-1/trial.json').read_text())
    first = anchor['response']['choices'][0]['logprobs']['content'][0]
    result = classify(report, trials, anchor_first=first) | {'needle_exit': code, 'identity': before}
    (out / 'verdict.json').write_text(json.dumps(result, indent=2) + '\n')
    print('SELECTION-REPLAY-VERDICT', result['verdict'], json.dumps(result['margins_nats']), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    from build_activation_trace import ssh
    record(a.out, ssh)


if __name__ == '__main__':
    main()
