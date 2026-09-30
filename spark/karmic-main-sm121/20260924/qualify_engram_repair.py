"""Run the unchanged correctness gates plus the content-transition regression.

Run under run_observed.py. This does not restart nodes, alter runtime settings,
or run benchmarks. Cross-rank injected-fault qualification remains separate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
NODES = ('dusty', 'toby', 'rusty', 'kirby')


def validate_identity(info, image_id, digest, expected_env):
    if (not info['State']['Running']
            or info['Image'].removeprefix('sha256:') != image_id
            or info['Config']['Labels'].get('local-inference.ds41.kit.sha256') != digest):
        raise RuntimeError('Serving image, kit or running state mismatch')
    env = dict(item.split('=', 1) for item in info['Config']['Env'])
    if any(env.get(key) != value for key, value in expected_env.items()):
        raise RuntimeError('Serving environment mismatch')
    if 'VLLM_DS41_L2_PREFETCH' in env:
        raise RuntimeError('Unexpected prefetch control override')
    if 'DS41_ROUTER_CONTROL_BLOCKS' in env:
        raise RuntimeError('Fixed-capacity diagnostic is not the serving qualification profile')
    return {key: info[key] for key in ('Id', 'Image', 'RestartCount')} | {
        'StartedAt': info['State']['StartedAt'], 'Env': sorted(info['Config']['Env']),
        'Cmd': info['Config']['Cmd'], 'Labels': info['Config']['Labels']}


def validate_pin(pin, expected_kind):
    if pin.get('diagnostic', {}).get('kind') != expected_kind or 'diagnostic_overlay' in pin:
        raise RuntimeError('Not the selected uninstrumented repair candidate')


def validate_router_build(pin, build, raw_lock):
    lock = json.loads(raw_lock)
    digest = hashlib.sha256(raw_lock).hexdigest()
    if (pin['image_id'] != build['image_id'] or build['lock_sha256'] != digest
            or pin['diagnostic']['lock_sha256'] != digest
            or pin['diagnostic']['vllm_tree'] != lock['trees']['vllm']
            or pin['diagnostic']['b12x_tree'] != lock['trees']['b12x']):
        raise RuntimeError('Router qualification differs from the reviewed build identity')


def snapshot(out, phase, expected_kind='engram-prequeued-native-io-fix'):
    from runtime import kit_digest
    from launch_contract import render
    pin = json.loads((ROOT / 'candidate.json').read_text())
    validate_pin(pin, expected_kind)
    if expected_kind == 'router-stage-release-candidate':
        validate_router_build(pin,
            json.loads((ROOT / 'receipts/router-release-build-receipt.json').read_text()),
            (ROOT / 'router-release.lock.json').read_bytes())
    if expected_kind == 'precision-release-candidate':
        import precision_release
        precision_release.validate_build(pin, ROOT)
    digest = kit_digest()
    result = {}
    for node in NODES:
        raw = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', node,
            'podman inspect ds41-flash-karmic-main-tp4'], text=True)
        (out / f'{node}-identity-{phase}.json').write_text(raw)
        expected = render(node)['env'] | {
            'B12X_DYNAMIC_DETERMINISTIC_OUTPUT': '1', 'B12X_DENSE_SPLITK_TURBO': '0'}
        result[node] = validate_identity(json.loads(raw)[0], pin['image_id'], digest, expected)
        if expected_kind == 'precision-release-candidate':
            info = json.loads(raw)[0]
            problems = precision_release.environment_problems(
                dict(item.split('=', 1) for item in info['Config']['Env']))
            marker = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', node,
                'podman logs ds41-flash-karmic-main-tp4 2>&1 | grep -F DS41-PRECISION-APPLIED || true'], text=True)
            (out / f'{node}-precision-marker-{phase}.log').write_text(marker)
            problems += precision_release.marker_problems(marker, node)
            if problems:
                raise RuntimeError(f'{node}: ' + '; '.join(problems))
    return result


def check_transition(rows):
    if len(rows) != 6:
        raise RuntimeError('Expected three prime/target transition cycles')
    targets = []
    for cycle in range(3):
        for offset, label, count, answer in ((0, 'prime', 2, '826493'),
                                             (1, 'target', 3, '739184')):
            row = rows[cycle * 2 + offset]
            if (row['cycle'] != cycle or row['prior'] != 385 or row['label'] != label
                    or row['returncode'] != 0 or len(row['answers']) != count
                    or any(text.strip() != answer for text in row['answers'])
                    or len(row['logprob_sha256']) != count
                    or len(set(row['logprob_sha256'])) != 1):
                raise RuntimeError('Content-transition regression failed: ' + str(row))
            if label == 'target':
                targets.extend(row['logprob_sha256'])
    if len(set(targets)) != 1:
        raise RuntimeError('Original prompt changes across content-transition cycles')


def check_residual_stress(summary, records):
    lengths = {127, 128, 143, 247, 248, 249, 250, 251, 252, 253, 254,
               255, 256, 257, 258, 271, 383, 384, 511, 512, 767, 1023}
    if (len(summary) != len(lengths) or {r['length'] for r in summary} != lengths
            or len(records) != 1100 or any(r['trials'] != 50 or r['unique_signatures'] != 1
                                          or r['nonmodal_trials'] != 0 for r in summary)):
        raise RuntimeError('Residual nondeterminism stress failed or incomplete')
    if any(r['unexplained_ttft_ge_5s'] for r in records):
        raise RuntimeError('Unexplained five-second TTFT gap in residual stress')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--router-fence', action='store_true', help='Require the reviewed uninstrumented router-fence candidate')
    parser.add_argument('--precision-release', action='store_true',
                        help='Require the gated precision release candidate on the normal profile')
    args = parser.parse_args()
    if args.router_fence and args.precision_release:
        parser.error('Choose one candidate kind')
    args.out.mkdir(parents=True, exist_ok=False)

    def run(script, options, marker, label):
        path = ROOT / script
        command = [sys.executable, '-u', str(path), *map(str, options)]
        (args.out / f'{label}-command.json').write_text(json.dumps({
            'command': command, 'script_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        }, indent=2) + '\n')
        print('ENGRAM-QUALIFICATION-START', label, flush=True)
        with (args.out / f'{label}.log').open('x') as stream:
            process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end='', flush=True)
            code = process.wait()
        if code or marker not in (args.out / f'{label}.log').read_text().splitlines():
            raise RuntimeError('Gate failed: ' + label)

    kind = ('router-stage-release-candidate' if args.router_fence else
            'precision-release-candidate' if args.precision_release else 'engram-prequeued-native-io-fix')
    (args.out / 'driver.json').write_text(json.dumps({
        'kind': kind, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'candidate_sha256': hashlib.sha256((ROOT / 'candidate.json').read_bytes()).hexdigest(),
    }, indent=2) + '\n')
    print('QUALIFICATION-CANDIDATE-KIND', kind, flush=True)
    before = snapshot(args.out, 'before', kind)
    try:
        run_gates(run, args)
    finally:
        after = snapshot(args.out, 'after', kind)
        if after != before:
            raise RuntimeError('Serving identity changed during the qualification envelope')
    print('ENGRAM-MODEL-CORRECTNESS-PASS: injected fault and benchmark gates remain', flush=True)


def run_gates(run, args):
    run('probe_length_transition.py', [
        '--out', args.out / 'transition', '--corpus', ROOT / 'receipts/determinism-corpus.json',
        '--prime-corpus', ROOT / 'receipts/fresh-lookup-corpus-20260925.json',
        '--prime-filler', ' meadow', '--prime-code', '826493',
        '--prime-lengths', '385', '--cycles', '3'],
        'TRANSITION-DIAGNOSTIC-COMPLETE; not a qualification verdict', 'transition')
    check_transition(json.loads((args.out / 'transition/summary.json').read_text()))
    print('ENGRAM-CONTENT-TRANSITION-PASS', flush=True)
    run('probe_interleaved_repeatability.py', ['--out', args.out / 'residual-stress',
        '--cycles', '50', '--seed', '20260925'], 'INTERLEAVED-DIAGNOSTIC-COMPLETE', 'residual-stress')
    check_residual_stress(json.loads((args.out / 'residual-stress/summary.json').read_text()),
                          json.loads((args.out / 'residual-stress/records.json').read_text()))
    print('ENGRAM-RESIDUAL-STRESS-PASS', flush=True)
    run('qualify_dense_release.py', ['--out', args.out / 'correctness'],
        'DENSE-RELEASE-CORRECTNESS-PASS: concurrency/cache and benchmark remain separate gates',
        'correctness')
    run('claude_gate_conversations.py', ['--out', args.out / 'conversations',
        '--mixed-needle-input', ROOT.parents[1] /
        'ds41/r38/receipts/20260916/admission-k7-1m-u80/needle-524288-input.json'],
        'CLAUDE-CONVERSATIONS-PASS', 'conversations')
    # A final checked output also observes any sticky fault from preceding work.
    run('probe_repeatability.py', ['--corpus', ROOT / 'receipts/determinism-corpus.json',
        '--out', args.out / 'final-check', '--lengths', '385', '--repeats', '3'],
        'REPEATABILITY-PASS', 'final-check')


if __name__ == '__main__':
    main()
