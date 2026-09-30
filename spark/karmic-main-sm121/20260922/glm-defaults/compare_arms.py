#!/usr/bin/env python3
"""Compare finite diagnostic arms with pinned images, profiles and harnesses."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from statistics import geometric_mean
import experiment

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('grid', ROOT.parents[2] / 'glm53/r38-spark/qualification/compare-qwen-grids.py')
grid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grid)


def compare(base_path, arm_path, arm, reference_arm='defaults'):
    if reference_arm == arm:
        raise SystemExit('Use distinct arms for this discriminator comparison')
    base, left = grid.load(base_path)
    trial, right = grid.load(arm_path)
    if set(left) != set(right):
        raise SystemExit('Grid coordinates differ')
    for key in ('version', 'decode_mode', 'duration_per_test', 'decode_warmup_seconds',
                'context_lengths', 'ignore_eos', 'max_tokens', 'chat_template_kwargs',
                'temperature', 'prefill_mode', 'concurrency_levels', 'model'):
        if base['metadata'][key] != trial['metadata'][key]:
            raise SystemExit('Protocol drift: ' + key)
    lock, record, digest = experiment.resolve((ROOT / 'profile.lock.json').read_bytes(), arm)
    for data in (base, trial):
        m = data['run_metadata']
        for key, value in {'image_id': lock['image_id'], 'checkpoint_revision': lock['checkpoint_revision'],
                           'profile_sha256': experiment.BASE_SHA256,
                           'recurrent_checkpoint_policy': 'request_boundaries'}.items():
            if m.get(key) != value:
                raise SystemExit('Pinned comparison identity mismatch: ' + key)
    if base['run_metadata'].get('experiment_arm', 'defaults') != reference_arm:
        raise SystemExit('Reference arm identity mismatch')
    reference, _, reference_digest = experiment.resolve((ROOT / 'profile.lock.json').read_bytes(), reference_arm)
    if reference_arm != 'defaults' and base['run_metadata'].get('effective_profile_sha256') != reference_digest:
        raise SystemExit('Reference effective profile mismatch')
    if trial['run_metadata'].get('experiment_arm') != arm or trial['run_metadata'].get('effective_profile_sha256') != digest:
        raise SystemExit('Candidate arm identity mismatch')
    def harness(data):
        m = data['run_metadata']
        return m.get('llm_decode_bench_sha256', m.get('harness_sha256'))
    if not harness(base) or harness(base) != harness(trial):
        raise SystemExit('Harness drift')
    def delta(a, b):
        return {reference_arm: a, arm: b, 'change_pct': 100 * (b / a - 1)}
    def window(profile):
        argv = profile['argv']
        return json.loads(argv[argv.index('--speculative-config') + 1])['num_speculative_tokens']
    for data, profile in ((base, reference), (trial, lock)):
        if int(data['run_metadata'].get('speculative_tokens', 3)) != window(profile):
            raise SystemExit('Speculative-window metadata mismatch')
        spec = json.loads(profile['argv'][profile['argv'].index('--speculative-config') + 1])
        if data['run_metadata'].get('rejection_sample_method', 'standard') != spec['rejection_sample_method']:
            raise SystemExit('Rejection-method metadata mismatch')
    metrics = ('aggregate_tps', 'server_steps_per_s', 'server_accept_len_effective')
    return {
        'arm': arm, 'environment_overrides': record['environment_overrides'], 'extra_argv': record['extra_argv'],
        'reference_arm': reference_arm,
        'speculative_tokens': {reference_arm: window(reference), arm: window(lock)},
        'step_rates_like_for_like': window(reference) == window(lock),
        'repeatability_established': False,
        'inputs': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in (base_path, arm_path)],
        'summary': [{'concurrency': c, **{m: delta(
            geometric_mean(r[m] for (cc, _), r in left.items() if cc == c),
            geometric_mean(r[m] for (cc, _), r in right.items() if cc == c)) for m in metrics}} for c in (1, 2, 4)],
        'cells': [{'concurrency': k[0], 'context_tokens': k[1],
                   **{m: delta(left[k][m], right[k][m]) for m in metrics}} for k in sorted(left)],
        'prefill': [{'context_tokens': int(k), **delta(base['prefill'][k]['tok_per_sec'], trial['prefill'][k]['tok_per_sec'])}
                    for k in sorted(base['prefill'], key=int)],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--arm', required=True, choices=set(experiment.ARMS) - {'defaults'})
    parser.add_argument('--reference-arm', default='defaults', choices=experiment.ARMS)
    args = parser.parse_args()
    print(json.dumps(compare(args.baseline, args.candidate, args.arm, args.reference_arm), indent=2, allow_nan=False))
