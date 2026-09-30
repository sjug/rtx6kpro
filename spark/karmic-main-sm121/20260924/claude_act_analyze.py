"""Offline analysis of claude_act_trace JSONL records; stdlib only.

Usage: python3 claude_act_analyze.py <dir-or-jsonl> [...] [--arm TOKEN] [--ranks 0,1,2,3] [--json out.json]

Inference rules (a step that fails any of them is counted but never used as
evidence):
  * One arm. Records from several arm tokens are refused unless --arm selects
    one. Within the arm each rank must have exactly one session (process);
    several sessions (restarts, combined boots) are refused.
  * Cross-rank identity is the target model's Engram epoch, which advances
    identically on every rank for every step. A step is ALIGNED only if every
    expected rank has exactly one record for that epoch and all of them agree
    on the step key (row count and the digests of input IDs and positions).
    Otherwise it is incomplete (a rank missing), duplicate, or misaligned
    (keys disagree). Records without an epoch are excluded.
  * A record is COVERED only if it did not overflow and holds at least one
    module digest besides <input>, <positions> and <output>. Output-only
    records can show that a step diverged but cannot localize it; they are
    reported separately and never count as clean module coverage.
  * The reference for each (rank, step key) is the modal digest vector of its
    aligned, covered records; a group with fewer than --min-group records or
    no strict majority is undecided.
Verdict: no-coverage, no-divergence-observed, or divergence-localized.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import sys

SPECIAL = ('<input>', '<positions>')


class AnalysisRefused(RuntimeError):
    pass


def load(paths):
    records = []
    for path in paths:
        path = Path(path)
        files = sorted(path.glob('*.jsonl')) if path.is_dir() else [path]
        for file in files:
            for number, line in enumerate(file.read_text().splitlines(), 1):
                if line.strip():
                    record = json.loads(line)
                    if record.get('type') == 'step':
                        record['_source'] = f'{file.name}:{number}'
                        records.append(record)
    return records


def step_key(record):
    names, hashes = record['names'], record['hashes']
    return (record['rows'],) + tuple(tuple(hashes[names.index(s)]) if s in names else None for s in SPECIAL)


def module_count(record):
    return sum(1 for name in record['names'] if name not in SPECIAL and not name.startswith('<output>'))


def analyze(records, *, arm=None, ranks=None, min_group=3):
    arms = sorted({r.get('arm') for r in records})
    if arm is None:
        if len(arms) != 1:
            raise AnalysisRefused(f'records span arms {arms}; select one with --arm')
        arm = arms[0]
    records = [r for r in records if r.get('arm') == arm]
    sessions = defaultdict(set)
    for r in records:
        sessions[r['rank']].add(r.get('session'))
    multi = {rank: sorted(map(str, s)) for rank, s in sessions.items() if len(s) != 1}
    if multi:
        raise AnalysisRefused(f'ranks with several sessions in arm {arm}: {multi}')
    expected = sorted(ranks) if ranks is not None else sorted(sessions)
    counts = Counter()
    by_epoch = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.get('epoch') is None:
            counts['no_epoch'] += 1
            continue
        by_epoch[r['epoch']][r['rank']].append(r)
    aligned = {}
    for epoch, per_rank in by_epoch.items():
        if any(len(v) > 1 for v in per_rank.values()):
            counts['duplicate'] += 1
        elif sorted(per_rank) != expected:
            counts['incomplete'] += 1
        elif len({step_key(v[0]) for v in per_rank.values()}) != 1:
            counts['misaligned'] += 1
        else:
            aligned[epoch] = {rank: v[0] for rank, v in per_rank.items()}
    usable, output_only = {}, {}
    for epoch, per_rank in aligned.items():
        if any(r.get('overflow') for r in per_rank.values()):
            counts['overflow'] += 1
        elif all(module_count(r) > 0 for r in per_rank.values()):
            usable[epoch] = per_rank
        else:
            output_only[epoch] = per_rank
            counts['uncovered'] += 1
    groups = defaultdict(list)
    for epoch, per_rank in usable.items():
        for rank, r in per_rank.items():
            groups[(rank, step_key(r))].append(r)
    modal, undecided = {}, []
    for (rank, key), members in groups.items():
        vectors = Counter((tuple(r['names']), tuple(map(tuple, r['hashes']))) for r in members)
        vector, count = vectors.most_common(1)[0]
        if len(members) < min_group or count * 2 <= len(members):
            undecided.append({'rank': rank, 'rows': key[0], 'records': len(members), 'modal_count': count})
        else:
            modal[(rank, key)] = vector
    divergent, clean, clean_ms = [], 0, []
    for epoch, per_rank in sorted(usable.items()):
        if any((rank, step_key(r)) not in modal for rank, r in per_rank.items()):
            counts['undecided_steps'] += 1
            continue
        findings = {}
        for rank, r in per_rank.items():
            names, hashes = tuple(r['names']), tuple(map(tuple, r['hashes']))
            ref_names, ref_hashes = modal[(rank, step_key(r))]
            if (names, hashes) == (ref_names, ref_hashes):
                continue
            if names != ref_names:
                index = next((i for i, (a, b) in enumerate(zip(names, ref_names)) if a != b),
                             min(len(names), len(ref_names)))
                findings[rank] = {'index': index, 'name': names[index] if index < len(names) else None,
                                  'structure_differs': True}
            else:
                index = next(i for i, (a, b) in enumerate(zip(hashes, ref_hashes)) if a != b)
                findings[rank] = {'index': index, 'name': names[index], 'structure_differs': False,
                                  'differing': sum(a != b for a, b in zip(hashes, ref_hashes))}
        if not findings:
            clean += 1
            clean_ms.extend(r['gpu_ms'] for r in per_rank.values() if r.get('gpu_ms') is not None)
            continue
        earliest = min(findings.values(), key=lambda f: f['index'])
        divergent.append({'epoch': epoch, 'rows': next(iter(per_rank.values()))['rows'],
                          'divergent_ranks': sorted(findings), 'clean_ranks': sorted(set(per_rank) - set(findings)),
                          'first_by_rank': {str(k): v for k, v in sorted(findings.items())},
                          'earliest_name': earliest['name'],
                          'wall': min(r['wall'] for r in per_rank.values()),
                          'gpu_ms': {str(k): r.get('gpu_ms') for k, r in sorted(per_rank.items())}})
    counts.update(records=len(records), aligned=len(aligned), covered=len(usable),
                  clean=clean, divergent=len(divergent))
    verdict = ('no-coverage' if clean + len(divergent) == 0
               else 'divergence-localized' if divergent else 'no-divergence-observed')
    timing = {}
    if clean_ms:
        ordered = sorted(clean_ms)
        timing = {'clean_median_ms': statistics.median(ordered),
                  'clean_p90_ms': ordered[int(0.9 * (len(ordered) - 1))]}
    return {'arm': arm, 'expected_ranks': expected, 'verdict': verdict, 'counts': dict(counts),
            'divergent_steps': divergent, 'undecided_groups': undecided, 'timing': timing,
            'output_only_epochs': sorted(output_only)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('paths', nargs='+')
    parser.add_argument('--arm')
    parser.add_argument('--ranks', help='expected TP ranks, e.g. 0,1,2,3 (default: ranks seen)')
    parser.add_argument('--min-group', type=int, default=3)
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    ranks = [int(r) for r in args.ranks.split(',')] if args.ranks else None
    try:
        report = analyze(load(args.paths), arm=args.arm, ranks=ranks, min_group=args.min_group)
    except AnalysisRefused as error:
        print(json.dumps({'verdict': 'refused', 'reason': str(error)}))
        sys.exit(2)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('arm', 'verdict', 'counts')}))
    for step in report['divergent_steps']:
        print(f"epoch {step['epoch']} rows {step['rows']}: divergent ranks {step['divergent_ranks']} "
              f"clean {step['clean_ranks']} earliest {step['earliest_name']}")
    sys.exit(0)


if __name__ == '__main__':
    main()
