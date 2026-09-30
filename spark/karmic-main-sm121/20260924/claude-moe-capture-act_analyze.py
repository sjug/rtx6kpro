"""Offline analysis of claude_act_trace JSONL records; stdlib only.

Usage: python3 claude-moe-capture-act_analyze.py <dir-or-jsonl> [...] [--arm TOKEN] [--ranks 0,1,2,3]
       [--require PATTERN[=COUNT] ...] [--json out.json]

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
  * --require (repeatable) is a coverage gate: every record of a step must
    contain at least one name (or exactly COUNT names) matching PATTERN,
    segment-wise as in the tracer ('layers.*.ffn.experts#post_reduce=40'
    requires all 40 post-collective seams). Steps that fail are counted as
    missing_required and are never evidence.
  * The reference for each (rank, step key) is the modal digest vector of its
    aligned, covered records; a group with fewer than --min-group records or
    no strict majority is undecided.
Verdict: no-coverage, no-divergence-observed, or divergence-localized.

v3 records: raw `<capture>` entries ([latched, epoch], not digests) are moved
out of the comparison into record['capture_word']; `type: capture` status
lines are summarized per (rank, session, generation), and a generation whose
latch word was seen without a `saved` status is reported as incomplete, so a
lost capture never reads as "no divergence"; seams add #input, #logits,
#topk_weights, #topk_ids and #input_after before #shared/#routed.

Reading MoE seams (v2 tracer): within one layer the call order is
NAME#shared, NAME#routed, NAME#pre_reduce, NAME#post_reduce. For every
divergent rank, `differing` lists all differing names in call order, so the
first divergent layer shows whether a rank-local partial (#shared or
#routed), the local combine (#pre_reduce with both partials modal) or the
collective (#post_reduce with every rank's #pre_reduce modal) changed first.
"""
import argparse
from collections import Counter, defaultdict
import fnmatch
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
                        strip_capture(record)
                        records.append(record)
    return records


def strip_capture(record):
    """Move raw <capture> latch words out of the compared vector."""
    if '<capture>' in record['names']:
        keep = [i for i, n in enumerate(record['names']) if n != '<capture>']
        record['capture_word'] = record['hashes'][record['names'].index('<capture>')]
        record['names'] = [record['names'][i] for i in keep]
        record['hashes'] = [record['hashes'][i] for i in keep]
    return record


def load_status(paths):
    lines = []
    for path in paths:
        path = Path(path)
        for file in (sorted(path.glob('*.jsonl')) if path.is_dir() else [path]):
            for line in file.read_text().splitlines():
                if line.strip():
                    record = json.loads(line)
                    if record.get('type') == 'capture':
                        lines.append(record)
    return lines


def capture_summary(records, status_lines, arm):
    """Per (rank, session, generation): latched epochs seen and final state."""
    groups = defaultdict(lambda: {'latched_epochs': set(), 'states': []})
    for r in records:
        info, word = r.get('capture'), r.get('capture_word')
        if r.get('arm') != arm or not info:
            continue
        group = groups[(r['rank'], r['session'], info['generation'])]
        if word and word[0]:
            group['latched_epochs'].add(word[1])
    for line in status_lines:
        if line.get('arm') != arm:
            continue
        group = groups[(line['rank'], line['session'], line['generation'])]
        group['states'].append(line['state'])
        if line.get('latched_epoch') is not None:
            group['latched_epochs'].add(line['latched_epoch'])
    summary, incomplete = [], []
    for (rank, session, generation), group in sorted(groups.items(), key=lambda kv: str(kv[0])):
        entry = {'rank': rank, 'session': session, 'generation': generation,
                 'latched_epochs': sorted(group['latched_epochs']), 'states': group['states']}
        summary.append(entry)
        if group['latched_epochs'] and 'saved' not in group['states']:
            incomplete.append(entry)
        elif 'error' in group['states']:
            incomplete.append(entry)
    return summary, incomplete


def step_key(record):
    names, hashes = record['names'], record['hashes']
    return (record['rows'],) + tuple(tuple(hashes[names.index(s)]) if s in names else None for s in SPECIAL)


def matches(name, pattern):
    parts, wanted = name.split('.'), pattern.split('.')
    return len(parts) == len(wanted) and all(fnmatch.fnmatchcase(a, b) for a, b in zip(parts, wanted))


def parse_require(text):
    pattern, sep, count = text.rpartition('=')
    if sep and count.isdigit():
        return pattern, int(count)
    return text, None


def meets(record, require):
    for pattern, count in require:
        found = sum(1 for name in record['names'] if matches(name, pattern))
        if found == 0 or (count is not None and found != count):
            return False
    return True


def module_count(record):
    return sum(1 for name in record['names'] if name not in SPECIAL and not name.startswith('<output>'))


def analyze(records, *, arm=None, ranks=None, min_group=3, require=(), status_lines=()):
    require = [parse_require(r) if isinstance(r, str) else tuple(r) for r in require]
    for record in records:
        strip_capture(record)
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
        elif not all(meets(r, require) for r in per_rank.values()):
            counts['missing_required'] += 1
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
                changed = [i for i, (a, b) in enumerate(zip(hashes, ref_hashes)) if a != b]
                findings[rank] = {'index': changed[0], 'name': names[changed[0]], 'structure_differs': False,
                                  'differing': len(changed),
                                  'differing_names': [names[i] for i in changed[:64]]}
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
    capture, capture_incomplete = capture_summary(records, status_lines, arm)
    return {'arm': arm, 'expected_ranks': expected, 'verdict': verdict, 'counts': dict(counts),
            'capture': capture, 'capture_incomplete': capture_incomplete,
            'divergent_steps': divergent, 'undecided_groups': undecided, 'timing': timing,
            'output_only_epochs': sorted(output_only)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('paths', nargs='+')
    parser.add_argument('--arm')
    parser.add_argument('--ranks', help='expected TP ranks, e.g. 0,1,2,3 (default: ranks seen)')
    parser.add_argument('--min-group', type=int, default=3)
    parser.add_argument('--require', action='append', default=[], help='PATTERN or PATTERN=COUNT')
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    ranks = [int(r) for r in args.ranks.split(',')] if args.ranks else None
    try:
        report = analyze(load(args.paths), arm=args.arm, ranks=ranks, min_group=args.min_group,
                         require=args.require, status_lines=load_status(args.paths))
    except AnalysisRefused as error:
        print(json.dumps({'verdict': 'refused', 'reason': str(error)}))
        sys.exit(2)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('arm', 'verdict', 'counts', 'capture_incomplete')}))
    for step in report['divergent_steps']:
        print(f"epoch {step['epoch']} rows {step['rows']}: divergent ranks {step['divergent_ranks']} "
              f"clean {step['clean_ranks']} earliest {step['earliest_name']}")
    sys.exit(0)


if __name__ == '__main__':
    main()
