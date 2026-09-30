#!/usr/bin/env python3
"""Offline verification of downloaded v3 MoE captures against their own trace receipts.

DIAGNOSTIC ONLY, CPU, no network. For every capture file it:
  1. loads it with torch.load(weights_only=True) on CPU;
  2. checks identity: arm, session (pid prefix in the file name), rank, node,
     generation, latched meta, file name pattern, and the matching `saved`
     status line (path, epoch, rows, runner, generation);
  3. checks layout: routed and input share shape and dtype, first dimension
     equals meta rows, top-k weights and ids share shape, ids lie in
     [0, logits.shape[1]), float32 logits and weights, int32/int64 ids;
  4. recomputes each tensor's digest with the tracer's own digest function
     and requires equality with the same arm/session/rank/epoch step record:
     input == #input_after (both taken after the op; #input, taken before
     the op, is reported), logits == #logits, weights == #topk_weights,
     ids == #topk_ids, routed == #routed, and routed == meta digest;
  5. classifies the captured call against the modal #routed digest of all
     aligned records with the same step key (rows, input-id and position
     digests) on that rank: `nonmodal` (the fault was captured), `modal` (the
     latch reference was itself the faulty first occurrence), or
     `undecided` (fewer than --min-group records or no strict majority). It
     also reports, per route input seam, whether the captured step was modal.

The tracer and analyzer used for digests and step keys must match the v3
lock (claude-moe-capture.lock.json), i.e. the code inside the built image.
Any identity, layout or digest mismatch fails (exit 1). Exit 0 means every
capture verified; classification is reported, never used to pass or fail.

    python3 claude_verify_moe_capture.py receipts/.../traces-00 --arm localize-...
        [--downloads receipts/.../traces-00/capture-downloads.json] [--json out.json]
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

import torch

HERE = Path(__file__).resolve().parent
ROUTE_SEAMS = ('#input', '#input_after', '#logits', '#topk_weights', '#topk_ids', '#shared')
FILE = re.compile(r'(?P<node>[a-z0-9]+)-(?P<pid>[0-9]+)-g(?P<gen>[0-9]+)-capture\.pt')


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Mismatch(Exception):
    pass


def require(condition, message):
    if not condition:
        raise Mismatch(message)


def expand(paths):
    """Each directory holding jsonl receipts (recursively), once, plus given files."""
    out = []
    for path in map(Path, paths):
        if path.is_file():
            out.append(path)
        else:
            out += sorted({p.parent for p in path.rglob('*.jsonl')})
            out += [p for p in path.rglob('*-capture.pt')]
    return list(dict.fromkeys(out))


def verify(paths, arm, *, downloads=None, min_group=3,
           tracer_path=HERE / 'claude-moe-capture-act_trace.py',
           analyzer_path=HERE / 'claude-moe-capture-act_analyze.py'):
    lock = json.loads((HERE / 'claude-moe-capture.lock.json').read_text())['inputs']
    for path in (Path(tracer_path), Path(analyzer_path)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != lock.get(path.name):
            raise SystemExit(f'CLAUDE-VERIFY-REFUSED: {path.name} is not the locked v3 file')
    tracer = _load('claude_verify_tracer', tracer_path)
    analyzer = _load('claude_verify_analyzer', analyzer_path)
    files = sorted({p for path in map(Path, paths)
                    for p in ([path] if path.is_file() else path.glob('*-capture.pt'))
                    if p.suffix == '.pt'})
    paths = [p for p in map(Path, paths) if not (p.is_file() and p.suffix == '.pt')]
    records = [analyzer.strip_capture(r) for r in analyzer.load(paths) if r.get('arm') == arm]
    status = [s for s in analyzer.load_status(paths) if s.get('arm') == arm]
    expected = {}
    if downloads is not None:
        for row in json.loads(Path(downloads).read_text()):
            expected[Path(row['file']).name] = row
    results, failures = [], []
    seen_names = set()
    for file in files:
        try:
            results.append(_verify_one(file, arm, records, status, tracer.digest, analyzer.step_key,
                                       expected, min_group))
        except Mismatch as error:
            failures.append({'file': str(file), 'error': str(error)})
        seen_names.add(file.name)
    for name in sorted(set(expected) - seen_names):
        failures.append({'file': name, 'error': 'listed in downloads but not found'})
    saved = {Path(s['path']).name for s in status if s.get('state') == 'saved'}
    for name in sorted(saved - seen_names):
        failures.append({'file': name, 'error': 'saved status line without a capture file'})
    return {'arm': arm, 'verified': results, 'failures': failures,
            'verdict': 'fail' if failures else ('pass' if results else 'no-captures')}


def _verify_one(file, arm, records, status, digest, step_key, expected, min_group):
    match = FILE.fullmatch(file.name)
    require(match, 'unexpected capture file name')
    if file.name in expected:
        row = expected[file.name]
        data = file.read_bytes()
        require(len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256'],
                'bytes or sha256 differ from capture-downloads.json')
    saved = torch.load(file, map_location='cpu', weights_only=True)
    meta, tensors = saved['meta'], saved['tensors']
    require(saved['arm'] == arm, f'arm {saved["arm"]} differs')
    require(saved['node'] == match['node'], 'node differs from file name')
    require(str(saved['session']).split('-', 1)[0] == match['pid'], 'session pid differs from file name')
    require(saved['generation'] == int(match['gen']) and saved['generation'] >= 1, 'generation differs')
    require(type(saved['rank']) is int and saved['rank'] >= 0, 'rank is not a non-negative int')
    require(meta['latched'] == 1, 'meta latch word is clear')
    require(saved['runner'] == saved['names'][meta['runner']], 'runner name and index disagree')
    runner, rows, epoch, rank = saved['runner'], meta['rows'], meta['epoch'], saved['rank']
    require(rows in saved['row_values'], 'captured rows not among the armed row values')

    lines = [s for s in status if s.get('state') == 'saved' and s.get('session') == saved['session']
             and s.get('rank') == rank and s.get('generation') == saved['generation']]
    require(len(lines) == 1, f'expected one saved status line, found {len(lines)}')
    line = lines[0]
    require(Path(line['path']).name == file.name and line['node'] == saved['node'], 'status path/node differ')
    require((line['epoch'], line['rows'], line['runner']) == (epoch, rows, runner),
            'status epoch/rows/runner differ from the capture meta')

    fields = ('input', 'logits', 'topk_weights', 'topk_ids', 'routed')
    require(set(tensors) == set(fields), f'tensor fields {sorted(tensors)}')
    inp, logits, weights, ids, routed = (tensors[k] for k in fields)
    require(routed.shape == inp.shape and routed.dtype == inp.dtype and routed.dim() == 2, 'routed/input layout')
    require(all(t.shape[0] == rows for t in tensors.values()), 'first dimension differs from meta rows')
    require(logits.dtype == torch.float32 and logits.dim() == 2, 'logits layout')
    require(weights.shape == ids.shape and weights.dtype == torch.float32, 'top-k weights layout')
    require(ids.dtype in (torch.int32, torch.int64), 'top-k ids dtype')
    require(bool(((ids >= 0) & (ids < logits.shape[1])).all()), 'top-k ids out of range')

    steps = [r for r in records if r['rank'] == rank and r.get('session') == saved['session']
             and r.get('epoch') == epoch]
    require(len(steps) == 1, f'expected one trace record at rank {rank} epoch {epoch}, found {len(steps)}')
    step = steps[0]
    require(step['rows'] == rows, 'trace record rows differ')
    require((step.get('capture') or {}).get('generation') == saved['generation'],
            'trace record belongs to another capture generation')
    index = {name: i for i, name in enumerate(step['names'])}
    for seam in ('#input', '#input_after', '#logits', '#topk_weights', '#topk_ids', '#routed'):
        require(runner + seam in index, f'trace record lacks {runner + seam}')
    got = {k: digest(t).tolist() for k, t in tensors.items()}
    trace = {seam: step['hashes'][index[runner + seam]] for seam in
             ('#input', '#input_after', '#logits', '#topk_weights', '#topk_ids', '#routed')}
    for field, seam in (('input', '#input_after'), ('logits', '#logits'), ('topk_weights', '#topk_weights'),
                        ('topk_ids', '#topk_ids'), ('routed', '#routed')):
        require(got[field] == trace[seam], f'{field} digest differs from trace {runner + seam}')
    require(got['routed'] == [meta['digest0'], meta['digest1']], 'routed digest differs from capture meta')

    key = step_key(step)
    group = [r for r in records if r['rank'] == rank and r.get('session') == saved['session']
             and step_key(r) == key and runner + '#routed' in r['names']]

    def modal(seam):
        values = Counter(tuple(r['hashes'][r['names'].index(runner + seam)]) for r in group
                         if runner + seam in r['names'])
        value, count = values.most_common(1)[0]
        decided = len(group) >= min_group and count * 2 > len(group)
        return (list(value) if decided else None), count

    routed_modal, count = modal('#routed')
    if routed_modal is None:
        classification = 'undecided'
    elif got['routed'] == routed_modal:
        classification = 'modal'
    else:
        classification = 'nonmodal'
    inputs_modal = {}
    for seam in ROUTE_SEAMS:
        value, _ = modal(seam)
        inputs_modal[seam] = None if value is None else step['hashes'][index[runner + seam]] == value
    return {'file': file.name, 'arm': arm, 'rank': rank, 'node': saved['node'], 'session': saved['session'],
            'generation': saved['generation'], 'epoch': epoch, 'rows': rows, 'runner': runner,
            'classification': classification, 'group_size': len(group), 'modal_count': count,
            'reference_is_modal': None if routed_modal is None else [meta['ref0'], meta['ref1']] == routed_modal,
            'route_inputs_modal': inputs_modal,
            'input_changed_by_op': trace['#input'] != trace['#input_after']}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('paths', nargs='+', help='trace directories (jsonl and *-capture.pt, recursively) or files')
    parser.add_argument('--arm', required=True)
    parser.add_argument('--downloads', type=Path)
    parser.add_argument('--min-group', type=int, default=3)
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    report = verify(expand(args.paths), args.arm, downloads=args.downloads, min_group=args.min_group)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    sys.exit(1 if report['verdict'] == 'fail' else 0)


if __name__ == '__main__':
    main()
