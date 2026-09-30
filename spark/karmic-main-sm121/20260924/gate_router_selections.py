"""Check observed post-boot selections before the matched router test.

A capacity-only verdict is explicitly conditional, not proof that the entire
runtime execution is identical. Preserve the per-rank comparison for review.
"""
import argparse
import hashlib
import json
from pathlib import Path

from claude_compare_selections import compare, load
from router_tuning_transfer import NODES, PARENT

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--parent-boot', type=Path, required=True)
    parser.add_argument('--candidate-boot', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    manifests = [json.loads((p / 'manifest.json').read_text()) for p in (args.parent, args.candidate)]
    build = json.loads((ROOT / 'receipts/router-release-build-receipt.json').read_text())
    if manifests[0]['image_id'] != PARENT or manifests[1]['image_id'] != build['image_id']:
        raise RuntimeError('Selection snapshots do not name the comparison images')
    for directory, manifest in zip((args.parent, args.candidate), manifests):
        if set(manifest['nodes']) != set(NODES):
            raise RuntimeError('Incomplete rank coverage')
        for node in NODES:
            if hashlib.sha256((directory / (node + '.json')).read_bytes()).hexdigest() != manifest['nodes'][node]['sha256']:
                raise RuntimeError('Selection receipt digest differs: ' + node)
    reports, problems = {}, []
    for node in NODES:
        parent = load(args.parent / (node + '.json'))
        report = compare(parent, load(args.candidate / (node + '.json')), seed=parent,
                         parent_log=args.parent_boot / (node + '-final.log'),
                         candidate_log=args.candidate_boot / (node + '-final.log'))
        reports[node] = report
        if report['verdict'] not in ('identical-selections', 'capacity-only (conditional)'):
            problems.append(node + ': ' + report['verdict'])
        if any(report['undecoded_mhc'].values()):
            problems.append(node + ': undecoded mHC selections')
        router = report['decoded'].get('router', {})
        mhc = sum(len(value) for key, value in report['decoded'].items() if key.startswith('mhc.'))
        if len(router) != 31 or mhc != 54:
            problems.append(node + ': router/mHC plan inventory differs')
        for rows in ('256', '8192'):
            expected = {'backend': 'prefill', 'rows_per_tile': 8}
            if router.get(rows, {}).get('candidate') != expected or not router.get(rows, {}).get('same'):
                problems.append(node + ': router selection differs at ' + rows)
    # Preparation progress is emitted only by rank zero, not by every rank.
    ready = reports['dusty']['logs']['candidate']['ready']
    for component, cached in (('comm.roce@1051', 874), ('gemm.block_fp8_linear@503', 168)):
        if ready.get(component) != {'measured': 0, 'cached': cached}:
            problems.append('Head preparation progress differs: ' + component)
    args.out.write_text(json.dumps({'reports': reports, 'problems': problems,
                                   'scope': 'Matched selections; capacity-dependent bind queries remain a disclosed confound'},
                                  indent=2, sort_keys=True) + '\n')
    if problems:
        raise RuntimeError('; '.join(problems))
    print('ROUTER-SELECTION-GATE-PASS', {n: r['verdict'] for n, r in reports.items()}, flush=True)


if __name__ == '__main__':
    main()
