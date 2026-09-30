"""Independent two-boot record of the original 524K retrieval, and descriptive parent timing.

record  (on a serving release boot, under run_observed.py): identity snapshot with the
        release gates, the unchanged qualify_original_needle.py (three cold trials), identity
        snapshot again. The full qualification's own original-524k result is boot A; a
        second cold boot recorded here is boot B.
compare (local, no node contact): each boot must pass on its own (all trials correct and
        identical within the boot), both on the same image, kit and lock, and on different
        containers with different start times on every rank. Cross-boot signature equality
        and all timing against the router parent's existing receipts are reported, never
        required, and no parent model run is needed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

import precision_release as release

ROOT = Path(__file__).resolve().parent
INPUT = ROOT.parents[1] / 'ds41/r38/receipts/20260916/admission-k7-1m-u80/needle-524288-input.json'
PARENT_GATES = ROOT / 'receipts/router-b2-full-qualification-20260926/gates'


def signature(trial):
    choice = trial['response']['choices'][0]
    payload = {key: choice[key] for key in ('message', 'finish_reason', 'logprobs')}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def boot(directory):
    """Identity and original-524k result of a record directory or a full qualification gates directory."""
    directory = Path(directory)
    identity_dir = next(d for d in (directory, directory / 'correctness')
                        if (d / 'dusty-identity-before.json').is_file())
    needle = next(d for d in (directory / 'original-524k', directory / 'correctness/original-524k')
                  if (d / 'report.json').is_file())
    identities = {}
    for node in release.NODES:
        info = json.loads((identity_dir / f'{node}-identity-before.json').read_text())[0]
        identities[node] = {'Id': info['Id'], 'StartedAt': info['State']['StartedAt'],
                            'Image': info['Image'].removeprefix('sha256:'),
                            'kit': info['Config']['Labels'].get('local-inference.ds41.kit.sha256')}
    report = json.loads((needle / 'report.json').read_text())
    trials = [json.loads((needle / f'{row["repeat"]}.json').read_text()) for row in report]
    return {'path': str(directory), 'identities': identities, 'report': report,
            'signatures': [signature(t) for t in trials]}


def compare(a, b, parent=PARENT_GATES):
    boots = [boot(a), boot(b)]
    for item in boots:
        if len(item['report']) < 3 or not all(r['correct'] and r['identical'] and r['cached_tokens'] == 0
                                              for r in item['report']):
            raise RuntimeError('A boot did not pass the original retrieval on its own: ' + item['path'])
        if len(set(item['signatures'])) != 1:
            raise RuntimeError('Signatures differ within one boot: ' + item['path'])
    for node in release.NODES:
        left, right = boots[0]['identities'][node], boots[1]['identities'][node]
        if (left['Image'], left['kit']) != (right['Image'], right['kit']):
            raise RuntimeError('Boots differ in image or kit on ' + node)
        if left['Id'] == right['Id'] or left['StartedAt'] == right['StartedAt']:
            raise RuntimeError('Not two independent boots on ' + node)
    parent_report = json.loads((parent / 'correctness/original-524k/report.json').read_text())
    return {'scope': 'two independent release boots of the unchanged original 524K gate; '
                     'cross-boot equality and timing are descriptive only',
            'boots': [{k: v for k, v in item.items() if k != 'report'} | {
                'elapsed_s': [r['elapsed_s'] for r in item['report']]} for item in boots],
            'cross_boot_signature_equal': boots[0]['signatures'][0] == boots[1]['signatures'][0],
            'parent_original_524k_elapsed_s': [r['elapsed_s'] for r in parent_report],
            'parent_original_524k_correct': [r['correct'] for r in parent_report]}


def stress_timing(release_gates, parent=PARENT_GATES):
    """Median request and server seconds per length, release against the parent's 1100-request stress."""
    def medians(path):
        records = json.loads((Path(path) / 'residual-stress/records.json').read_text())
        lengths = sorted({r['length'] for r in records})
        return {n: {'elapsed_s': statistics.median(r['elapsed_s'] for r in records if r['length'] == n),
                    'server_s': statistics.median(r['server_seconds'] for r in records if r['length'] == n)}
                for n in lengths}
    left, right = medians(parent), medians(release_gates)
    if set(left) != set(right):
        raise RuntimeError('Stress length sets differ')
    return {'scope': 'cross-boot, same request set and profile; descriptive, not a gate',
            'per_length': {n: {'parent': left[n], 'release': right[n],
                               'release_over_parent_elapsed': right[n]['elapsed_s'] / left[n]['elapsed_s']}
                           for n in left}}


def record(out):
    from qualify_engram_repair import snapshot
    out.mkdir(parents=True, exist_ok=False)
    before = snapshot(out, 'before', release.KIND)
    command = [sys.executable, '-u', str(ROOT / 'qualify_original_needle.py'), '--input', str(INPUT),
               '--out', str(out / 'original-524k'), '--repeats', '3']
    (out / 'original-524k-command.json').write_text(json.dumps({'command': command, 'script_sha256':
        hashlib.sha256((ROOT / 'qualify_original_needle.py').read_bytes()).hexdigest()}, indent=2) + '\n')
    with (out / 'original-524k.log').open('x') as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
    after = snapshot(out, 'after', release.KIND)
    if after != before:
        raise RuntimeError('Serving identity changed during the boot record')
    if code or 'ORIGINAL-NEEDLE-PASS' not in (out / 'original-524k.log').read_text().splitlines():
        raise SystemExit('PRECISION-RELEASE-BOOT-FAIL: original retrieval did not pass on this boot')
    print('PRECISION-RELEASE-BOOT-RECORDED', out, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    r = sub.add_parser('record')
    r.add_argument('--out', type=Path, required=True)
    c = sub.add_parser('compare')
    c.add_argument('--boot-a', type=Path, required=True, help='full qualification gates directory')
    c.add_argument('--boot-b', type=Path, required=True, help='second-boot record directory')
    c.add_argument('--release-gates', type=Path, help='release gates directory with residual-stress')
    c.add_argument('--out', type=Path, required=True)
    a = parser.parse_args()
    if a.command == 'record':
        record(a.out)
        return
    result = compare(a.boot_a, a.boot_b)
    if a.release_gates:
        result['residual_stress_timing'] = stress_timing(a.release_gates)
    with a.out.open('x') as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print('PRECISION-RELEASE-TWO-BOOT-PASS', json.dumps({'cross_boot_signature_equal': result['cross_boot_signature_equal']}),
          flush=True)


if __name__ == '__main__':
    main()
