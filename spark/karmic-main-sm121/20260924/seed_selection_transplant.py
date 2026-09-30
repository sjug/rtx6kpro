"""Seed the selection-transplant root: passing selections with exactly the 275 changed records from the release.

Root-cause discrimination only; never a serving profile or a numerical-policy choice.

Per rank, from receipts only:
  passing  receipts/decision-row-matched8192-nccl-standard-upstream-capture-20260926T233036Z/<rank>-selection.json
           (the full-selection replay's source, re-verified by seed_selection_replay.check)
  release  receipts/ratio1-namespace-snapshot-20260927T010531Z/<rank>.json
           (the release selections the control and return boots ran with; hashes checked against snapshot.json)
The transplant set is every shared key whose config or assignment differs. It must be exactly 275 keys
with the frozen digest below, identical on all ranks, with config and assignment differing together.
Those records are copied whole (assignment, config, coverage, programs) from the release; every other
passing record, including the 536 shared records that differ only in programs or coverage metadata and
the passing-only capacity records, is kept byte-for-byte as parsed. The result is written as JSON only
into a new root inside the existing remote task tree; an existing root is refused.

check  (local): derive, verify and print per-rank digests.
seed   (all nodes, idle): write each rank's transplanted file with the same fail-closed command as the replay.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(ROOT))
import seed_selection_replay as replay  # noqa: E402  (used unchanged)

NODES = replay.NODES
SELECTOR = 'transplant275-release-into-233036Z'
HOST_ROOT = '/home/jugs/git/ds41-r38/karmic-main-20260924/selection-transplant-cache-275-233036Z'
RELEASE_SNAPSHOT = 'receipts/ratio1-namespace-snapshot-20260927T010531Z'
TRANSPLANT_COUNT = 275
TRANSPLANT_KEYS_SHA256 = '63f34e9f7d937f709db7dffc5b226364e48939bccd51986962f9c3fb988ee45e'
METADATA_ONLY_COUNT = 536
# Contingent narrowing: only the two mHC 8192-row records, rebuilt from claude_compare_selections.mhc_key.
MHC8192_KEYS = {'mhc.pre.8192': '2394c70075c7fe2b87a1bd135e1ba48faaf75dfbb048b7f683b8df73d6de75c3',
                'mhc.pre.expanded.8192': '1b6c36a14eaadabaa5b558f21ee4755eb16d6f5c5bc7c1f8ff4daae6b64f10a6'}
SETS = {
    'transplant275': {'selector': SELECTOR, 'host_root': HOST_ROOT, 'count': TRANSPLANT_COUNT,
                      'keys_sha256': TRANSPLANT_KEYS_SHA256},
    'mhc8192': {'selector': 'transplant2-mhc8192-release-into-233036Z',
                'host_root': '/home/jugs/git/ds41-r38/karmic-main-20260924/selection-transplant-cache-mhc8192-233036Z',
                'count': 2, 'keys_sha256': 'd22ef4017be3bbd839e069d806cf0cc0f9388618b224476b2443a2d91ad60012'},
    # One-key split of the pair above: exactly one release record each.
    'mhc-pre8192': {'selector': 'transplant1-mhc-pre8192-release-into-233036Z',
                    'host_root': '/home/jugs/git/ds41-r38/karmic-main-20260924/selection-transplant-cache-mhc-pre8192-233036Z',
                    'count': 1, 'keys_sha256': '44d51d40d8ffad5a1e31bb27d8d20b4e26620dd466b1a920822f3217f92c11b2'},
    'mhc-expanded8192': {'selector': 'transplant1-mhc-expanded8192-release-into-233036Z',
                         'host_root': '/home/jugs/git/ds41-r38/karmic-main-20260924/selection-transplant-cache-mhc-expanded8192-233036Z',
                         'count': 1, 'keys_sha256': 'e8f3ee60f8d71f8cbb912bc22a7da48729ed150fec5da6a06e7de30ff2533a93'},
}
MHC_SUBSETS = {'mhc8192': tuple(MHC8192_KEYS), 'mhc-pre8192': ('mhc.pre.8192',),
               'mhc-expanded8192': ('mhc.pre.expanded.8192',)}
# First-token references (token-0 full top-20 record) the transplant is compared against.
REFERENCES = {
    'passing-anchor': 'receipts/decision-row-matched8192-nccl-standard-upstream-capture-20260926T233036Z/unarmed-1/trial.json',
    'bf16-ratio1-variant': 'receipts/ratio1-variant-20260927T0141Z/original-524k/0.json',
    'release-control': 'receipts/ratio1-control-20260927T0122Z/original-524k/0.json',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def keys_digest(keys):
    return sha('\n'.join(sorted(keys)).encode())


def transplant(passing, release):
    """(transplanted payload, changed keys, metadata-only keys) for one rank; raises on any surprise."""
    if passing['identity'] != release['identity']:
        raise RuntimeError('Passing and release selection identities differ')
    p, r = passing['records'], release['records']
    shared = set(p) & set(r)
    config = {k for k in shared if p[k]['config'] != r[k]['config']}
    assignment = {k for k in shared if p[k]['assignment'] != r[k]['assignment']}
    if config != assignment:
        raise RuntimeError('Config and assignment differences are not the same key set')
    if len(config) != TRANSPLANT_COUNT or keys_digest(config) != TRANSPLANT_KEYS_SHA256:
        raise RuntimeError(f'Transplant set is not the frozen 275 keys: {len(config)}')
    metadata = {k for k in shared - config if p[k] != r[k]}
    if len(metadata) != METADATA_ONLY_COUNT:
        raise RuntimeError(f'Unexpected metadata-only difference count: {len(metadata)}')
    records = {k: (r[k] if k in config else p[k]) for k in p}
    return {'identity': passing['identity'], 'records': records}, config, metadata


def select(passing, release, name='transplant275'):
    """The named frozen set; the full 275 derivation and its guards always run first."""
    payload, changed, metadata = transplant(passing, release)
    if name == 'transplant275':
        return payload, changed, metadata
    if name not in MHC_SUBSETS:
        raise ValueError('Unknown transplant set: ' + repr(name))
    from claude_compare_selections import mhc_key
    rebuilt = {'mhc.pre.8192': mhc_key('pre', 8192), 'mhc.pre.expanded.8192': mhc_key('pre', 8192, True)}
    if rebuilt != MHC8192_KEYS:
        raise RuntimeError('mHC 8192 keys no longer rebuild from source')
    if keys_digest(set(MHC8192_KEYS.values())) != SETS['mhc8192']['keys_sha256']:
        raise RuntimeError('mHC 8192 pair digest changed')
    keys = {MHC8192_KEYS[label] for label in MHC_SUBSETS[name]}
    if not keys <= changed or keys_digest(keys) != SETS[name]['keys_sha256']:
        raise RuntimeError('mHC 8192 keys are not inside the frozen 275 set')
    records = {k: (release['records'][k] if k in keys else passing['records'][k]) for k in passing['records']}
    return {'identity': passing['identity'], 'records': records}, keys, metadata


def verify(payload, passing, release, changed):
    records, p, r = payload['records'], passing['records'], release['records']
    if set(records) != set(p):
        raise RuntimeError('Transplant changed the key set')
    for key, record in records.items():
        expected = r[key] if key in changed else p[key]
        if record != expected:
            raise RuntimeError('Transplanted record differs from its source: ' + key)


def check(root=ROOT, name='transplant275'):
    """Per-rank transplanted bytes, verified against both sources and the frozen key set."""
    replay.check(root)  # passing source, anchor bytes, identity equality: unchanged replay checks
    snapshot = json.loads((root / RELEASE_SNAPSHOT / 'snapshot.json').read_text())
    namespace = json.loads((root / 'ds41-precision-release.lock.json').read_text())['cache_fingerprint']
    if snapshot['source_namespace'] != namespace:
        raise RuntimeError('Release snapshot is not the release namespace')
    files, key_sets = {}, set()
    for node in NODES:
        release_raw = (root / RELEASE_SNAPSHOT / f'{node}.json').read_bytes()
        if sha(release_raw) != snapshot['nodes'][node]['sha256']:
            raise RuntimeError('Release snapshot file changed: ' + node)
        passing_raw = (root / replay.SOURCE / f'{node}-selection.json').read_bytes()
        passing, release = json.loads(passing_raw), json.loads(release_raw)
        payload, changed, _ = select(passing, release, name)
        if len(changed) != SETS[name]['count']:
            raise RuntimeError('Transplant set size changed')
        verify(payload, passing, release, changed)
        data = (json.dumps(payload, sort_keys=True, separators=(',', ':')) + '\n').encode()
        files[node] = {'bytes': data, 'sha256': sha(data), 'passing_sha256': sha(passing_raw),
                       'release_sha256': sha(release_raw), 'records': len(payload['records'])}
        key_sets.add(frozenset(changed))
    if len(key_sets) != 1:
        raise RuntimeError('Ranks disagree on the transplant key set')
    return namespace, files


def references(root=ROOT):
    out = {}
    for name, path in REFERENCES.items():
        choice = json.loads((root / path).read_text())['response']['choices'][0]
        out[name] = choice['logprobs']['content'][0]
    return out


def seed(out, run, root=ROOT, name='transplant275'):
    namespace, files = check(root, name)
    host_root = SETS[name]['host_root']
    out.mkdir(parents=True, exist_ok=False)
    for node in NODES:
        command = replay.seed_command(host_root, namespace, files[node]['sha256'])
        (out / f'{node}-command.txt').write_text(command + '\n')
        text = run(node, command, files[node]['bytes'])
        if f'SEEDED {host_root}' not in text.splitlines():
            raise RuntimeError('Seeding did not confirm on ' + node)
    (out / 'seeded.json').write_text(json.dumps({
        'set': name, 'selector': SETS[name]['selector'], 'host_root': host_root, 'namespace': namespace,
        'transplant_keys_sha256': SETS[name]['keys_sha256'], 'transplant_count': SETS[name]['count'],
        'nodes': {n: {k: v for k, v in f.items() if k != 'bytes'} for n, f in files.items()}}, indent=2) + '\n')


def main():
    import subprocess
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=('check', 'seed'))
    p.add_argument('--set', choices=tuple(SETS), default='transplant275')
    a = p.parse_args()
    if a.action == 'check':
        namespace, files = check(name=a.set)
        print('SELECTION-TRANSPLANT-SOURCE-OK', namespace, json.dumps({n: f['sha256'] for n, f in files.items()}))
        return
    from build_activation_trace import idle
    idle()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')

    def run(node, command, data):
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', node, command], input=data,
                                capture_output=True, timeout=120, check=True)
        return result.stdout.decode(errors='replace')
    seed(ROOT / 'receipts' / f'selection-transplant-{a.set}-seed-{stamp}', run, name=a.set)
    print('SELECTION-TRANSPLANT-SEED-OK', flush=True)


if __name__ == '__main__':
    main()
