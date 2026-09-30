"""Seed the ratio-1 diagnostic cache namespace from the exact release namespace, never overwriting.

snapshot  (nodes idle, release stopped): fetch each rank's release selection file into a new
          receipt directory and record its sha256 and eight-plan regime at the pinned KV count.
seed      (nodes idle): on each node copy the whole release namespace directory, read-only, to
          the diagnostic namespace named by ds41-ratio1.lock.json. The destination must not
          exist; the copy goes to a temporary name and is renamed only after its selection file
          and full file listing digest match the source and the snapshot.

The release namespace and every container are left untouched. Selection records keep their
assignment, config and program identities byte for byte; nothing is edited.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parent
NODES = ('dusty', 'toby', 'rusty', 'kirby')
HOST_JIT = '/home/jugs/.cache/vllm-jj-ds41-tp4/jit'
SELECTION = 'b12x/preparation/6115b03c7a814701d5610b3e6aecf82e30fdbb46a122d70441228538b9bd1e5e.json'


def namespaces(root=ROOT):
    ratio1 = json.loads((root / 'ds41-ratio1.lock.json').read_text())
    return ratio1['base_cache_fingerprint'], ratio1['cache_fingerprint'], ratio1['kv_blocks']


def listing_command(directory):
    """Stable digest of every regular file (relative path and content) under one namespace."""
    d = shlex.quote(directory)
    return (f'cd {d} && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d" " -f1')


def seed_command(source, destination, expected):
    src, dst, tmp = (shlex.quote(p) for p in (source, destination, destination + '.seeding'))
    sel_src, sel_tmp = shlex.quote(source + '/' + SELECTION), shlex.quote(destination + '.seeding/' + SELECTION)
    return ' && '.join([
        f'test -d {src}', f'test ! -e {dst}', f'test ! -e {tmp}',
        f'test "$(sha256sum {sel_src} | cut -d" " -f1)" = {shlex.quote(expected)}',
        f'cp -a --reflink=auto {src} {tmp}',
        f'test "$(sha256sum {sel_tmp} | cut -d" " -f1)" = {shlex.quote(expected)}',
        f'test "$({listing_command(source)})" = "$({listing_command(destination + ".seeding")})"',
        f'mv -T {tmp} {dst}',
        f'echo SEEDED {dst}'])


def check_snapshot(snapshot, source_ns, blocks):
    if set(snapshot['nodes']) != set(NODES) or snapshot['source_namespace'] != source_ns or snapshot['kv_blocks'] != blocks:
        raise RuntimeError('Snapshot does not cover the release namespace on all four nodes at the pinned KV count')
    regimes = {json.dumps(entry['regime'], sort_keys=True) for entry in snapshot['nodes'].values()}
    if len(regimes) != 1:
        raise RuntimeError('Ranks disagree on the release attention regime')
    if json.loads(regimes.pop())['ratio1.extend']['v41_compute_mode'] != 'fp8':
        raise RuntimeError('Release snapshot does not hold the fp8 ratio-1 extend selection under test')


def snapshot(out, ssh, root=ROOT):
    from claude_decode_sparse_mla import decode, regimes
    source_ns, _, blocks = namespaces(root)
    out.mkdir(parents=True, exist_ok=False)
    result = {'source_namespace': source_ns, 'kv_blocks': blocks, 'nodes': {}}
    for node in NODES:
        text = ssh(node, 'cat ' + shlex.quote(f'{HOST_JIT}/{source_ns}/{SELECTION}'))
        (out / f'{node}.json').write_text(text)
        records = json.loads(text)['records']
        found, _ = decode(records, [blocks], None)
        result['nodes'][node] = {'sha256': hashlib.sha256(text.encode()).hexdigest(), 'records': len(records),
                                 'regime': regimes(records, found).get(blocks)}
    check_snapshot(result, source_ns, blocks)
    (out / 'snapshot.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return result


def seed(snapshot_path, out, ssh, root=ROOT):
    source_ns, target_ns, blocks = namespaces(root)
    snap = json.loads(Path(snapshot_path).read_text())
    check_snapshot(snap, source_ns, blocks)
    out.mkdir(parents=True, exist_ok=False)
    results = {}
    for node in NODES:
        command = seed_command(f'{HOST_JIT}/{source_ns}', f'{HOST_JIT}/{target_ns}', snap['nodes'][node]['sha256'])
        (out / f'{node}-command.txt').write_text(command + '\n')
        text = ssh(node, command)
        if f'SEEDED {HOST_JIT}/{target_ns}' not in text.splitlines():
            raise RuntimeError('Seeding did not confirm on ' + node)
        results[node] = {'target': f'{HOST_JIT}/{target_ns}', 'selection_sha256': snap['nodes'][node]['sha256']}
    (out / 'seeded.json').write_text(json.dumps({'source_namespace': source_ns, 'target_namespace': target_ns,
                                                 'nodes': results}, indent=2, sort_keys=True) + '\n')
    return results


def main():
    from build_activation_trace import idle, ssh
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=('snapshot', 'seed'))
    p.add_argument('--snapshot', type=Path, help='seed: the snapshot.json recorded by the snapshot action')
    a = p.parse_args()
    idle()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = ROOT / 'receipts' / f'ratio1-namespace-{a.action}-{stamp}'
    if a.action == 'snapshot':
        snapshot(out, ssh)
    else:
        if a.snapshot is None:
            p.error('--snapshot is required for seed')
        seed(a.snapshot, out, ssh)
    print('RATIO1-NAMESPACE-' + a.action.upper() + '-OK', out, flush=True)


if __name__ == '__main__':
    main()
