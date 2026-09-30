"""Seed the selection-replay host cache root with the passing arm's exact per-rank B12X selections.

Source: receipts/decision-row-matched8192-nccl-standard-upstream-capture-20260926T233036Z, the
standard-NCCL 8192 arm that answered correctly with signature fba61148... Each rank receives its
own recorded selection file byte for byte, placed where the unchanged release image looks for
its selections (/cache/jit/<release namespace>/b12x/preparation/<identity>.json) inside a
separate host root that only the replay arm mounts at /cache. Nothing else is copied: no
compiled artifact from any image is assumed interchangeable, so every kernel is compiled on the
replay boot from the cached (assignment, config) pairs. B12X looks records up by assignment and
re-lowers them to the saved config (b12x a7d7d29b session._lookup); program lists are not read.

check  (local): the source receipt passed, is the recorded geometry, and its selection identity
       equals the release's; prints per-rank digests.
roce   (dusty, idle): digest of the image-owned /opt/b12x-roce-cache in the release image and in
       the passing image; they must be equal, since RoCE selections are not in the replay root.
seed   (all nodes, idle): create the root under a temporary name, write and verify the file,
       write SELECTION-REPLAY-SEEDED.json, then rename. An existing root is refused.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parent
NODES = ('dusty', 'toby', 'rusty', 'kirby')
SOURCE = 'receipts/decision-row-matched8192-nccl-standard-upstream-capture-20260926T233036Z'
PASSING_SIGNATURE = 'fba6114819113ce773e8bae15429a81db022eadd4217347d8848ac23be828430'
PASSING_IMAGE = '58b0720238b8d03abb8d11b32dcac8ad66daec75aa1cddc83bf2869c94339c09'
HOST_ROOT = '/home/jugs/git/ds41-r38/karmic-main-20260924/selection-replay-cache-standard8192-233036Z'
SELECTION = 'b12x/preparation/6115b03c7a814701d5610b3e6aecf82e30fdbb46a122d70441228538b9bd1e5e.json'
MARKER = 'SELECTION-REPLAY-SEEDED.json'
RELEASE_IDENTITY = 'receipts/precision-release-full-qualification-20260927-r2/initial-selections/dusty.json'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify_anchor(directory):
    """Recompute the passing anchor from responses and cold-request accounting."""
    salts = set()
    expected = {'model': 'DeepSeek-V4.1-Flash', 'temperature': 0, 'max_tokens': 64,
                'logprobs': True, 'top_logprobs': 20, 'chat_template_kwargs': {'thinking': False}}
    for subdir in ('unarmed-1', 'unarmed-2', '.'):
        path = directory / subdir
        trial = json.loads((path / 'trial.json').read_text())
        req, response = trial['request'], trial['response']
        messages = req['messages']
        if ({k: v for k, v in req.items() if k not in ('messages', 'cache_salt')} != expected
                or len(messages) != 1 or messages[0].get('role') != 'user'
                or sha(messages[0]['content'].encode()) != 'b410c6b19d492e83ecb775805266d91eed15ded5e3afd0cf264c25aae41bc7e4'
                or not req.get('cache_salt') or req['cache_salt'] in salts):
            raise RuntimeError('Passing anchor request differs from the frozen input')
        salts.add(req['cache_salt'])
        choice = response['choices'][0]
        payload = {k: choice[k] for k in ('message', 'finish_reason', 'logprobs')}
        signature = sha(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
        if (signature != PASSING_SIGNATURE or choice['message']['content'] != '739184, 482617'
                or choice['finish_reason'] != 'stop' or response['usage']['prompt_tokens'] != 524288):
            raise RuntimeError('Passing anchor response does not reproduce its signature')
        metrics = json.loads((path / 'trial-cache.json').read_text())
        before, after = metrics['before'], metrics['after_samples'][-1]
        if any(after[k] - before[k] != delta for k, delta in
               {'queries': 524288, 'hits': 0, 'successes': 1, 'preemptions': 0}.items()):
            raise RuntimeError('Passing anchor cold accounting differs')


def check(root=ROOT):
    summary = json.loads((root / SOURCE / 'summary.json').read_text())
    row = json.loads((root / SOURCE / 'response-row.json').read_text())
    if not (summary['correct'] and summary['signature_equals_control'] and summary['blocks'] == 81389
            and summary['chunk_rows'] == 8192 and summary['nccl_arm'] == 'standard-upstream'
            and summary['nccl_geometry'] is None and summary['image_id'] == PASSING_IMAGE
            and row['response_signature'] == PASSING_SIGNATURE):
        raise RuntimeError('Source receipt is not the recorded passing standard-NCCL 8192 arm')
    verify_anchor(root / SOURCE)
    release = json.loads((root / 'ds41-precision-release.lock.json').read_text())
    identity = json.loads((root / RELEASE_IDENTITY).read_text())['identity']
    files = {}
    for node in NODES:
        raw = (root / SOURCE / f'{node}-selection.json').read_bytes()
        payload = json.loads(raw)
        if payload['identity'] != identity:
            raise RuntimeError('Passing selection identity differs from the release identity: ' + node)
        files[node] = {'sha256': sha(raw), 'records': len(payload['records']), 'bytes': raw}
    configs = {n: {k: v['config'] for k, v in json.loads(f['bytes'])['records'].items()} for n, f in files.items()}
    if any(configs[n] != configs['dusty'] for n in NODES):
        raise RuntimeError('Passing ranks disagree on selected configs')
    return release['cache_fingerprint'], files


def seed_command(host_root, namespace, expected):
    root, tmp = shlex.quote(host_root), shlex.quote(host_root + '.seeding')
    target = shlex.quote(f'{host_root}.seeding/jit/{namespace}/{SELECTION}')
    return ' && '.join([
        f'test ! -e {root}', f'test ! -e {tmp}',
        f'mkdir -p "$(dirname {target})"', f'cat > {target}',
        f'test "$(sha256sum {target} | cut -d" " -f1)" = {shlex.quote(expected)}',
        f'printf %s {shlex.quote(json.dumps({"source": SOURCE, "selection_sha256": expected, "namespace": namespace}))}'
        f' > {tmp}/{MARKER}',
        f'mv -T {tmp} {root}', f'echo SEEDED {root}'])


def roce_command(image):
    return shlex.join(['podman', 'run', '--rm', '--pull=never', '--network=none', '--entrypoint', '/bin/bash', image,
                       '-c', 'set -euo pipefail; cd /opt/b12x-roce-cache; '
                             'test -n "$(find . -type f -print -quit)"; '
                             'find . -type f -print0 | LC_ALL=C sort -z | '
                             'xargs -0 sha256sum | sha256sum | cut -d" " -f1'])


def seed(out, run, root=ROOT):
    namespace, files = check(root)
    out.mkdir(parents=True, exist_ok=False)
    for node in NODES:
        command = seed_command(HOST_ROOT, namespace, files[node]['sha256'])
        (out / f'{node}-command.txt').write_text(command + '\n')
        text = run(node, command, files[node]['bytes'])
        if f'SEEDED {HOST_ROOT}' not in text.splitlines():
            raise RuntimeError('Seeding did not confirm on ' + node)
    (out / 'seeded.json').write_text(json.dumps({'source': SOURCE, 'host_root': HOST_ROOT, 'namespace': namespace,
        'nodes': {n: {k: v for k, v in f.items() if k != 'bytes'} for n, f in files.items()}}, indent=2) + '\n')


def main():
    import subprocess
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=('check', 'roce', 'seed'))
    a = p.parse_args()
    if a.action == 'check':
        namespace, files = check()
        print('SELECTION-REPLAY-SOURCE-OK', namespace, json.dumps({n: f['sha256'] for n, f in files.items()}))
        return
    from build_activation_trace import idle, ssh
    idle()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = ROOT / 'receipts' / f'selection-replay-{a.action}-{stamp}'
    if a.action == 'roce':
        release = json.loads((ROOT / 'receipts/precision-release-build-receipt.json').read_text())['image_id']
        digests = {image: ssh('dusty', roce_command(image)).strip() for image in (release, PASSING_IMAGE)}
        out.mkdir(parents=True, exist_ok=False)
        (out / 'roce.json').write_text(json.dumps(digests, indent=2) + '\n')
        if len(set(digests.values())) != 1:
            raise SystemExit('SELECTION-REPLAY-ROCE-DIFFERS: image-owned RoCE selections differ; replay is not matched')
        print('SELECTION-REPLAY-ROCE-EQUAL', digests[release])
        return

    def run(node, command, data):
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', node, command], input=data,
                                capture_output=True, timeout=120, check=True)
        return result.stdout.decode(errors='replace')
    seed(out, run)
    print('SELECTION-REPLAY-SEED-OK', out, flush=True)


if __name__ == '__main__':
    main()
