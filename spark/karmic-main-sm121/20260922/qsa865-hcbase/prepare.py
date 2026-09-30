#!/usr/bin/env python3
"""Apply only PR865 to the exact earlier HC-off Spark source payload."""
import io
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from contracts import file_sha, git_tree, manifest, require, sha

BASE = '88867036403cb3c9bc3026eb8c907ad64bc1af9c28d316d22b670dde4f561152'
HEAD = '26b42cf9eb715b118e55f91f53fa0ab412a79c09'
PATCH_SHA = '4a70945def2dda5540d04c4a18cf138f3908ae5329745b62753a5c106061f113'
PATHS = (
    'tests/models/qwen4_exp/test_b12x_qsa.py',
    'tests/v1/cudagraph/test_cudagraph_manager.py',
    'vllm/models/qwen4_exp/nvidia/b12x_qsa.py',
    'vllm/models/qwen4_exp/nvidia/model_state.py',
    'vllm/v1/worker/gpu/cudagraph_utils.py',
    'vllm/v1/worker/gpu/input_batch.py',
)


def compose():
    require(file_sha(ROOT / 'upstream.patch') == PATCH_SHA, 'Patch changed')
    lock = json.loads((ROOT.parent / 'source.lock.json').read_text())
    row = lock['sources']['vllm']
    require(row['commit'] == '6afb99982576a7a2eb53d667189e859629e22739', 'Wrong HC-off source')
    archive = ROOT.parent / 'payload/vllm.tar'
    require(file_sha(archive) == row['archive_sha256'], 'Base archive changed')
    files = {}
    with tarfile.open(archive) as bundle:
        for name, entry in row['files'].items():
            member = bundle.getmember(name)
            data = member.linkname.encode() if member.issym() else bundle.extractfile(member).read()
            require(sha(data) == entry['sha256'], f'Base file changed: {name}')
            files[name] = (entry['mode'], data)
    require(git_tree(files) == row['refreshed_tree'], 'Wrong base tree')
    before = dict(files)
    with tempfile.TemporaryDirectory(prefix='qsa865-hcbase-') as scratch:
        work = Path(scratch)
        for name in PATHS:
            dest = work / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(files[name][1])
        for flags in (['--check'], []):
            subprocess.run(['git', 'apply', '--whitespace=error', *flags,
                            str(ROOT / 'upstream.patch')], cwd=work, check=True)
        for name in PATHS:
            files[name] = (files[name][0], (work / name).read_bytes())
    changed = sorted(p for p in files if files[p] != before[p])
    require(set(changed) == set(PATHS), 'Not exactly the six PR865 paths')
    old_tree = row['refreshed_tree']
    row.update(base_files=manifest(before), files=manifest(files), changed_paths=changed,
               refreshed_tree=git_tree(files),
               package_tree=git_tree({p[5:]: v for p, v in files.items() if p.startswith('vllm/')}),
               local_backport={'pr': 865, 'head': HEAD, 'patch_sha256': PATCH_SHA})
    with tarfile.open(ROOT / 'refresh.tar', 'w') as bundle:
        for name in changed:
            mode, data = files[name]
            item = tarfile.TarInfo('vllm/' + name)
            item.mode = int(mode[-3:], 8)
            item.size = len(data)
            bundle.addfile(item, io.BytesIO(data))
    lock['base_image_id'] = BASE
    lock['composition'] = 'Previous Karmic HC-off baseline plus PR865 only; B12X and LMCache unchanged'
    lock['cache_fingerprint'] = 'karmic-main-hcbase-qsa865-' + sha((BASE + row['refreshed_tree']).encode())[:20]
    (ROOT / 'runtime.lock.json').write_text(json.dumps(lock, indent=2, sort_keys=True) + '\n')
    pin = dict(base_image_id=BASE, upstream_head=HEAD,
               upstream_base='e77be22511ab91ecf217760524b7579c366cca2a',
               applied_to_commit=row['commit'], patch_sha256=PATCH_SHA,
               before={p: manifest(before)[p] for p in PATHS},
               after={p: row['files'][p] for p in PATHS},
               base_tree=old_tree, result_tree=row['refreshed_tree'], package_tree=row['package_tree'],
               base_lock_sha256=file_sha(ROOT.parent / 'source.lock.json'),
               runtime_lock_sha256=file_sha(ROOT / 'runtime.lock.json'),
               refresh_sha256=file_sha(ROOT / 'refresh.tar'), cache_fingerprint=lock['cache_fingerprint'],
               b12x_tree=lock['sources']['b12x']['refreshed_tree'])
    (ROOT / 'backport.lock.json').write_text(json.dumps(pin, indent=2, sort_keys=True) + '\n')
    runner = (ROOT.parent / 'run-qwen-hc-diagnostic.sh').read_text()
    replacements = {
        ':karmic-main-spark-sm121': ':karmic-main-hcbase-qsa865-spark-sm121',
        'NAME=${NAME:-qwen38-flash-next-nvfp4-karmic-main-tp2}':
            'NAME=${NAME:-qwen38-flash-next-nvfp4-karmic-main-hcbase-qsa865-tp2}',
        old_tree: row['refreshed_tree'],
        '# Diagnostic only: unset preserves the image default; no general env passthrough.':
            '# Qualified HC-off profile; explicit HC=1 is diagnostic only.\n'
            'VLLM_QWEN3_8_FLASH_NEXT_HC_TP=${VLLM_QWEN3_8_FLASH_NEXT_HC_TP:-0}\n'
            '[[ $MAX_NUM_SEQS == 4 && $NUM_SPECULATIVE_TOKENS == 3 ]] || { echo "QSA865 requires four sequences and MTP3" >&2; exit 78; }',
        '  --recurrent-checkpoint-policy "${RECURRENT_CHECKPOINT_POLICY}"\n)':
            '  --recurrent-checkpoint-policy "${RECURRENT_CHECKPOINT_POLICY}"\n'
            '  --max-cudagraph-capture-size 32\n)',
        'ROLE=${ROLE:?set ROLE=head|worker|stop}':
            'ROLE=${ROLE:?set ROLE=head|worker}\n'
            '[[ $ROLE != stop ]] || { echo "Stop gracefully and retain containers" >&2; exit 78; }',
    }
    for old, new in replacements.items():
        require(runner.count(old) == 1, f'Runner anchor changed: {old}')
        runner = runner.replace(old, new)
    # Keep the retained-container contract structural, not just an early guard.
    require(runner.count('  stop)\n') == 1, 'Inherited stop case changed')
    start = runner.index('  stop)\n')
    end = runner.index('    ;;\n', start) + len('    ;;\n')
    runner = runner[:start] + runner[end:]
    require('podman rm' not in runner, 'Unexpected removal command')
    (ROOT / 'run-qwen.sh').write_text(runner)
    print(json.dumps({k: pin[k] for k in ('base_tree', 'result_tree', 'package_tree')}, indent=2))


if __name__ == '__main__':
    compose()
