"""Generate a pinned DS4.1-only diagnostic overlay, without source-repo writes."""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
COMMIT = 'a7d7d29b2ef8869086e0ceaa787321f17544e3c9'
TARGET = 'b12x/attention/dsa_indexer/mxfp4.py'
MODULE = 'b12x/attention/dsa_indexer/deterministic_topk.py'
REPO = Path.home() / 'git/b12x'


def object_id(kind, data):
    return hashlib.sha1(kind.encode() + b' ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def tree_id(entries):
    directories, files = {}, {}
    for path, entry in entries.items():
        first, sep, rest = path.partition('/')
        if sep:
            directories.setdefault(first, {})[rest] = entry
        else:
            files[first] = entry
    for name, content in directories.items():
        files[name] = ('40000', tree_id(content))
    data = b''
    for name, (mode, digest) in sorted(files.items(), key=lambda x: (x[0] + ('/' if x[1][0] == '40000' else '')).encode()):
        data += mode.encode() + b' ' + name.encode() + b'\0' + bytes.fromhex(digest)
    return object_id('tree', data)


def main():
    old = subprocess.check_output(['git', '-C', str(REPO), 'show', f'{COMMIT}:{TARGET}']).decode()
    new = old.replace('from .tiled_topk import run_row_topk\n',
                      'from .tiled_topk import run_row_topk\nfrom .deterministic_topk import repair_topk\n')
    sites = [
        ('    out_values = binding.output_scores if binding.output_scores is not None else s["sorted_values"]\n',
         '    repair_topk(s["logits"], s["lengths"], s["values"], s["indices"], gather=rt.candidate_indices)\n'),
        ('        launch("sort", (caps.candidate_topk_blocks, True),\n',
         '        repair_topk(s["block_logits"], s["block_lengths"], s["block_values"], s["block_indices"])\n'),
    ]
    for anchor, insertion in sites:
        if new.count(anchor) != 1:
            raise RuntimeError(f'Unexpected insertion anchor: {anchor}')
        new = new.replace(anchor, insertion + anchor)
    module = (ROOT / 'deterministic_topk.py').read_bytes()
    listing = subprocess.check_output(['git', '-C', str(REPO), 'ls-tree', '-rz', COMMIT])
    entries = {}
    for raw in listing.split(b'\0'):
        if raw:
            metadata, name = raw.split(b'\t', 1)
            mode, _, digest = metadata.decode().split()
            entries[name.decode()] = (mode, digest)
    baseline_tree = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', f'{COMMIT}^{{tree}}'], text=True).strip()
    if tree_id(entries) != baseline_tree:
        raise RuntimeError('Independent baseline tree reconstruction failed')
    entries[TARGET] = ('100644', object_id('blob', new.encode()))
    entries[MODULE] = ('100644', object_id('blob', module))
    patch = ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                       fromfile='a/' + TARGET, tofile='b/' + TARGET))
    (ROOT / 'diagnostic-topk.patch').write_text(patch)
    (ROOT / 'diagnostic-mxfp4.py').write_text(new)
    lock = {'base_image_id': json.loads((ROOT / 'candidate.json').read_text())['image_id'],
            'base_b12x_commit': COMMIT, 'base_b12x_tree': baseline_tree,
            'b12x_tree': tree_id(entries), 'target_path': TARGET, 'module_path': MODULE,
            'input_sha256': hashlib.sha256(old.encode()).hexdigest(),
            'output_sha256': hashlib.sha256(new.encode()).hexdigest(),
            'module_sha256': hashlib.sha256(module).hexdigest(),
            'patch_sha256': hashlib.sha256(patch.encode()).hexdigest(),
            'scope': 'Diagnostic stable threshold repair, DS4.1 MXFP4 indexer only; native artifacts unchanged'}
    (ROOT / 'diagnostic-topk.lock.json').write_text(json.dumps(lock, indent=2, sort_keys=True) + '\n')
    print(json.dumps(lock, indent=2))


if __name__ == '__main__':
    main()
