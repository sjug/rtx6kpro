"""Compose a one-file circular mapping fix over the clean dense-fence image."""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
from prepare_deterministic_diagnostic import object_id, tree_id

root = Path(__file__).resolve().parent
repo = Path.home() / 'git/vllm'
commit = '1794dcf18454900263e0c66711af8ea4a1283ac1'
target = 'vllm/models/deepseek_v4_1/sparse_mla.py'
old = subprocess.check_output(['git', '-C', str(repo), 'show', f'{commit}:{target}']).decode()
anchor = ('    input_slot = tl.load(InputSlots + t, t < nt, other=-1)\n'
          '    valid = valid & (pos >= 0) & (input_slot >= 0)\n')
replacement = ('    valid = valid & (pos >= 0)\n'
               '    if not CIRCULAR:\n'
               '        # CircularBufferSpec disables generic slot mapping. Its\n'
               '        # request-owned ring is addressed from Table and pos below.\n'
               '        input_slot = tl.load(InputSlots + t, t < nt, other=-1)\n'
               '        valid = valid & (input_slot >= 0)\n')
if old.count(anchor) != 1:
    raise RuntimeError('Circular fix anchor changed')
new = old.replace(anchor, replacement)
compile(new, target, 'exec')
patch = ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
    fromfile='a/' + target, tofile='b/' + target))
entries = {}
for row in subprocess.check_output(['git', '-C', str(repo), 'ls-tree', '-rz', commit]).split(b'\0'):
    if row:
        meta, name = row.split(b'\t', 1)
        mode, _, oid = meta.decode().split()
        entries[name.decode()] = (mode, oid)
base_tree = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', commit + '^{tree}'], text=True).strip()
if tree_id(entries) != base_tree:
    raise RuntimeError('Independent base tree reconstruction failed')
# This is the upstream-plus-ring-fix tree, not the existing Spark arch overlay.
entries[target] = ('100644', object_id('blob', new.encode()))
for name, content in [('ring-fix-sparse_mla.py', new), ('ring-fix.patch', patch)]:
    (root / name).write_text(content)
names = ['ring-fix-sparse_mla.py', 'ring-fix.patch', 'prepare_ring_fix.py',
         'install_ring_fix.py', 'Dockerfile.ring-fix', 'test_compressor_ring_mapping.py',
         'build_ring_fix.py', 'run_ring_regression.py']
lock = {
    'base_image_id': '3413799408e09ad3a1bb4ea2297384fa03927ca8b4e8e72f2af61aef8951d54c',
    'b12x_tree': '2d34ffa9a1cb4380b766e2752fe6d3705841d81e',
    'vllm_commit': commit, 'upstream_tree': base_tree,
    'upstream_plus_fix_tree': tree_id(entries), 'source_path': target,
    'input_sha256': hashlib.sha256(old.encode()).hexdigest(),
    'output_sha256': hashlib.sha256(new.encode()).hexdigest(),
    'scope': 'Restore circular compressor ring writes; generic mapping remains required for noncircular caches',
    'inputs': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names},
}
(root / 'ring-fix.lock.json').write_text(json.dumps(lock, indent=2, sort_keys=True) + '\n')
(root / 'ring-fix.ignore').write_text('**\n' + '\n'.join('!' + name for name in names + ['ring-fix.lock.json']) + '\n')
print('RING-FIX-PREPARED', hashlib.sha256((root / 'ring-fix.lock.json').read_bytes()).hexdigest())
