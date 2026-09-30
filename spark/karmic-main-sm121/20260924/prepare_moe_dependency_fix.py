"""Declare the deterministic dynamic MoE's existing ordered reduction program."""
import difflib
import hashlib
import json
import subprocess
from prepare_deterministic_diagnostic import ROOT, REPO, COMMIT, object_id, tree_id

target = 'b12x/moe/fused_moe/_preparation.py'
old = subprocess.check_output(['git', '-C', str(REPO), 'show', f'{COMMIT}:{target}']).decode()
anchor = '            launches.append(launch)\n    return tuple(launches)\n\ndef compile_fused_moe('
insert = '''            launches.append(launch)
        if plan.deterministic_output:
            # Runtime _launch_dynamic_topk_sum executes this ordered reduction
            # after the per-route dynamic kernel. It must be declared as well
            # as compiled; otherwise preparation rejects the first real call.
            from b12x.moe._shared.kernels.w4a16.kernel import compile_w4a16_topk_sum
            for float32_output in (False, True):
                launches.append(compile_w4a16_topk_sum(
                    m=exact_m, topk=plan.num_topk, hidden_size=plan.k,
                    element_dtype=_impl._w4a16_element_dtype(plan.dtype),
                    float32_output=float32_output,
                ))
    return tuple(launches)

def compile_fused_moe('''
if old.count(anchor) != 1:
    raise RuntimeError('MoE dependency patch anchor drift')
new = old.replace(anchor, insert)
patch = ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                   fromfile='a/' + target, tofile='b/' + target))
listing = subprocess.check_output(['git', '-C', str(REPO), 'ls-tree', '-rz', COMMIT])
entries = {}
for raw in listing.split(b'\0'):
    if raw:
        metadata, name = raw.split(b'\t', 1)
        mode, _, digest = metadata.decode().split()
        entries[name.decode()] = (mode, digest)
base_tree = tree_id(entries)
entries[target] = ('100644', object_id('blob', new.encode()))
(ROOT / 'moe-dependency-preparation.py').write_text(new)
(ROOT / 'moe-dependency.patch').write_text(patch)
lock = {'base_image_id': json.loads((ROOT / 'candidate.json').read_text())['image_id'],
        'base_b12x_tree': base_tree, 'b12x_tree': tree_id(entries), 'target_path': target,
        'input_sha256': hashlib.sha256(old.encode()).hexdigest(),
        'output_sha256': hashlib.sha256(new.encode()).hexdigest(),
        'patch_sha256': hashlib.sha256(patch.encode()).hexdigest(),
        'scope': 'Declaration-only repair for dynamic deterministic ordered reduction; kernel arithmetic unchanged'}
(ROOT / 'moe-dependency.lock.json').write_text(json.dumps(lock, indent=2, sort_keys=True) + '\n')
print(json.dumps(lock, indent=2))
