"""Freeze the reviewed, two-file diagnostic delta without changing launch pins."""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
pin = '1794dcf18454900263e0c66711af8ea4a1283ac1'
source = 'vllm/models/deepseek_v4_1/nvidia/model_state.py'
base = '3413799408e09ad3a1bb4ea2297384fa03927ca8b4e8e72f2af61aef8951d54c'
original = subprocess.check_output(['git', '-C', str(Path.home() / 'git/vllm'),
                                    'show', f'{pin}:{source}']).decode()
anchor = '            if isinstance(module, DeepseekV4Model) and module.disk_engram\n        )\n'
addition = ('        from .claude_stale_state_probe import install as _stale_probe_install  # DIAGNOSTIC ONLY\n'
            '        _stale_probe_install(self, vllm_config)\n')
if original.count(anchor) != 1:
    raise RuntimeError('Diagnostic hook anchor changed')
patched = original.replace(anchor, anchor + addition)
patch = ''.join(difflib.unified_diff(original.splitlines(True), patched.splitlines(True),
                                   fromfile='a/' + source, tofile='b/' + source))
if patch != (root / 'claude-stale-state-model_state.patch').read_text():
    raise RuntimeError('Reviewed diagnostic patch differs')
compile(patched, source, 'exec')
for name in ('claude_stale_state_probe.py', 'install_stale_probe.py'):
    compile((root / name).read_bytes(), name, 'exec')
(root / 'stale-probe-model_state.py').write_text(patched)
files = ['stale-probe-model_state.py', 'claude_stale_state_probe.py',
         'claude-stale-state-model_state.patch', 'install_stale_probe.py',
         'Dockerfile.stale-probe', 'build_stale_probe.py', 'prepare_stale_probe_image.py',
         'claude_test_stale_state_probe_torch.py']
lock = {'base_image_id': base, 'vllm_commit': pin, 'source_path': source,
        'input_sha256': hashlib.sha256(original.encode()).hexdigest(),
        'output_sha256': hashlib.sha256(patched.encode()).hexdigest(),
        'b12x_tree': '2d34ffa9a1cb4380b766e2752fe6d3705841d81e',
        'status': 'diagnostic-only-not-qualified',
        'inputs': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}}
(root / 'stale-probe.lock.json').write_text(json.dumps(lock, indent=2, sort_keys=True) + '\n')
(root / 'stale-probe.ignore').write_text('**\n' + '\n'.join('!' + n for n in files + ['stale-probe.lock.json']) + '\n')
print('STALE-PROBE-PREPARED', hashlib.sha256((root / 'stale-probe.lock.json').read_bytes()).hexdigest())
