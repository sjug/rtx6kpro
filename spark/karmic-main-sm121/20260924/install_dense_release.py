"""Verify all tracked base source, install the two-line fence, preserve natives."""
import hashlib
import json
from pathlib import Path
import shutil

here = Path(__file__).resolve().parent
root = Path('/opt/jovian-judgement/b12x')
lock = json.loads((here / 'dense-release.lock.json').read_text())
prior = json.loads((here / 'determinism.lock.json').read_text())
baseline = json.loads((here / 'runtime.lock.json').read_text())['sources']['b12x']['files']


def digest(path):
    data = str(path.readlink()).encode() if path.is_symlink() else path.read_bytes()
    return hashlib.sha256(data).hexdigest()


for name, expected in lock['inputs'].items():
    if digest(here / name) != expected:
        raise RuntimeError('Recipe input mismatch: ' + name)
expected_sources = {name: entry['sha256'] for name, entry in baseline.items()}
expected_sources[prior['moe_dependency']['target_path']] = prior['moe_dependency']['output_sha256']
expected_sources[prior['selector_path']] = prior['selector_output_sha256']
for name, expected in expected_sources.items():
    if digest(root / name) != expected:
        raise RuntimeError('Base source mismatch: ' + name)
target = root / lock['target_path']
if digest(target) != lock['input_sha256']:
    raise RuntimeError('Dense source mismatch')
native = {str(path): digest(path) for path in Path('/opt/jovian-judgement').rglob('*.so')}
shutil.copyfile(here / 'dense_gemm-fence-before-release.py', target)
expected_sources[lock['target_path']] = lock['output_sha256']
for name, expected in expected_sources.items():
    if digest(root / name) != expected:
        raise RuntimeError('Installed source mismatch: ' + name)
if native != {str(path): digest(path) for path in Path('/opt/jovian-judgement').rglob('*.so')}:
    raise RuntimeError('Native artifacts changed')
attention = Path('/opt/jovian-judgement/vllm/vllm/models/deepseek_v4_1/attention.py')
if digest(attention) != 'd3b92eb0e9d64de567185e24bc04d6001b76da5653cdaa4fb512f7f5a8432752':
    raise RuntimeError('Instrumented or unexpected attention source')
for helper in ('trace_attention.py', 'trace_attention_ops.py',
               'diagnostic_overlay.py', 'freeze_ops_overlay.py'):
    if list(attention.parents[2].rglob(helper)):
        raise RuntimeError('Diagnostic helper present in release image: ' + helper)
(here / 'native-preserved.json').write_text(json.dumps(native, indent=2, sort_keys=True) + '\n')
print('DENSE-RELEASE-INSTALL-PASS', flush=True)
