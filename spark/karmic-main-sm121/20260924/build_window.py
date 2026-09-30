"""Build the approved window instrumentation, never launch or promote it."""
from build_decision_row import main

SMOKE = '''
import hashlib, importlib, json
from pathlib import Path
import torch
if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
    raise RuntimeError('A visible GB10 is required')
root = Path('/opt/jovian-judgement')
lock = json.loads(Path('/opt/ds41-window/claude-window.lock.json').read_text())
loaded = {}
for target, entry in lock['targets'].items():
    if not target.startswith('vllm/vllm/') or not target.endswith('.py'):
        raise RuntimeError('Unexpected non-Python window target: ' + target)
    name = target.removeprefix('vllm/').removesuffix('.py').replace('/', '.')
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve()
    if path != (root / target).resolve():
        raise RuntimeError('Wrong loaded module: ' + name)
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry['output_sha256']:
        raise RuntimeError('Wrong loaded bytes: ' + name)
    loaded[name] = module
attention = loaded['vllm.models.deepseek_v4_1.attention']
window = loaded['vllm.models.deepseek_v4_1.claude_window']
decision = loaded['vllm.models.deepseek_v4_1.claude_decision_row']
if attention._claude_window is not window or attention._claude_row is not decision:
    raise RuntimeError('Attention does not reference the window helper')
if window.SCHEMA != 'claude-window-v1' or decision.SCHEMA != 'claude-decision-row-v3':
    raise RuntimeError('Unexpected capture schemas')
for name in ('begin', 'rotated', 'attention', 'wo_partial', 'wo_reduced'):
    if not callable(getattr(window, name, None)):
        raise RuntimeError('Missing window boundary: ' + name)
stub = json.loads(Path('/opt/ds41-decision-row/claude-decision-row.lock.json').read_text())
if stub['trees'] != lock['trees']:
    raise RuntimeError('Decision-row provenance stub has different source trees')
print('WINDOW-HELPER-IMPORT-PASS')
'''


if __name__ == '__main__':
    main(kind='window', smoke=SMOKE, install_marker='WINDOW-INSTALL-PASS',
         smoke_marker='WINDOW-HELPER-IMPORT-PASS', wrapper='build_window.py')
