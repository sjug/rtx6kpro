"""Compose a diagnostic wrapper without modifying the arithmetic implementation."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
COMMIT = '1794dcf18454900263e0c66711af8ea4a1283ac1'
SOURCE = 'vllm/models/deepseek_v4_1/attention.py'
BASE = '30ef9c2d529e9df0a3ea348c773d8ff55284389a90a40922ac1b0d93ccffdd9b'
original = subprocess.check_output(['git', '-C', '/home/jugs/git/vllm', 'show', f'{COMMIT}:{SOURCE}'])
needle = b'    return layer._forward(positions, hidden)\n'
if original.count(needle) != 1:
    raise RuntimeError('Opaque attention wrapper changed')
replacement = b'''    from .trace_attention import before, after
    trace = before(hidden, positions, prefix)
    output = layer._forward(positions, hidden)
    after(trace, output)
    return output
'''
patched = original.replace(needle, replacement)
compile(patched, SOURCE, 'exec')
(ROOT / 'trace-patched-attention.py').write_bytes(patched)
lock = {'base_image_id': BASE, 'vllm_commit': COMMIT, 'source_path': SOURCE,
        'input_sha256': hashlib.sha256(original).hexdigest(),
        'output_sha256': hashlib.sha256(patched).hexdigest(),
        'helper_sha256': hashlib.sha256((ROOT / 'trace_attention.py').read_bytes()).hexdigest(),
        'purpose': 'diagnostic-only; snapshot copies can alter allocator history and timing'}
(ROOT / 'attention-trace.lock.json').write_text(json.dumps(lock, indent=2) + '\n')
print(json.dumps(lock, indent=2))
