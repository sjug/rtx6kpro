"""One hash-pinned trace helper overlay, never an unrestricted source mount."""
import hashlib
from pathlib import Path

TARGET = '/opt/jovian-judgement/vllm/vllm/models/deepseek_v4_1/trace_attention.py'
SOURCE = 'trace_attention_ops.py'
IMAGE = '335a15d1ffed84e0203d0105032c143d68fdb1ab85b56e480ebcd813afcb31fe'


def validate(pin, root, *, installed=False):
    overlay = pin.get('diagnostic_overlay')
    if overlay is None:
        return None
    if (pin['image_id'] != IMAGE or pin.get('diagnostic', {}).get('kind') != 'attention-trace'
            or set(overlay) != {'source', 'target', 'sha256'}
            or overlay['source'] != SOURCE or overlay['target'] != TARGET):
        raise RuntimeError('Unqualified diagnostic overlay')
    path = Path(TARGET) if installed else Path(root) / SOURCE
    if hashlib.sha256(path.read_bytes()).hexdigest() != overlay['sha256']:
        raise RuntimeError('Diagnostic overlay digest mismatch')
    return overlay
