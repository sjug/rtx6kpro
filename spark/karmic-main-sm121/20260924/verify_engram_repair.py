"""Check loaded source paths, preserved GPU ops and the real ARM64 I/O binary."""
import hashlib
import importlib
import json
from pathlib import Path
import platform
import subprocess

import torch
import vllm
import b12x
from vllm.platforms import current_platform


def main():
    if not __debug__ or platform.machine() != 'aarch64' or torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError('Expected non-optimized Python on a real ARM64 GB10')
    for module, expected in [(vllm, '/opt/jovian-judgement/vllm/vllm'),
                              (b12x, '/opt/jovian-judgement/b12x/b12x')]:
        if not Path(module.__file__).resolve().is_relative_to(expected):
            raise RuntimeError('Unexpected module origin: ' + str(module.__file__))
    if type(current_platform).__name__ != 'NvmlCudaPlatform':
        raise RuntimeError('GPU platform not selected')
    importlib.import_module('vllm._C_stable_libtorch')
    importlib.import_module('vllm._moe_C_stable_libtorch')
    x = torch.linspace(-2, 2, 256, device='cuda', dtype=torch.float32).reshape(2, 128).to(torch.bfloat16)
    weight = torch.ones(128, device='cuda', dtype=torch.bfloat16)
    output = torch.empty_like(x)
    torch.ops._C.rms_norm(output, x, weight, 1e-6)
    reference = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(output.float(), reference, rtol=.01, atol=.01)
    from b12x.loader._native import load
    native = load()
    for name in ('ple_batch', 'ple_batch_launch', 'ple_batch_result'):
        if not callable(getattr(native, name, None)):
            raise RuntimeError('Missing native entry point: ' + name)
    binary = Path(native.__file__)
    ldd = subprocess.check_output(['ldd', str(binary)], text=True)
    if 'not found' in ldd or 'liburing' not in ldd:
        raise RuntimeError('I/O binary linkage failed: ' + ldd)
    print(ldd, flush=True)
    lock = json.loads(Path('/opt/ds41-engram-repair/engram-repair.lock.json').read_text())
    for name, target in lock['targets'].items():
        actual = hashlib.sha256((Path('/opt/jovian-judgement') / name).read_bytes()).hexdigest()
        if actual != target['output_sha256']:
            raise RuntimeError('Runtime source mismatch: ' + name)
    print(json.dumps({'native_path': str(binary), 'native_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
                      'platform': type(current_platform).__name__, 'trees': lock['trees']}), flush=True)
    print('ENGRAM-NATIVE-GATE-PASS', flush=True)


if __name__ == '__main__':
    main()
