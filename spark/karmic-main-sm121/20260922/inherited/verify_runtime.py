#!/usr/bin/env python3
"""Device-attached foundation, source, extension and NCCL gate, not model qualification."""
import ctypes
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
import re
import subprocess
from pathlib import Path


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    import torch
    import vllm
    import b12x
    from vllm.platforms import current_platform

    require(torch.cuda.get_device_capability() == (12, 1), 'GPU is not SM121')
    require(current_platform.is_cuda(), f'Unexpected platform: {current_platform}')
    require(metadata.version('torch') == '2.14.0a0+4fdf77b940.nv26.8.63802676', 'Torch changed')
    require(torch.version.cuda == '13.4', 'CUDA foundation changed')
    root = Path('/opt/jovian-judgement/vllm')
    require(Path(vllm.__file__).resolve().is_relative_to(root / 'vllm'), 'Wrong vLLM import')
    require(Path(b12x.__file__).resolve().is_relative_to(Path('/opt/jovian-judgement/b12x')), 'Wrong B12X import')
    manifest = json.loads(Path('/opt/karmic-build/vllm-payload.json').read_text())
    for category in ('tracked', 'build_products'):
        for relative, expected in manifest[category].items():
            path = root / relative
            if path.is_symlink():
                require(hashlib.sha256(os.readlink(path).encode()).hexdigest() == expected, f'Changed symlink {path}')
                continue
            require(path.is_file(), f'Missing {path}')
            with path.open('rb') as stream:
                require(hashlib.file_digest(stream, 'sha256').hexdigest() == expected, f'Changed {path}')
    for module in ('vllm._C_stable_libtorch', 'vllm._moe_C_stable_libtorch',
                   'vllm._flashkda_C', 'vllm._qutlass_C', 'vllm.cumem_allocator',
                   'vllm.spinloop', 'vllm.fs_io_C', 'vllm._rust_tool_parser',
                   'vllm.vllm_flash_attn._vllm_fa2_C', 'vllm.vllm_flash_attn._vllm_fa3_C',
                   'lmcache.cuda_ops', 'lmcache.lmcache_native', 'lmcache.lmcache_fs',
                   'lmcache.lmcache_redis'):
        loaded = importlib.import_module(module)
        linkage = subprocess.check_output(['ldd', loaded.__file__], text=True)
        require('not found' not in linkage, f'Unresolved linkage: {module}\n{linkage}')
        print('NATIVE-IMPORT-PASS', module, loaded.__file__, flush=True)
    torch.manual_seed(20260920)
    value = torch.randn(16, 256, device='cuda', dtype=torch.bfloat16)
    weight = torch.ones(256, device='cuda', dtype=torch.bfloat16)
    output = torch.empty_like(value)
    torch.ops._C.rms_norm(output, value, weight, 1e-6)
    reference = value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(output.float(), reference, atol=0.02, rtol=0.01)
    torch.cuda.synchronize()
    from b12x.comm.roce import _proxy
    require(_proxy.load().roce_abi_version() == 4, 'Wrong RoCE proxy ABI')
    library = ctypes.CDLL('/opt/nccl/lib/libnccl.so.2.31.2')
    version = ctypes.c_int()
    require(library.ncclGetVersion(ctypes.byref(version)) == 0 and version.value == 23102, 'Wrong NCCL')
    nccl_paths = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if '/libnccl.so' in line}
    require(nccl_paths == {'/opt/nccl/lib/libnccl.so.2.31.2'}, f'Mixed NCCL objects: {nccl_paths}')
    flashkda = root / 'vllm/_flashkda_C.abi3.so'
    cubins = subprocess.check_output(['cuobjdump', '--list-elf', str(flashkda)], text=True)
    print(cubins, flush=True)
    require(re.search(r'sm_(121a|120f)(?:[. _\-]|$)', cubins), 'FlashKDA lacks SM121 architecture/family target')
    require('checkpoint_indptr' in str(torch.ops._flashkda_C.fwd.default._schema), 'Missing checkpoint operator contract')
    require(os.environ.get('LMCACHE_ENABLED') == '0', 'LMCache must stay disabled')
    print('KARMIC-RUNTIME-NATIVE-SMOKE-PASS', flush=True)


if __name__ == '__main__':
    main()
