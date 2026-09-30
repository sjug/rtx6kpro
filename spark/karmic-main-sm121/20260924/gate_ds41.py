#!/usr/bin/env python3
"""DS4.1 native, stock NCCL and CLI gates. This does not qualify the model."""
import ctypes
import importlib
import json
import os
import resource
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, '/opt/karmic-main-build')
from contracts import file_sha, require
from launch_contract import NCCL, NODES, render


def check_nccl():
    import torch
    require(torch.cuda.get_device_capability() == (12, 1), 'Expected SM121')
    require(os.environ.get('LD_PRELOAD') == NCCL and os.environ.get('VLLM_NCCL_SO_PATH') == NCCL,
            'DS4.1 must select the approved stock NCCL object')
    library = ctypes.CDLL(NCCL)
    version = ctypes.c_int()
    require(library.ncclGetVersion(ctypes.byref(version)) == 0 and version.value == 23007,
            'Stock NCCL version mismatch')
    # Version is a sanity check; the maps equality below proves object identity.
    require(torch.cuda.nccl.version() == (2, 30, 7), 'Torch NCCL version mismatch')
    paths = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
             if '/libnccl.so' in line}
    require(paths == {NCCL}, f'Mixed NCCL mappings: {paths}')
    kit = Path('/opt/ds41-main-refresh')
    lock = json.loads((kit / 'build.lock.json').read_text())
    require(file_sha(kit / 'build.lock.json') == os.environ['DS41_BUILD_LOCK_SHA256'], 'Build lock drift')
    require(file_sha(NCCL) == lock['nccl_component']['library_sha256'], 'Stock NCCL digest drift')
    for name, digest in json.loads((kit / 'native-reuse.json').read_text()).items():
        require(file_sha(name) == digest, f'Declared native artifact changed: {name}')
    import vllm
    import b12x
    require(Path(vllm.__file__).resolve().is_relative_to('/opt/jovian-judgement/vllm/vllm'), 'Wrong vLLM')
    require(Path(b12x.__file__).resolve().is_relative_to('/opt/jovian-judgement/b12x/b12x'), 'Wrong B12X')
    for name in ('vllm._C_stable_libtorch', 'vllm._moe_C_stable_libtorch'):
        native = importlib.import_module(name)
        require('not found' not in subprocess.check_output(['ldd', native.__file__], text=True), name)
    x = torch.arange(512, device='cuda', dtype=torch.float32).reshape(2, 256).to(torch.bfloat16) / 256
    y = torch.empty_like(x)
    torch.ops._C.rms_norm(y, x, torch.ones(256, device='cuda', dtype=x.dtype), 1e-6)
    expected = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(y.float(), expected, atol=0.02, rtol=0.02)
    torch.cuda.synchronize()
    print('DS41-STOCK-NCCL-NATIVE-PASS', flush=True)


def check_cli():
    from vllm.entrypoints.launchers.cli_args import make_arg_parser
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    for node in NODES:
        parser = make_arg_parser(FlexibleArgumentParser(prog='vllm serve'))
        args = parser.parse_args(render(node)['model'][2:])
        require(args.tensor_parallel_size == 4 and args.decode_context_parallel_size == 1, 'TP/DCP drift')
        require(args.max_model_len == 600000 and args.max_num_seqs == 4, 'Envelope drift')
        require(args.kv_cache_memory_bytes is None and args.gpu_memory_utilization == 0.85, 'KV sizing drift')
        require(args.load_format == 'b12x', 'Loader drift')
    print('DS41-FOUR-RANK-CLI-PASS', flush=True)


def check_loader():
    from b12x.loader._native import load
    native = load()
    require(native.ABI_VERSION == 1, 'Wrong disk loader ABI')
    linkage = subprocess.check_output(['ldd', native.__file__], text=True)
    require('not found' not in linkage and 'liburing.so' in linkage, 'Disk loader liburing missing')
    print('DS41-LOADER-LINKAGE-PASS', native.__file__, flush=True)


def check_limits():
    shm = os.statvfs('/dev/shm')
    shm_bytes = shm.f_frsize * shm.f_blocks
    pids = Path('/sys/fs/cgroup/pids.max').read_text().strip()
    memlock = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    require(shm_bytes == 64 * 1024**3, 'Expected approved upstream 64 GiB shm ceiling')
    from vllm.distributed.device_communicators.shm_broadcast import check_shm_free_space
    check_shm_free_space(160 * 1024**2)
    require(pids == 'max', f'Unexpected PID limit: {pids}')
    require(memlock[0] == resource.RLIM_INFINITY, f'Bounded memlock: {memlock}')
    print('DS41-RESOURCE-RECEIPT', json.dumps({'shm_bytes': shm_bytes, 'pids_max': pids, 'memlock': memlock,
          'nofile': resource.getrlimit(resource.RLIMIT_NOFILE)}), flush=True)


if __name__ == '__main__':
    require(__debug__, 'Optimized Python is not admitted')
    check_limits()
    check_nccl()
    check_cli()
    check_loader()
