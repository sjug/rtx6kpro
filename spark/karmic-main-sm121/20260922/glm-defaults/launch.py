#!/usr/bin/env python3
"""Execute a digest-verified upstream-resolved command in the pinned Spark image."""
import hashlib
import ctypes
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NCCL_PATH = '/opt/nccl/lib/libnccl.so.2.31.2'

def runtime_environment(inherited, policy):
    environment = dict(inherited)
    for key in list(environment):
        if key.startswith(('VLLM_', 'B12X_DYNAMIC_', 'NCCL_')) and key not in policy and key not in {'VLLM_HOST_IP', 'VLLM_NCCL_SO_PATH'}:
            environment.pop(key)
    environment.update(policy)
    if environment.get('VLLM_NCCL_SO_PATH') != NCCL_PATH:
        raise SystemExit('Pinned NCCL library path missing or changed')
    return environment

def main():
    for file, key in [('profile.lock.json', 'GLM_PROFILE_SHA256'), ('launch.py', 'GLM_LAUNCH_SHA256'),
                      ('experiment.py', 'GLM_EXPERIMENT_SHA256')]:
        if hashlib.sha256((ROOT / file).read_bytes()).hexdigest() != os.environ[key]:
            raise SystemExit('Runtime file digest mismatch: ' + file)
    import experiment
    lock, experiment_record, effective = experiment.resolve((ROOT / 'profile.lock.json').read_bytes(), os.environ['GLM_EXPERIMENT_ARM'])
    if effective != os.environ['GLM_EFFECTIVE_PROFILE_SHA256']:
        raise SystemExit('Effective diagnostic profile mismatch')
    print(json.dumps({'experiment': experiment_record, 'effective_profile_sha256': effective}), flush=True)
    import vllm
    import b12x
    if not str(vllm.__file__).startswith('/opt/jovian-judgement/vllm/vllm/'):
        raise SystemExit('Unexpected vLLM import path')
    if not str(b12x.__file__).startswith('/opt/jovian-judgement/b12x/'):
        raise SystemExit('Unexpected B12X import path')
    rank = int(os.environ['GLM_NODE_RANK'])
    if rank not in range(4):
        raise SystemExit('Invalid rank')
    environment = runtime_environment(os.environ, lock['environment'])
    cuda = ctypes.CDLL('libcuda.so.1')
    version = ctypes.c_int()
    if cuda.cuDriverGetVersion(ctypes.byref(version)) != 0:
        raise SystemExit('Cannot query CUDA driver API version')
    mappings = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                       if '/libcuda.so' in line})
    expected = '/usr/local/cuda-13.4/compat/lib.real/libcuda.so.615.65.02'
    if version.value != 13040 or mappings != [expected]:
        raise SystemExit(f'CUDA initialization mismatch: version={version.value}, mappings={mappings}')
    print(json.dumps({'cuda_driver_api': version.value, 'libcuda_mappings': mappings,
                      'LD_LIBRARY_PATH': environment.get('LD_LIBRARY_PATH')}), flush=True)
    argv = lock['argv'] + ['--nnodes', '4', '--node-rank', str(rank),
                          '--master-addr', '10.11.11.1', '--master-port', '25000']
    if rank:
        argv.append('--headless')
    print(json.dumps({'argv': argv, 'profile_sha256': os.environ['GLM_PROFILE_SHA256'],
                      'policy_environment': lock['environment']}), flush=True)
    os.execvpe(argv[0], argv, environment)

if __name__ == '__main__':
    main()
