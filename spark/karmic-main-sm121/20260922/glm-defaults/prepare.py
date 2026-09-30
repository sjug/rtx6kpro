#!/usr/bin/env python3
"""Freeze the upstream resolver and its GLM defaults without touching source checkouts."""
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = Path('/home/jugs/git/blackwell-llm-docker')
RECIPE = 'c37f4a0ce9be0f2d35ca2f8b599f09fb3ad0a457'
IMAGE = '1a7a8acff71044e3b8bf97dacb7617cd1bdb50d0ab2fc2e35364a4d4dc165dcc'
REVISION = '46aaae8a82032f77100f2f03e9cc11b391df3b4d'

def main():
    source = ROOT / 'upstream'
    source.mkdir(exist_ok=True)
    archive = subprocess.check_output(['git', '-C', str(REPO), 'archive', RECIPE, 'runtime'])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(source, filter='data')
    sys.path.insert(0, str(source))
    from runtime.launcher import resolve
    raw = resolve('glm53-flash', 'native', env={}, runtime_identity=IMAGE)
    overrides = {
        'model': '/root/.cache/huggingface/hub/models--local-inference-lab--GLM-5.3-Flash-NVFP4/snapshots/' + REVISION,
        'served-model-name': 'GLM-5.3-Flash',
        'gpu-memory-utilization': 0.85,
    }
    # Transport belongs to the four-host Spark deployment, not the RTX profile.
    transport = {
        'NCCL_NET_PLUGIN': 'none',
        'VLLM_ENABLE_PCIE_ALLREDUCE': '0',
        'VLLM_USE_B12X_PCIE_DMA': '0',
        'VLLM_PCIE_DMA_FP8': '0', 'B12X_PCIE_DMA_FP8': '0',
        'VLLM_ENABLE_ROCE_ALLREDUCE': '1',
        'VLLM_ROCE_ALLREDUCE_MAX_SIZE': '2MB',
        'VLLM_ROCE_ALLGATHER_MAX_SIZE': '16MB',
        'B12X_ROCE_SPIN_LIMIT': '50000000',
        'B12X_ROCE_CACHE_DIR': '/opt/b12x-roce-cache',
        'B12X_ROCE_HCA': 'rocep1s0f0,roceP2p1s0f0',
        'B12X_ROCE_GID_INDEX': '3',
        'NCCL_DEBUG': 'WARN', 'NCCL_IB_DISABLE': '0',
        'NCCL_IB_HCA': 'rocep1s0f0,roceP2p1s0f0',
        'NCCL_IB_GID_INDEX': '3', 'NCCL_IB_TC': '106',
        'NCCL_IB_MERGE_NICS': '0', 'NCCL_IB_SUBNET_AWARE_ROUTING': '1',
        'NCCL_PROTO': 'LL,Simple',
        'NCCL_SOCKET_IFNAME': 'enp1s0f0np0,enP2p1s0f0np0',
        'GLOO_SOCKET_IFNAME': 'enp1s0f0np0',
        'CUTE_DSL_ARCH': 'sm_121a', 'TORCH_CUDA_ARCH_LIST': '12.1a',
    }
    plan = resolve('glm53-flash', 'native', env={},
                   config={'options': overrides, 'environment': transport}, runtime_identity=IMAGE)
    if plan.cache_service is not None:
        raise RuntimeError('Unexpected external cache service')
    parent = json.loads((ROOT.parent / 'qsa865/runtime.lock.json').read_text())
    lock = {
        'image_id': IMAGE, 'recipe_commit': RECIPE, 'checkpoint_revision': REVISION,
        'vllm_tree': parent['sources']['vllm']['refreshed_tree'],
        'b12x_tree': parent['sources']['b12x']['refreshed_tree'],
        'options': plan.values, 'environment': plan.environment, 'argv': plan.argv,
        'upstream_options': raw.values, 'upstream_environment': raw.environment,
        'deployment_option_overrides': overrides, 'deployment_environment_overrides': transport,
        'source_files': {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(source.rglob('*')) if p.is_file() and '__pycache__' not in p.parts},
    }
    (ROOT / 'profile.lock.json').write_text(json.dumps(lock, indent=2, sort_keys=True) + '\n')
    print('GLM-DEFAULTS-PREPARED')

if __name__ == '__main__':
    main()
