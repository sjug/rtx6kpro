#!/usr/bin/env python3
"""Render or launch one reviewed GLM rank. Never replaces or stops containers."""
import argparse
import hashlib
import json
import os
import shlex
import socket
import subprocess
from pathlib import Path
import experiment

ROOT = Path(__file__).resolve().parent
NODES = {'sparky': (0, '10.11.11.1'), 'buddy': (1, '10.11.11.2'),
         'rocky': (2, '10.11.11.4'), 'lucky': (3, '10.11.11.3')}

def checked(*args):
    return subprocess.check_output(args, text=True).strip()

def command(node, home, lock, arm='defaults'):
    rank, address = NODES[node]
    digest = hashlib.sha256((ROOT / 'profile.lock.json').read_bytes()).hexdigest()
    launch_digest = hashlib.sha256((ROOT / 'launch.py').read_bytes()).hexdigest()
    _, _, effective = experiment.resolve((ROOT / 'profile.lock.json').read_bytes(), arm)
    experiment_digest = hashlib.sha256((ROOT / 'experiment.py').read_bytes()).hexdigest()
    cache = str(home / '.cache/vllm-karmic-main-glm-defaults')
    cmd = ['podman', 'run', '-d', '--name', experiment.name(arm), '--pull=never',
           '--device', 'nvidia.com/gpu=all', '--device', '/dev/infiniband',
           '--security-opt', 'label=disable', '--network', 'host', '--ipc', 'host',
           '--init', '--ulimit', 'memlock=-1', '--ulimit', 'stack=67108864',
           '--ulimit', 'nofile=500000:500000',
           '-v', f'{home}/.cache/huggingface:/root/.cache/huggingface:ro',
           '-v', f'{cache}:/cache:rw', '-v', f'{ROOT}:/opt/glm-defaults:ro',
           '--label', f'local-inference.glm-profile.sha256={digest}',
           '--label', f'local-inference.glm-effective-profile.sha256={effective}',
           '--label', f'local-inference.glm-experiment.arm={arm}',
           '-e', f'GLM_EXPERIMENT_ARM={arm}', '-e', f'GLM_EFFECTIVE_PROFILE_SHA256={effective}',
           '-e', f'GLM_EXPERIMENT_SHA256={experiment_digest}',
           '-e', f'GLM_PROFILE_SHA256={digest}', '-e', f'GLM_LAUNCH_SHA256={launch_digest}',
           '-e', f'VLLM_HOST_IP={address}', '-e', f'GLM_NODE_RANK={rank}',
           '-e', 'HF_HUB_OFFLINE=1', '-e', 'TRANSFORMERS_OFFLINE=1',
           '-e', 'TMPDIR=/cache/tmp',
           # Bash sources the foundation's BASH_ENV and activates CUDA compat.
           '--entrypoint', '/bin/bash', lock['image_id'], '-c',
           'exec /opt/venv/bin/python /opt/glm-defaults/launch.py']
    return cmd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--node', choices=NODES, default=socket.gethostname().split('.')[0])
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--arm', choices=experiment.ARMS, default='defaults')
    args = parser.parse_args()
    if args.node not in NODES:
        raise SystemExit('Explicit known node required')
    lock = json.loads((ROOT / 'profile.lock.json').read_text())
    cmd = command(args.node, Path.home(), lock, args.arm)
    if args.dry_run:
        print(shlex.join(cmd))
        return
    if socket.gethostname().split('.')[0] != args.node or os.uname().machine != 'aarch64':
        raise SystemExit('Wrong host or architecture')
    if checked('podman', 'ps', '-q'):
        raise SystemExit('Host still has running containers; authorized graceful stop required')
    exists = subprocess.run(['podman', 'container', 'exists', experiment.name(args.arm)]).returncode
    if exists != 1:
        raise SystemExit('Container exists or container probe failed; refusing replacement')
    image = json.loads(checked('podman', 'image', 'inspect', lock['image_id']))[0]
    if image['Id'].removeprefix('sha256:') != lock['image_id']:
        raise SystemExit('Image ID mismatch')
    labels = image.get('Labels') or image['Config']['Labels']
    if labels.get('vllm.source-tree') != lock['vllm_tree'] or labels.get('b12x.source-tree') != lock['b12x_tree']:
        raise SystemExit('Source labels mismatch')
    snapshot = Path.home() / '.cache/huggingface/hub/models--local-inference-lab--GLM-5.3-Flash-NVFP4/snapshots' / lock['checkpoint_revision']
    index = json.loads((snapshot / 'model.safetensors.index.json').read_text())
    shards = set(index['weight_map'].values())
    if len(shards) != 44 or index['metadata']['total_size'] != 198042331512 or not all((snapshot / s).is_file() for s in shards):
        raise SystemExit('Pinned checkpoint is incomplete')
    cache = Path.home() / '.cache/vllm-karmic-main-glm-defaults'
    cache.mkdir(exist_ok=True)
    (cache / 'tmp').mkdir(exist_ok=True)
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()
