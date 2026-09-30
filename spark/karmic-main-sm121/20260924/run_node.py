#!/usr/bin/env python3
"""Launch one reviewed upstream-derived rank, never replace another container."""
import argparse
import hashlib
import ipaddress
import json
from pathlib import Path
import platform
import resource
import socket
import subprocess

from launch_contract import NODES, SELECTION_ROOTS, render, validate_chunking_image
from runtime import kit_digest, require
from diagnostic_overlay import validate as validate_overlay

ROOT = Path(__file__).resolve().parent
NAME = 'ds41-flash-karmic-main-tp4'
REVISION = 'fb2764a5cf321eaa5070ca8f9e892818f477c16d'


def read(argv):
    return subprocess.check_output(argv, text=True, timeout=30).strip()


def fabric(node):
    suffix = NODES[node][1].rsplit('.', 1)[1]
    for hca, nic, subnet in [('rocep1s0f0', 'enp1s0f0np0', '10.11.11'),
                             ('roceP2p1s0f0', 'enP2p1s0f0np0', '10.11.12')]:
        ip = subnet + '.' + suffix
        addresses = json.loads(read(['ip', '-j', '-4', 'address', 'show', 'dev', nic]))
        require(any(a['local'] == ip for interface in addresses for a in interface['addr_info']), f'Missing fabric IP {ip}')
        for peer, (_, address) in NODES.items():
            if peer != node:
                destination = subnet + '.' + address.rsplit('.', 1)[1]
                route = json.loads(read(['ip', '-j', 'route', 'get', destination]))[0]
                require(route.get('dev') == nic and route.get('prefsrc') == ip, f'Wrong fabric route: {destination}')
        port = Path('/sys/class/infiniband') / hca / 'ports/1'
        require('ACTIVE' in (port / 'state').read_text(), f'Inactive HCA {hca}')
        require('RoCE v2' in (port / 'gid_attrs/types/3').read_text(), f'Wrong GID on {hca}')
        require((port / 'gid_attrs/ndevs/3').read_text().strip() == nic, f'Wrong GID netdev on {hca}')
        require(str(ipaddress.IPv6Address((port / 'gids/3').read_text().strip()).ipv4_mapped) == ip,
                f'Wrong GID address on {hca}')


def checkpoint(cache):
    manifest = json.loads((ROOT / 'model-manifest.json').read_text())
    require(manifest['revision'] == REVISION, 'Checkpoint manifest drift')
    snapshot = cache / 'hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots' / REVISION
    for entry in manifest['files']:
        path = snapshot / entry['path']
        for part in (path, *path.parents):
            if part == cache:
                break
            require(not part.is_symlink() or not part.readlink().is_absolute(),
                    f'Absolute checkpoint link cannot survive remapping: {part}')
        require(path.resolve().is_relative_to(cache.resolve()), f'Checkpoint escapes HF mount: {path}')
        require(path.is_file() and path.stat().st_size == entry['size'], f'Checkpoint incomplete: {path}')
        if 'git_blob' in entry:
            data = path.read_bytes()
            digest = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
            require(digest == entry['git_blob'], f'Checkpoint metadata mismatch: {path}')


def command(node, pin, digest):
    profile = render(node)
    validate_chunking_image(pin, profile['env'])
    if 'DS41_DECISION_ROW_BLOCKS' in profile['env']:
        require(pin.get('diagnostic', {}).get('kind') in ('decision-row-capture', 'window-capture', 'indexer-capture', 'precision-capture',
                                                              'precision-release-candidate', 'ratio1-bf16-diagnostic',
                                                              'mhc-expanded-capture'),
                'KV block pin is restricted to decision-row-capture or window-capture')
    if 'DS41_NCCL_GEOMETRY' in profile['env']:
        require(pin.get('diagnostic', {}).get('kind') in ('window-capture', 'indexer-capture', 'precision-capture'),
                'NCCL geometry diagnostic is restricted to window-capture')
    if 'DS41_NCCL_ARM' in profile['env']:
        require(pin.get('diagnostic', {}).get('kind') == 'precision-capture',
                'Standard NCCL arm is restricted to precision-capture')
    hf = Path.home() / '.cache/huggingface'
    # The selection replay mounts its own seeded host root; the normal root is never written by it.
    cache = Path.home() / (SELECTION_ROOTS[profile['env']['DS41_SELECTION_REPLAY']]
                           if 'DS41_SELECTION_REPLAY' in profile['env'] else '.cache/vllm-jj-ds41-tp4')
    env = dict(profile['env'], DS41_NODE=node, DS41_KIT_SHA256=digest, PYTHONUNBUFFERED='1', PYTHONOPTIMIZE='0')
    argv = ['podman', 'run', '-d', '--pull=never', '--name', NAME, '--init',
            '--network=host', '--device', 'nvidia.com/gpu=all', '--device', '/dev/infiniband',
            *profile['container_resource_args'], '--security-opt', f'seccomp={ROOT / "seccomp-io-uring.json"}',
            '-v', f'{hf}:/root/.cache/huggingface:ro', '-v', f'{cache}:/cache:rw',
            '-v', f'{ROOT}:/opt/ds41-adapter:ro', '--label', f'local-inference.ds41.kit.sha256={digest}']
    overlay = validate_overlay(pin, ROOT)
    if overlay is not None:
        argv += ['-v', f'{ROOT / overlay["source"]}:{overlay["target"]}:ro',
                 '--label', 'local-inference.ds41.diagnostic-overlay.sha256=' + overlay['sha256']]
    for key, value in sorted(env.items()):
        argv += ['-e', f'{key}={value}']
    # The contract owns the tuning unset list; the approved NCCL geometry arm renders it empty.
    body = ('unset ' + ' '.join(profile['unset']) + '; ') if profile['unset'] else ''
    body += 'export LD_LIBRARY_PATH=/opt/nccl-2.30.7/lib:${LD_LIBRARY_PATH:-}; '
    body += 'exec /opt/venv/bin/python /opt/ds41-adapter/runtime.py'
    argv += ['--entrypoint', '/bin/bash', pin['image_id'], '-c', body]
    return argv, hf, cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node', choices=NODES, default=socket.gethostname().split('.')[0])
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    digest = kit_digest()
    pin = json.loads((ROOT / 'candidate.json').read_text())
    argv, hf, cache = command(args.node, pin, digest)
    if args.dry_run:
        print(json.dumps({'command': argv, 'profile': render(args.node)}, indent=2))
        return
    require(__debug__, 'Optimized Python is not admitted')
    require(args.node == socket.gethostname().split('.')[0] and platform.machine() == 'aarch64', 'Wrong host')
    require(not read(['podman', 'ps', '-q']), 'A container is running; explicit serving window required')
    require(not read(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits']), 'GPU busy')
    exists = subprocess.run(['podman', 'container', 'exists', NAME], timeout=30)
    require(exists.returncode == 1, 'Candidate exists or probe failed; never replace it')
    if cache.name != 'vllm-jj-ds41-tp4':
        require((cache / 'SELECTION-REPLAY-SEEDED.json').is_file(), 'Selection replay cache root is not seeded')
    require(resource.getrlimit(resource.RLIMIT_MEMLOCK)[0] == resource.RLIM_INFINITY, 'Host memlock is bounded')
    image = json.loads(read(['podman', 'image', 'inspect', pin['image_id']]))[0]
    require(image['Id'].removeprefix('sha256:') == pin['image_id'], 'Image mismatch')
    labels = image.get('Labels') or image['Config']['Labels']
    for label, value in [('local-inference.ds41-build-lock.sha256', pin['build_lock_sha256']),
                         ('local-inference.source-lock.sha256', pin['source_lock_sha256']),
                         ('local-inference.serving.recipe.sha256', pin['recipe_sha256'])]:
        require(labels.get(label) == value, f'Image label mismatch: {label}')
    if 'diagnostic' in pin:
        diagnostic = pin['diagnostic']
        for label, key in [('local-inference.ds41.diagnostic.kind', 'kind'),
                           ('local-inference.ds41.diagnostic.lock.sha256', 'lock_sha256'),
                           ('b12x.source-tree', 'b12x_tree')]:
            require(labels.get(label) == diagnostic[key], f'Diagnostic identity mismatch: {label}')
    fabric(args.node)
    checkpoint(hf)
    if args.preflight_only:
        print('DS41-HOST-PREFLIGHT-PASS', args.node)
        return
    cache.mkdir(parents=True, exist_ok=True)
    subprocess.run(argv, check=True, timeout=120)


if __name__ == '__main__':
    main()
