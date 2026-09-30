#!/usr/bin/env python3
"""Read pinned safetensors headers only; no tensor payload or GPU allocation."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

REVISION = '46aaae8a82032f77100f2f03e9cc11b391df3b4d'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('snapshot', type=Path)
    args = parser.parse_args()
    root = args.snapshot
    if root.name != REVISION:
        raise SystemExit('Pinned checkpoint revision required')
    index_bytes = (root / 'model.safetensors.index.json').read_bytes()
    index = json.loads(index_bytes)
    shards = sorted(set(index['weight_map'].values()))
    if len(shards) != 44 or index['metadata']['total_size'] != 198042331512:
        raise SystemExit('Checkpoint inventory mismatch')
    matrices, headers = [], []
    seen = set()
    for shard in shards:
        if Path(shard).name != shard:
            raise SystemExit('Unexpected shard path')
        with (root / shard).open('rb') as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise SystemExit('Truncated safetensors length')
            size = struct.unpack('<Q', prefix)[0]
            if not 2 <= size <= 16 * 1024 * 1024:
                raise SystemExit('Unexpected header size')
            raw = stream.read(size)
            if len(raw) != size:
                raise SystemExit('Truncated header')
        headers.append({'shard': shard, 'header_bytes': size, 'sha256': hashlib.sha256(raw).hexdigest()})
        for name, entry in json.loads(raw).items():
            if name == '__metadata__':
                continue
            if name in seen or index['weight_map'].get(name) != shard:
                raise SystemExit('Duplicate or inconsistent tensor index')
            seen.add(name)
            shape = entry['shape']
            if entry['dtype'] != 'BF16' or len(shape) != 2:
                continue
            byte_count = entry['data_offsets'][1] - entry['data_offsets'][0]
            if byte_count != 2 * shape[0] * shape[1]:
                raise SystemExit('BF16 matrix storage mismatch')
            matrices.append({'name': name, 'shape': shape, 'checkpoint_bytes': byte_count})
    if seen != set(index['weight_map']):
        raise SystemExit('Missing tensor headers')
    print(json.dumps({'revision': REVISION, 'index_sha256': hashlib.sha256(index_bytes).hexdigest(),
                      'headers': headers, 'bf16_matrices': sorted(matrices, key=lambda row: row['name']),
                      'caveat': 'Checkpoint sizes, not per-rank traffic; sharding, replication, packing and execution frequency need source attribution.'}, indent=2))


if __name__ == '__main__':
    main()
