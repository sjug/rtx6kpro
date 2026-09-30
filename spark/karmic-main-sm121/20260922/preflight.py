#!/usr/bin/env python3
"""Verify all frozen inputs before taking a build window."""
import argparse
import json
from pathlib import Path
from contracts import file_sha, load_lock, require, sha

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--values', action='store_true')
    parser.add_argument('--manifest', action='store_true')
    args = parser.parse_args()
    lock = load_lock(ROOT)
    require(file_sha(ROOT / 'inherited/base-build.lock.json') == lock['base_build_lock_sha256'], 'Base lock changed')
    for path, expected in lock['assets'].items():
        require(file_sha(ROOT / path) == expected, f'Inherited asset changed: {path}')
    for name, row in lock['sources'].items():
        require(file_sha(ROOT / 'payload' / f'{name}.tar') == row['archive_sha256'], f'Archive changed: {name}')
    inputs = {str(path.relative_to(ROOT)): file_sha(path) for path in ROOT.rglob('*')
              if path.is_file() and path.suffix in ('.py', '.sh')
              and not any(part in ('__pycache__', 'build-receipts', 'payload') for part in path.relative_to(ROOT).parts)}
    for name in ('Dockerfile', '.containerignore', 'source.lock.json', 'recipe-origin.json'):
        inputs[name] = file_sha(ROOT / name)
    if args.manifest:
        print(json.dumps(inputs, sort_keys=True, indent=2))
    elif args.values:
        print(inputs['source.lock.json'])
        print(sha(json.dumps(inputs, sort_keys=True).encode()))
        print(lock['cache_fingerprint'])
        for model in ('qwen38-flash-next', 'glm53-flash'):
            print(lock['assets'][f'inherited/launchers/serve-{model}-karmic-spark.sh'])
        for name in ('vllm', 'b12x'):
            print(lock['sources'][name]['refreshed_tree'])
    else:
        print('LOCAL-INPUT-INTEGRITY-PASS')

if __name__ == '__main__':
    main()
