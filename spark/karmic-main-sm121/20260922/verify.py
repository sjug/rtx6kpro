#!/usr/bin/env python3
"""Final source and native reuse audit after package installation."""
import json
import os
from pathlib import Path

from contracts import file_sha, git_tree, load_lock, require
from install import B12X, KIT, VLLM, verify_files

if not __debug__:
    raise RuntimeError('GPU gates require assertions enabled')
lock = load_lock(KIT)
require(file_sha(KIT / 'source.lock.json') == os.environ['SOURCE_LOCK_SHA256'], 'Lock mismatch')
for name, root in (('vllm', VLLM), ('b12x', B12X)):
    row = lock['sources'][name]
    verify_files(root, row['files'])
    package = {}
    for path, entry in row['files'].items():
        if path.startswith(name + '/'):
            source = root / path
            data = os.readlink(source).encode() if source.is_symlink() else source.read_bytes()
            package[path[len(name) + 1:]] = (entry['mode'], data)
    require(git_tree(package) == row['package_tree'], f'{name} package tree mismatch')
for path, expected in json.loads((KIT / 'native-reuse.json').read_text()).items():
    require(file_sha(path) == expected, f'Native reuse mismatch: {path}')
import vllm
import b12x
require(Path(vllm.__file__).is_relative_to(VLLM / 'vllm'), 'Wrong vLLM import')
require(Path(b12x.__file__).is_relative_to(B12X / 'b12x'), 'Wrong B12X import')
from b12x.comm.roce import _proxy
require(_proxy.load().roce_abi_version() == 4, 'Wrong proxy ABI')
print('KARMIC-MAIN-IDENTITY-PASS', flush=True)
