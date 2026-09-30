#!/usr/bin/env python3
"""Retain upstream tests and strengthen both graph replays against stale outputs."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

source, destination = map(Path, sys.argv[1:])
if destination.exists():
    raise RuntimeError('Refusing to replace a test receipt')
shutil.copytree(source, destination)
path = destination / 'test_vsplit.py'
original = path.read_bytes()
before = '        saved.fill_(float("nan"))\n        graph.replay()'
after = ('        saved.fill_(float("nan"))\n'
         '        output.fill_(float("nan"))\n'
         '        final.fill_(float("nan"))\n'
         '        graph.replay()')
# Keep the transformation explicit and fail if upstream changes either site.
text = original.decode()
if text.count(before) != 2:
    raise RuntimeError('Unexpected FlashKDA graph replay structure')
path.write_text(text.replace(before, after))
(destination / 'spark-test-adaptation.json').write_text(json.dumps({
    'original_sha256': hashlib.sha256(original).hexdigest(),
    'adapted_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    'change': 'Poison output and final state as well as checkpoint state before both replay loops',
}, indent=2) + '\n')
