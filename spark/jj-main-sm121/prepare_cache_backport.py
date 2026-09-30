#!/usr/bin/env python3
"""Port only upstream changed definitions, proving AST-equal prerequisite code.

Reads existing git objects only. Emits JSON with the patch and its source lock;
does not write objects, refs, checkouts or files.
"""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

from prepare_overlay import adapt_arches, adapt_head
from verify_sources import object_id, tree_id

BASE = '8e1f1e587f8d24faf606f334a1c4bdaaa6bd4368'
FIX = '0f60770e859ec95fd161b2a64c4925dfcfbd86ed'
REPO = Path.home() / 'git/vllm'
PATHS = ('vllm/v1/worker/b12x_startup.py', 'tests/v1/executor/test_b12x_startup.py')

def read(ref, path):
    return subprocess.check_output(['git', '-C', str(REPO), 'show', f'{ref}:{path}'], text=True)

def definitions(source):
    return {node.name: node for node in ast.parse(source).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))}

def span(node):
    return min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1, node.end_lineno

def port(path):
    target, before, after = read(BASE, path), read(FIX+'^', path), read(FIX, path)
    current, old, new = [definitions(s) for s in (target, before, after)]
    lines, source_lines = target.splitlines(keepends=True), after.splitlines(keepends=True)
    replacements, additions, changed = [], [], []
    for name, node in new.items():
        prior = old.get(name)
        if prior is not None and ast.dump(prior) == ast.dump(node):
            continue
        changed.append(name)
        start, end = span(node)
        content = ''.join(source_lines[start:end])
        if prior is None:
            if name in current:
                raise RuntimeError(f'Unexpected existing definition: {name}')
            additions.append(content)
        else:
            if ast.dump(current[name]) != ast.dump(prior):
                raise RuntimeError(f'Prerequisite differs: {name}')
            start, end = span(current[name])
            replacements.append((start, end, content))
    for start, end, content in sorted(replacements, reverse=True):
        lines[start:end] = [content]
    result = ''.join(lines).rstrip() + '\n\n\n' + '\n\n'.join(additions) + '\n'
    if path.startswith('vllm/'):
        anchor = 'from contextlib import nullcontext\n'
        if result.count(anchor) != 1:
            raise RuntimeError('Import anchor differs')
        result = result.replace(anchor, anchor + 'from typing import TYPE_CHECKING\n\n'
            'if TYPE_CHECKING:\n    from b12x.preparation import TuningCacheRequirement\n')
    # Every changed definition is exactly the upstream AST, and all other
    # candidate definitions remain unchanged. No unrelated Karmic tests imported.
    final = definitions(result)
    for name, node in final.items():
        expected = new[name] if name in changed else current[name]
        if ast.dump(node) != ast.dump(expected):
            raise RuntimeError(f'Unexpected semantic change: {name}')
    patch = f'diff --git a/{path} b/{path}\n' + ''.join(difflib.unified_diff(
        target.splitlines(keepends=True), result.splitlines(keepends=True), f'a/{path}', f'b/{path}'))
    return target, result, patch, changed

def generate():
    changes, records, overrides = [], {}, {}
    for path in PATHS:
        before, after, patch, changed = port(path)
        changes.append(patch)
        records[path] = dict(before_sha256=hashlib.sha256(before.encode()).hexdigest(),
            after_sha256=hashlib.sha256(after.encode()).hexdigest(), definitions=changed)
        overrides[path.encode()] = object_id('blob', after.encode())
    for path, transform in [('CMakeLists.txt', adapt_arches),
            ('vllm/models/glm5next/nvidia/mtp_draft_head.py', adapt_head)]:
        overrides[path.encode()] = object_id('blob', transform(read(BASE, path)).encode())
    raw = subprocess.check_output(['git', '-C', str(REPO), 'ls-tree', '-rz', BASE])
    entries = []
    for record in raw.split(b'\0'):
        if record:
            meta, path = record.split(b'\t', 1)
            mode, _, oid = meta.split()
            entries.append((path, mode, overrides.get(path, oid.decode())))
    patch = ''.join(changes)
    lock = dict(base_image_id='e7926f763859ba5800cc24198ef297a12322fdaac93b55ac89b88abe02c1b1fb',
        base_vllm_tree='182f7f4ac37b980a3bb81e74797a78e9dbf822cb', upstream_fix=FIX,
        base_commit=BASE, vllm_tree=tree_id(entries), files=records,
        patch_sha256=hashlib.sha256(patch.encode()).hexdigest())
    return dict(patch=patch, lock=lock)

if __name__ == '__main__':
    print(json.dumps(generate()))
