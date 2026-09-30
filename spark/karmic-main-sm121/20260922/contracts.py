"""Fail-closed source and payload contracts shared by preparation and installation."""
import hashlib
import json
from pathlib import Path


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_lock(root):
    return json.loads((Path(root) / 'source.lock.json').read_text())


def git_tree(files):
    """Reproduce Git tree identity without writing objects to a source checkout."""
    tree = {}
    for name, (mode, data) in files.items():
        parts = name.split('/')
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        require(parts[-1] not in node, f'Duplicate path {name}')
        node[parts[-1]] = (mode, data)

    def object_id(kind, data):
        return hashlib.sha1(kind + b' ' + str(len(data)).encode() + b'\0' + data).digest()

    def walk(node):
        entries = []
        for name, value in node.items():
            directory = isinstance(value, dict)
            mode, oid = ('40000', walk(value)) if directory else (value[0], object_id(b'blob', value[1]))
            entries.append((name.encode() + (b'/' if directory else b''),
                            mode.encode() + b' ' + name.encode() + b'\0' + oid))
        return object_id(b'tree', b''.join(value for _, value in sorted(entries)))
    return walk(tree).hex()


def overlay(files):
    """Carry only the previously qualified SM121 architecture and draft-head gate."""
    result = dict(files)
    for name, before, after, count in (
        ('CMakeLists.txt', ';11.0;12.0")', ';11.0;12.0;12.1")', 2),
        ('vllm/models/glm5next/nvidia/mtp_draft_head.py',
         'if (major, minor) != (12, 0):', 'if (major, minor) not in ((12, 0), (12, 1)):', 1),
        ('vllm/models/glm5next/nvidia/mtp_draft_head.py',
         'f"12.0; got {major}.{minor}"', 'f"12.0 or 12.1; got {major}.{minor}"', 1),
    ):
        mode, data = result[name]
        require(data.count(before.encode()) == count, f'Overlay context changed: {name}')
        result[name] = (mode, data.replace(before.encode(), after.encode()))
    return result


def manifest(files):
    return {name: {'mode': mode, 'sha256': sha(data)} for name, (mode, data) in sorted(files.items())}


def safe_path(root, name):
    path = Path(name)
    require(not path.is_absolute() and '..' not in path.parts, f'Unsafe path: {name}')
    result = Path(root) / path
    require(result.parent.resolve().is_relative_to(Path(root).resolve()), f'Escaping parent: {name}')
    return result
