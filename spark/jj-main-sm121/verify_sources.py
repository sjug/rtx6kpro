#!/usr/bin/env python3
"""Verify frozen inputs and compute the overlay tree without writing Git objects."""
import hashlib
import json
from pathlib import Path
import subprocess

from prepare_overlay import adapt_arches, adapt_head

ROOT = Path(__file__).resolve().parent


def object_id(kind, payload):
    return hashlib.sha1(kind.encode() + b" " + str(len(payload)).encode() + b"\0" + payload).hexdigest()


def read(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def tree_id(entries):
    """Build nested Git trees in memory, preserving executable and symlink modes."""
    root = {}
    for path, mode, oid in entries:
        node = root
        parts = path.split(b"/")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = (mode, oid)

    def digest(node):
        payload = bytearray()
        for name in sorted(node, key=lambda key: key + (b"/" if isinstance(node[key], dict) else b"")):
            value = node[name]
            mode, oid = (b"40000", digest(value)) if isinstance(value, dict) else value
            payload.extend(mode + b" " + name + b"\0" + bytes.fromhex(oid))
        return object_id("tree", bytes(payload))
    return digest(root)


def verify(repos=Path.home() / "git"):
    lock = json.loads((ROOT / "source-selection.json").read_text())
    for name, pin in lock["sources"].items():
        actual = read(repos / name, "rev-parse", pin["commit"] + "^{tree}").decode().strip()
        if actual != pin["tree"]:
            raise ValueError(f"{name}: expected {pin['tree']}, got {actual}")
    pin = lock["sources"]["vllm"]
    repo = repos / "vllm"
    overrides = {}
    for path, transform in (("CMakeLists.txt", adapt_arches),
                            ("vllm/models/glm5next/nvidia/mtp_draft_head.py", adapt_head)):
        content = read(repo, "show", f"{pin['commit']}:{path}")
        overrides[path.encode()] = object_id("blob", transform(content.decode()).encode())
    entries = []
    for record in read(repo, "ls-tree", "-rz", pin["commit"]).split(b"\0"):
        if not record:
            continue
        metadata, path = record.split(b"\t", 1)
        mode, _, oid = metadata.split()
        entries.append((path, mode, oid.decode()))
    if tree_id(entries) != pin["tree"]:
        raise ValueError("Independent tree reconstruction disagrees with Git")
    patched = tree_id([(p, m, overrides.get(p, oid)) for p, m, oid in entries])
    emitted = subprocess.check_output(["python3", str(ROOT / "prepare_overlay.py"), "--repos", str(repos)])
    if emitted != (ROOT / "spark-overlay.patch").read_bytes():
        raise ValueError("Overlay differs from pinned source transformation")
    return {"vllm_overlay_tree": patched,
            "overlay_sha256": hashlib.sha256(emitted).hexdigest()}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
