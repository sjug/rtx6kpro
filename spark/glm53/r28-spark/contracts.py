"""Small build contracts, shared by preparation and in-image verification."""

if not __debug__:
    raise RuntimeError("R28 verification requires Python assertions enabled")

import hashlib
import json
import re
from pathlib import PurePosixPath

NATIVE_VLLM = frozenset(
    {
        "cmake/external_projects/apply_flashkda_checkpoint_patch.cmake",
        "cmake/external_projects/flashkda.cmake",
        "cmake/external_projects/patches/flashkda-packed-checkpoints.patch",
        "csrc/flashkda_registration.cpp",
    }
)
FROZEN_TREES = {
    "vllm": "4b935ffc1d9ff38196ddddbc359110ea35c0a128",
    "b12x": "c4bfeee9f3c9457400d191c870eb2e44fbcd5c2e",
    "lmcache": "247f81b6fd3b156c402a0dd31cd4a21ae8e21160",
}


def validate_paths(name, paths):
    for path in paths:
        p = PurePosixPath(path)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError(f"invalid changed path: {path}")
        if name == "vllm":
            allowed = path in NATIVE_VLLM or (
                p.suffix in (".py", ".sh", ".md")
                and p.parts[0] not in ("csrc", "cmake", "requirements")
                and path != "setup.py"
            )
        elif name == "b12x":
            # No compiled source changed in this frozen B12X refresh.
            allowed = p.suffix in (".py", ".md", ".json", ".gz", ".txt", ".toml", ".sh")
        elif name == "lmcache":
            allowed = p.suffix == ".py" or path in (
                "csrc/storage_backends/fs/connector.cpp",
                "csrc/storage_backends/fs/connector.h",
            )
        else:
            raise ValueError(name)
        if not allowed:
            raise ValueError(f"unclassified {name} refresh: {path}")


def cache_fingerprint(lock):
    inputs = {n: lock[n]["tree"] for n in FROZEN_TREES}
    inputs.update(
        base=lock["base_image_id"],
        flashkda=lock["flashkda"]["base_commit"],
        flashkda_patch=lock["flashkda"]["patch_sha256"],
        build=lock["native_build_contract"],
    )
    digest = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    return "cu133-torch213-jj-r28-sm121-" + digest[:20]


def require_sm121a(output):
    # A generic sm_121 cubin is not proof of the architecture-specific target.
    if not re.search(r"(?<![A-Za-z0-9_])sm_121a(?=[.\s]|$)", output):
        raise RuntimeError("required sm_121a cubin absent")
