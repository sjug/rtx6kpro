"""Small build contracts, shared by preparation and in-image verification."""

if not __debug__:
    raise RuntimeError("R32 verification requires Python assertions enabled")

import hashlib
import json
import re
from pathlib import PurePosixPath

NATIVE_VLLM = frozenset()
FROZEN_TREES = {
    "vllm": "80a18accc688cdee2974f4c5e03e9416b2896087",
    "b12x": "c4bfeee9f3c9457400d191c870eb2e44fbcd5c2e",
    "lmcache": "dc1dad48e97cb849fcccf4be289e86bf8449988d",
}


def validate_paths(name, paths):
    for path in paths:
        p = PurePosixPath(path)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError(f"invalid changed path: {path}")
        if name == "vllm":
            allowed = (
                p.suffix == ".py"
                and p.parts[0] not in ("csrc", "cmake", "requirements")
                and path != "setup.py"
            )
        elif name == "b12x":
            # No compiled source changed in this frozen B12X refresh.
            allowed = False  # B12X is byte-identical, not merely native-compatible.
        elif name == "lmcache":
            allowed = (
                p.suffix == ".py"
                and p.parts[0] not in ("csrc", "rust", "setup_extensions", "requirements")
                and path != "setup.py"
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
    return "cu133-torch213-jj-r32-sm121-" + digest[:20]


def require_sm121a(output):
    # A generic sm_121 cubin is not proof of the architecture-specific target.
    if not re.search(r"(?<![A-Za-z0-9_])sm_121a(?=[.\s]|$)", output):
        raise RuntimeError("required sm_121a cubin absent")


def require_linkage(output, *, allow_missing_driver=False):
    missing = {line.split("=>", 1)[0].strip() for line in output.splitlines()
               if "=> not found" in line}
    allowed = {"libcuda.so.1"} if allow_missing_driver else set()
    if missing - allowed:
        raise RuntimeError(output)
