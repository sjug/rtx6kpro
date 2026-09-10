"""Small build contracts, shared by preparation and in-image verification."""

if not __debug__:
    raise RuntimeError("R29 verification requires Python assertions enabled")

import hashlib
import json
import re
from pathlib import PurePosixPath

NATIVE_VLLM = frozenset(
    {
        "csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu",
        "csrc/libtorch_stable/ops.h",
        "csrc/libtorch_stable/torch_bindings.cpp",
    }
)
FROZEN_TREES = {
    "vllm": "ec2025d80cdade8a1083e1ce61f0607bb227f5d1",
    "b12x": "c4bfeee9f3c9457400d191c870eb2e44fbcd5c2e",
    "lmcache": "cd8a219da05eec6c7d6c24712f9ee1a9186c3567",
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
            allowed = p.suffix == ".py"
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
    return "cu133-torch213-jj-r29-sm121-" + digest[:20]


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
