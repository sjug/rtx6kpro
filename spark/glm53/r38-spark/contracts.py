"""Frozen source identities and shared native verification helpers."""

import re

if not __debug__:
    raise RuntimeError("Build verification requires assertions enabled")

FROZEN_TREES = {
    "vllm": "077347fdeab296404d0b0cb297d316176acc635a",
    "b12x": "6abad73444018f7ce1f0bdc3f0992649a4c21722",
    "lmcache": "5a88a1ea9d2627c76288d056e7193f7454669b64",
}

ASYNC_COUNTS_COMMIT = "27f0745fc11f2346bacb646c2d4e1864f6dacd89"

# Reviewed exact-input transitions, not a filename-only permission to change
# native code. CMake preserves the same default SM121 compilation and patched
# header bytes; the Python changes only exclude wheel tags from version lookup.
NATIVE_INPUT_UPDATES = {
    "CMakeLists.txt": (
        "1c3b2e49607cc386a407b0a80fdc0c4dea97469b",
        "93f0c054f565aaad99d8298c4dcc2df5e5deb82e",
    ),
    "setup.py": (
        "ae64a13daa0f1facc255afcdb4ffcad264776b98",
        "ce6681d21b58433f37679e0e1655dc83e097aa56",
    ),
    "tools/build_rust.py": (
        "a000ec8169fd7acafac3f99a4350998f350fa32a",
        "ce60248b38036cfb59140d87e26ce0fdb131a869",
    ),
}


def require_native_transition(path, before, after):
    if path in NATIVE_INPUT_UPDATES:
        if (before, after) != NATIVE_INPUT_UPDATES[path]:
            raise RuntimeError(f"Unreviewed native build-input change: {path}")
    elif before != after:
        raise RuntimeError(f"Native source changed: {path}")


def require_sm121a(output):
    if not re.search(r"(?<![A-Za-z0-9_])sm_121a(?=[.\s]|$)", output):
        raise RuntimeError("required sm_121a cubin absent")


def require_linkage(output, *, allow_missing_driver=False):
    missing = {line.split("=>", 1)[0].strip() for line in output.splitlines()
               if "=> not found" in line}
    if missing - ({"libcuda.so.1"} if allow_missing_driver else set()):
        raise RuntimeError(output)
