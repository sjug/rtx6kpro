"""Frozen source identities and shared native verification helpers."""

import re

if not __debug__:
    raise RuntimeError("Build verification requires assertions enabled")

FROZEN_TREES = {
    "vllm": "61e4d99858fefebc8cbe460d8643b0e04e3ab6bc",
    "b12x": "480ea232d52fff4271b0412470d4ec340196735a",
    "lmcache": "5a88a1ea9d2627c76288d056e7193f7454669b64",
}


def require_sm121a(output):
    if not re.search(r"(?<![A-Za-z0-9_])sm_121a(?=[.\s]|$)", output):
        raise RuntimeError("required sm_121a cubin absent")


def require_linkage(output, *, allow_missing_driver=False):
    missing = {line.split("=>", 1)[0].strip() for line in output.splitlines()
               if "=> not found" in line}
    if missing - ({"libcuda.so.1"} if allow_missing_driver else set()):
        raise RuntimeError(output)
