# September 24 Karmic main DS4.1 artifacts

This directory contains the September 24 DS4.1 composition and its later
correctness/determinism investigations. It was formerly `ds41-20260924/`.

Start with [EXECUTION.md](EXECUTION.md) for qualification outcomes and the
history of diagnostic arms. `build.lock.json`, `runtime.lock.json` and
`nccl.lock.json` retain their frozen composition identities. The many diagnostic
scripts, patches, amendments and receipts remain together to preserve their
relationships and historical evidence.

The build uses the [September 22 foundation](../20260922/README.md), including
its `qsa865/` source lock and inherited gates. `prepare.py` and `preflight.py`
resolve that dependency explicitly. `build.sh` and `build-nccl.sh` share the
pair lock at `../.build.lock` with the other Karmic builders.

Local input check and build dry-run:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 preflight.py
PYTHONDONTWRITEBYTECODE=1 DRY_RUN=1 bash build.sh
```

A passing input check is not model qualification. No build or deployment was
performed as part of the directory reorganization. Historical receipt paths
and the archived CPU environment remain unchanged records of earlier runs.
