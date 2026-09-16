# Historical Podman maintenance helpers

The three `podman-image-*` Python scripts are retained with the September 15
inventory, plan, execution, and post-inventory receipts under `logs/`.
They were not executed during the September 16 repository cleanup.

These scripts are not an unattended maintenance policy. In particular,
`podman-image-prune-exec.py` defaults to execution, not dry run, and its
leftover handling can remove containers and force image removal. Its
header's claim that leftover handling is unimplemented is stale. The
keep-latest-three policy does not by itself preserve every deliberate
rollback or component image.

Treat the saved commands and scripts as historical evidence. Reuse needs
an explicitly approved, current per-node keep/remove inventory and a safety
review of failure handling and leftover removal. Do not run an old plan to
clean the current fleet. Never use `podman system reset`.
