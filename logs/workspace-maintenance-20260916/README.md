# Workspace maintenance, 2026-09-16

Scope: this rtx6kpro checkout only. No serving-node actions, benchmark-repo
changes, pushes, model-cache changes, or source-repository edits.

## Retention

- R37's corrected, superseded recipe and failed-build narrative are retained,
  without claiming a successful R37 image. The older pre-gate-fix source
  snapshot is preserved under its `build-receipts/pre-gate-fixes/` directory.
  Historical R37 node-side build logs were not present in this checkout;
  this pass did not retrieve or invent them.
- R38 implementation and its complete build receipts are retained separately
  from qualification records. The original build-kit archive preserves the
  bytes used by the dirty build, independently of later documentation and
  qualification-tool additions. Image labels and old manifests are unchanged.
- All qualification failures, excluded contaminated GLM runs, comparisons,
  and Qwen restoration evidence remain. Rejected experiments are identified
  as historical, with the user's R32 pause decision recorded.
- DSv4 Vision R38 implementation and qualification records are retained.
- Podman cleanup scripts and receipts are retained as historical operations,
  not certified general-purpose maintenance tools. See
  `../../scripts/PODMAN-MAINTENANCE.md` before considering any reuse.
- The dated kernel-thread review is retained as a research snapshot. Its
  upstream status claims were not refreshed during this maintenance task.

## Disposable files and intentionally retained caches

Remove only generated composition indexes and Python bytecode in the R37/R38
artifact trees, plus the cached bytecode of the newly retained Podman planner.
These files can be regenerated; the removed-path inventory is `removed.txt`.
The pre-gate-fix source snapshot is moved out of ignored scratch storage only
after the destination is byte-checked. No source or receipt is discarded.

The ignored `.compose/repos` repositories remain intentionally. They preserve
pinned upstream/composed source objects for offline reconstruction and analysis.
R38's LMCache and FlashInfer repositories use alternates in R37's cache, so
deleting R37's cache alone would break them. They are not disposable indexes
and are not added to Git. Older unrelated caches and nested checkouts are out
of scope.

## Local validation

R37: 34 offline tests and build dry run passed. R38: 36 recipe/native/gate
tests, two grid-admission tests, two launcher-permission tests, build dry run,
and both runner suites passed. Shell syntax checks passed for the current
R37/R38/DSv4 artifacts; DSv4 head and worker host renders passed. No image,
CUDA, remote, or full DSv4 container-render test was rerun.

Python syntax parsing also passed for 63 current artifact/helper files.
Standard credential-pattern scans of candidate text files found no matches.
A separate scan of 196 JSON objects found no nonempty sensitive environment
fields; the original build-kit archive passed the credential-pattern scan
and matched its recorded SHA-256. These are bounded checks, not a claim
that arbitrary logs can never contain sensitive data.

## Retained commits

- `d5b96c8`: corrected R37 recipe and failed-build history.
- `86e4b1d`: R38 SM121 build and serving implementation.
- `128973d`: R38 build, qualification, and Qwen rejection evidence.
- `7ed3a03`: DSv4 Vision R38 deployment tooling.
- `6e1da9b`: DSv4 Vision R38 qualification receipts.
- The maintenance commit containing this record: historical Podman cleanup
  tooling and receipts, the dated kernel review, and workspace cleanup.

The documentation is folded into the maintenance commit, not a seventh commit.
Hardware-key authentication required retries; no unsigned fallback was used.
Signatures are checked against the configured SSH public security key with
a temporary allowed-signers file, without changing Git configuration. The
temporary commit-message and signer files are removed after the final amendment.
No push is part of this maintenance pass.
