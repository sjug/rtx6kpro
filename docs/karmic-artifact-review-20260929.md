# September 29 beta review and Karmic artifact organization

Scope: review the prepared `spark/karmic-beta-sm121/20260929` kit and organize
local Karmic-main artifacts by dated build lineage. No builds, GPU execution,
remote changes, fetches, branch changes, commits or publication were performed.

## Beta review: standards

One P2 provenance finding: `runtime.lock.json` lines 38-39 retain the old main
`published_image` and `recipe_commit` as top-level fields, while
`build.lock.json` and the Dockerfile identify the September 29 beta recipe
`f28aea04babb`. `prepare.py` copies the old lock and adds a new composition
without separating these historical fields. The installer writes that runtime
lock as the image's canonical source lock. Per the upstream release workflow,
foundation and new publication provenance should be named separately. This is
not a source-byte mismatch. Suggested correction is explicit foundation
metadata plus beta publication/recipe metadata, with deterministic regeneration
of the new beta runtime/build locks. Review only: no source lock was rewritten.

No other actionable documented-standard violation was found.

## Beta review: specification

No actionable spec-correctness finding was confirmed. The kit reproduces the
specified vLLM `99cbe782f3b6` and B12X `1b6cd278626a` trees over base image
`1a7a8acf`, carries the SM121 overlay, checks the merged PR865 patch, and
explicitly defers Rust DSML native code and the LMCache refresh. The installer
checks the complete tracked-file delta and native-byte preservation. The
build publishes its tag only after the declared gates.

Status is prepared and locally checked, not built or model-qualified. Native
ABI and GPU regression execution still belong to the authorized build window;
model qualification follows separately.

## Layout and compatibility

See [main index](../spark/karmic-main-sm121/README.md).

- `spark/karmic-main-sm121/20260922/` holds the September 22 foundation and its
  September 23 Qwen/GLM derivatives. Keeping the derivatives together preserves
  their source-lock and inherited-asset relationships.
- `spark/karmic-main-sm121/20260924/` holds the DS4.1 composition and diagnostic
  history, formerly `ds41-20260924/`.
- Today's beta explicitly references the foundation under `20260922/`.
- All six builders use the shared `spark/karmic-main-sm121/.build.lock`.
- A small `20260922/qualify.py` wrapper resolves the new repository depth for
  the frozen `qualify_qwen.py` asset. Its original bytes stay intact because
  they appear in source locks embedded in existing images.
- Operational paths, affected ancestor traversal, container gate mounts, and
  repository documentation links were updated. Remote staging must preserve
  the new layout; current remote files and running services were untouched.
- Raw receipts and their recorded historical paths, payloads, source locks,
  images and archived environments were not rewritten or removed. The
  September 20 beta tree was outside this main-tree cleanup.

## Validation

- 82,325 previously inventoried artifact files/symlinks remain present after
  relocation, including ignored files. 81,074 protected evidence, payload,
  inherited and archived-environment entries retained inode, size and mtime.
- All 75 inventoried frozen lock files retain their original SHA256.
- Main, both QSA variants, DS4.1 and September 29 beta preflights pass.
- 65 focused tests pass: 16 foundation contracts, 7 QSA wide, 7 QSA HC-base,
  16 GLM profile, 13 DS4.1 launcher/wheel contracts, and 6 beta integrity/replay
  checks. The beta replay reaches both locked result trees.
- The beta dry-run and ShellCheck pass. Python syntax was checked for 463
  operational source files. Qwen qualification wrapper help resolves correctly.
- Beta's preparation-determinism test was excluded because it regenerates
  locks and the archive; no preparation or frozen-source regeneration was
  necessary for this organization.

Review totals: standards 1 provenance finding; specification 0 confirmed findings.

## P2 resolution verified, September 29

The provenance finding is resolved in the prepared beta kit. Inspection confirms
that all six inherited fields (`published_image`, `recipe_commit`,
`base_build_lock_sha256`, `native_inputs_sha256`, `native_metadata`, and
`lmcache_refresh`) moved unchanged into `foundation`, alongside the base image
and source-lock identity. The separate `publication` block identifies the beta
tag, registry digest, recipe, channel, and both source commits, and explicitly
limits the amd64/SM120 publication to source identity. Result manifests remain
under `sources` and `assets`.

The regenerated runtime lock SHA256 is
`d995a5140e257030e0f15afba2156089a7fc9573fa87682407f5ac685595dc1c`,
and `build.lock.json` pins that exact digest. The earlier statement that all
75 locks were unchanged describes the reorganization checkpoint; this subsequent
provenance fix intentionally regenerates the two beta locks.

Independent verification reran the provenance regression, installer tree replay,
and preflight tests under normal Python and `-O`: all six executions passed.
The author's complete eight-test normal/optimized runs, deterministic regeneration,
ShellCheck and dry-run results were reported separately and were not all rerun
in this focused verification. No build, node access or source regeneration was
performed to close this finding.

Current review totals: standards 0 open findings (1 resolved); specification
0 confirmed findings. Status remains prepared, not built or model-qualified.
