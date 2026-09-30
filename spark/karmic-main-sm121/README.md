# Karmic main SM121 build artifacts

Build lineages are grouped by their source-composition date. Keep descendants,
locked inputs, and qualification evidence with the foundation they depend on.

| Directory | Contents | Status |
| --- | --- | --- |
| [20260922](20260922/README.md) | September 22 foundation; September 23 `qsa865`, `qsa865-hcbase`, and `glm-defaults` derivatives; build receipts and qualification | Historical built/tested lineages; see each execution record |
| [20260924](20260924/README.md) | September 24 DS4.1 composition, NCCL component, subsequent diagnostic arms and receipts | Historical DS4.1 investigation; see [EXECUTION.md](20260924/EXECUTION.md) |

The newer beta kit is [September 29](../karmic-beta-sm121/20260929/README.md).
It reuses the locked foundation in `20260922/` through its explicit FOUNDATION
path. Source refresh pins and image identities were not changed by organization.

All five main builders and the new beta builder share `.build.lock` here for
the dusty/kirby pair. Staging future build kits must preserve this dated layout.
Remote serving containers and existing remote files were not changed.

## Migration from the flat layout

- Former top-level build files, `inherited/`, `payload/`, `qsa865/`,
  `qsa865-hcbase/`, `glm-defaults/`, and their evidence now live in `20260922/`.
- Former `ds41-20260924/` is now `20260924/` and explicitly imports its
  foundation from `../20260922/`.
- Historical receipts, archived environments, raw results and source locks keep
  their original bytes and recorded historical paths. They are evidence, not
  rewritten descriptions of the new local layout.
- `20260922/qualify_qwen.py` is a frozen, hash-locked asset embedded in existing
  image provenance. Use `20260922/qualify.py` to run it from the dated layout;
  the wrapper adjusts only its repository root. Execution scripts use this wrapper.
- No compatibility symlink tree, artifact deletion, image build or deployment was
  introduced. The September 20 beta tree is outside this main-tree cleanup.
