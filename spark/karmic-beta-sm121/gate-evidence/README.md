# Retained build evidence

Copied without editing from dusty, September 20/21, 2026:

- `runtime/`: `build-receipts/runtime-20260921T002742Z-1655697/`.
- `serving/`: `build-receipts/serving-20260921T003918Z-1667340/`.

The initial regressions log intentionally contains the stopped obsolete-test
selection, after the 235-case pool-tail and seven-case cleanup groups passed.
The continuation log records the replacement six-case MoE test and remaining
groups. Do not present the initial invocation alone as an all-green gate.
The poisoned FlashKDA replay is the final 12-case gate.

`build.lock.json` records the actual component/runtime/serving image IDs and
wheel checksums read from the built image. Source selection remains a separate
publication record. The component receipts on dusty retain the recipes actually
used. Subsequent local hardening (receipt hashing, compiler-flag assertion,
extra identity labels) is for future builds and does not describe new bytes in
the current image. In particular, the existing vLLM component built with
MAX_JOBS=20 and NVCC_THREADS=4, which yielded Ninja -j5; the current recipe uses
NVCC_THREADS=1 for future 20-job builds. No scheduling-only rebuild was performed.

The actual NCCL build log was checked read-only and contained no
`-march`, `-mtune` or `-mcpu` options. A future Dockerfile gate now enforces this.
No claim is made about switched TP4 transport until the separate GLM window.

Known candidate reproducibility limitations: apt dependency selection was not
version-pinned; the retained installed-package inventory records what shipped.
B12X was fetched by immutable commit and tree-checked inside runtime assembly,
not built as an independent component. Its evidence is the runtime build and
native smoke, not an invented separate B12X BUILD-OK.
