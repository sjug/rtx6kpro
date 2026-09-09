# JJ R28 Spark build candidate

Status: **built on dusty; all image gates passed on 2026-09-08**. Image ID
`cd93d80b3f9547f70e1e4608cd42af7c4b0f2ff33d92d5e520913cc9bab2d8d1`.
Qwen qualification passed on dusty/kirby on 2026-09-08. Aligned MTP3 remains
serving after the auto and MTP0 controls. The separately authorized GLM rollout
passed qualification at 19:20 EDT on 2026-09-08; R28 aligned MTP3 is left serving
on sparky/buddy/rocky/lucky, with R27 stopped as rollback. GLM performance is
effectively at R27 parity, not the Qwen-sized gain. Nous remains untouched. See the
[GLM rollout record](qualification/glm-20260908/RESULTS.md).
See [QUALIFICATION.md](QUALIFICATION.md) for results and serving identities,
and [BUILD-STATUS.md](BUILD-STATUS.md) for historical build receipts and retries.
After the user's dusty storage reset, R28 was restored from kirby's archive;
the old R27 containers are not a rollback on dusty.

## Frozen source identities

Published R28 lock SHA-256:
`15a9a649559830822cb943ea0c3c6a644c8b69c9e54d7be3e265801851843932`.
Recipe reference: blackwell-llm-docker commit
`53031598b6ba4c176e9a64512c17cad18a8bf829`, not the moving recipe branch tip.

| Component | Published commit | Candidate tree |
| --- | --- | --- |
| vLLM | `2531689fa50b956d3e1156e1ab80d119aaf34c1e` | `4b935ffc1d9ff38196ddddbc359110ea35c0a128` |
| B12X | `3edbcbce70f491741b82f5eab9c1b30b39447228` | `c4bfeee9f3c9457400d191c870eb2e44fbcd5c2e` |
| LMCache | `617a1b47a790de6b86eea92f59deb232a9eff87d` | `247f81b6fd3b156c402a0dd31cd4a21ae8e21160` |

vLLM retains the qualified R27 CMake architecture and optional NVFP4
draft-head capability overlay. The production profile still selects BF16.
B12X and LMCache are byte-exact upstream trees, without new runtime patches.
PR 674 is not reapplied: R28 widens the restore scalar arguments with
`tl.full(..., tl.int64)` before arithmetic and carries a strengthened GPU test.

`prepare_sources.py` uses private Git indices, reconstructs the R27 Spark base,
and proves every generated patch replays to the declared tree. It does not
check out branches, reset source repositories, or create worktrees. Scratch
indices and fetched FlashKDA objects live on persistent disk in `.compose/`.
It does write composed objects and retain refs under `refs/spark/jj-r28/`
in the source repositories, preventing pruning without moving working branches.
Exact upstream whitespace in generated patches is preserved.

## Native boundary

Base image: `ef669fa1cde3e99936c02575eca8f610990bb6c64dd1bbfcb04fd42dde87afae`.

- Rebuild FlashKDA at base `3b225bf2` with R28 packed-checkpoint patch
  `a9537532...`. The registered `fwd` operator gains `checkpoint_indptr`.
  Reusing R27's object would violate the new API. Require an exact `sm_121a`
  cubin, clean linkage, and all 12 packed/dense checkpoint GPU cases, without
  skips. Graph replays poison output, final state, and checkpoints first.
- Rebuild the LMCache SM121 wheel. The native filesystem connector changed;
  the cuMem interposer is rebuilt and tracked alongside it. Keep LMCache off.
- Reuse the remaining vLLM native objects, Rust binaries, copied FlashAttention
  Python, and Triton sources. Hash them before and after source/metadata
  installation; retain and resolve the site-packages Triton symlink.
- Build/load the RoCEnante proxy from the exact candidate source, outside the
  mounted JIT cache. Loading with `CC=/bin/false` must pass. API 1 / proxy ABI 3.

The base runtime inspection found CMake and Ninja but neither uv nor cargo.
Bootstrap pinned uv 0.12.9 only through pip, then use uv for packages/builds.
LMCache's selected setuptools extension path does not invoke Rust; vLLM reuses
the existing Rust products. No Rust toolchain is presumed installed.

LMCache builds from the locally prepared, digest-pinned Git bundle, not a
build-time repository fetch. Its CUDA object must contain `sm_121a`; CPU-only
modules and the interposer must import/link cleanly. LMCache remains disabled
and none of these checks qualify a cache-enabled serving profile.

The cache fingerprint includes all three full source trees, the base image,
the FlashKDA base/patch and the Dockerfile/compiler contract. Native changes
outside package subtrees therefore invalidate caches too. Every image-owned JIT cache path is rebased onto it,
including paths that R27 still inherited from R11. Hugging Face storage is
unchanged and must never be cleared.

## Serving contract and release boundary

Carry the R27 Qwen TP2 and GLM TP4 profiles with aligned checkpoints,
InstantTensor BUFFERED, existing revision pins, MTP3, and existing memory,
capture and transport settings. Do not import upstream RTX L2-prefetch,
persistent-grid, split-MoE, 16-channel NCCL or NVFP4 proposal-head defaults.
Qwen keeps PyNCCL; GLM keeps RoCEnante plus PyNCCL.

The normal candidate tag is `localhost/voipmonitor/vllm:jj-r28-spark-sm121`.
Compilation uses untagged immutable image IDs. The FlashKDA builder ID and
binary SHA are frozen in `build-receipts/<run>/native.txt` before assembling
the runtime. Assembly, labels and GPU checks must match that independently
recorded SHA. Only a fully gated runtime receives the normal tag. A failure
retains its logs and IDs, does not publish the tag, and does not block retry.

The source lock records both launcher hashes. A mounted GLM launcher must
match the image's launcher label; its in-container preflight checks the baked
expected hash again. The recipe label records dirty/untracked state honestly.
An uncommitted build requires explicit `ALLOW_DIRTY_BUILD=1`, and its recipe
content manifest, not the ancestor commit alone, identifies its inputs.
Node runners refuse a real launch until `EXPECTED_IMAGE_ID` is supplied from
the completed build receipt. No image identity is invented before building.
The existing R27 deployment remains the comparison and rollback.

Local validation:

```bash
uv run --no-project --python python3 spark/glm53/r28-spark/prepare_sources.py
DRY_RUN=1 bash spark/glm53/r28-spark/build.sh
```

The actual build is scoped to dusty and refuses running containers on dusty
or kirby. It never stops them implicitly. Wait for approval to gracefully
stop Qwen worker first, head last; then build on dusty. No GLM window is
authorized by this build step. No automatic image transfers or launches.

After GPU build gates, qualify Qwen on the pair first: semantic output before
timing, MTP0/MTP3 boundaries and native context, padded transitions, identical
bursts and head-of-line reproducer, and the standard `run_bench.sh` grid versus
R27. GLM follows in its own authorized window using the existing frozen cache
pairs/triples, semantic and 1M retrieval battery, and the same standard grid.
Auto policy and 256-token recurrent retention are separate controls, not
silent defaults. Before filing the R27 report, run both the identical-prompt
burst and the fresh-request head-of-line reproducer under R28 `auto` on each
model, then compare with aligned under matched settings. R28 specifically
changes waiting-head admission. Its scheduler regression shows the second
boundary-hit request admitted while another decodes and both decoding in the
next step. Isolated admission does not imply whole-burst serialization.
Neither defect is presumed to remain or to be fixed on Spark before execution.

## Remaining execution gates

Build steps 1-3, distribution to dusty/kirby, and Qwen qualification completed
on 2026-09-08. GLM distribution and aligned qualification subsequently passed
in their separately authorized window that evening. Auto GLM controls remain
unrun. The frozen source lock's status describes its
pre-build creation; build and qualification completion are recorded separately.

1. Local negative-path and runner tests pass; review the finalized candidate diff.
2. Stage the recipe and `inputs/lmcache.bundle` on dusty. The roughly 52 MB
   bundle is Git-ignored and will not arrive with a clone or pull. Transfer the
   exact prepared bundle using the 200G mesh for any inter-Spark hop, then
   verify its SHA-256 against `lmcache.bundle_sha256` in `source.lock.json`
   and run the build dry run on dusty before taking down Qwen. Do not regenerate
   or replace the bundle without updating and reviewing the lock.
3. Approved Qwen stop, then native compilation and all GPU gates on dusty.
4. Pin the resulting image ID and native receipts; distribute unchanged Docker
   archives over 200G only, verify image identity on each receiver.
5. Qwen qualification, then separately authorized GLM qualification.

Nothing in the source replay or dry run is a claim of build or serving success.
