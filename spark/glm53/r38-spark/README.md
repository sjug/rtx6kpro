# Jovian Judgement R38 on DGX Spark

Recorded qualification as of 2026-09-16: R38 passed the bounded GLM-5.3-Flash
and DSv4 Vision profiles. Qwen remains on R32 by user decision after a
repeatable small R38 prefill cost and a rejected FlashInfer/CUDA-decode
experiment. See `ROLLOUT.md`, `qualification/GLM-QUALIFICATION.md`,
`qualification/flashinfer-direct-20260915/RESULTS.md`, and
`../../ds4-vision/r38/QUALIFICATION.md`. This is the recorded deployment
decision, not a fresh live-node check. Qwen experiments are paused.

The original build window used idle rusty/toby. Future builds require a
fresh authorized idle window; the historical preflight is not permission
to build while these nodes are serving.

The image is derived from qualified R32
`74e53e710bef141f6f68e722582569f9c6aa388bce405ad6f1423566a2300c9c`.
The frozen upstream source recipe is
`69a86060a158bcdd14af182b0381be9e92649db3`, published in the R38 runbook at
rtx6kpro commit `99de462ef9c573ae4f79c92a6f8ddf62b7879bfb`.
vLLM is frozen at `66c293578412417476f842c1da5805d3a3d959a8`, B12X at
`ce419b52681b7922bb0972d4b58b590a3fd005b2`. Add only the merged vLLM PR #756
commit `27f0745fc11f2346bacb646c2d4e1864f6dacd89`, not the mutable JJ tip.
See `source.lock.json` for every source tree, patch, input and launcher digest.

## What changes

Tracked vLLM, B12X and LMCache sources are replayed exactly. The unchanged
Spark overlay admits SM121 native builds and the GLM draft-head gate.
The published R38 Torch schema, Torch mutable-argument and CuTe sentinel
corrections are applied only to their exact expected dependency bytes.

vLLM's native kernel sources, Rust sources, toolchain, shell build script,
MANIFEST.in and FlashKDA inputs match the R32 base. Three build/versioning
inputs differ: `CMakeLists.txt`, `setup.py` and `tools/build_rust.py`.
`contracts.py` pins both sides of each reviewed transition; the lock carries
the exact review diff. CMake preserves the default SM121 kernel build and
patched-header contents, while avoiding needless header timestamp changes.
The Python changes exclude source-addressed wheel tags from version lookup;
the metadata install already uses an explicit version. Any other native-input
change fails closed. The existing native
objects, Rust frontend, CuTe Python, Triton source and resolved site-packages
symlink are retained and checked before and after installation. RoCEnante's
proxy source and ABI 3 are unchanged and its in-image cache is retained.

PR #756 snapshots verified draft-token counts before asynchronous handoff,
preventing a later mutation from changing the delivered counts. Its two real
CUDA regression cases are required. This field is used by adaptive verification;
inclusion is not evidence that every MTP profile was previously affected.

FlashInfer is rebuilt at `803c4664` for `12.1a`, including the changed
sparse-MLA prefill CUDA source and its AOT cache. LMCache's filesystem C++
source changed: rebuild its three CPU modules and retain its source-identical
CUDA extension and cuMem interposer. The pre-reinstall inventory covers
distribution-owned files outside the package too. Unknown payloads fail.

B12X includes new PLE reader and storage C code. Install pinned Ubuntu
liburing 2.5 headers and library and compile/import the host loader once,
requiring resolved liburing linkage as well as ABI 1. This does not enable
offload or the loader: the preserved
serving profiles use InstantTensor with `VLLM_PLUGINS` empty. Its ordinary
runtime cache remains source/compiler-keyed and may JIT if enabled later.

The Qwen and GLM launch settings remain the qualified R32 settings, including
aligned recurrent checkpoints, utilization 0.85, MTP3, and GLM's BF16 head.
Qwen's public alias is `Qwen3.8-Flash-Next`. LMCache stays disabled. Neither
RTX tuning nor the new DS4.1 serving defaults are adopted automatically.

## Build

Build parallelism follows the existing component recipes:

- FlashInfer: `MAX_JOBS=20`, `FLASHINFER_NVCC_THREADS=1`, preserving the
  effective policy from
  `Dockerfile.glm53-jj-r11-spark-sm121` at build-repo commit
  `79f688fc866ed2ee4a7ad65e63c34a5a198d380e`, also used by II r18p.
  Both the R11 and R38 FlashInfer code read `FLASHINFER_NVCC_THREADS`, default
  1, not the generic `NVCC_THREADS=4` in those older image recipes. We pin 1
  explicitly, not an unmeasured four NVCC threads per job. Ninja consumes
  `MAX_JOBS` directly. The unused `AOT_MAX_JOBS_CAP` override is removed.
- LMCache CPU rebuild: `MAX_JOBS=8`, `NO_GPU_EXT=1`, retaining the reviewed R37
  build policy. R38's LMCache source is unchanged from R37, but our derivation
  base is R32, so its filesystem CPU delta still needs rebuilding. Its CUDA
  extensions are reused, not rebuilt in this stage.

Contract tests pin these component-specific policies. Podman's `--jobs=1`
controls build-stage concurrency, not compiler parallelism, and stays intact.
Source versions, SM121 targets and serving settings are unchanged by this
correction. Do not overwrite the input kit of a running build: its receipts
must continue to describe the files and settings it actually used.

Keep this directory and its receipts on persistent disk. Composition uses
isolated bare repositories under ignored `.compose/repos`; it adds refs there,
not in the developer source repositories. Generated patches are sufficient
for image assembly; bare repositories are not transferred into the image.

1. Run `prepare_sources.py` with the pinned upstream/base objects available.
   It reconstructs the R32 Spark base, verifies the R38 trees, proves vLLM
   native-source equality and the three exact reviewed build-input transitions,
   and pins native build recipes in the fingerprint.
   The pinned R32 lock and Spark overlay are bundled in this directory, so
   sibling release directories need not be checked out. B12X's entire
   non-Python changed-path set is allowlisted, not just selected C suffixes.
2. On rusty, `bash build-flashinfer.sh` builds a component under a unique
   `localhost/voipmonitor/build-components:jj-r38-flashinfer-*` retention tag.
   Both rusty and toby must be idle. Its persistent receipt contains its ID,
   tag, image inspection, and exact Dockerfile and build-script copies.
   The component also embeds those inputs and their digests beside the wheels.
3. Set `FLASHINFER_IMAGE` to that completed component ID and run `bash build.sh`.
   For an uncommitted exported kit, supply its real `RECIPE_COMMIT` and explicit
   `ALLOW_DIRTY_BUILD=1`; both dirty state and recipe bytes are recorded.
   `DRY_RUN=1 bash build.sh` runs offline recipe checks without node actions.
   Before assembly, the component's embedded recipe and input receipts are
   copied into the runtime receipt and verified against this kit's lock.
   A different recipe is rejected even when source and architecture match.
4. Runtime assembly uses a unique
   `localhost/voipmonitor/build-candidates:jj-r38-runtime-*` retention tag.
   Its inspection is saved immediately after the build, before any gate.
   These are non-serving tags. Only a passing candidate also receives
   `localhost/voipmonitor/vllm:jj-r38-spark-sm121`.

No serving tag exists for a failed candidate. The installed rootless
`podman-auto-update.service` on rusty runs `podman image prune -f` after its
auto-update step. The daily timer has a 15-minute random delay and
`Persistent=true`, so a missed run can also execute after the user manager
starts. The former untagged R37 outputs and 34 other images were removed by
that service on September 14. Intermediate build caches are not durable.

Retention tags protect completed outputs from that dangling-only prune, not
from `prune -a`, explicit removal or system reset. The host service is unchanged.
Schedule compilation outside the local 00:00 to 00:15 prune window and check
for a running or catch-up invocation before starting. Interaction with
in-flight multi-stage build images has not been qualified; output tags do
not establish that guarantee. No prune, reset, cache clearing, transfer,
serving restart, commit or push is implicit.

## Acceptance boundary

Gates preserve the R32 native/custom-op checks, Mamba/pool-tail matrices,
checkpoint scalar restore, prefix lifecycle, draft head, persistent MXFP8
GEMM regression, and 12 FlashKDA cases. Added R38 checks cover the three
dependency patches, ten NVFP4 FC1 fragment-rebind cases, three small-tile
cooperative MoE cases, all 24 sparse-MLA dual-cache prefill cases, 20 scheduler
admission cases, and four native LMCache filesystem delete cases. The gate
also includes the two PR #756 asynchronous-count snapshot cases. The same
selection list is collected first, before numerical execution. Collection
needs a visible GPU because some modules inspect capability at import. Every
selected suite must then actually run; collection alone is not acceptance,
and skips, wrong counts and zero cases fail. Dependency checks also verify
that imported modules resolve to the patched files.

Build gates are not multi-node model qualification, quality evaluation or
benchmarks. Qwen/GLM/DS4 serving remains a separate phase with checkpoint,
runner, fabric and image identities, semantic output checks, and the normal
`run_bench.sh` campaign. No performance improvement is claimed by this build.
Relative to R32, the GB10 policy profile has eight new or changed components
out of 23, including `moe.decode`; qualification must measure decode against
R32. Decoded profile contents are unchanged between R37 and R38.
