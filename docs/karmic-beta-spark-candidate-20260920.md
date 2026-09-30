# Karmic beta Spark candidate

Status: the September 20 build completed, and native gates passed. Image
`f30dc6d9a2a6f6fc0ac9ff8cddb04d9f631f4a87b48ca7cac69254802fe83233`
is undergoing Qwen qualification on dusty/kirby, with R32 retained stopped.
Initial MTP3 correctness passed; performance and the remaining model gates are
pending. GLM and DS4 Vision have not been changed. Follow the live
[qualification record](../spark/karmic-beta-sm121/QUALIFICATION.md).
The sections below retain the preparation and build history, not a claim of
model promotion or an executable build lock.

## Frozen publication

- Image reference: `ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20260920-828a46ee4bbf1aa4`.
- Published image digest: `sha256:b737367417a357a09088796553cf8d0ebe240cb3dc3a38f51c379a34a3c28081`.
- Build recipe: `8c5aa7f828689e6385fa309acd0cce168fb38ec7` in `local-inference-lab/blackwell-llm-docker`.
- Release: `karmic-kraken-beta-828a46ee4bbf1aa4fd0440d8ad930d0990977a4a1885ae0473eaa5a92749ae94`.
- Runtime manifest SHA-256: `34bfe7f059c1dbc433b6aa18b382661533acd82be48fc425680f4d86cdafb354`.
- Assembly SHA-256: `4478b3e7251938ec64b70b3102eebc2e7dc6c91e013f33e399fa97927a7bae52`.
- Container release SHA-256: `d622ce8b037b0618f7c13813718f5dde14b66562ef17ba8134ec3fca6f714b89`.

| Component | Source commit | Git tree, where checked locally |
| --- | --- | --- |
| vLLM | `57a80980bbf4b40398de7ed851b23e55a3a4c50e` | `a431c22c423738bb8807c8b9e60a67db3bea1958` |
| B12X | `e9ce547767ff9ee6509faf294fa1b4e2380dfbf5` | `9f20c0e6b9d49a42bc19f45429ac44758cf5fe76` |
| FlashInfer | `2206a14e46387a56c093860a46bbbdd00596b75b` | `dbbe29fea63e3135dbd980132cacb168473363ef` |
| LMCache | `688bee14e157b64623d93c07fc0d4db93470e12f` | `fa24eb3adba2015056d76a42fcd6f660e590460e` |
| InstantTensor | `95d4729b6d6a991bb8de61877147a9d9d9100b23` | Not replayed locally |
| NCCL | `93fe05d9f9b6963ef841166a69cd0b30e4efe97b` | Not replayed locally |
| FlashKDA | `b59532f1f464fbd536272780e30df5bf6a2ccc02` | Includes both recipe patches below |

FlashKDA requires both `flashkda-packed-checkpoints.patch` and
`flashkda-sm120-cta-copy.patch` from the selected vLLM source. A commit pin alone
does not identify its compiled source.

## ARM64 foundation and required adaptation

Registry metadata for `nvcr.io/nvidia/pytorch:26.08-py3` supplies ARM64 manifest
`sha256:237ecf9ac7373daf91b31bb4f86651ce1ce57b676366ed435aa1aba61dad81d5`.
Its config declares CUDA `13.4.1.012`, Torch `2.14.0a0+4fdf77b`, and NCCL
`2.30.7+cuda13.3`. Only manifest/config metadata was inspected; neither native
execution nor dependency patch compatibility has been verified on this base.

The upstream runtime recipe defaults to the AMD64 NGC manifest. Its published
wheels and architecture-specific native support closure cannot simply be reused
on ARM64. Build the native components against the selected ARM64 foundation;
do not carry R38 CUDA 13.3/Torch 2.13 binaries into it. Install and verify the
selected coherent NCCL 2.31.2 object rather than silently retaining NGC's 2.30.7.

The selected vLLM tree still needs the Spark architecture-list addition and the
SM121 draft-head capability adaptation. Adapt those narrowly: preserve the new
chunked draft-head scale calculation. The prior async verified-count cloning
fix is already present and must not be applied twice.

Use the published runtime dependency manifest rather than blindly installing
the generic vLLM CUDA requirements, which describe different dependency versions.
Check the Torch and CuTe dependency patch input hashes on the ARM64 foundation.
The old Torch 2.13 mutation-tracking patch is not a presumed requirement of 2.14.

## Build and qualification boundaries

### R38 input disposition checklist

These are build requirements, not claims that the pending port has passed:

| R38 input | Candidate disposition |
| --- | --- |
| `vllm.patch` | Replace the R32-to-R38 refresh delta with the pinned Karmic source; do not replay a release delta as a local fix. |
| `b12x.patch` | Replace the release delta with the pinned Karmic B12X source. |
| `lmcache.patch` | Replace the release delta with LMCache `688bee14`; audit its four changed R38 paths for retained behavior. Rebuild natives, not the R38 CPU-only refresh/reuse scheme. |
| `spark-overlay.patch` | Re-derive only the architecture-list and draft-head capability changes, preserving the new chunked scale reduction and capability tests. |
| `vllm-pr756.patch` | Drop the duplicate patch; the commit is already an ancestor. Verify behavior, not equality to the old blob. |
| `torch-schema-enumeration.patch` | Use the selected foundation recipe's corresponding patch and byte gates; verify ARM64 installed inputs. |
| `torch-mutable-argument-metadata.patch` | Do not replay the Torch 2.13 patch blindly; verify the 2.14 implementation and regression coverage. |
| `cutlass-sentinel-identity.patch` | This is the CuTe `jit_executor.py` identity-sentinel fix, not a separate CUDA kernel patch. Re-derive from the selected dependency patch manifest with installed-byte gates. |
| Dependency patch manifest, installer and tests | Replace with the pinned recipe's versions; retain fail-closed input/output digest and imported-path checks. |
| `pack_lmcache_native_reuse.py` | Drop cross-foundation native reuse; build the candidate native payload afresh and inventory it. |
| R38 FlashKDA patch and binary digest | No binary reuse. Compare the old patch with the new base plus both upstream patches, documenting subsumption or any remaining delta before building. |

### Explicit native and transport gates

The R38 `b12x.patch` channel-variable matches are in two upstream evidence
scripts under `docs/evidence/nvfp4-immutable-input/`:
`launch_glm_nvfp4_split_local.sh` and
`run_qwen_cookbook_source_qualification.sh`. Both explicitly select 16 channels;
neither is a Spark serving runner or an implicit B12X runtime default. These
matches do not establish a channel pin to carry into the candidate.

- **FlashKDA GB10 CTA-copy gate:** the new patch tests `__CUDA_ARCH__ >= 1200`,
  so its source branch includes SM121 despite the filename. This is not proof of
  correctness. Verify actual emitted architecture/family targets and execute the
  checkpoint, replay and numerical-reference tests on GB10. Do not reject a
  legitimate family target merely because its spelling differs from `sm_121a`.
- **NCCL ARM64 gate:** build the pinned 2.31.2 source for ARM64. Audit the selected
  branch's host flags and dependencies for AMD Turin/x86 assumptions; verify
  architecture, linkage, loaded object identity and distributed behavior.
  Do not copy the historical II channel contract into current JJ profiles:
  R38 Qwen and GLM runner tests explicitly reject `NCCL_MIN_NCHANNELS` and
  `NCCL_MAX_NCHANNELS`. Preserve their model-specific transport configuration.
  Historical exact-four/launch-order findings remain separate evidence, not a
  universal image default.
- **Foundation smoke before serving:** check Torch CUDA execution, representative
  rebuilt native operators, dependency patches and NCCL identity. Then run the
  single-model correctness screen. A failure in the latter does not by itself
  distinguish toolchain from engine; attribution needs an isolated reproducer or
  matched source/foundation control. That control is conditional investigation
  work, not part of the initial window: if attribution requires it, assess and
  separately build R38 sources against the CUDA 13.4/Torch 2.14 foundation,
  recording any compatibility changes. Do not assume that combination builds
  unchanged or that an existing image supplies this control.
- Run GLM draft-head, FlashKDA and other single-GPU build tests on dusty in the
  same build window. The later four-node window still owns distributed GLM
  admission, correctness and performance qualification.

1. Finish the ARM64 recipe and lock, including dependency closure, patch hashes,
   composed trees, native architecture checks and executable regression gates.
2. Obtain a dusty/kirby interruption window before stopping Qwen or building.
   Both nodes are needed for its current TP2 service; they cannot keep that
   service running while dusty builds.
3. Retain the established FlashInfer settings: `MAX_JOBS=20` and
   `FLASHINFER_NVCC_THREADS=1`. Protect component/candidate images from the
   automatic dangling-image prune, record inspect data before gates, and publish
   the candidate's normal image tag only after build gates pass. A tag is not
   model promotion; per-model qualification remains pending until completed.
4. Qualify Qwen on dusty/kirby against R32: real outputs before timing, retrieval,
   MTP acceptance, concurrent and repeated-request checks, then the standard
   benchmark grid and matched prefill measurements. Pin checkpoint and serving
   settings; keep aligned checkpoint policy and LMCache disabled.
5. Qualify GLM independently against its R38 serving profile in a separately
   authorized four-node window. DS4 Vision on rusty/toby follows independently.
   No result on one model qualifies another.

Do not edit the separate benchmark repository. Bulk transfers use only the
200G fabric and preserve image identity with the established Docker archive
workflow. Keep rollback images and model caches. DS4.1 remains outside this
rollout: neither its failed 524K gate nor the Qwen R38 prefill deficit is claimed
fixed by selecting this candidate.

## Execution, September 20

- ARM64 foundation pulled on dusty, image ID
  `0abf0782ad35f4d9ef6497c8442fd9c59a91ba64c0044a645283d95533b3887f`.
  Tagged as a non-serving build component to protect it from dangling-image prune.
- Qwen stopped worker-first using `podman stop -t 60`. Retained container IDs:
  dusty `74e75ce3baff7b59d5d2008c19c04754b54b54464ca571b55f07ff473d809a44`,
  kirby `cae069abb64543e5424c7ffa7dfb62f5f3e3d62eda0d346dce6e9efaf4a74e5f`.
  Both use R32 image `74e53e710bef141f6f68e722582569f9c6aa388bce405ad6f1423566a2300c9c`.
- Foundation GPU gate passed on dusty with driver `580.173.02`: Torch
  `2.14.0a0+4fdf77b940.nv26.08`, CUDA `13.4`, Triton `3.8.0`, BF16 GEMM and
  freshly compiled Triton output correct. This is not candidate native or model
  qualification. The base's NCCL remains `2.30.7` pending replacement.
- Small build kit lives at `spark/karmic-beta-sm121/` locally, staged inside the
  existing dusty build workspace at
  `/home/jugs/git/bld-jj-r38-spark/karmic-beta-sm121/`. Its directory ancestry
  does not indicate R38 binary reuse. No new checkout was created.
- Three overlay preparation tests pass normally and under `python -O`; source
  trees verify against the selection. The local vLLM checkout is shallow, so an
  ancestry query cannot independently establish PR756 history here. The pinned
  source contains the verified-count snapshot implementation; no duplicate patch
  is emitted.
- FlashInfer's first attempt stopped before compilation because uv respected the
  source's Python 3.10 preference. The recipe now explicitly selects the base's
  `/usr/bin/python3` (3.12). The retry is compiling 2,882 Ninja actions with
  `-j20`, `--threads=1` and `compute_121a,code=sm_121a`.
  The managed unit is `karmic-flashinfer-build.service`; logs and recipe copies
  are under the remote kit's `build-receipts/` directory. No final image exists
  yet and the remaining native/runtime recipes and model gates are pending.

### NCCL source audit finding

The selected NCCL commit is 13 publication/build commits beyond `fb6f4099`.
Inspecting `fb6f4099` itself shows that its compatibility claim is broader than
its guards: `ncclMaybePromoteAmdZen5EightGpuRing` is explicitly gated to x86 AMD
Zen5 and the narrow same-host eight-GPU signature, but the change to
`ncclTopoCompareGraphs` prefers more channels at equal aggregate bandwidth
without an architecture guard. Removing the AMD exception from the asymmetric
channel search changes AMD behavior; that search was already allowed on ARM.
The common makefile does not add `-march`/`-mtune` host tuning flags in the
inspected compiler flag definitions.

Therefore the pinned source is buildable as an ARM64 candidate in principle,
but its graph-selection behavior must not be described as unchanged solely
because the numeric NCCL version matches. Preserve the frozen source selection
and require distributed transport receipts, selected channel counts, loaded
library identity and correctness before qualification. The draft component
recipe does not add channel pins or remove upstream graph changes.

The retained R32 image's local NCCL labels identify NVIDIA source
`7b83616df3ae082a1f32bb74c27458bfe8153a13`, ref `v2.31.2-1`, rather than the
candidate's canonical fork. Its inherited `com.nvidia.nccl.version=2.30.7`
label is not the declared local replacement version; inspect the loaded object
in the eventual comparison, not that inherited label alone.

### Serial native build continuation

The current FlashInfer receipt is
`build-receipts/flashinfer-20260920T215648Z-1131249` inside the remote kit.
`karmic-native-continuation.service` waits for that exact build to finish and
requires both its successful status and `BUILD-OK`. It then builds NCCL, vLLM,
LMCache and InstantTensor sequentially, stopping at the first failure. It does
not assemble a runtime, start models or publish a serving tag.

ARM64 toolchain images were separately resolved and pulled:

- uv 0.11.30: `sha256:2f0a1c36ce6323e61bc81d39e692319669c03e385b592d670b6ba36dc884e3aa`.
- Rust 1.95.0 bookworm: `sha256:8e45ae5b178fa788bbbd818b42a1f93a6e2c03e7144badd5e0a37087537177e1`.

These replace upstream's architecture-specific tool images, not its selected
tool versions. The vLLM build-tool lock likewise changes the CMake and Ninja
wheel hashes to ARM64 at the same versions. Local preparation/policy tests total
eight and pass normally and under Python optimization; shellcheck passes for
the build scripts. Those checks are not substitutes for the pending build and
GPU gates.

### Overlay failure and restart

FlashInfer completed as `358f888657b631b53a0706a3e5c88a680c0e363f48502fac49af2dbfcf135c73`;
NCCL completed as `ee19802e04c9ee6923dbb7a3f19147e4eae82edd015c14618a931eb9e51bd719`.
The vLLM component then failed at 22:24:50 UTC before compilation with
`does not match index` on the two overlay paths. The continuation stopped;
LMCache and InstantTensor had not started. This failure was missed until the
user requested a live status check at 22:34 UTC.

Replaying the failed command in cached layer `11dff6511485` reproduced it.
`git diff --exit-code HEAD` confirmed no source-content or mode difference;
`git update-index --refresh` followed by the same indexed apply passed.
The cause was stale index stat metadata across the container layer boundary.
The recipe now refreshes that metadata, retains indexed apply checks, and
asserts the resulting tree `02a457e2e933d0fb20d5786835110546acf7d8f2`.
`test-overlay-layer.sh` passes on the actual cached layer. The managed queue
was restarted at vLLM, not at FlashInfer or NCCL.

### Native completion and runtime assembly, September 21 UTC

All five component receipts record exit status zero and BUILD-OK. Component IDs:

| Component | Image ID |
| --- | --- |
| FlashInfer | `358f888657b631b53a0706a3e5c88a680c0e363f48502fac49af2dbfcf135c73` |
| NCCL | `ee19802e04c9ee6923dbb7a3f19147e4eae82edd015c14618a931eb9e51bd719` |
| vLLM | `83f8183cbcc1063d50e1dffc8e9c95d751c7da6e4658c95e1c6da98c4f3a7f41` |
| LMCache | `bc4722058e7d27f8683546fc7cd234c8acdfe829abce9964dbca882fb4c1cd2f` |
| InstantTensor | `ecfffd85f816b5a82bea260e68977b1256ca15d1c741ef3a48f945dfa9a9a5ee` |

The vLLM build actually used Ninja `-j5`: its setup divides MAX_JOBS=20 by
NVCC_THREADS=4. This was not the intended 20-job contract. Its original recipe
and inputs remain in its component receipt and image. The future recipe now
uses NVCC_THREADS=1, yielding 20 Ninja jobs; the completed native component is
not rebuilt solely to change compilation scheduling. Do not claim its recipe
hash equals the subsequently corrected local Dockerfile.

Runtime external dependencies retain upstream's exact versions, with ARM64
hash resolution instead of the x86-only overlay hash. The dependency image is
`ad008e629e810957ba8c8b18b89712717816f52b35960bddf6f9b495bdab7673`.
Runtime assembly completed at 00:28:39 UTC as
`627e4cf62a02a73a5f01aea3fc8194b9f567a2e475ecf0cec82de091d148ec14`.
This is a retained component, not a serving or model-qualified tag.

Passed on dusty: native custom-op execution, source and build-product hashes,
all selected extension imports and linkage, one mapped NCCL 2.31.2 library,
RoCE proxy ABI 3 with compiler disabled, and actual FlashKDA `sm_121a` cubin.
The 12 packed/dense FlashKDA cases passed, then passed again with all three
output buffers poisoned before graph replay. The 235-case GLM pool-tail matrix
and seven sparse-cleanup cases passed. The old warmup selector no longer exists;
Karmic's six-case prefill-capacity/exact-decode-reuse matrix replaces that gate.
Remaining regressions and model qualification are still pending.
