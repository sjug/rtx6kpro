# DS4/DSpark on DGX Spark (GB10, SM121, aarch64)

Goal: replicate the x86 DS4 vLLM images for the DGX Spark nodes `rusty` and
`toby`. This directory holds the investigation results and the per-version
build/launch script sets:

```text
v9/    ds4dspark-v9 (Eldritch Enlightenment) — built, deployed, validated
v10/   ds4dspark-v10 (Fathomless Firmament)  — built + GPU smoke 2026-07-09
v16/   unified GLM-5.2 + DS4 v16             — built 2026-07-14, serving DS4
                                               TP2 on rusty+toby
v17/   GLM-5.2 v17 NVFP4/NF3 hybrid          — built + DRY_RUN-validated
                                               2026-07-14 on rusty
v18.4/ Gilded Gnosis Grid48 for SM121         — source-pinned build + TP4
                                               switched-200G launcher
blackwell-llm-docker/   arch-parameterized build infra (branch spark/sm121-arm64)
src/    component clones used by the SM121 audits
```

## v18.4: SM121 Grid48 source build

`v18.4/` pins the reviewed B12X and vLLM Grid48 branches directly. B12X is
`6b10833` (Grid48, both v18 compatibility fixes, and review cleanup) and
vLLM is `df7a0b7`. The build performs a real CuTe Grid48 compile on the
48-SM GB10 and admits only a 48-CTA, one-CTA-per-SM, spill-free kernel. The
four-node launcher retains TP4/DCP1 and NCCL over the two switched 200 GbE
rails. See [`v18.4/README.md`](v18.4/README.md) for pins and commands.

## v17: GLM-5.2 hybrid TP4 for the 4-node cluster (prepared 2026-07-14)

Target: `madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid` (models/glm5.2_v17.md) at
TP4 across the sparky/buddy/rocky/lucky cluster (one GB10 per node; the
~744B hybrid checkpoint cannot fit fewer than four 121 GB nodes).

- `v17/build-ds4dspark-v17-spark-sm121-cu132.sh` — v17 delta on the v16
  Spark build: only the vLLM (`6ccc3eb`) and B12X (`1377d5f`) pins change,
  plus the `serve-glm52-hybrid-v17.sh` launcher. Runs on the existing
  `spark/sm121-arm64-v16` branch; reuses the arm64 base images. Post-build
  DRY_RUN checks ported from the canonical v17 recipe (podman, `2>&1`).

Built 2026-07-14 on rusty (three runs):

```text
localhost/voipmonitor/vllm:fathomless-firmament-v17-spark-sm121-vllm6ccc3eb-b12x1377d5f-fi801d57a-cu132-20260714
image id 56a90d60ce31, 24.6 GB
vllm 0.11.2.dev280+fathomless.firmament.v17.vllm6ccc3eb.b12x1377d5f.fi801d57a.sm121.cu132.20260714
```

Two SM121-side fixes were needed (r1 and r2 failure logs preserved on rusty
as `~/build-v17-20260714-r{1,2}.log`):

1. `patches/vllm-sm121-cuda13-supported-archs-20260714.patch` (**required**,
   applied via `VLLM_PATCH_FILE`): vLLM's CUDA>=13 `CUDA_SUPPORTED_ARCHS`
   lacks 12.1, so `TORCH_CUDA_ARCH_LIST=12.1a` collapsed to plain `12.0`
   and the v17 NVFP4 KV-cache additions to
   `csrc/libtorch_stable/cache_kernels.cu` failed at ptxas
   (`cvt.e2m1x2.f32` needs an arch-specific target). One line; x86-neutral;
   **upstream candidate** — any SM121 CUDA-13 vLLM build hits this.
   Verified in the final image: `_C_stable_libtorch.abi3.so` carries 63
   `sm_121a` cubins. Note this also means the v16 Spark image compiled its
   vLLM kernels family-generic (no 12.1a) — worked because v16 had no
   arch-specific instructions in base sources; a v16 rebuild with the patch
   would give native SM121 codegen.
2. `launchers/serve-glm52-hybrid-v17.sh` (new) and
   `launchers/serve-glm52-v16.sh` (DCP-workspace-gate update) pulled from
   upstream `blackwell-llm-docker` @ `6d3d0aa` — the spark branch pin
   (`d104659` merge) predates them, and the required-launcher check fails
   without the hybrid preset. Cheap fix: `COPY launchers/` sits after the
   vLLM compile step, so only the image tail rebuilt.

All post-build DRY_RUN checks passed (hybrid TP4/DCP4 and TP4/DCP1,
workspace-off override, DS4 tp2 dspark, cooperative B12X, fathomless
glm52); the hybrid TP4/DCP4 check was additionally re-run manually against
the tagged image.

### Four-node serving results (2026-07-14/15)

Deployed on the sparky/buddy/rocky/lucky cluster: TP4/DCP1, MTP0,
checkpoint rev `68babde`, NCCL over the two switched 200G rails
(`rocep1s0f0,roceP2p1s0f0`), `GPU_MEM=0.86`, `MAX_MODEL_LEN=262144`.
Weights 82.56 GiB/rank (314 s InstantTensor cold load; ~8.5 min full
boot warm-JIT), KV cache 16.04 GiB = 472,064 tokens (1.80x full-length
concurrency). Correctness probe clean — first NF3 execution on SM121.
An earlier bring-up profile (`GPU_MEM=0.85`, `MAX_MODEL_LEN=131072`,
431,168 KV tokens) measured identically within noise; note the 128k
decode band REQUIRES max_model_len > ~135k or the bench's
prompt+generation budget 400s at request validation.

Decode (aggregate tok/s, sustained; llm-inference-bench defaults):

| ctx \ cc | 1 | 2 | 4 |
|---|---:|---:|---:|
| 0 | 15.7 | 25.3 | 40.7 |
| 16k | 15.3 | 24.2 | 38.9 |
| 32k | 15.2 | 24.2 | 38.9 |
| 64k | 15.1 | 24.1 | 38.6 |
| 128k | 14.9 | 23.9 | needs >524k KV |

Prefill: 906 tok/s @8k (TTFT 9.1 s) declining gently to 827 tok/s @128k
(TTFT 155.8 s). Reference x86 (4x RTX PRO 6000, TP4/DCP1, v17 page):
49.9 tok/s decode cc1, 4,469 tok/s prefill @8k — the Spark cluster
delivers ~30% of decode and ~20% of prefill on ~15% of the memory
bandwidth, with near-flat context scaling (-5% decode from 0 to 128k).
Results JSON: `llm-inference-bench/benchmark_results-glm52-v17-spark-
tp4-dcp1-gpu086-len256k_20260714_225402.json` (plus the gpu085 baseline
and 127k-prefill probe files from the same evening).

### MTP3 speculative decoding (2026-07-15)

First MTP run on the hybrid checkpoint anywhere — x86 v17 validates MTP
off, and x86 v14's MTP3 campaign used the standard NVFP4 checkpoint.
Same profile as above plus `MTP=3` (launcher passes the helper-identical
`--speculative-config` with `method=mtp`, `moe_backend=b12x`,
probabilistic draft sampling). vLLM resolves the layer-78 head as
`DeepSeekMTPModel` and runs the single MTP layer three times per step
(it warns late draft positions accept less — confirmed below). Memory:
draft weights +1.56 GiB/rank (84.12 GiB total), KV cache 14.16 GiB =
404,032 tokens (down from 472,064). Boot ~9 min warm.

Decode (aggregate tok/s; concurrency widened to 1,2,3,4,8 to probe the
404k KV budget; MTP0 baseline in parens):

| ctx \ cc | 1 | 2 | 3 | 4 | 8 |
|---|---:|---:|---:|---:|---:|
| 0 | 24.8 (15.7) | 35.2 (25.3) | 47.3 | 52.2 (40.7) | 75.2 |
| 16k | 21.5 (15.3) | 33.1 (24.2) | 48.0 | 51.1 (38.9) | 73.2 |
| 32k | 22.6 (15.2) | 34.4 (24.2) | 44.8 | 52.3 (38.9) | 73.6 |
| 64k | 22.4 (15.1) | 33.3 (24.1) | 43.4 | 49.9 (38.6) | ∅ |
| 128k | 22.2 (14.9) | 31.6 (23.9) | ∅ | ∅ | ∅ |

+41–58% at cc1, +32–42% at cc2, +28–34% at cc4 — the same band as x86
v14's +45–55% — and peak cluster throughput nearly doubles, 40.7 →
75.2 tok/s at cc8 (9.1–9.4 tok/s per stream). ∅ = cell does not fit
the 404k-token KV budget (cc8 needs ~516k at 64k; at 128k even cc3 —
3×131k prompts, 97% occupancy — never drains once generation slots are
added). Prefill pays ~2.4% for the draft pass: 885 tok/s @8k → 807
@128k (was 906 → 827).

Acceptance (cumulative over probe + sweep, 11,436 drafts): 64.6% of
draft tokens accepted; per-position 85.0% / 63.3% / 45.4%; mean
acceptance length 2.94 of max 4 tokens per step. Coding prompts run
hotter (interval peaks 84% acceptance; a cc1 code probe did 26.1 tok/s
wall-clock including prefill vs the 15.7 MTP0 decode baseline); prose
sits near 58%. Results JSON: `llm-inference-bench/benchmark_results-
glm52-v17-spark-tp4-dcp1-mtp3-gpu086-len256k_20260715_093449.json`.

Untested so far: DCP>1 (impossible cross-node), KV bump via
`--kv-cache-memory-bytes` (~22.5 GiB available per vLLM's report under
MTP3), raising `MAX_BATCHED_TOKENS` beyond 2048 (with MTP3 vLLM warns
`max_num_scheduled_tokens=2048` may be suboptimal for draft slots).
- `v17/run-glm52-v17-hybrid-tp4-node.sh` — four-node launcher. Replicates
  the vllm serve command (v10-launcher style) because the GLM helpers have
  no `EXTRA_VLLM_ARGS` hatch and hard-export single-node x86 env
  (`CUTE_DSL_ARCH=sm_120a`, `NCCL_IB_DISABLE=1`,
  `VLLM_ENABLE_PCIE_ALLREDUCE=1`). Node rank derives from the hostname
  (sparky=0/head, buddy=1, rocky=2, lucky=3); start workers first, head
  last; `MASTER_ADDR` (sparky's fabric IP) is required. DCP is pinned to 1
  (B12X DCP pool is CUDA-IPC, single-node only); MTP defaults to 0 (the
  x86-v17-validated mode; `MTP=3` validated on the cluster 2026-07-15 —
  see the MTP3 section above). Checkpoint revision `68babde` must be
  staged on all four nodes. First-bring-up profile: gpu_mem 0.85,
  max_model_len 131072, seqs 8, graph 32.

v10 image (built on rusty, copied to toby):

```text
localhost/voipmonitor/vllm:fathomless-firmament-dspark-spark-sm121-vf5f4af3-b12x90172a5-cu132-20260709
vllm 0.11.2.dev279+fathomless.firmament.f5f4af3.b12x90172a5.sm121.cu132.20260709
b12x 0.30.0, flashinfer 0.6.14+cu132, deep_gemm 2.5.0, nvtx 0.2.15,
pynvvideocodec 2.1.0
```

v10 build notes: whole Fathomless compile stack passed at `12.1a` first try;
the only SM121-side fix needed was infra support for
`VLLM_RUNTIME_EXTRA_PACKAGES` (fathomless vLLM requires nvtx +
PyNvVideoCodec; both have aarch64 wheels). GPU smoke on GB10 passed.
Two-node serving with the v10 image is not yet validated (DSpark mode is
also unvalidated upstream on Fathomless — v10 launcher has the note).

Each `vN/` has `build-ds4dspark-vN-spark-sm121-cu132.sh` (runs inside
`../blackwell-llm-docker`) and `run-ds4-dspark-tp2-node.sh` (two-node TP2
launcher). v10 pins per `models/ds4dspark-v10.md`: vLLM `f5f4af3`
(`codex/ff-v15-mxfp4-online-mxfp8-20260709`), B12X `voipmonitor/b12x @
90172a5`, FlashInfer `5a73a36`, DeepGEMM `a6b593d`, InstantTensor `85e7c5f`,
humming-kernels 0.1.10. v10 drops the vLLM warmup patch (B12X `90172a5`
ships `fused_indexer_decode_warmup_rows`); the Spark-only DeepGEMM SM121
MQA-logits patch still applies at `a6b593d` and stays default (disable with
`DEEPGEMM_PATCH_FILE=""`). v10 reuses the v9 arm64 base images
(`BUILD_BASE_IMAGE=0`). Note: the x86 v10 page validates the standard
checkpoint only — DSpark on Fathomless is unvalidated there too.

The v9-specific investigation below is kept as-is for the record.

## Build Outcome (2026-07-08)

**Built and GPU-smoke-tested on rusty:**

```text
localhost/voipmonitor/vllm:eldritch-enlightenment-dspark-spark-sm121-v45c1582-b12xf3686b5-cu132-20260708
image id 31821f9b7b3e, 24 GB
vllm 0.11.2.dev279+eldritch.enlightenment.v45c1582.b12xf3686b5.fi25dd814.sm121.cu132.20260708
```

Smoke results (rootless podman, `--device nvidia.com/gpu=all`): GB10 reported
as capability `(12, 1)`, CUDA matmul executes (CUDA 13.2 userspace on the
580.159.03 driver works), runtime `ncclGetVersion=23004` with only
`/opt/libnccl.so.2.30.4` mapped, and pinned versions verified (`b12x 0.23.0`,
`flashinfer-python 0.6.13+cu132`, `deep_gemm 2.5.0+local`).
Note: `torch.cuda.nccl.version()` prints `(2, 29, 7)` — that is torch's
compile-time header version, not the loaded library (same on x86).

Build log: `rusty:~/blackwell-llm-docker/build-spark-sm121-20260708-r7.log`
(runs r1–r6 are the failed iterations below).

Issues found and fixed on the way (all committed on `spark/sm121-arm64`):

1. podman has no `--progress` flag — made docker-only.
2. `pip check` failed: NVIDIA aarch64 wheels (cusparselt) record the bogus
   platform tag `manylinux2014_sbsa`; pip ≥ 25 rejects it. Added a
   normalization step (tag + RECORD hash), no-op on x86.
3. `[[: not found` in later stages: podman's default OCI image format drops
   `SHELL` at stage boundaries. Re-declared SHELL in derived stages and pass
   `--format docker` for podman.
4. Pin drift: `nv_dev` (DeepGEMM), `master` (B12X), and
   `dev/eldritch-enlightenment` (vLLM) have all moved past the v9 pins; the
   commit-verify guards caught each. Helper now fetches pinned commits
   directly.
5. Final stage NCCL surgery assumed `/opt/libnccl.so.2.30.4` pre-exists (true
   only on the legacy x86 base images) — materialize it before the checks.
6. torch's CUDA-deps preloader globs `nvidia/nccl/lib/libnccl.so.*` in
   arbitrary order and could dlopen the 2.29.7 backup instead of the patched
   2.30.4 symlink (latent on x86 too). Backup now parked at
   `/opt/libnccl-torch-bundled-2.29.7.orig`.

**Verdict: feasible with no source-code changes to vLLM or B12X.** The whole
stack at the exact v9 pins already has explicit SM121/GB10 support. The real
work is (a) an aarch64-native rebuild — DGX Spark is ARM, the x86_64 v9 image
cannot run there — and (b) deployment differences: one 121 GB unified-memory
GPU per node, so the 149–156 GB checkpoints need TP2 across both Sparks.

## Per-component findings (at the exact v9 pins)

### vLLM `dev/eldritch-enlightenment` @ `45c1582e9`

Already systematically SM121-clean. `CUDA_SUPPORTED_ARCHS` includes 12.1;
`cuda_archs_loose_intersection` cross-matches `12.0f`↔`12.1a`; every SM12x
kernel group lists `12.0a;12.1a` (zero occurrences of `12.0a` without
`12.1a`). Runtime gating is family-inclusive throughout (`major == 12` /
`is_device_capability_family(120)`); nothing uses `== (12, 0)`. Comments
explicitly name "SM121 DGX Spark GB10". aarch64 source build is clean (the
x86-pinned `requirements/test/cuda.txt` affects tests only).

Notes:
- The `vllm-b12x-indexer-warmup-fallback-20260704.patch` was **never merged**
  upstream (13 commits after 45c1582 on the branch; none touch the warmup
  file). It is still required.
- FlashInfer-b12x NVFP4 kernels are deliberately excluded from auto-selection
  pending an upstream CUTLASS SM121 MMA guard fix
  (`vllm/model_executor/kernels/linear/__init__.py:419`,
  `fused_moe/oracle/nvfp4.py:180`); explicit `--linear-backend
  flashinfer_b12x` opts in.
- `triton_decode_attention.py:544` tuning table is keyed on `(12, 0)`; GB10
  falls back to default tuning (perf, not correctness).

### B12X `master` @ `f3686b5`

README literally says "SM120/SM121 CuTe DSL kernel library"; master contains
GB10 commits ("SM121 fixes", "DGX Spark fixes", Spark-specific MoE decode
tuning). Pure-Python package — kernels JIT per device via nvidia-cutlass-dsl,
so GB10 gets `sm_121a` automatically. No capability gate excludes (12,1).
aarch64-clean. Operational cautions:
- Never set `CUTE_DSL_ARCH=sm_120a` on GB10 (some bench recipes do);
  leave unset or use `sm_121a`.
- `nvidia-cutlass-dsl>=4.5.2` required for native sm_121a (pinned in image).
- `B12X_MLA_SM120_UNIFIED` **does not exist** in either the vLLM or B12X tree
  at these pins (it lives on B12X `dev/sparse_mla_unified`/`dev/w4a8`
  branches). The v9 doc's env line is a no-op at these pins; carry it or drop
  it, nothing changes.
- The PCIe one-shot allreduce is single-node-only and lazy-loaded; on Spark
  TP2 (cross-node) set `VLLM_ENABLE_PCIE_ALLREDUCE=0` and let NCCL/RoCE do it.

### FlashInfer @ `25dd814` (0.6.13)

SM121 is first-class: `sm121a_nvcc_flags`, `is_sm121a_supported`, sparse MLA
docstrings "Requires SM120a / SM121a", DSv4 sparse routing sends major==12 to
the source-compiled path (not trtllm-gen cubins, which are SM10x-only — so
`FLASHINFER_BUILD_CUBIN=0` stays correct). aarch64 wheels are a supported CI
target. **Critical:** FlashInfer builds a separate cubin per SM12x variant and
refuses to run SM120 code on SM121 (`compilation_context.py:39-41`), and its
sm121 AOT modules are only enabled when `12.1a` is in the arch list — so the
Spark build must use `FLASHINFER_CUDA_ARCH_LIST=12.1a`, not the x86 build's
`12.0f`. CUDA >= 12.9 required for sm_121a (we're on 13.2). Pin bump to
0.6.14 optional, not needed for bring-up.

### DeepGEMM `nv_dev` @ `2073ddb`

Contains an explicit SM121 commit (`6b8a2c3`): a (12,1) device with a >= 12.9
compiler reuses the SM120 **family** cubin (`compute_120f/sm_120f`), which
loads natively on GB10 — the FP8 linear path needs nothing. One latent bug:
the fp8 (paged) MQA-logits code generators (used by the DeepSeek sparse
indexer) can emit `#include <deep_gemm/impls/sm121_...cuh>` (nonexistent) if
they run before the JIT compiler singleton initializes on a (12,1) device.
One-line fix prepared as
`patches/deepgemm-sm121-mqa-logits-arch-number-20260708.patch`.

## Environment findings

| | rusty | toby |
|---|---|---|
| GPU | NVIDIA GB10, compute cap **12.1**, driver 580.159.03 | same |
| Arch / OS | aarch64, Ubuntu 24.04.4, CUDA 13.0 host toolkit | same |
| CPU / RAM | 20 cores, 121 GB unified | same |
| Disk free | 2.7 TB | 3.0 TB |
| Podman | 6.0.0 rootless, overlay, CDI `/etc/cdi/nvidia.yaml`, nvidia-ctk 1.19.1 | 6.0.0 |
| Interconnect | 2× 200 GbE CX7 RoCE ports up (`rocep1s0f1`, `roceP2p1s0f1`) | linked |
| Models | Both DS4 checkpoints already in `~/.cache/huggingface/hub` (149 G + 156 G) | — |

Other checks that passed:
- `nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04` is multi-arch (arm64 present).
- All pinned NVIDIA apt packages exist at identical versions in the arm64
  (sbsa) repo: `cuda-compat-13-2=595.71.05-1ubuntu1`,
  `libcublas13-cuda-13=13.4.1.2-1`, `libcudnn9*-cuda-13=9.22.0.52-1`.
- All pinned Python deps have aarch64 wheels: `torch==2.12.0+cu132`,
  `torchvision==0.27.0+cu132` (manylinux_2_28_aarch64 on the cu132 index),
  `nvidia-cutlass-dsl(-libs)==4.5.2`, `cuda-tile>=1.4.0`,
  `nvidia-cuda-tileiras==13.2.78`, `humming-kernels` (pure-python).
- CUDA 13.2 userspace on the 580 (CUDA 13.0) driver rides CUDA 13 minor-version
  compatibility, and the image installs `cuda-compat-13-2` besides. Expected to
  work; confirm with the first smoke run.

## What was prepared (branch `spark/sm121-arm64` in `spark/blackwell-llm-docker/`)

Clone of `local-inference-lab/blackwell-llm-docker` (main @ d25e349) with one
commit on top:

1. `Dockerfile.vllm-b12x-cu132` made arch-neutral: multiarch lib dir
   (`$(uname -m)-linux-gnu`) and CUDA targets dir (`x86_64-linux` vs
   `sbsa-linux`) computed at build time; CUDA arch lists turned into build
   args (`TORCH_CUDA_ARCH_LIST_ARG`, `CMAKE_CUDA_ARCHITECTURES_ARG`,
   `FLASHINFER_CUDA_ARCH_LIST_ARG`, defaults unchanged for x86);
   `DEEPGEMM_PATCH_FILE` hook added.
2. `build-vllm-b12x-cu132.sh`: `CONTAINER_ENGINE` switch (docker/podman) and
   the new build args threaded through.
3. `patches/vllm-b12x-indexer-warmup-fallback-20260704.patch` — **recovered
   from the pushed v9 image** (`voipmonitor/vllm@sha256:7703639a...`, which
   ships `/opt/vllm` with the patch applied). It applies cleanly to 45c1582
   and reproduces the shipped file byte-for-byte. Its own sha256 is
   `8d8c794e4a6d...` (differs from the documented `c1441b5...` because the
   file was regenerated, not copied; get the original from
   `/root/vllm/blackwell-llm-docker/patches/` on the 16-GPU host if exact
   provenance matters).
4. `patches/deepgemm-sm121-mqa-logits-arch-number-20260708.patch` — the
   one-line DeepGEMM fix above.
5. `build-ds4dspark-v9-spark-sm121-cu132.sh` — the Spark build helper: exact
   v9 source pins, SM121 arch targets (12.1a/121a/12.1a), podman, both
   patches, `BUILD_BASE_IMAGE=1` (arm64 bases must be built fresh),
   `MAX_JOBS=12` for the 20-core/121 GB node, refuses to run on non-aarch64.

Component audits left full clones for reference under `spark/src/`
(`vllm`, `b12x`, `flashinfer`, `DeepGEMM`).

## How to build (on rusty or toby)

```bash
rsync -a --exclude .git spark/blackwell-llm-docker/ rusty:~/blackwell-llm-docker/
ssh rusty
cd ~/blackwell-llm-docker
./build-ds4dspark-v9-spark-sm121-cu132.sh
```

Expect a long first build (NCCL + FlashInfer AOT jit-cache + vLLM CUDA build
on 20 cores); ccache and the split base images make rebuilds cheap. All build
stages are compile-only, so rootless podman needs no GPU access. Quick smoke
after the build:

```bash
podman run --rm --device nvidia.com/gpu=all \
  voipmonitor/vllm:eldritch-enlightenment-dspark-spark-sm121-v45c1582-b12xf3686b5-cu132-20260708 \
  python -c "import torch, flashinfer, deep_gemm, b12x, vllm; \
             print(torch.cuda.get_device_capability(), vllm.__version__)"
```

## Deployment differences vs the v9 doc (phase 2)

- **The model does not fit one node**: 149–156 GB FP8 weights vs 121 GB
  unified memory. Minimum is TP2 across rusty+toby (242 GB total) over the
  CX7 200 GbE RoCE link — vLLM multi-node (Ray) with NCCL on
  `rocep1s0f1`/`roceP2p1s0f1`. Weights ~75–78 GB per node leave roughly
  30–40 GB for KV cache and the OS; `max_model_len=262144` with
  `gpu_memory_utilization=0.93` will not carry over — expect to reduce
  context or utilization on first bring-up. TP4 rows are out of reach.
- `run-ds4-v9-server.sh` needs a podman/CDI variant: `podman run --device
  nvidia.com/gpu=all` instead of `docker run --gpus all --runtime nvidia`,
  no `ENABLE_TOPO_PIN` core maps (single-socket 20-core), plus the Ray
  multi-node bootstrap.
- Env deltas from the v9 common B12X env: set `VLLM_ENABLE_PCIE_ALLREDUCE=0`
  (cross-node TP2; the b12x PCIe one-shot allreduce is single-node), leave
  `CUTE_DSL_ARCH` unset, and `B12X_MLA_SM120_UNIFIED=1` is a no-op at these
  pins.
- GB10 is an integrated-GPU UMA platform; vLLM's memory profiler knows
  (`is_integrated_gpu`), but `VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1` +
  utilization values need re-derivation on this topology.
- Benchmark comparability: single-user decode (cc1) is the realistic target;
  the cc64/prefill numbers from the 16-GPU host are a different class of
  machine.

## Two-Node TP2 Deployment (2026-07-08) — WORKING

DeepSeek-V4-Flash-DSpark serves across rusty+toby with **no Ray**: this vLLM
fork natively supports `--nnodes/--node-rank/--master-addr/--headless`
(torch-distributed), the same mechanism as the Aiden GB10 stack
(`~/git/spark-vllm-docker/DeepSeek-v4-DSpark-Aidendle94-GB10-ServingStack`),
whose RoCE/multi-node handling this deployment adapts.

Launcher: `spark/v9/run-ds4-dspark-tp2-node.sh` (deployed to
`~/ds4-tp2/` on both nodes). Start worker first, then head:

```bash
ssh toby  'ROLE=worker ds4-tp2/run-ds4-dspark-tp2-node.sh'
sleep 15
ssh rusty 'ROLE=head   ds4-tp2/run-ds4-dspark-tp2-node.sh'
# endpoint: rusty:8000 (default), served model DeepSeek-V4-Flash-DSpark
```

Defaults: port 8000, `max_num_seqs=8`, `max_model_len=524288` (KV cache
1,560,680 fp8 tokens at gpu_mem 0.85). NOTE: the cluster instance started
2026-07-08 predates the port default change and serves on 8100 until its
next restart.

Config: TP2, dspark/5 speculative, `FLASHINFER_MLA_SPARSE_DSV4` attention,
`--moe-backend b12x` with `VLLM_USE_B12X_MOE=0` (Aiden tuning), kv fp8,
block 256, `max_model_len=262144`, `gpu_mem=0.85`, NCCL over both 200G RoCE
rails (`NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1`, GID auto-detected),
`VLLM_ENABLE_PCIE_ALLREDUCE=0`, `CUTE_DSL_ARCH=sm_121a`.

Validated: `/health` 200, coherent chat completions, server-side
time-per-output-token 10–25 ms (≈40–100 tok/s single-stream) on first-boot
smoke. Model snapshot on both nodes is `62af8ff...` (newer than the v9 doc
pin; the launcher resolves the local snapshot dynamically).

Rootless-podman notes: `--shm-size` invalid with `--ipc host` (host /dev/shm
used); ulimit nofile capped at the host hard limit 500000;
`--device /dev/infiniband` for RoCE.

## v10 (Fathomless Firmament) — build + validation (2026-07-10)

Image `localhost/voipmonitor/vllm:fathomless-firmament-dspark-spark-sm121-vf5f4af3-b12x90172a5-cu132-20260709`
built via `spark/v10/build-ds4dspark-v10-spark-sm121-cu132.sh` (v15-era doc
pins; reused v9 arm64 base images; new `VLLM_RUNTIME_EXTRA_PACKAGES` support
for nvtx/PyNvVideoCodec). Doc-parity launcher
`spark/v10/run-ds4-v10-tp2-node.sh` (9 documented deviations, port 8000,
`seqs=4`, `gpu_mem=0.85` after measuring 0.90 → 0 host headroom on UMA).

Validated cells (standard checkpoint, `b12x-a16`, cc = client concurrency,
sustained decode tok/s at ctx 0 → 128k, `llm-inference-bench`):

| Cell | cc1 | cc2 | cc4 | KV tokens | notes |
|---|---:|---:|---:|---:|---|
| `standard-mtp0` | 29.1→28.5 | 49.8→48.7 | 74.6→73.2 | 901,767 | flat vs ctx; ITL ~34.5 ms |
| `standard-mtp2` | 37.0→39.9 | 57.2→56.8 | 89.7→90.3 | 799,512 | accept 0.52–0.79; ITL ~26 ms cc1 |

Prefill ~2.3k→1.9k tok/s (0→128k). mtp2 dominates mtp0 at every
concurrency for ~3 GiB KV; ~6.2× slower than the x86 TP4 reference,
consistent with the memory-bandwidth ratio. Graph-cap learning: full-graph
capture self-bounds at `seqs×(1+draft)`; padding only widens the piecewise
ladder.

## v16 unified image — build + validation (2026-07-14)

Upstream rewrote `models/ds4dspark-v10.md` in place as the **v16 unified
GLM+DS4 release**: all pins moved (vLLM `8f86f42`, B12X `fe06f49`,
FlashInfer fork `801d57a` with the SM120 DSV4 `topk=256` fixes, source-built
canonical NCCL `dfab7c1`), the launch contract moved into the image
(`/usr/local/bin/serve-ds4-flash.sh`, env-only), InstantTensor became the
default loader (`BUFFERED`), and DSpark returned as the validated default
mode (fixed K=5).

Port: upstream `d104659` merged into branch `spark/sm121-arm64-v16`
(two conflicts: kept podman `PROGRESS_ARGS`; kept `pip check` after
FIXSBSA — the extras install can pull sbsa-tagged NVIDIA wheels).
Base images reused (nccl-canonical pin unchanged since our base build).
Build wrapper `spark/v16/build-ds4dspark-v16-spark-sm121-cu132.sh`; image
`localhost/voipmonitor/vllm:fathomless-firmament-v16-spark-sm121-vllm8f86f42-b12xfe06f49-fi801d57a-cu132-20260714`
(24.6 GB). One wrapper-only rerun: the canonical recipe's post-build DRY_RUN
capture reads stdout, but the helper prints to stderr — capture `2>&1`
(likely latent upstream bug too).

Launcher `spark/v16/run-ds4-v16-tp2-node.sh` is now a thin env wrapper
around the in-image helper; deviations are down to 7, all via supported
knobs: podman/CDI+IB, two-node flags through `EXTRA_VLLM_ARGS`
(`--nnodes 2 --node-rank N --master-addr 10.11.1.1`, worker `--headless`),
RoCE NCCL block (helper defaults are single-host), `CUTE_DSL_ARCH=sm_121a`,
`ALLREDUCE_MODE=nccl` (PYNCCL-only dispatch confirmed), `gpu_mem=0.85`,
`seqs=4` (graph cap left on the helper's own derivation).

Cold boots dropped to ~5 min: InstantTensor BUFFERED loads the 149–156 GB
checkpoints at ~1,100 tensors/s on aarch64.

Validated cells (sustained decode tok/s, ctx 0 → 128k band):

| Cell | cc1 | cc2 | cc4 | KV tokens | notes |
|---|---:|---:|---:|---:|---|
| `b12x-a16 mtp0` | 28.3–28.9 | 48.6–49.6 | 73.2–74.8 | 808,254 | v10 parity (x86 canary predicted -0.1%) |
| `b12x-a16 mtp2` | 37.8–44.2 | 60.3–66.9 | 91.8–97.9 | 722,207 | +2–8% vs v10; accept 0.46–0.94 |
| `b12x-a16 dspark` | 38.1–56.5 | 57.9–93.2 | 83.4–104.8 | 553,311 | K=5; floor and peak both above v9 dspark |
| `lucifer-cutlass dspark` | 44.2–71.1 | 60.8–72.1 | 82.9–107.0 | 513,991 | **FlashInfer DSV4 + CUTLASS MoE work on SM121**; best cc4 cell; x86's 64k+ prefill collapse does NOT reproduce (2062 tok/s at 128k, best measured) |

Result files: `~/git/llm-inference-bench/benchmark_results-ds4dspark-v16-*.json`.
5-minute cooldown observed between sweep cells (thermal settling).

## Open items

1. `mtp3` cell (validated upstream in v16) — boot + sweep when wanted.
2. `lucifer-default` backend row; deeper contexts (Aiden runs 524k–1M at
   0.85 with kv fp8).
3. Optional: push `spark/sm121-arm64-v16` upstream — the FIXSBSA, SHELL,
   NCCL-materialization, and DRY_RUN-capture fixes affect fresh x86
   rebuilds too.
4. Upstream v17 exists already (`blackwell-llm-docker` main moved past
   `d104659`); revisit when a v17 DS4 doc lands.
