# JJ main Spark candidate

Status: cache-agreement derivative `6633e678fee7` built and build-gated on
September 22 UTC; Qwen correctness and the full grid passed, not promoted.
Parent `e7926f763859` passed single-GPU
gates but could not start TP2 because its coordinator lacked B12X's cache
handshake. The explicitly approved Python-only upstream backport `0f60770e85`
preserves autotuning. See `cache-agreement.lock.json`, `QUALIFICATION.md`, and
`gate-evidence/cache-agreement/`. GLM and DS4 remain untouched.

This kit targets the published JJ-main engine sources, not Karmic main and not
a numbered R39. Source selection and completed build identities are separate:
`source-selection.json` pins inputs; `build.lock.json` remains the frozen input
contract with null output IDs. Completed IDs are in `BUILD-STATUS.md` and the
gate receipt's image inspection. Null never means use a tag or an older image.

## Composition

- vLLM `8e1f1e58`, B12X `0f3a8cbf`, recipe `23d674e8`.
- ARM64 NGC CUDA 13.4 / Torch 2.14 foundation. The pinned JJ wheel recipe itself
  normalizes Torch to `2.14.0a0+4fdf77b940.nv26.8.63802676` and FlashInfer to
  `0.6.18`. Source requirements naming Torch 2.13 are not the wheel contract.
- Rebuild vLLM natives and FlashKDA `3b225bf2` with JJ's packed-checkpoint patch.
  Do not copy Karmic vLLM extensions or its SM120 CTA-copy patch.
- Reuse the exact FlashInfer, NCCL, LMCache and InstantTensor component IDs in
  the lock, subject to their runtime/native gates. LMCache remains disabled.
- The runtime dependency input and dependency patches are byte-identical between
  recipe `8c5aa7f8` and `23d674e8`. Their ARM64 lock and dependency base are reused,
  not an inferred dependency closure from the Karmic engine. The inherited
  `/opt/karmic-build/runtime-inputs` path deliberately identifies that base's
  retained upstream patch tools. It is not a Karmic engine import path.
- Regenerated overlay changes only two architecture lists and the draft-head
  capability check. PR756 is already present. The independently computed result
  is tree `182f7f4ac37b980a3bb81e74797a78e9dbf822cb`.

`verify_sources.py` reads existing repositories, reconstructs Git trees in memory,
and writes neither Git objects nor refs. The local build tooling versions match
the upstream build lock; cmake/ninja hashes select ARM64 wheels.

## Build boundary

Local validation and independent review precede an approved Qwen build window.
The build script refuses non-idle dusty or kirby and fails closed on probe errors.
Compile parallelism is 20 jobs with one NVCC thread per job. Podman's `--jobs=1`
serializes image stages, not the native compiler.

Run `DRY_RUN=1 bash build-component.sh vllm` locally. In an approved idle window,
build vLLM, then pass its exact component ID to `build-component.sh runtime ID`.
Components receive non-serving retention tags immediately, and inspect receipts
are written before GPU gates. Never prune or reset storage to retry a build.
Every component build first inventories all reused local image IDs and ARM64
architecture, retaining their original recipe labels in the build receipt. Missing
components fail before compilation. This inventory passed on dusty for this build.

`build-serving.sh RUNTIME_ID` runs native,
FlashKDA, draft-head, regression and live CLI gates after packaging. GPU test
counts must actually collect and pass, without skips. Independent review and
the node-runner identity bindings are required before publishing a serving tag.
The Qwen distribution script requires the exact build receipt and image ID;
the runner independently checks the image and source identities.

## Qualification contract

The current authorized window is Qwen on dusty/kirby, MTP3 only, compared with
the existing R32 return measurements. No fresh R32 baseline is planned. GLM and
DS4 stay serving. Any later GLM window remains a separate cluster qualification.
Keep TP4/DCP1/MTP3, aligned checkpoints, BF16 draft head, existing checkpoint
revision, RoCEnante, LL/Simple and automatic NCCL channels. Preserve the existing
utilization, context and batch envelope. Explicit expandable-segments allocation
is mandatory, not a diagnostic passthrough. Do not add RTX-only policy knobs.

Require native operator execution, linkage, coherent NCCL, proxy ABI/load without
a compiler, FlashKDA emitted architecture and checkpoint GPU tests, and real
launcher preflights (DRY_RUN alone bypasses those checks). Semantic exchanges must
retain full requests and responses before timing. A formatting failure is not
evidence of numerical corruption and does not authorize an automatic rollback.

Model correctness precedes the unchanged external benchmark harness. Do not edit
the benchmark repository. Report engine steps separately from output throughput
and acceptance. Build gates permit a candidate tag, not production promotion.

## Independent review, September 21

Claude reproduced the local checks and found no hard component-build blocker.
The follow-up adds poisoned output/final/checkpoint replay tests, parsing of the
actual launcher command with JJ's parser, the R38 JIT cache directories, and
local shared-component inventories. The new FlashKDA first-launch stack probe
is informational: a growth receipt requires review before model qualification,
not an automatic CTA-copy patch. The JJ FlashKDA source/patch matches R38's;
its CUDA 13.4 native execution passed all twelve checkpoint cases. The first-launch
stack probe grew from 1,024 to 15,984 bytes; review admission implications during
the separate model qualification window.

The six-case MoE warmup selection, inline draft-head capability gate and workspace
fixture were checked against `8e1f1e58`, not assumed from Karmic. Before commit,
copy the small build-gate logs, image inspect and input manifests into tracked
`gate-evidence/`; bulk logs and wheels remain in durable build receipts on dusty.
