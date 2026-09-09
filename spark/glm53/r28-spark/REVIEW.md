# R28 Spark review resolution, 2026-09-08

Status: implementation and local verification only. No build, node stop,
transfer, serving change, commit, push, or upstream filing performed here.

## Findings and disposition

| Finding | Assessment | Resolution |
| --- | --- | --- |
| Failed serving probe interpreted as idle | Confirmed P1 | Capture status and output separately on both nodes. Failed or nonempty probes return 78. Mocked local/remote failures and busy results tested. No automatic stops. |
| Normal tag assigned before gates, blocking retry | Confirmed P1 | Compile to untagged immutable IDs; tag only after the independent gate process passes. Failed gate publication tested. Persist each attempt's IDs, source lock, inputs, log and exit status. |
| Python assertions removable under optimization | Confirmed P1 validation gap | Explicit `__debug__` refusal in every Python entrypoint and inline runner/launcher preflight. All local entrypoints tested under both `-O` and `PYTHONOPTIMIZE=1`. Shell/Git checks were not affected by Python optimization. |
| Circular FlashKDA expected digest | Confirmed P1 | Separate native-builder stage ID and digest receipt before runtime assembly. Pass that digest as a build argument; installer, label and GPU verifier must match. No fake pre-build binary digest in the source lock. |
| Lost refreshed-path restrictions | Confirmed P1 | Shared path contract in preparer and in-image refresher, plus fixed target trees. Exactly four vLLM native paths are admitted for the explicit FlashKDA rebuild. Unexpected CUDA/C++/build changes tested as rejection cases. |
| LMCache local patch ignored by image builder | Correct provenance caveat, not wrong source | The preparer replays all three source deltas. The image applies two refresh patches and builds LMCache from a digest-pinned local bundle at the verified third tree. No LMCache repository fetch during assembly. Dependency/bootstrap downloads remain. |
| LMCache architecture check absent | Confirmed missing proof, not proof of wrong compilation | `TORCH_CUDA_ARCH_LIST=12.1a` is consumed by CUDAExtension. The CMake environment variable alone is not a CMake define. Check `sm_121a` in CUDA ops, plus native imports and clean linkage including the interposer. CPU-only objects do not need a GPU architecture. |
| Overlay digest/target not explicitly anchored | Hardening gap, already partly constrained by R27 reconstruction | Pin the overlay SHA and all candidate trees, retain exact R27 replay. Composition refs preserve the trees without moving working branches or indices. |
| Fingerprint omitted native inputs | Confirmed P2 | Hash full source trees, base image, FlashKDA source/patch and Dockerfile/compiler contract. Unit tests change only full-tree/native-build inputs and require a new fingerprint. |
| Graph test can accept stale eager results | Confirmed P2 | Poison output and final state as well as checkpoint storage before both replay loops. Exact two-site adaptation records input/output test-source hashes. GPU execution pending. |
| Cubin substring accepts generic target | Confirmed P2 | Require exact architecture-specific `sm_121a`, with generic-target rejection tests. |
| FlashKDA subset or skips could pass | Confirmed P2 | Require exactly 12 collected and passed cases, no skip, xfail, failed setup or failed teardown. Negative count cases tested. |
| Draft-head test prose, loose thresholds/storage bound | Confirmed P2/P3 | R28 banner, exact packed-weight/scales/global-scalar byte accounting, RMSE below 0.12 and cosine above 0.99. Prior fixed-seed SM121 receipts are about 0.095 and 0.9955, so dividing the old RMSE threshold by four would reject known-good output. New bounds still need the GPU run. BF16 remains the serving default. |
| GLM host launcher can shadow gated file | Confirmed P2 | Host file must match the image's launcher SHA label. In-container preflight checks the mounted file against image-owned expected SHA. Image gate exercises tampered-launcher rejection; execution pending. |
| Lost no-pull/context/receipt/architecture safeguards | Confirmed | Restore no-pull, build-context exclusions, durable receipts and aarch64 host check. Strict 0/1 dry-run parsing. Dirty builds require explicit opt-in and carry dirty/content provenance. |

## R27 gate comparison

Retained native-product hashes, actual custom-op execution, source-tree-first
imports, Triton symlink resolution, image/lock identities, proxy ABI/load
checks, launcher render checks, LMCache-disabled rejection and hollow-install
preflight rejection. Restored the additional R27 execution battery:

- RoCEnante health and PR 674 scalar restore.
- Partial-prefix cache, Mamba chunk alignment and resume indices.
- Draft-head mode/capability and numerical checks.
- FlashKDA near-collinear finiteness.
- MXFP8 persistent-CTA store completion.

R28 also gates the scheduler admission regression, MLA storage, shared-expert
stream tests and the complete packed/dense FlashKDA checkpoint matrix.

## Scheduler finding changes qualification, not the default

R28 commit `2531689f` explicitly changes waiting-head admission. Its
`test_full_boundary_hit_is_admitted_while_another_request_decodes` verifies
isolated admission followed by both requests decoding together. Therefore,
unchanged isolated-step rules do not prove whole-burst serialization remains.

On each model, test both the identical-prompt burst and the fresh-request
head-of-line reproducer under `auto`, then the matched aligned arm. Keep the
results separate. Do not file the R27 report as an unresolved R28 defect
without these runs. Aligned remains the candidate default until qualified.

## Local evidence and pending execution

- Exact three-component source replay: PASS. Candidate trees unchanged.
- Eight CPU-only contract tests: PASS, including failure injection and optimized Python.
- Qwen and GLM runner render/negative suites: PASS.
- Build dry run: PASS, normal final tag, dirty provenance reported honestly.
- Ruff and shell syntax/checks: checked separately from GPU tests.

FlashKDA/LMCache compilation, architecture inspection of the rebuilt objects,
image negative controls, GPU tests and serving qualification are pending.
The local LMCache bundle is generated input, excluded from Git and pinned by
the source lock. Preserve or regenerate it before transporting the build kit.

Next execution requires releasing dusty/kirby from Qwen serving. The build
script refuses a busy or uninspectable pair. GLM and nous are outside that
build window. R27 remains the qualified serving image until R28 passes its
model-specific qualification.
