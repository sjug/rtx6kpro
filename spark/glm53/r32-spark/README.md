# R32 Spark refresh candidate

Status, 2026-09-10: BUILD-OK on dusty at 14:54 UTC, image `74e53e710bef`.
All local and GPU build gates passed. Image distribution, the initial Qwen
correctness battery, the standard grid and MTP0 control passed. The second-boot
repeat and contemporary R29 C4 control completed. Fresh-control C4 output is
-1.18%, engine steps -0.54%, and prefill differs by at most 0.72%. No material
overall regression is demonstrated; long-context acceptance remains variable.
R32 aligned MTP3 was restored and completion-verified at 12:45:15 EDT.
The authorized GLM cutover completed at 14:42:12 EDT with a correct first
completion. Correctness and cache checks passed. The first grid was rejected
because separate Pi traffic overlapped later C2/C4 cells. The authorized quiet
repeat passed all 15 cells at 16:14:16 EDT. R32 GLM is qualified for the frozen
aligned/MTP3 contract and stays serving. Engine geomeans differ from R29 by
+0.91/+0.53/-0.68% at C1/C2/C4, with no material overall regression demonstrated;
Qwen and DS4 are untouched by that continuation. See
`qualification/BUILD-STATUS.md` for execution state and provenance.

## Composition

The input is the published R32 lock at rtx6kpro commit
`59f01d1c3ab25c7a6d39c4e56de75c9df2f63944`, not moving source branches.
`source.lock.json` records complete identities, changed-path manifests and
patch digests. The existing Spark architecture/draft-head overlay is retained.

| Component | R32 candidate | Change from corrected R29 |
| --- | --- | --- |
| vLLM | commit `5576927057cf71b6ec61d120932338b333efa089`, Spark tree `80a18accc688cdee2974f4c5e03e9416b2896087` | 15 Python files, no native changes |
| B12X | tree `c4bfeee9f3c9457400d191c870eb2e44fbcd5c2e` | Byte-identical |
| LMCache | commit `35ad809fdddd430c3970777a2ff3d984bf8d1963` | 19 Python files, native inputs unchanged, remains disabled |
| FlashKDA | `3b225bf26bb8e218928a1fe14751cb48cf31d11b` plus the qualified patch | Binary retained |
| FlashInfer | Spark `1ac6942776b383c6b03c7a5805a22e72a3e3349f` | Retained, no RTX wheel substituted |

The derivation base is the corrected R29 image, not its original launcher-bug
build: `0b15723cb87646bb4628c5cd67aa2a7c879ac5a27105109ea5db06c30113dc0b`.
Its source-lock digest and independent native receipts are required inputs.
Preparation compares both declared binary digests with the R29 native receipt
and the FlashKDA digest with R29's lock before composing any source trees.
The kit carries the unchanged receipt as `inputs/r29-native.txt`; the local
contracts verify its recorded digest and both launcher digests before a build.
In the full checkout they also check the original R29 receipt for later drift.
The normal output tag is `localhost/voipmonitor/vllm:jj-r32-spark-sm121`.

Notable included changes are the GLM short-pool tail correction (#715), Mamba
sparse-block cleanup (#718), queued request-boundary admission (#721), and
MoE warmup buffer reuse (#724). The admission change principally concerns
request-boundary checkpoints, not the retained `aligned` baseline. No claim
of a performance gain or of fixing every `auto`-policy symptom is made here.
Newer PLE disk-offload work and new B12X loader behavior are outside this lock.

## Native reuse, not a native rebuild

The preparer independently reconstructs the corrected R29 base, reapplies the
Spark overlay to R32, and round-trips every generated patch. Every changed path
must be Python; native build inputs and dependency specifications must match.
Source preparation uses private indices and writes `refs/spark/jj-r32/*` in the
three source repositories. It never checks out, resets or rebases their branches.
All preparation files are on persistent disk under `.compose/`.

The image refreshes the tracked source trees that runtime Python imports.
Native objects, Rust frontend, copied CuTe modules, Triton sources and its
site-packages link must survive. Before/after manifests reject native changes,
removal or additions. The GPU gate still executes real custom ops, checks SM121a
cubins, import paths, linkage and the corrected caller-owned-output schema.

LMCache is packaged from a clean tracked-source export with `NO_NATIVE_EXT=1`,
then its wheel receives the exact base image's compiled ARM payload. Its wheel
tag is `cp312-cp312-linux_aarch64`, its RECORD is regenerated, and installation
must preserve all native digests, including the separate cuMem interposer.
No source bundle or network fetch is needed in image assembly. `uv` runs offline
without dependency resolution or build isolation against the pinned environment.
Old generated version files and build outputs are not copied into the wheel.
Before either reinstall, a recorded inventory classifies the old distribution's
RECORD entries (including paths outside `lmcache/`) and all package files as
tracked source, generated files, metadata or preserved natives. Unclassified
payloads fail with the old files still intact. Every inventoried native must also
have a byte-identical replacement in the wheel. Generated `_version.py`, console
scripts and bytecode are not mistaken for missing native artifacts. The inventory
is saved as `/opt/local-inference/r32-lmcache-preinstall-inventory.json`.

A CPU-only read-only inspection of base `0b15723c` on dusty (2026-09-10)
confirmed that `/opt/lmcache/lib/liblmcache_cumem_shareable.so` is not owned by
LMCache's RECORD, including resolved-path checks. R29 installs it separately.
Its SHA-256 is `6c3b32b7892571c0a6a94b7197f87a3bad9735eb897779fc9c9d5981a583499d`;
the inspected RECORD digest is
`3c03e2361afdc31782faa65746a99fb566095c5565585bdd2d9bc667fb72843e`.
No interposer wheel relocation is needed for this exact base. The native
manifest still verifies that its bytes survive the refresh.

## Matched serving contract

Alias update applied 2026-09-11: the Qwen host runner now defaults to the
single served name `Qwen3.8-Flash-Next`. The pinned NVFP4-4p89 repository,
revision and R32 image are unchanged. The runner supplies this value through
`SERVED_MODEL_NAME` to the existing in-image launcher, so no rebuild is needed.
The dusty/kirby pair was recreated worker-first using the same image ID and
settings. At 13:25:41 EDT the restart smoke passed: `/v1/models` advertised
only `Qwen3.8-Flash-Next`, arithmetic returned exact `333` with normal stop,
and requests using the former long name returned HTTP 404. No qualification
battery or benchmark was rerun for this alias-only change. Logs and smoke
receipts are in `qualification/qwen-alias-20260911/`. Historical qualification
receipts and their dated drivers retain the original served name; new client
requests and probes must use the shorter name.

Both profiles retain InstantTensor BUFFERED, aligned checkpoints, MTP3, full
CUDA graphs, the qualified checkpoint revisions, and utilization 0.85.

| Profile | Nodes | Context | Max sequences | Batched tokens |
| --- | --- | ---: | ---: | ---: |
| Qwen3.8-Flash-Next NVFP4-4p89 | dusty/kirby, TP2 | 262,144 | 4 | 4,096 |
| GLM-5.3-Flash NVFP4 | sparky/buddy/rocky/lucky, TP4/DCP1 | 1,048,576 | 8 | 4,096 |

GLM retains BF16 draft head, FlashKDA, RoCEnante, materialized-queue MoE and
disabled L2 prefetch. Both retain their R29 transport contracts and `LL,Simple`.
LMCache stays off. This refresh does not switch `clear_thinking`, sampling,
fairness, speculative decoding method, loader or checkpoint policy.

## Build and qualification sequence

1. Review these artifacts. The authorized build proceeds with explicit dirty
   provenance while the separate signed-records commit remains pending;
   source-lock and recipe digests identify the exact exported files.
   `DRY_RUN=1 bash build.sh` performs local checks only. A real build
   requires idle dusty and kirby, the exact corrected R29 base image and explicit
   approval for the Qwen interruption. No build while either node is serving.
2. Build by image ID, with durable receipts under `build-receipts/`. Assign the
   normal tag only after the complete image gate passes. The inherited R29
   native, FlashKDA, draft-head, recurrent-cache and RoCEnante gates remain.
   Added gates require all 235 sparse-pool cases, seven cleanup cases, seven
   retirement cases, the pooled-indexer tail test and the warmup reuse test.
   Each group runs in a fresh process with exact counts and no skips. The GPU
   groups explicitly set `B12X_GLM53_GPU_TEST=1`. All 251 R32 cases passed on
   GB10 in the 2026-09-10 build; local tests cover failure-handling mechanics.
3. Save and transfer the Docker archive without conversion or compression over
   the switched 200G fabric. Load on kirby and verify identical image IDs.
   Start worker before head. Never clear the Hugging Face cache.
4. Qualify Qwen first: meaningful completion, semantic/tool/vision probes,
   MTP0/MTP3 boundaries and native context, fixed-token acceptance, padded-batch
   transition, repeated-prompt burst, head-of-line and prefix-reuse probes.
   Follow with the normal `~/git/llm-inference-bench/run_bench.sh` full sweep
   against the qualified R29 receipts. Repeat any apparent regression before
   classifying it. Report engine steps, output throughput and acceptance separately.
   Record startup KV tokens per rank and the boot configuration for both models:
   #724 can reduce warmup headroom, but more KV capacity is not guaranteed.
5. After Qwen passes, stage GLM and obtain its cutover window. Preserve its
   R29 profile and request/template parameters, including `clear_thinking=false`
   for the matched comparison. Run semantic/tool, cache/multi-turn and native
   1M checks, then the same standard benchmark grid. Log cached tokens so warm
   prefix hits are not mistaken for faster fresh-token prefill. Explicitly cover
   short GLM prompts with complete/incomplete pool tails and the 2048-token
   boundary. #715 corrects sparse attention for short history with an incomplete
   tail; judge answers on semantic correctness, not exact R29 output equality.
6. Test `auto` checkpoint policy only as a separate arm if the retained aligned
   candidate passes. Before that arm, run all five frozen
   `tests/v1/core/test_boundary_admission*.py` suites. They remain queued rather
   than part of the aligned build gate. Its identical-burst, fresh-arrival and divergent-prefix
   behavior must be measured on both model families before changing defaults.

No DS4/nous changes, automatic production cutover, cache deletion, image pruning,
Podman reset, upstream posting or push is part of artifact authoring.

The GLM distribution and qualification drivers completed under
`qualification/`. The cutover driver refuses to operate
without `GLM_CUTOVER_APPROVED=1`, then verifies the exact running corrected R29
image before stopping workers and head. All four image transfers and identity
checks passed. Its added seven-case short-prompt retrieval check, 15-case
semantic battery and exact retrieval through 1M passed on the live cluster.
The GLM campaign completed after the clean concurrency repeat;
`qualification/GLM-QUALIFICATION.md` records the full grid, comparisons and
first-run exclusions. R32 stays serving, with R29 stopped and preserved.
