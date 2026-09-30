# Previous Karmic HC-off baseline plus PR865 only

Prepared September 23, 2026. Build and all GPU gates passed on dusty after
independent review. Image `a25bedd43581dffecca2f9eefb40e60b170af6ae19124fef1c292850b2a0483e`
is the user-selected Qwen serving image on dusty/kirby as of September 23, 2026.
Distribution completed at 21:28:53 UTC with identical image IDs on both nodes.
Public correctness, the two authorized private crash replays, and two standard
full benchmark grids passed. The user approved keeping this exact HC-off plus
PR865 profile serving. See EXECUTION.md for measured results and limits.
This decision applies only to Qwen; other clusters are unchanged.
The wider candidate was stopped worker-first, exit 0 without
OOM kill on either node, and its containers are retained. GLM and DS4 are outside
this window and were not touched.

Build receipt: `../build-receipts/qsa865-hcbase-20260923T211351Z-1355127/`.
Build completed at 21:22:51 UTC. Source/native identity, FlashKDA 12/12, all
inherited regressions and memory gates, draft-head gate, and PR865's exact 1+2
cases passed. Normal tag publication occurred only after the gates completed.

The user approved preparing this narrow derivative after the wider September 23
refresh lost performance versus the previous HC-off image. Claude must review
the artifacts before any build or serving interruption. That review is complete:
Claude independently reproduced both trees and all six patched blobs, checked
the earlier-base prerequisites, and cleared build and serving scripts. Nine local
tests and shellcheck pass. The inherited removal branch was deleted from the
runner generator, regenerated and tested before staging and building.

## Composition

Base image: `88867036403cb3c9bc3026eb8c907ad64bc1af9c28d316d22b670dde4f561152`.
Base vLLM: `6afb99982576a7a2eb53d667189e859629e22739`, Spark tree
`31cb715f3511073f893900248ba8e737646ca50d`. Apply the exact six-file PR865 patch
at `26b42cf9eb715b118e55f91f53fa0ab412a79c09` without fuzzy or three-way apply.
The patch applies directly to this earlier base, with no prerequisite cherry-pick.
Four runtime Python files and two tests change. No other source changes are
admitted by the preparer, installer and preflight.

Result tree: `196f6e87a37480bca3b609a960e8969bcb9aa367`.
vLLM package tree: `2ff7182c64300ed2324278e28ced8099db460432`.
B12X stays at `4f3028b19c1d8290dc72b6f483aba40de23eae5a`, tree
`6a30d32d796df362b5b6a1568f7de86a8c1693d2`. LMCache and all native components,
including NCCL, FlashKDA, FlashInfer and the RoCE proxy, are inherited unchanged.
The September 23 broader source refresh is not included.

The source-derived cache fingerprint changes. Therefore tuning choices may be
measured again; the narrow source delta alone does not establish identical tuning
choices or a controlled kernel-performance A/B. Keep existing caches intact and
record the boot's tuning receipts. Never claim the earlier crash-exposed image
is safe merely because its benchmark was faster.

## Gates and serving contract

Run `python prepare.py`, `python -m unittest -v test_backport.py`, shellcheck,
and `DRY_RUN=1 bash build.sh`. Build offline against the pinned local base only
after artifact review and an authorized idle dusty/kirby window. The build reuses
all parent GPU gates, then executes both PR865 test selections in separate
processes, requiring exactly one and two passes with no skipped cases. Native
bytes and source-tree import identity remain fail-closed gates. The normal tag
is published only after build gates pass, not as model promotion.

Runner: `karmic-main-hcbase-qsa865-spark-sm121`, explicit built image ID required;
TP2, HC-off, MTP3, four sequences, aligned, capture32, 262144 context, utilization
0.85, same checkpoint, InstantTensor loader and transport. No capture16 workaround.
Gracefully stop worker before head and retain containers; start worker before
head. R32 remains the known-good rollback.

Qualification must cover cold completion, semantics, retrieval, the long counting
and mixed-load reproducers, and private local replay under the user's existing
scope. Do not claim observed graph dispatch from prompt lengths. Use the standard
unchanged benchmark harness and saved previous HC-off/R32 controls. Keep timing
separate from concurrent probes and record request history and tuning state.
The source patch is a candidate fix until these gates pass on this derivative.
