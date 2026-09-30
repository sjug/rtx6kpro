# Karmic-main QSA capture fix candidate

Prepared September 23, 2026. **Not built, deployed, or model-qualified.**
R32 stays serving on dusty/kirby. Build requires a separately approved pair
window; no other cluster may be interrupted. No MTP0 or new R32 sweep is needed.

## Frozen composition

Derive from image `88867036403cb3c9bc3026eb8c907ad64bc1af9c28d316d22b670dde4f561152`,
not from a mutable tag or the published amd64 image. Refresh to the sources of
`karmic-kraken-20260923-f37d447a16799e72`, then apply the four runtime Python
files and two test files from [vLLM PR865](https://github.com/local-inference-lab/vllm/pull/865)
at `26b42cf9eb715b118e55f91f53fa0ab412a79c09`, relative to PR base
`e77be22511ab91ecf217760524b7579c366cca2a`. The release-note JSON is deliberately
not installed. The publication pins vLLM `e77be22511`, B12X `10a553ef98`, and
LMCache `413ac98733`. Its recipe is `459593df3d`; this derivative retains our
Spark launchers and native foundation rather than claiming recipe equivalence.

The input patch SHA256 is
`4a70945def2dda5540d04c4a18cf138f3908ae5329745b62753a5c106061f113`.
Tracked Spark tree changes from `31cb715f3511073f893900248ba8e737646ca50d`
to `509f53e2e40d7391a7fcb2e6a611ba809af137e5`, with vLLM package tree
`20c53d88f184e575ccb20c6c707f74deb6ab8a86` and B12X tracked tree
`5bab5ba61f2b511111f1944d3d21642cb6582bcf`.
vLLM native inputs and the RoCEnante proxy are unchanged. B12X loader C sources
change and are compile/link-gated separately; that loader remains inactive in
our InstantTensor serving profile. LMCache's only package change is its MQ Python
fix; LMCache remains disabled. Distribution
version metadata still identifies the base; explicit tree/backport labels are
the authority. The cache fingerprint changes with the patched tree.

`prepare.py` verifies the frozen parent tar and applies the patch without fuzzy
or three-way matching in temporary scratch, not in the vLLM checkout. It writes
the derived lock and runner. The parent lock, runners and source payload remain
unchanged. `runtime.lock.json` retains parent archive metadata as provenance;
installation uses the digest-pinned changed-file `refresh.tar`, not that original
archive, for the final trees. The inherited tracked-file manifest is refreshed
without replacing its untracked native/build-product inventory.

## Checks and build

```
python prepare.py
python -m unittest -v test_backport.py
shellcheck build.sh run-qwen.sh
DRY_RUN=1 bash build.sh
```

On an authorized idle dusty/kirby pair: `bash build.sh`. Stage the complete parent
kit, including this directory, because the unchanged parent native/model kernel
gates are reused. The base image must already be present. Image assembly has no
network access or changed compile parallelism. The updated loader is compiled
in a fresh disposable cache inside the GPU-attached gate container, where the
driver libraries are available. It is not compiled during image assembly.
Source manifests and native
bytes are checked before and after installation. The parent GPU gates run first,
then PR865's two regression tests run in separate processes with exact counts of
one and two passed cases and no skips. Additional exact-count gates cover
RecoverSSM boundaries (10), block-FP8 capacity/replay (2), DS4 WO preparation/replay
(2), and the LMCache malformed-header regression (1). Each runs in a separate
process. These tests need the image environment;
local artifact tests are not substitutes for them.

September 23 local review: Claude independently replayed the composition and
verified both source manifests, native-input equality, refresh archive contents,
LMCache blobs and selected test counts. The identified loader-gate placement risk
was corrected and re-reviewed with no remaining build blocker. Seven local tests,
shellcheck, preflight (including Python optimization) and dry run pass. These are
artifact checks only; no GPU build gates or serving qualification have run yet.

Build uses a retention tag, saves inspect before gates, and publishes the normal
tag `localhost/voipmonitor/vllm:karmic-main-qsa865-spark-sm121` only after gates.
That tag is not production promotion. Failed images are retained, never pruned.

## Qualification, not yet executed

Runner defaults: HC-off, MTP3, TP2, four sequences, aligned checkpoints, native
262144 context, utilization 0.85, same model revision, loader and transport.
`EXPECTED_IMAGE_ID` must be set to the new BUILD-OK image on both nodes. The
runner verifies the new source tree and inherited launcher digest. Start worker
then head; stop gracefully worker-first and retain old containers.

The runner explicitly preserves the original capture maximum of 32, including
the affected 24-row and 32-row graphs. There is no 16-row workaround in the
patched serving profile: qualification must prove the source fix works without
avoiding the affected graphs.

This is a wider source refresh, not a one-variable PR865 experiment. A pass
qualifies the combined baseline and backport; it cannot independently attribute
the recovery to PR865. The inherited non-editable B12X site-packages copy remains
as ancestry; fail-closed import checks require the refreshed source tree first.

1. Build gates, then a real cold completion and existing semantic/retrieval gates.
2. With capture 32, upstream's long greedy counting workload plus concurrent short
   requests. Verify exact continuation text, not merely health or absence of Xid.
3. Replay the private proxy request after checking its reconstruction, and exercise
   long-context tails around 16/17/18/24/32/33 rows and a warm repeated conversation.
   Async scheduler metadata alone does not prove a prefix hit or faulting step.
4. Confirm the final HC-off profile and run the unchanged
   standard benchmark harness. Compare saved R32 and HC-off receipts; do not edit
   the benchmark repository. Record cache state and per-rank memory/KV.

Both captured proxy JSON files contain private conversation/identity information.
They are not build inputs and must not be committed or posted with a bug report.
No GPU replay, daemon inspection container, or production load is part of local
preparation. The upstream diagnosis is a strong match, not a proven local fix yet.
