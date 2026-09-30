# Karmic main Spark candidate, September 22

## Incident and rollback, September 23

The HC-on return boot faulted during serving with an illegal GPU memory write.
R32 was restored on dusty/kirby and passed a real completion. The saved HC-off
profile below is historical experiment evidence, not current serving state or a
claim of immunity. See [the QSA865 backport candidate](qsa865/README.md), which is
prepared separately and is not yet built or qualified.

## Saved Qwen profile, September 23

The user selected HC-off after the on/off/on comparison. Use
`ROLE=worker bash run-qwen-profile.sh` on kirby, then
`ROLE=head bash run-qwen-profile.sh` on dusty after an authorized graceful
pair stop. The wrapper pins image `88867036403c...`, verifies the diagnostic
runner digest, and explicitly sets `VLLM_QWEN3_8_FLASH_NEXT_HC_TP=0`.
Defaults remain TP2, MTP3, aligned, memory utilization 0.85, native 262144 context,
four sequences and 4096 batched tokens with the existing checkpoint and alias.
Other inherited runner overrides are controls, not newly qualified profiles.

Saving this profile does not activate it: the last verified serving pair is the
HC-on return. No restart was performed to save it. The frozen build lock and
original control runner remain unchanged. Use the diagnostic runner for HC-on
controls, not this saved profile. Retain prior containers with graceful stop and
rename rather than the inherited destructive `ROLE=stop` operation.

HC-off improves long-context prefill and engine steps against HC-on, but the
16K deficit versus R32 and startup allocation warnings remain open. See
[qualification](QUALIFICATION.md) for output/acceptance tradeoffs and receipts.

## Initial candidate record

Status: build gates and Qwen MTP3 correctness/full grid passed. Performance is
mixed; see [QUALIFICATION.md](QUALIFICATION.md). Candidate remains evaluation
serving, not promoted. Small build-gate evidence is in `evidence/build-gates/`.
No model is qualified by build gates alone. User authorized pausing Qwen on dusty/kirby and building
on dusty only after Claude reviews and clears the artifacts. GLM and DS4 Vision
must remain serving throughout this window.

## Composition

This is a non-linear beta-to-main replacement, not a descendant release update.
Neither old source pin is an ancestor of the new pin. Native reuse rests on
direct tree equality. Main omits beta's shared-expert-aware MoE tuning context;
that is a known tuning-fidelity delta for Qwen, not evidence of unchanged
performance. Other carried code is checked by content rather than ancestry.

`source.lock.json` freezes the September 22 Karmic-main publication: vLLM
`6afb99982576a7a2eb53d667189e859629e22739`, B12X
`4f3028b19c1d8290dc72b6f483aba40de23eae5a`. The existing Spark Karmic image
`f30dc6d9a2a6f6fc0ac9ff8cddb04d9f631f4a87b48ca7cac69254802fe83233` is
the native foundation, not the amd64 published image.

The preparer reads existing local repositories without writing Git objects or
refs. It asserts unchanged vLLM native/dependency inputs, applies only the prior
SM121 architecture and draft-head gate overlay, and freezes full tracked-file
manifests and package Git trees. Installation updates those tracked files only,
preserving native objects, generated CuTe Python and the Triton symlink. Native
bytes are checked before/after refresh and again after B12X installation.
Distribution version metadata stays at the base vLLM version; source manifests,
not that version string, identify this refresh.

CUDA 13.4, PyTorch 2.14, FlashInfer, FlashKDA (including both upstream patches),
LMCache, InstantTensor, CUTLASS/Python dependency fixes and NCCL 2.31.2 are reused
from that content-addressed base. B12X's RoCE proxy is the native exception: its
source changed and ABI is now 4. It is compiled at build time into the same
in-image cache path; source-keyed filenames avoid the old object. A compiler-
disabled load proves it is present before serving. LMCache stays disabled.

Qwen and GLM launchers retain the corrected existing launch settings, including
aligned recurrent checkpoints. Only GLM's proxy ABI assertion changes from 3 to
4. No NCCL channel pin, loader change, new model revision or tuning preset is
introduced. The cache fingerprint includes both source archives and base image.

ABI 4 also changes RoCEnante's default traffic class: it now follows NCCL_IB_TC
(106 in our runners) instead of hard-coded 0. Qwen disables custom all-reduce
and uses NCCL, so this does not change its transport. A future GLM window must
qualify the TC/QoS interaction or explicitly preserve RoCEnante TC 0. The GLM
runtime contract is not claimed unchanged merely because launcher arguments are.

The image also explicitly redirects CUDA, DeepGEMM, TVM, Numba and Torch extension
caches into the new fingerprint namespace and enables assertion checks. Some
legacy cache variables are retained but not read by current B12X. Expect a cold
first boot and record cold/warm state alongside memory and KV capacity.

## Local checks and review

```
# Regenerates frozen inputs; this is preparation, not a read-only check.
python prepare.py
python -m unittest -v test_contracts.py
shellcheck build.sh gate.sh
DRY_RUN=1 bash build.sh
```

The retained FlashKDA tests under `inherited/flashkda-tests` came from the prior
Karmic build receipt and include the two all-buffer poison adaptations. Their
bytes are locked alongside the inherited gates. No network is used in assembly.

## Build window and gates

After independent review, record exact running Qwen container/image identities.
Stop kirby's worker then dusty's head gracefully with `podman stop -t 60`, retaining
both containers for rollback. Do not interrupt another cluster. Stage this task
directory on dusty and run `bash build.sh` with durable logging and monitoring.
The build fails closed if either idle probe fails or any container is running.

Native reuse avoids recompilation, not validation. MAX_JOBS remains 20. GPU gates
check source manifests, native imports/linkage and actual RMSNorm, NCCL identity,
FlashKDA targets and 12 poisoned checkpoint cases, CLI renders, the inherited
235/7/6/1/7 regressions, GLM draft-head, and new MHC/workspace memory regressions.
Matrices run in separate processes and reject skipped/missing cases; collection
is GPU-visible. Final live model correctness, capacity, multi-turn and benchmark
qualification remain separate work, one cluster at a time.

A non-serving retention tag protects the candidate from daily image prune.
Inspect/recipe/exit receipts are written before gates. The normal serving tag
`localhost/voipmonitor/vllm:karmic-main-spark-sm121` is published only after all
build gates pass; this is not a promotion. No prune or reset is part of this kit.
Transfers between Sparks use only switched 200G links and Docker archive without
conversion or compression. The separate benchmark repository is not modified.

The user additionally authorized Qwen MTP3 qualification in this same window.
The new node runner pins the new tag/tree labels and still requires an explicit
EXPECTED_IMAGE_ID. The copied correctness driver runs the existing semantic,
native-context, fixed-token acceptance, padded-transition and prefix probes.
Do not run MTP0 or a fresh R32 benchmark. Compare saved R32 and current JJ-main
receipts, including startup pressure and per-rank KV capacity, not just tok/s.
