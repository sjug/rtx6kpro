# Karmic beta SM121 build work

Candidate contract: [September 20 selection](../../docs/karmic-beta-spark-candidate-20260920.md).

As-built identities and wheel hashes: `build.lock.json`. Small native gate logs
are retained under `gate-evidence/`, including their original failed selection
and successful continuation. Current image-ID pins remain authoritative; the
additional component identity labels in the Dockerfile apply to future builds.

The image is built and has passed the native build gates. Model qualification
is in progress; it is not yet a production-qualified replacement.
`verify_foundation.py` checks the ARM64 NGC foundation's
Torch, BF16 cuBLAS and Triton execution on GB10, including the existing driver.
It does not qualify FlashInfer, vLLM, B12X, FlashKDA or the final NCCL replacement.
Run only with the node idle, with GPU attached, against the pinned base digest.
The check uses explicit exceptions and remains active under Python optimization.

The active build window is Qwen on dusty/kirby. Preserve the R32 containers and
image for restoration. The requested program continues through independent GLM
and DS4 Vision qualification on their existing nodes, with rollback retained.
Neither production deployment has been changed during native assembly.

## Current work

The foundation gate passed on dusty with driver 580.173.02. The Qwen R32
containers are stopped and retained; the candidate is serving on dusty/kirby.
All five native components and runtime assembly finished successfully.
Receipts are inside the deployed kit
at `/home/jugs/git/bld-jj-r38-spark/karmic-beta-sm121/build-receipts/`.

`build-component.sh` refuses concurrent FlashInfer compilation and
serving on either node. It only emits a retained component tag, never a serving
tag. Runtime dependency assembly and B12X/proxy preparation passed. The full
native gate set passed. Do not treat component `BUILD-OK` receipts alone as those
gates. Image ID: `f30dc6d9a2a6f6fc0ac9ff8cddb04d9f631f4a87b48ca7cac69254802fe83233`.

The managed native continuation completed. Check per-component receipts rather
than treating an inactive collected systemd unit as success.

Qwen's initial MTP3 correctness battery passed: semantic 18/18, exact retrieval
at 2,848, 2,849, 131,072 and 262,000 tokens, fixed-token acceptance, padded
concurrency transitions without collapse, and prefix reuse. Receipts are under
`qualification/qwen-mtp3-initial/`. Burst/queue probes and the first standard
benchmark passed; the second Karmic MTP3 sweep completed with consistent engine
steps (within 0.6% of the first run) and no new kernel allocation warnings.
Post-benchmark semantic checks passed. The first run's allocation warning remains
recorded, not claimed fixed; see `QUALIFICATION.md` for the mixed R32 comparison.
MTP0 is excluded by user instruction. GLM and DS4 Vision remain unchanged pending
their separate windows. Startup allocation-pressure warnings on both Qwen nodes
are retained for review, not silently treated as a clean kernel log.

Local checks:

```sh
python3 test_preparation.py
python3 test_build_policy.py
python3 test_runtime_kit.py
python3 test_build_lock.py
python3 -O test_preparation.py
python3 -O test_build_policy.py
shellcheck build-flashinfer.sh build-component.sh
```

## Subsequent cluster windows

Only one cluster is taken down at a time. Leave the other two models serving.
For GLM, stop the retained R38 workers on buddy/rocky/lucky gracefully with
`podman stop -t 60`, then the head on sparky. For DS4 Vision, stop
`ds4-vision-jj-r38-tp2` on toby first, then rusty, also with `podman stop -t 60`.
Keep stopped containers and images for rollback; do not remove them.

Only after the selected cluster is idle, run `distribute-clusters.sh` on dusty
with `QUALIFICATION_WINDOW=glm` or `ds4` and that cluster's hostnames. It verifies
the switched 200G route, available disk and idle state, loads the same Docker
archive, verifies image identity, then removes only that receiver's archive.
The source archive on dusty is retained. Never overlap this transfer with Qwen
timing. Stage the matching runner/launcher kit, then start workers before head.
