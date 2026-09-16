# R37 SM121 build

2026-09-13: User authorized an R37 GB10/SM121 build on idle rusty/toby.
No serving cutover or work on dusty/kirby or the GLM cluster is included.

- Artifact branch: existing `spark`; upstream documentation remains on `master`.
- Build host: rusty, aarch64/GB10, no running Podman containers, GPU idle,
  approximately 116 GiB MemAvailable and 2.2 TiB disk available at preflight.
- Test peer: toby, no running containers, GPU idle, approximately 117 GiB
  MemAvailable and 3.0 TiB disk available at preflight.
- Available base on both nodes: R32 image
  `74e53e710bef141f6f68e722582569f9c6aa388bce405ad6f1423566a2300c9c`.
- Target: the frozen R37 source lock published at rtx6kpro master
  `7a36aecdf0627f4e1efbd878532d630b3489a494`, including its dependency patches.
- Normal output tag: `localhost/voipmonitor/vllm:jj-r37-spark-sm121`.
  Publish only after image gates pass.
- Native audit: all vLLM native inputs and FlashKDA match R32. Reuse those
  binaries and RoCEnante; rebuild FlashInfer for SM121 and LMCache CPU modules.
  LMCache CUDA and the interposer have unchanged native inputs. Retain
  canonical libaio and coherent NCCL.
- Preserve existing qualified serving settings for later comparisons; no
  adoption of RTX communication, memory, speculative, or checkpoint defaults.
- Build contexts, receipts, and source preparation use persistent disk, not
  tmpfs. Never clear the Hugging Face cache or reset Podman storage.
- Bulk transfers between Sparks use verified switched 200G routes only.

Current phase, 2026-09-13 23:29 UTC: FlashInfer AOT compilation is running on
rusty in `jj-r37-flashinfer-build.service`, at 1017 of 3406 CUDA build steps.
The last memory check reported 113 GiB available. No serving qualification
has run and the final R37 tag has not been published.

Runtime assembly and GPU gates are staged and queued in
`jj-r37-runtime-build.service` (invocation
`aa18ed92a17f41bfb9937055bd040a66`). This service waits for the exact component
receipt `build-receipts/flashinfer-20260913T230211Z` to complete successfully,
checks its image ID, then runs `build.sh`. It does not start a second build
concurrently. All 22 local contract/native-inventory checks and the dry run
also passed on rusty. Shellcheck passed on the workstation.

Remote kit: `/home/jugs/git/bld-jj-r37-spark`. The native component and final
runtime each write persistent build logs, status, image ID and inspection
receipts below `build-receipts/`. The queued process fails closed if the
component fails, the wait exceeds four hours, either node starts serving,
or any image gate fails. It never starts a model.

The recipe is uncommitted, intentionally recorded with `ALLOW_DIRTY_BUILD=1`
and exporting commit `f7073c09067a9145ac7d09918ade6b32a4c05925`.
The final recipe digest is computed over the actual staged input files.
No commit or push was requested.

## Build-policy correction, 2026-09-13 evening

The workspace artifacts now restore the prior Spark FlashInfer policy:
`MAX_JOBS=20`, `NVCC_THREADS=4`, with the unused `AOT_MAX_JOBS_CAP=4`
override removed. The LMCache CPU-only stage now uses the pinned upstream
R37 policy, `MAX_JOBS=8`, instead of the unsupported four-job substitution.
The FlashInfer entrypoint runs the policy/digest checks before compilation
and snapshots its Dockerfile and build script into each new receipt.

All 24 local tests, the dry run, and shellcheck pass. Source replay still
produces the same vLLM, B12X and LMCache trees. Only `native_build_inputs`
and `cache_fingerprint` changed in the lock; source pins, patches and
launcher hashes did not change. The corrected source-lock SHA-256 is
`e32cd17b279d1379c6e22aabc76e53e44aae62e1dcf0fa2e1ffe0f5ed240a90f`.

The active build on rusty was not interrupted or rewritten. At inspection
it was at 2364/3406 steps, with the runtime continuation still waiting.
That run retains its original source lock
`0deea7c4fb7257e7486396980c86f858acf5e76552f3382f72e5694eb46555f0`
and FlashInfer Dockerfile digest
`3d2fa3bc986adde463835ab8c2ab2b0b6c7f6bf6d942f241974255fadec688d7`.
The corrected workspace kit has not replaced the active remote kit: changing
an artifact cannot change a running Ninja process, and the old receipt must
not be relabeled as a twenty-job build. These settings apply on the next
build invocation using the corrected artifacts.
