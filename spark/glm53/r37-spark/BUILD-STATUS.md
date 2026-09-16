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

Historical progress, 2026-09-13 23:29 UTC: FlashInfer AOT compilation was running on
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

Remote kit: `/home/jugs/git/bld-jj-r37-spark`. Both builds wrote persistent
logs, status and image IDs below `build-receipts/`. The old component saved
its inspection, but runtime inspection was incorrectly deferred until after
successful gates. The queued process fails closed if the
component fails, the wait exceeds four hours, either node starts serving,
or any image gate fails. It never starts a model.

The recipe is uncommitted, intentionally recorded with `ALLOW_DIRTY_BUILD=1`
and exporting commit `f7073c09067a9145ac7d09918ade6b32a4c05925`.
The final recipe digest is computed over the actual staged input files.
No commit or push was requested.

## Build-policy correction, 2026-09-13 evening

The first workspace correction restored the prior recipe declarations:
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

## Failure and automatic prune, verified September 15

FlashInfer completed at 2026-09-14 00:24:46 UTC with component ID
`987f9fb58a9a9221c167503992f1dc19ba6c67bc1fa4a2a764bbd73740bbc376`.
Runtime assembly completed, but the gate failed closed at 00:26:11 UTC on
candidate `37db089076b67361a9fe68a0171eab409384551b1c77dede38091a01f51e61eb`.
No final tag was published. The GLM digest argument collided with the
inherited R32 environment variable. Three further gate defects were present:
the wrong FlashInfer metadata attribute, scheduler count 1 instead of 20,
and sparse-MLA count 16 for a 24-case selection.

The enabled rootless `podman-auto-update.timer` invoked its service at
00:00:50 EDT (04:00:50 UTC) on September 14. Its
`ExecStartPost=/usr/bin/podman image prune -f` removed 36 images, including
the component at 04:00:51 UTC and runtime candidate at 04:00:52 UTC. The
service journal attributes those removals to Podman PID 2389558. This was
automatic pruning, not an operator action. The tagged R32 base survived.
The component wheels and intermediate cache from that run are unavailable;
FlashInfer needs rebuilding. The old receipts remain unchanged. Runtime
inspection was never saved, so the deleted candidate cannot be reinspected.

## Corrected local kit, September 15

- Both launcher build arguments have distinct `BUILD_*` names and their
  files are checked by SHA-256 during assembly. FlashInfer checks
  `__git_commit__`. The scheduler and sparse-MLA counts are 20 and 24.
- Both build commands assign non-serving retention tags at image commit.
  Runtime inspection is saved before gates, and the serving tag remains
  publish-after-gates only. Host timers and services were not changed.
- The component embeds its recipe files, manifest and input identity. Future
  runtime builds copy and verify them against the current lock before reuse.
- Effective FlashInfer parallelism is explicitly `MAX_JOBS=20` and
  `FLASHINFER_NVCC_THREADS=1`. The generic `NVCC_THREADS=4` declaration in
  R11 never controlled FlashInfer; its actual compiler used `--threads=1`.
  The same is true of the failed R37 run. No increase in per-job threading
  has been introduced.
- Collection and execution share one selection list. Collection runs with
  CDI before numerical tests and rejects unexpected counts or import skips.
  Four native filesystem delete cases are included, B12X's seed fixture is
  restored, imported dependency paths and liburing linkage are checked, and
  native-source coverage includes Rust and all B12X non-Python changes.
- Immediate-native-base labels consistently describe R32. Older ancestry
  is preserved in the base inspection receipt rather than mixed into that
  family. Version, rebuild and description labels describe R37.

Static validation: 33 local tests, dry run and shellcheck pass. Source replay
still produces vLLM `61e4d998`, B12X `480ea232` and LMCache `5a88a1ea`.
Source pins, patches, changed-path manifests and both launcher hashes are
unchanged. Lock SHA-256:
`0c2ce1cdcf6fa045a896d72fccc054c0d673d7792738e4089b5fcee3149551d9`.
The recipe changes the cache fingerprint to
`cu133-torch213-jj-r37-sm121-870c7d5c5bc8053a11f9`.

This correction was authored while the checkout was on `master`, confined
to the untracked R37 kit. No branch switch, commit, remote kit replacement,
build, GPU gate, serving restart or cleanup was performed in this pass.
Full collection and GPU acceptance remain pending the rebuilt image.

## Gate import-mode correction, September 15

The LMCache filesystem selection runs from `/opt/lmcache/source-r37`, whose
test package carries `__init__.py` at every level. Pytest's default prepend
mode inserts that root at `sys.path[0]`, so `import lmcache` would resolve to
the source tree, which holds the Python files but never the rebuilt
extensions. `native_connector_l2_adapter` imports `lmcache.lmcache_native` at
module level, so collection would have failed in the preflight and the build
with it. That selection now passes `--import-mode=importlib`, which imports by
spec and leaves `sys.path` alone; the other selections keep prepend mode,
which is what gives vLLM and B12X their intended source-tree imports.

Both modes were reproduced outside the image with the invocation shape
`run_required.py` uses, a script outside the tree calling `pytest.main`:
prepend imported the source tree, importlib imported the installed
distribution. A contract test pins the mode for this selection. The four cases
cannot pass against a source tree that lacks the extensions, so a passing gate
is itself evidence that the installed rebuilt modules were exercised.

Validation: 34 local tests, dry run and shellcheck pass. The source lock is
unchanged at `0c2ce1cdcf6fa045a896d72fccc054c0d673d7792738e4089b5fcee3149551d9`
and the cache fingerprint is unchanged. Only the recipe digest changes, to
`f6b2f6470a450d8367c706a90ad6de826cf75403a8772149dd255e4afc968268`.
No build, GPU gate, commit or host change was performed.

The former `build-receipts/staged/build-flashinfer.sh` is now filed as
`build-receipts/runtime-20260914T002458Z-1762619/build-flashinfer.sh`, the run
whose recipe manifest names its digest `d4c32f66`. No staged path is read any
more, and no bytes were discarded. The script that built the pruned FlashInfer
component was the earlier `6ab99398` revision, whose digest that component
receipt records; builds since the correction copy their own scripts into their
receipts.
