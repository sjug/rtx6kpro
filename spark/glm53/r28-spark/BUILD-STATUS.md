# R28 Spark build, 2026-09-08

Authorized scope: stop Qwen on dusty/kirby and build R28 on dusty.
GLM and nous are untouched. No image distribution or serving cutover yet.

## Final result: BUILD-OK

Completed at 2026-09-08 18:51:36 UTC (14:51:36 EDT), exit status 0.

- Tag: `localhost/voipmonitor/vllm:jj-r28-spark-sm121`.
- ID: `cd93d80b3f9547f70e1e4608cd42af7c4b0f2ff33d92d5e520913cc9bab2d8d1`.
- Podman-reported image size: 33,327,153,776 bytes.
- All 226 selected pytest cases passed, including all 12 FlashKDA checkpoint
  cases with no skips. Runtime/custom-op verification and standalone draft-head
  numerics passed. Launcher render and refusal tests passed.
- Source trees and native hashes passed. FlashKDA and LMCache CUDA objects
  contain `sm_121a`; native imports and linkage checks passed.
- Exact final receipt: [BUILD-OK](build-receipts/20260908T184825Z-100138/BUILD-OK).
  All four attempt directories were copied back here from dusty.
- Both nodes are idle, with the R27 Qwen containers retained in `exited` state.
  The R28 image is on dusty only. No model-serving qualification is claimed.

The recipe remains uncommitted and is labeled dirty. Final recipe SHA is
`9695c1ca7d7acf12c67010772a94d25d0c1c93492e408cea2d4acb369e25aecd`.

## Initial launch record

- Qwen worker on kirby stopped first, then head on dusty, using `podman stop -t 60`.
  Both existing R27 containers were retained, not removed.
- Persistent build directory: `/home/jugs/git/bld-jj-r28-spark` on dusty.
- Unit: `jj-r28-build-20260908.service`, started at 14:32:25 EDT.
- Invocation: `RECIPE_COMMIT=783ad369b1b8650df2e5448dce0f0a9f14236503
  ALLOW_DIRTY_BUILD=1 PYTHONUNBUFFERED=1 bash build.sh`.
- Source lock SHA: `a0e3f833042ac0be453160255464c8827734ea2ccc302de9fff531cbdfac8110`.
- Recipe SHA: `8d6b1a90eddc003b73a1248c3a93b6bd0e35b0e58449c3e8f0ea122a1b54443b`.
- Bundle, patches and recipe bytes verified before shutdown. Eight local tests
  and both runner suites passed on dusty.
- Host adaptations: use its installed `python3`; accept the exporting checkout's
  commit with explicit dirty provenance for a standalone kit; sort manifest paths
  using `LC_ALL=C` so workstation and build-host hashes agree.

Durable attempt receipts are under `build-receipts/` in the build directory;
initial startup is also in
the user journal. The final normal image tag is assigned only after gate success.

## Assembly retries

FlashKDA compiled and passed SM121a/linkage checks with SHA
`69347e79224ee3a48d1d3604d302dc275a595188b89db53be2a09593dbd07d98`.
LMCache's CUDA object, CPU modules and interposer passed their binary checks.

Attempt 1 stopped at the installed LMCache tracked-content check: copying
`lmcache/v1/distributed/bitmap_ops/README.md` dereferenced a tracked symlink.
Attempt 2 preserved symlinks but hit the regular README already installed by
the wheel. Neither published a final image tag.

The complete fix replaces only paths declared as symlinks in the frozen source,
then copies with `symlinks=True`. The installer passed directly in the existing
pre-install image `56f45ad98d8f` before attempt 3 was started. Source trees,
native objects and serving settings are unchanged.

Attempt 3 passed runtime verification and every GPU gate, including the exact
12-case FlashKDA matrix and the tighter draft-head thresholds. It stopped at
an overly-specific error-message match in the hollow-install negative test:
Qwen returned the required exit 78 with `runtime source-path preflight failed`,
while the test expected `runtime preflight failed`. No tag was published.
The corrected match retains exit 78 and the refusal message; the complete
launcher-only tail, including tampered-launcher rejection, passed against
the same image before another build was started.

Attempt 4 completed as `jj-r28-build-20260908-attempt4.service`, receipt directory
`build-receipts/20260908T184825Z-100138`. It incorporates both corrections in
recipe SHA `9695c1ca7d7acf12c67010772a94d25d0c1c93492e408cea2d4acb369e25aecd`,
reused the native layers, and passed the full gate ladder before tagging.
