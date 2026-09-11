# R32 Spark build execution

Started 2026-09-10 14:49 UTC following authorization to build and proceed.

## Inputs and scope

- Build host: dusty, `/home/jugs/git/bld-jj-r32-spark`, persistent filesystem.
- Base image: `0b15723cb87646bb4628c5cd67aa2a7c879ac5a27105109ea5db06c30113dc0b`.
- Source-lock SHA-256: `c28c09ef0fabe37d5ebd46a35aceee875a996cd3d5dafd813976050ce3940207`.
- Recipe SHA-256: `71afbe019c5e31ffe737d6732f1d930cc03bf9397bb2f2b76abba1c7caefda24`.
- Exporting checkout HEAD: `43b34e2bec1e211e961ebc81639ee55e0daadae4`.
- `ALLOW_DIRTY_BUILD=1`, with dirty status recorded. HEAD predates the R32
  artifacts and is not a claim that they are committed.
- Service: `jj-r32-build.service`, invocation
  `6e2a6add953e4953adb13a7e6730697a`.
- Durable build output and status: remote `build-receipts/` plus user journal.

The 25 local tests, both runner render suites and the dry run passed on both
the workstation and dusty. The staged recipe and source-lock digests matched.
An initial remote dry run rejected a shortened RECIPE_COMMIT argument before
any build or serving change; rerunning with the full HEAD above passed.

## Node transition

Only `qwen38-flash-next-nvfp4-jj-r29-tp2` was stopped, using `podman stop -t 60`
on kirby first, then dusty. Both were confirmed stopped and both hosts idle
before starting the build. Containers and images were retained, not removed.
Their serving image was
`ee996ef8e531eb6e9c3208ebe30b141dc41702da1466e6ab2c47daa666d5694f`,
with aligned checkpoints and no automatic container restart policy.

GLM and DS4 were not touched. No caches were cleared and no images were pruned.
The staged R29 records were not altered and no commit or push was attempted.

## Current state

BUILD-OK at 2026-09-10 14:54:03 UTC. Published
`localhost/voipmonitor/vllm:jj-r32-spark-sm121`, image
`74e53e710bef141f6f68e722582569f9c6aa388bce405ad6f1423566a2300c9c`.
All native, 251 R32 regression cases, inherited kernel/cache/communication,
12-case FlashKDA and launcher negative gates passed. Receipt:
`build-receipts/20260910T144935Z-1388331/`.

The Qwen continuation is `jj-r32-qwen-qualification.service` on the workstation.
Its first attempt stopped before transfer because the local receipt parent
directory was absent. The driver now creates that directory explicitly;
the first attempt's log and exit status are preserved separately. No gate was
waived. The resumed sequence transfers the unchanged Docker archive over the
verified `10.11.11.7 -> 10.11.11.8` switched link, checks receiver identity,
starts Qwen worker first, and requires correct output before any timed grid.

The initial Qwen sequence completed successfully at 11:34:49 EDT, including
the standard R29-matched sweep, MTP0 boundary control and return to aligned
MTP3. Review of the first sweep is in `QWEN-QUALIFICATION.md`. A matched repeat
on the restored boot completed successfully at 12:18:21 EDT under
`jj-r32-qwen-repeat.service`. The engine/prefill dips did not persist; the C4
output/acceptance residual was checked against a contemporary R29 control
under `jj-r32-qwen-c4-control.service` (12:26:36 to 12:45:15 EDT, exit 0).
C4 output is -1.18%, engine steps -0.54%, and prefill within 0.72% of the fresh
control. The aligned Qwen battery passes without a demonstrated material
overall regression; no speedup or statistical equivalence is claimed.
Both nodes are back on R32 aligned MTP3, image IDs verified and completion
exactly `333`, finish reason `stop`. Final engine KV capacity is 5392381 tokens.
No auto-policy change is in this sequence.

## Authorized GLM continuation

The user authorized the GLM transfer, cutover and qualification after the Qwen
review. `jj-r32-glm-qualification.service` started at 14:33:05 EDT. Four parallel
switched-200G archive transfers, digest checks, loads and image-ID checks
completed by 14:36:16. R29 was then stopped workers first and retained.
R32 passed its first completion at 14:42:12, all-rank identity/profile checks,
seven short-pool semantic checks and the 15-case semantic battery. The cache
and native-context checks also passed. The sweep finished at 15:08:57, but its
comparison exited 1 on capacity-limited cells caused by overlapping Pi traffic.
C1 and prefill preceded the interference and were at R29 engine/prefill parity.
After the user paused Pi traffic, `jj-r32-glm-quiet-repeat.service` ran the
full grid from 16:01:42 through 16:14:13, without a restart or configuration
change. All 15 cells, both comparisons, final completion and rank health
checks passed at 16:14:16, exit 0. No new JIT or runtime error was observed.
R32 is qualified for GLM aligned/MTP3 and remains serving, with R29 stopped
and preserved. Engine geomeans are +0.91/+0.53/-0.68% versus R29 at C1/C2/C4;
no material overall regression is demonstrated. See
`GLM-QUALIFICATION.md` and `glm-20260910/` for receipts. Qwen and DS4 are not
restarted or reconfigured by this continuation.
