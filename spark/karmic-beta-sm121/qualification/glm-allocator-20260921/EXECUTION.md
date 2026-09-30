# Allocator-only GLM retry

User approved one controlled retry after the loading failure. Image remains
9b23ca237881b90856adcea7f20be5c9937036b74863d668515ca1e58b4a6103.
The sole intended runtime change is
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, matching retained R38.
No NCCL, model, precision, utilization, checkpoint policy or image changes.

Runner render test proves only this environment argument is added. The driver
also checks actual container environments, image and command against the failed
boot before qualification. Prior failed containers are renamed and retained;
R38 containers remain available. Other model clusters are not touched.

This tests an allocator hypothesis, not a proven fix. If sustained loading
pressure recurs, abort and restore R38 rather than add further variables.
Memory/buddyinfo/vmstat, GPU and timestamped container/kernel logs are retained.
No benchmark repository edits, commits or promotion are authorized by this arm.

## Outcome

Actual identity/environment comparison passed on all four ranks. Main model
loading completed at 11:18:45 EDT: 46.93 GiB in 68.25 seconds. KV admission
reported 6,867,183 tokens. This is one successful allocator-enabled boot, not
proof of the allocation mechanism; page-cache history also differs from the
failed boot. The first real completion returned 333 at 11:22:34 EDT.

All seven short-pool checks passed. The three-repetition semantic battery
passed 14 checks then failed the third tool round trip: the model emitted
`17 × 23 = 391\n\nFINAL: 391` instead of exactly `FINAL: 391`.
The tool name, arguments and result were correct; this is exact-format
noncompliance, not an arithmetic failure. Five isolated unchanged tool-gate
repeats then passed, with complete exchanges in tool-format-diagnostics.jsonl.

A second full battery on the same boot, stored separately under
../glm-allocator-confirmation-20260921/, reproduced the same third-round failure
after 14 passes. No gate was weakened. No concurrency, long-context or benchmark
gate ran. Qualification remains blocked. R38 restoration was initiated after
the repeated failure; its final state is recorded under
../glm-r38-restored-after-allocator-20260921/.

Startup kernel snapshots through the first battery retain NV_ERR_NO_MEMORY
counts sparky=30, buddy=24, rocky=9, lucky=13. These warnings also occurred on
R38; they alone do not explain the answer-format finding.

Claude reviewed the local receipts and confirmed the narrow control. R38's
allocator setting came from its inherited base environment. Important
correction: DS4 Vision exports the allocator in its own launcher and Karmic DS4
logs already show expandable_segments:True. This GLM result therefore does not
explain the DS4 prefill regression. Other foundation/loader/environment changes
remain distinct, untested factors.

R38 restoration completed with a real completion of 333 and finish=stop.
All three head endpoints (dusty, rusty, sparky) returned health HTTP 200.
The candidate is stopped and retained, not promoted. Claude's completed second
review agrees exact-format noncompliance is established; sequence dependence
is only suggested by the repeated position, not proven. Failing full exchanges
were not captured by the original semantic harness, only its assertion output.
The next diagnostic should retain complete exchanges and compare the same
unchanged small battery on R38 and Karmic, rather than relax the gate or claim
a cache/kernel cause. No such new window was started.
