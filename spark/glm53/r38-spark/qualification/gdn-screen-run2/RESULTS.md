# GDN screen: startup blocked, no performance result

2026-09-15, dusty/kirby only. GLM and DS4 were not touched.

Final state: qualified R32 restored on both nodes at 20:06:52 EDT.
Both image-ID assertions passed and the completion returned exactly `333`
with `finish_reason: stop`. Driver status is `test_exit_status=1`,
`restore_exit_status=0`. No host reboot, GPU reset, cache deletion or
benchmark-source change was performed.

The experiment did not reach its first correctness or benchmark pass.
The FlashInfer arm was never launched. No conclusion about the GDN
prefill-performance hypothesis can be drawn from this attempt.

## Attempt history

1. The first attempt failed before model loading because the experiment
   launcher lacked its executable bit. This was an orchestration error.
   Receipts remain under `../gdn-screen/`. The permission was corrected,
   a fail-closed executable check was added, and container-level dry runs
   passed on both nodes. R32 recovery passed at 19:58:11 EDT.
2. The corrected retry began at 19:58:33 EDT. It reached R38 startup with
   B12X prefill and decode, the unchanged control selection. At 20:01:02
   EDT, kirby's worker raised `CUDA_ERROR_ILLEGAL_INSTRUCTION` (715)
   during `determine_available_memory -> profile_run -> _dummy_run ->
   speculator.propose -> MTP prefill`. The traceback surfaces the error
   at B12X `_w4a16_topk_sum_launch_flat`. CUDA errors may be asynchronous;
   this identifies the reporting path, not a proven faulty instruction.
3. Kirby's kernel journal at the same timestamp reports Xid 13,
   `Graphics SM Warp Exception: Out Of Range Register`, followed by
   Xid 43 for the worker. This is not an OOM receipt. Both nodes remained
   reachable and retained substantial available memory before KV sizing.
4. Both rank logs and kernel journals were archived before teardown.
   The worker was stopped, then the head. The head exceeded its 60-second
   graceful-stop deadline and Podman killed that container. The driver
   restarted the preserved R32 worker, then head. See `status.txt` and
   `restore-r32/` for the final recovery result.

## Configuration comparison

On both nodes, the failed control and the preserved earlier R38 container
have the same image ID and node command arguments. The only environment
differences are `GDN_SCREEN_ARM=b12x` and the matching experiment-launcher
SHA. The launcher diff adds an arm selector, clears an otherwise absent
decode override, and selects the original `--gdn-decode-kernel b12x` for
this arm. The extra read-only launcher mount is deliberate and recorded.
Model revision, MTP3, aligned policy, memory settings, NCCL, source packages
and cache paths are unchanged.

Evidence: `b12x-boot1/kirby-failure.log`, `dusty-failure.log`,
`kirby-kernel.log`, `dusty-kernel.log`, the per-rank container inspections,
and the preserved-R38 inspections at this directory's root.

## Next boundary

Do not proceed to timing without a healthy control. First distinguish a
reproducible stock-R38 startup problem from a transient or experiment
launch-context difference, preferably using the preserved R38 container
without any launcher override. Do not delete caches, change MTP, disable
graphs, alter memory limits, or patch kernels as an unrecorded workaround.
The GDN backend screen remains pending. No new image was built and no
production replacement was qualified.

The separate benchmark repository's newly staged campaign file still says
`active`. An attempt to record this blocked outcome there was rejected by
the approval boundary citing the instruction to leave that repository
untouched. No workaround was attempted. This local record is the current
status; there are no raw benchmark results to interpret or modify.
