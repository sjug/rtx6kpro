# Original R38 container replay

2026-09-15, dusty/kirby Qwen only. User requested the last successful run
be recreated before adding further experimental changes.

## Execution and result

The pair was idle before interruption. The saved R38 container Id, Image,
Config, HostConfig, Mounts, Args and Path matched the successful
`../contrast-r38-return/` inspections exactly on both nodes.

Stopped the R32 worker, then head with `podman stop -t 60`. Started the
preserved `qwen38-flash-next-nvfp4-jj-r38-tp2` container on kirby, then dusty,
using `podman start`. No container recreation, launcher bind mount,
environment change, cache deletion, image rebuild, or GPU reset.

Image: `ea031e1d3d051033f077fc986bf6f8fce04cf9ab52483d5a719ba13114567fc5`.
Checkpoint: `c374e7e24b54f6cb0017d0c2e6d26823d2f2fb5d`.
Profile: TP2, MTP3, aligned, B12X GDN prefill and decode, utilization 0.85.

- Both target and MTP profiling completed at 20:49:39 EDT. Kirby's MTP
  profiling took 0.17 seconds, passing the stage that failed at 20:01.
- Both ranks completed CUDA graph capture and kernel warmup.
- Engine reported 5,316,408 KV tokens. Per-rank available KV memory was
  43.19 GiB on dusty and 44.20 GiB on kirby for this boot.
- The arithmetic completion returned exactly `333`, finish reason `stop`.
- Neither archived kernel journal contains an Xid in this replay window.
  Dusty logged one `NV_ERR_NO_MEMORY` allocation warning at 20:49:49 during
  graph capture; startup and the completion nevertheless succeeded.
- At the final identity checks, both original R38 containers were running.
  R32 containers remain stopped and preserved. R38 is the investigation
  control, not a new production promotion.

## Interpretation and limits

The earlier illegal instruction did not recur in the exact preserved
container replay. This does not identify its cause or categorically clear
the experimental launcher, cache state, or kernel execution path. The same
named AOT artifacts were loaded, but the receipts do not contain historical
binary checksums. Both fresh creation and restart had prior successes.

Claude reviewed the archived execution-state evidence locally through
Herdr. No node operations or edits were delegated. This run is startup and
completion validation, not a repeated performance benchmark or full model
qualification. No benchmark-repository writes occurred. GLM and DS4 were
not touched. The FlashInfer arm remains unexecuted.
