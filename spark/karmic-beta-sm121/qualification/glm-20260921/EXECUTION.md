# GLM Karmic window, September 21

Authorized GLM-only qualification; Qwen R32 and DS4 Vision R38 stay serving.
The four R38 containers were checked, stopped worker-first with a 60-second
grace period, and retained unchanged. No container was removed.

The initial execution driver exited before transfer because its source was
edited while Bash was executing it. Its execution.log and status.txt describe
that failed controller, not a model or GPU failure. Never edit an executing
shell script. Resumption uses a separate driver and receipts.

Claude's requested review found two stale paths in the GLM launcher's live
preflight. The original candidate reproduced exit 78 with --help, before any
model loading. The repair changes only the launcher filename and proxy path;
the generator, manifest and runner digest were updated together. Native/source
bytes and cache fingerprint are inherited unchanged. Image identities and
scope are in ../../glm-serving.lock.json; the original build.lock.json remains
the historical original image identity used in the Qwen/DS4 experiments.

The repair built on idle sparky and passed native smoke plus the real launcher
preflight with --help and CC=/bin/false. Evidence is in
../../gate-evidence/glm-serving-fix/. Distribution uses uncompressed Docker
archives and the 200G fabric, never the public NIC for image payloads.

Qualification was blocked during loading. The corrected driver validates the grid and compares
the digest-pinned R38 r04 receipt, but its completion marker is not promotion:
kernel warnings, KV capacity, request isolation, clocks and performance require
review. Historical R38 is not a same-day multi-boot control.

## Boot outcome and restoration

Candidate loading slowed to about 16 MB/s near 26/184 GB. Sampled minimum
MemAvailable was 3.09/1.10/1.87/2.30 GiB on sparky/buddy/rocky/lucky, with
several GiB of swap consumed. Pre-abort kernel snapshots contain respectively
15/178/110/74 NV_ERR_NO_MEMORY messages. These are snapshot counts, not a
complete final lifetime count. The attempt was aborted before a completion,
so no correctness or benchmark result exists. Workers exited 0; sparky exceeded
the 60-second stop grace period and Podman used SIGKILL (exit 137). All four
OOMKilled flags were false. No containers were removed.

R38 was restored worker-first, unchanged. Model loading on the return boot
finished in about 59 seconds, and real completion returned 333 with finish=stop
at 11:07 EDT. Effective KV capacity was 6,255,843 tokens. Restoration receipts
are in ../glm-r38-restored-20260921/. R38 also emitted startup allocation warnings
(37/0/2/43), so warnings alone do not distinguish this failure; sustained pressure,
swap consumption and loader collapse are the material differences. All three
model endpoints returned HTTP 200 from /health after restoration.

Claude's completed local review found verified inherited-environment drift:
R38 GLM has PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; the Karmic GLM
container does not. Qwen's launcher sets this independently. NCCL plugin
environment also differs. The missing allocator setting is a plausible loading
cause, not proven. No diagnostic retry or serving-config changes were performed.
The review's loader comparison used a non-R38 source anchor and is provisional;
its assertion that shrinking io_depth is unique to Karmic is contradicted by
the R38 return boot. Neither claim should be used as attribution evidence.
