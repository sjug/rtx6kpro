# DS4 Vision JJ-main candidate

Authorized test window: rusty and toby only. Keep Qwen and GLM untouched.
Candidate image: `6633e678fee74f5e1060290a01df812d34d10edcf30e34dd1c9bf7db88260ef3`.
Source is JJ-main at `8e1f1e587f` with the pinned Spark overlay and upstream
cache-agreement backport `0f60770e85`. See the parent build locks.

The in-image DS4 launcher remains byte-identical to R38. The wrapper and runner
preserve the qualified TP2/DCP1/K3, 524288 context, utilization 0.85, FP8 KV,
InstantTensor BUFFERED, DGLIN and pair-local NCCL contract. No loader or transport
experiment is combined with this image comparison. Native preflight executes
rms_norm on each GB10 before launch. Retain the stopped R38 containers for rollback.

Run CPU render/parser checks in the candidate image before executing the driver.
`DS4_JJ_MAIN_APPROVED=1 bash execute.sh` starts worker then head, observes both
nodes, requires a correct real completion, runs the existing semantic/vision/tool
and long-context battery, then 32 concurrent constrained-JSON requests with K3
enabled. Only then does it invoke the unchanged standard benchmark harness.

The structured probe is a stress test of speculative grammar handling, not a
reproduction of the lost original request that crashed R38. Passing it does not
prove the historical crash fixed. Preserve every request/response and kernel logs.
No benchmark-repository modification, commit, push, or production promotion is
part of this window. Results belong in the parent qualification/ds4-vision tree.
