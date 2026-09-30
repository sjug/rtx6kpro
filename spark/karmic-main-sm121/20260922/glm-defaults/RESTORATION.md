# R38 restoration, September 24, 2026

User explicitly selected Restore R38 after the slower candidate grid.
Pre-stop metrics showed zero running and waiting requests. The exact retained
containers and image identities were inspected on all four hosts.

Candidate `glm53-flash-karmic-main-defaults-tp4` was stopped with
`podman stop -t 60`, workers before head. All four exited 0 without OOM kill.
It remains retained, as does the earlier failed Python-entry candidate.

Starting at 00:53:49 UTC, the original
`glm53-flash-nvfp4-jj-r38-spark-tp4` containers were started on buddy, rocky,
lucky, then sparky. No configuration, image, cache or weight changes.

Image: `ea031e1d3d051033f077fc986bf6f8fce04cf9ab52483d5a719ba13114567fc5`.

| Host | Original container ID |
| --- | --- |
| sparky | c07c4251dffd46f2b2e2094090a3aa3402c58cc5b527e302e4f360a47d18bc0b |
| buddy | 2ba050c4eccf5d1bee8d3f9786e67969d4a69ef261d5d81d6e1dce5c70d18bcb |
| rocky | 67cc8b2f9808a9198f0440a9ee182f68cd38f149289f15990f943e5c487785b3 |
| lucky | de839293b69d0030c6709118a692eae1c22360a422662438be6625fc3986943b |

The bounded readiness check completed successfully by 00:57:26 UTC. A real
chat completion returned exact 333 with finish_reason=stop. All four original
container IDs and the R38 image match, all are running and none is OOM-killed.
This boot reports 6,261,779 KV tokens at native 1M context.

Kernel journals since restoration have one NV_ERR_NO_MEMORY allocation warning
on sparky, with none on the three workers and no Xid or OOM kill. This remains
an observed startup allocation-pressure caveat, not a resolved condition.

Evidence is under `qualification/entrypoint-20260923/`: the restored completion
JSON, readiness log, and four `*-r38-restored-health.log` and container logs.
Qwen and DS4 remain untouched. No temporary systemd unit was created for
restoration. No container or image was deleted, and no commit or push was made.
