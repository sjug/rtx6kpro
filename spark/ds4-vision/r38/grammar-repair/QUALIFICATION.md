# R38p DS4 Vision qualification

Date: 2026-09-22. Status: qualification complete with the explicitly recorded
supplement below. R38p remains serving on rusty/toby; original R38 rollback
containers remain retained and stopped.

## Candidate and scope

- Image: `ab3ed5285a81c9fd4df3c9001021c420368e7667177470aa2e055420f89ecdcf`.
- Base: qualified R38 `ea031e1d3d051033f077fc986bf6f8fce04cf9ab52483d5a719ba13114567fc5`.
- Sole new source fix: `8e1f1e587f8d24faf606f334a1c4bdaaa6bd4368`.
- Existing PR756 remains present. No native, B12X, allocator, NCCL, loader,
  model revision, serving flag, or KV configuration change.
- Rusty/toby only. Qwen, GLM and nous were not changed.
- Original R38 containers are stopped and retained for rollback.

## Build evidence

Unmodified R38: exactly 12 semantic failures and 20 passes in the adapted
32-case GPU grammar test. All failures concern grammar-invalid input token
positions. The adapter removes only the unsupported argument, not assertions.

Final candidate: 32/32 original upstream grammar GPU cases, 2/2 batch-sharding,
2/2 PR756 ownership, 1/1 scheduler bookkeeping, and native RMS-norm smoke passed.
`receipts/build/regressions.log`, `native.log`, `BUILD-OK` and
`image-inspect.json` retain the evidence. Six local kit tests also pass under
normal Python and `-O`; shellcheck passes at warning severity.

Offline fixture setup failures are preserved separately. The OPT scheduler
fixture attempts an optional safetensors metadata lookup while offline, logs
that failure, then passes. No fixture model weights were downloaded. This is
not a serving error or a skipped test.

The initial OCI-format build was rejected by the transfer manifest gate before
serving. It is retained in the historical receipts, not the qualified candidate.
The accepted build explicitly uses Docker v2. Transfer to toby verified archive
SHA, image ID, manifest type, configuration, history and rootfs layers, using only
the switched 200G route. The rejected image is no longer protected by the current
build tag; its inspect and archive remain available as evidence.

Claude independently reviewed composition, semantic red/green evidence, recipe,
final image lineage, runner identity-only changes, and qualification drivers.
No code blockers remained. Runtime manifests verified on both hosts before boot.

## Serving qualification

Worker-first startup began at 15:51 EDT after verified transfer. The automated
driver requires a real arithmetic completion, then semantic/vision/tool/concurrent
checks with 16K and 524K retrieval, 32 structured-JSON requests, the standard
15-cell benchmark, and post-grid semantic checks.

Completed: 20/20 semantic checks, including vision, tool use and exact 524K
retrieval, and 32/32 constrained-JSON requests with speculation active. The
structured stress recorded 285 drafted tokens and 259 accepted tokens.

The benchmark ran all 15 cells, but validation correctly rejected C4/16K:
`warmup_timed_out=true`, `capacity_limited=true`, `queue_reqs=1`. A foreign
client, `192.168.2.33` (m4-3), sent a chat request at 16:10:37 EDT alongside
the four benchmark requests. The engine reported four running and one waiting
request. No inference errors occurred. This cell is invalid, not evidence of a
candidate regression. Preserve the original run and flags unchanged.

The original driver exited 1 at 16:15:55 before post-grid semantic checks and
final inspections. Claude independently reviewed that saved grid and its logs.
The user then approved a quiet repeat. `recheck.sh` used the unchanged standard
harness with `--contexts 16k --skip-prefill`, `CONCURRENCY=4`, repetition 2.
The separate repeat finished successfully at 16:41:13, with no foreign inference,
no measurement flags, 78.45 output tok/s and 37.51 engine steps/s (rounded below
to 78.5 and 37.5). All 18 post-grid semantic checks passed. Final inspections
confirm the same image and container IDs remain running, not OOM-killed.

`receipts/c4-16k-recheck/` retains the successful status, validation, comparison,
post-grid checks, inspections and logs. Both nodes' new window logs contain no
NVRM, Xid, traceback or scheduler assertion. Each GPU log contains 84 samples
with zero clock-event flags. The original failed status and grid are unchanged.
This closes measurement coverage using 14 original cells plus one separately
identified repeat; it is not represented as a single clean 15-cell sweep.
Claude independently reviewed the supplement and found no blocking issue.
The repeat used a 16K rather than 128K pre-decode warmup context and occurred
outside the full sweep's sequence and thermal history (maximum GPU temperature
71 C versus 86 C). Its 30-second decode measurement policy and readiness rule
were unchanged. These limitations remain attached to the comparison.
The 18 post-grid receipts comprise 17 checked output labels plus the validated
tool-call first leg of the round trip.

Comparison with historical R38, geometric means over the completed coverage:

| Cells | Output tok/s, R38 to R38p | Engine steps/s, R38 to R38p |
| --- | --- | --- |
| C1, five contexts | 36.20 to 37.21 | 17.35 to 17.11 |
| C2, five contexts | 56.01 to 56.22 | 26.33 to 26.27 |
| C4, five contexts with separate 16K repeat | 79.03 to 77.99 | 37.25 to 36.77 |

Prefill scouts were 2169, 2269, 2230, 2154 and 2021 tok/s at 8K through
128K. These observations do not establish a speedup. The 524K retrieval took
330.6 seconds with 16,128 cached tokens and is not a cold throughput reference;
another foreign request appeared during that long-request window.

KV capacity was 1,651,283 tokens. Capacity varies across boots and is not
attributed to this grammar patch. Rusty recorded three startup
`NV_ERR_NO_MEMORY` messages during weight loading; toby recorded none.
Benchmark kernel journals were clean. Remote telemetry recorded minimum
MemAvailable of approximately 2.7 GiB and 5.5 GiB respectively over the captured
window. Use the retained remote telemetry for Spark hardware claims, not the
benchmark's locally collected hardware summary. No scheduler assertion or
engine failure appeared. The subsequent post-grid health verification also
passed, as recorded above. Existing startup allocation pressure is not resolved
by this grammar-only change.

Historical performance reference is the qualified September 15 R38 grid. This
is a single candidate boot, not a matched multi-boot performance claim. The
separate benchmark repository is not modified.

## Attribution boundary

The GPU test proves the known deferred-grammar rejection defect and its fix on
GB10. The production request that triggered the prior scheduler assertion was
not captured. Passing this qualification cannot prove that unknown request had
the same cause. The patch retains the scheduler assertion and adds diagnostic
counts if it recurs.
