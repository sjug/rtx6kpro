# Narrow HC-off backport execution, September 23, 2026

Candidate: `a25bedd43581dffecca2f9eefb40e60b170af6ae19124fef1c292850b2a0483e`.
Only PR865 is applied to the previous Karmic HC-off base. B12X, LMCache and
native components are unchanged. The source-derived cache namespace changed,
so this boot retuned; this is not an identical-tuning performance experiment.

## Completed

- Independent artifact and execution review, nine local tests, build gates.
- Docker-archive transfer over 200G from dusty to kirby, matching image IDs.
- Worker-first launch, HC-off, MTP3, aligned, capture32, utilization 0.85.
- First completion and semantic 18/18; exact retrieval at 2848, 2849, 131072,
  and 262000 tokens, normal stop.
- Fixed-token acceptance 2.26465; padded concurrency transitions without collapse.
- Counting: 21/21 long requests with 193 overlapping short requests, plus 12/12
  serial cold/warm boundary cases. Graph dispatch is not observed directly.
- Head-of-line probe: fresh maximum TTFT 0.2623 seconds.
- Two reconstructed private replays completed with tool calls, valid JSON
  argument structure, no tool execution, no proxy credentials forwarded.
  This establishes replay completion, not correctness of the private task.

Head KV capacity: 5,237,990 tokens. Both nodes had startup NV_ERR_NO_MEMORY
warnings; dusty also had an admission warning at 21:38:55 UTC. These are retained
as unresolved health observations. Journals from 21:39 UTC through post-replay
review show no NVRM/Xid/OOM events. Both containers remain running, OOMKilled=false.

## Two full benchmarks complete

The unchanged standard run_bench.sh full 15-cell grid ran 21:55 to 22:08 UTC,
exit 0, no error, underfill, capacity or warmup-timeout flags. Its file is
`20260923T175545-0400__karmic-main-hcbase-qsa865-hc-off-mtp3__r01.json`
under the standard campaign's throughput directory. SHA256:
`b1fd315f3c6949fe5af3346c907f33f87730dc25d97d0d49060f17fc00a9f5cd`.

Geometric means over the five contexts:

| Profile | c1 tok/s | c2 tok/s | c4 tok/s | c1 steps/s | c2 steps/s | c4 steps/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R32 September 21 | 44.19 | 74.92 | 115.79 | 22.07 | 36.58 | 55.79 |
| HC-off September 22 | 48.25 | 79.18 | 120.57 | 23.60 | 38.37 | 58.03 |
| Narrow backport r01 | 49.01 | 79.20 | 121.70 | 23.46 | 38.13 | 57.24 |
| Narrow backport r02 | 47.45 | 78.33 | 116.46 | 23.23 | 37.83 | 56.39 |

Prefill at 8K/16K/32K/64K/128K: 3065/2984/2886/2741/2495 tok/s.
At 32K through 128K it is within 0.6% of the saved HC-off run. Against R32,
prefill is -1.9% to -0.2%, and aggregate engine steps are +6.3/+4.2/+2.6%.
Individual c4/64K is -4.2% steps versus HC-off and -2.7% versus R32; do not
hide it behind aggregate gains. No claim of statistical equivalence from one run.

Benchmark kernel logs are empty on both nodes; no in-window JIT warnings or
throttle masks. Minimum MemAvailable: dusty 6259 MiB, kirby 6762 MiB.
Earlier qualification window warning counts are 170 dusty and 98 kirby; the
same signature exists on prior base boots, but its mechanism is unresolved.
Two first-request Triton compiles preceded timing.

Claude independently reviewed these receipts and arithmetic after its wait
completed. It recommended a same-boot repeat because the wider candidate
drifted on repetition. That repeat ran 22:10 to 22:23 UTC, exit 0, all 15 cells
without errors, underfill, capacity or warmup-timeout flags. No new R32 run.
GLM and DS4 were not touched.

Coverage clarification: serial counting's printed "cached token usage retained"
does not establish cache-hit counts. The raw completions endpoint returned null
prompt-token details; warm timing supports reuse but does not prove graph dispatch.
The separate 32K prefix probe does record 28,480 cached tokens.

## Repeat and verdict

Repeat receipt: `20260923T181038-0400__karmic-main-hcbase-qsa865-hc-off-mtp3__r02.json`.
SHA256: `0e13329df8e76e381fa11bb8409ce3d3c9566449059a0a7444ccd8ccd34e7351`.
Prefill 8K/16K/32K/64K/128K: 3045/2976/2843/2727/2463 tok/s.
Compared with r01, engine steps fell 1.0/0.8/1.5%. Against saved R32, repeat
steps remain +5.3/+3.4/+1.1%; against saved HC-off they are -1.6/-1.4/-2.8%.
The c4/64K cell repeats at 55.48 steps/s versus 55.54 in r01, 57.09 in R32
and 57.97 in the prior HC-off control. This is a measured residual, not explained.

Both runs' prefill is within 2.6% of R32. Versus previous HC-off, prefill is
within 1.2% except the 16K cell, which is about 7% faster. The wider candidate's
large 64K/128K prefill deficit does not recur in these two runs.
The saved controls are historical, and the cache namespace and tuning changed.
Neither exact performance parity nor a causal attribution to PR865 is established.

Repeat health: no kernel entries or in-window engine warnings on either node;
zero clock-throttle masks, no direct-reclaim/allocstall/compaction counter growth.
Minimum MemAvailable: dusty 6220.8 MiB, kirby 6747.1 MiB. Only the benchmark
client's address appears among in-window inference POSTs. Both image IDs remain
the qualified candidate ID, running and OOMKilled=false. All local one-off
qualification/benchmark units and observer processes have exited; remote build
and transfer transient units are also gone. No retained containers or caches
were removed.

Claude reviewed the repeat after its wait completed and independently reproduced
the arithmetic. Conclusion: the scoped backport passes this qualification, is
close to the previous HC-off performance, and is preferable to the wider source
refresh on these measurements. Do not claim a universal crash cure or absence
of small regressions. Leave this image serving on the Qwen pair for evaluation,
with R32 retained for rollback. No additional cluster rollout, commit or push.

Receipts: qualification/mtp3/ and qualification/window/. Private responses are
ignored local evidence and must not be committed or disclosed.

## User serving decision

September 23, 2026: user approved serving Qwen with the previous Karmic HC-off
image plus only PR865. Live inspection confirmed both dusty and kirby running
image a25bedd43581dffecca2f9eefb40e60b170af6ae19124fef1c292850b2a0483e,
HC=0, MTP3, aligned policy and capture32, neither OOM-killed. No restart was
needed or performed. Client endpoint remains http://dusty:8000/v1 with alias
Qwen3.8-Flash-Next. R32 remains retained for rollback. Earlier qualification
caveats remain valid; this approval does not establish a universal crash cure.
