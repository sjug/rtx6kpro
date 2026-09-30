# September 23 Qwen QSA865 window

User authorized pausing Qwen on dusty/kirby, building on dusty, and qualifying
Qwen in the same window. GLM and DS4 are out of scope and remain untouched.

R32 worker and head were stopped gracefully, in that order, both exit 0 and
OOMKilled=false. Containers were retained, not removed. Rollback is the retained
`qwen38-flash-next-nvfp4-jj-r32-tp2`, worker on kirby before head on dusty.

## Build

- Image: `1a7a8acff71044e3b8bf97dacb7617cd1bdb50d0ab2fc2e35364a4d4dc165dcc`.
- Tag: `localhost/voipmonitor/vllm:karmic-main-qsa865-spark-sm121`.
- Receipt: `../build-receipts/qsa865-20260923T173427Z-925019/`, retained on dusty
  and copied locally without the archive.
- All inherited identity/native/FlashKDA/CLI/regression/memory/draft-head gates
  passed. Additional loader ABI/linkage, PR865 (1+2), RecoverSSM (10), block-FP8
  replay (2), WO replay (2), and LMCache MQ (1) gates passed.
- Inherited Bash initialization emitted `PS1: unbound variable` but returned to
  the build command, which completed successfully. GPU logs include upstream
  deprecation warnings, the known SM100-only DeepSelect absence, and an unknown
  pytest CUDA mark in the copied LMCache test. None was a test failure or skip.
- No compiler-job reduction or native vLLM rebuild was introduced.

## Serving qualification

Image distribution completed over 10.11.11.7 to 10.11.11.8 using Docker archive,
with archive digest and exact image ID verified. Candidate started worker-first
at 17:50 UTC with HC-off, MTP3, four sequences, aligned and capture size 32.
First real completion returned 333, finish=stop. KV admission: 5,069,308 tokens.

Completed checks:

- Semantic/tool/image: 18/18.
- Exact retrieval: 2,848, 2,849, 131,072 and 262,000 tokens, all stop normally.
- Fixed-token acceptance: 2.198211, 12 requests; outputs are not bitwise
  repeatable across waves. This is acceptance evidence, not determinism proof.
- Padded concurrency transitions: all six phases pass, acceptance 2.516 to 2.607,
  no collapse. Dispatch coverage remains predicted, not instrumented.
- Counting: 21/21 mixed-load continuations, 191 short requests without errors;
  12/12 serial cold/warm long continuations. All 21 mixed long requests overlap
  short client requests. Exact continuation oracle checks every returned text.
- Serial repeats hit 54,112 tokens rather than the predicted 56,960. Their
  success alone does not establish the exact 18-row incident geometry. Source
  review of that distinction is pending.
- Aligned head-of-line probe: maximum fresh TTFT 0.240s. Identical c4 burst
  129.7 tok/s versus distinct 124.4 and 126.1 in this diagnostic, not the grid.
- Health: startup NV_ERR_NO_MEMORY counts 177 on dusty and 73 on kirby, last
  events 17:57:08 and 17:56:58 UTC. No Xid or OOM kill, and no further driver
  allocation warnings after first serving admission. All POSTs in the retained
  window log are from the qualification workstation, 192.168.2.2.

Private replay initially stopped at the approval gate before sending anything.
The user then explicitly approved local private replay. Both reconstructed
requests completed with valid tool calls; no returned tools were executed and
proxy credentials were excluded. Private responses remain in ignored receipts.
The explicit replay health review passed before timing began.
Candidate remains evaluation-only, not promoted.
Both R32 containers remain retained for rollback.

Claude reviewed the amended execution/counting/benchmark/private replay scripts;
nine local tests pass. Raw qualification receipts are under `qualification/`.

## First grid and focused repeat

The standard 15-cell grid completed with exit 0. Its transient service was
collected automatically. Benchmark observers were started and stopped by
`benchmark.sh`; both benchmark-window kernel journals have no entries.

Geometric-mean output throughput at c1/c2/c4 was 48.769/78.042/116.284 tok/s;
engine steps were 22.762/37.299/56.965 per second. Relative to the saved R32
control, steps changed +3.15/+1.96/+2.10 percent. Relative to the prior Karmic
HC-off control, steps changed -3.57/-2.78/-1.83 percent.

Prefill at 8K/16K/32K/64K/128K was 3087/2993/2746/2692/1955 tok/s. The 128K
result is 21.23 percent below prior HC-off (2482) and 21.83 percent below R32
(2501). This is a single scout, not yet a repeatable regression claim. Claude's
completed read-only review confirms no measured clock, reclaim, JIT or foreign
traffic explanation. Promotion is held. A focused 128K/c1 repetition 2 is
running on the unchanged boot with its own observers and receipt directory
`qualification/mtp3-128k-repeat/`; neither control is being rerun.

The focused repeat finished with exit 0 at approximately 18:52 UTC. At 128K,
prefill recovered from 1955 to 2402 tok/s (53.32s TTFT), versus prior HC-off
2482 and R32 2501. The 21 percent loss did not repeat. C1 measured 45.3 tok/s,
22.0 steps/s and 2.06 acceptance. The harness also ran its automatic 8K and 64K
scouts, at 3074 and 2495 tok/s. Repeat-window kernel journals contain Ubuntu Pro
AppArmor perfmon denials on both hosts, but no NVRM, Xid or OOM messages.
The repeat is not a full-grid replication and does not establish the cause of
the first run's excursion or eliminate the remaining smaller differences.

At the user's request, a second full 15-cell grid started at 18:53:57 UTC on
the same candidate boot, as repetition 3 (repetition 2 was the focused test).
It uses the unchanged standard harness, contexts 0/16K/32K/64K/128K and
concurrency 1/2/4, with separate receipts in
`qualification/mtp3-full-repeat/`. Image/container identity checks passed and
the endpoint was idle before starting. No control rerun, restart or runtime
configuration change was made. GLM and DS4 remain untouched.

The full repetition 3 completed at approximately 19:07 UTC, exit 0, with both
benchmark-window kernel journals empty. Its transient unit was collected.
Grid comparison validates the saved protocol. Output-throughput geometric means
at c1/c2/c4 were 46.452/79.221/111.667 tok/s and engine steps were
22.618/36.717/54.758 per second. Against prior Karmic HC-off, engine steps are
down 4.18/4.30/5.63 percent; output throughput changes -3.72/+0.05/-7.38 percent.

Prefill at 8K/16K/32K/64K/128K was 3052/2901/2872/2471/2321 tok/s. Against
prior HC-off this is -0.78/+4.05/-0.17/-9.62/-6.49 percent. Thus the initial
21 percent 128K deficit did not recur, but the second full grid does not support
performance parity with prior HC-off. The cause is unresolved. This is a
same-boot repeat, not independent multi-boot qualification. Promotion stays held.

Result: `runs/qwen3.8-flash-next/nvfp4-4p89/2026-09-karmic-main-sm121-qualification/throughput/20260923T145358-0400__karmic-main-qsa865-hc-off-mtp3__r03.json`
(path relative to repository root). Raw logs and telemetry are in the separate
full-repeat directory named above. Neither private prompts nor private replay
responses are included in this public summary.

## Completed performance review

Claude reviewed r01/r02/r03 together and corrected its initial interpretation
after independent checking: the 1472 measurements and 3309 compilations beside
the PLE progress line are cumulative preparation counters, largely accumulated
while GDN prefill was being prepared. They are not evidence of a new PLE-specific
tuning event; the previous cold main boot had comparable preparation counts.

The second full run versus R32 measures engine-step changes +2.50/+0.37/-1.85
percent at c1/c2/c4 and prefill changes -10.60/-7.20 percent at 64K/128K.
This is a material long-prefill loss, not a uniform 21 percent slowdown.
The measured GPU clocks, temperatures, throttle flags, reclaim and kernel logs
do not explain it. Unsampled CPU state, process history, source changes and
tuning choices remain unisolated; no root cause is established.

A proposed next discriminator is a fresh candidate boot with its current tuning
and compiled caches retained, followed by matched repeated timing before the
heavy counting/replay battery. Such a test diagnoses boot/history dependence,
not source versus tuning by itself. It has not been executed. Re-tuning or
reverting to the older crash-exposed main build must not be presented as a fix.
The old Qwen R32 rollback remains available; no other clusters were changed.

Counting includes mixed traffic and serial repeated prompts at 56,960 plus
16/17/18/24/32/33 tokens. Actual cache reuse must be read from receipts; these
lengths alone do not prove that a particular graph was replayed. The private
trace has no raw wire request and is a reconstructed local replay, not byte-exact.
