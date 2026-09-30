# DeepSeek-V4.1-Flash on four DGX Sparks

Status: **DS4.1 Flash is down; previous serving restored** (2026-09-16
21:36 EDT). dusty/kirby serve Qwen R32 and rusty/toby serve DS4 Vision R38
again from their retained containers. DS4.1 qualification ended blocked at
the 524K retrieval gate; see [EXECUTION.md](EXECUTION.md), the
[verdict](receipts/20260916/needle-524k-diagnostics/VERDICT.md) and the
[attribution](receipts/20260916/needle-524k-diagnostics/ATTRIBUTION.md). All
DS4.1 containers are retained stopped. The GLM cluster is untouched.

## Frozen admission profile

| Setting | Candidate |
| --- | --- |
| Image | `localhost/voipmonitor/vllm:jj-r38-spark-sm121` |
| Image ID | `ea031e1d3d051033f077fc986bf6f8fce04cf9ab52483d5a719ba13114567fc5` |
| Model | `deepseek-ai/DeepSeek-V4.1-Flash` |
| Checkpoint revision | `fb2764a5cf321eaa5070ca8f9e892818f477c16d` |
| Served name | `DeepSeek-V4.1-Flash`, one name only |
| Distribution | TP4, PP1, DCP1, one GPU per node, multiprocessing, no Ray |
| Engram | Local disk, no resident scales, no prefetch, projection TP off |
| Admission context | Initially 131,072 tokens; current boot native 1,048,576, not qualified (524K gate failed) |
| Memory | Utilization 0.85 initially, 0.80 for the K0 retry and native boot; no explicit KV byte allocation |
| Scheduling | 4 sequences, 4,096-token budget, one prefill, round-robin, compute share 0.4 |
| Speculation | DSpark K7, greedy proposal, standard rejection, adaptive verification |
| Graphs | FULL_AND_PIECEWISE, capture cap `sequences * (1 + K)`, initially 32 |
| Cache | Native B12X DS4.1 format (upstream auto), main block 256, SWA block 128, prefix caching enabled |
| Reasoning | Thinking on, maximum effort by default |
| Transport | PyNCCL, switched f0 rails, LL/Simple, no channel pin |
| Other backends | B12X attention, linears and MoE; LMCache and custom all-reduce off |

The rank map is deliberate and fixed:

| Host | Rank | Primary switched address | Role |
| --- | --- | --- | --- |
| dusty | 0 | 10.11.11.7 | API/head |
| toby | 1 | 10.11.11.6 | headless worker |
| rusty | 2 | 10.11.11.5 | headless worker |
| kirby | 3 | 10.11.11.8 | headless worker |

Only `rocep1s0f0,roceP2p1s0f0` and their f0 network devices are admitted.
The f1 back-to-back links cannot connect this four-node group. The host
preflight checks both rails' address ownership, peer routes, HCA state, and
GID index 3's netdev, RoCE v2 type and mapped IPv4 address. Secondary addresses
are the same host suffix on `10.11.12.0/24`. These fail-closed assertions
passed on all four nodes before the first launch. SSH management uses hostnames.
Bulk model/image transfers must originate on a Spark and use verified
switched 200G addresses, never the public NIC. Keep Docker-archive identity,
no conversion or on-the-fly compression. The completed transfer is recorded
in EXECUTION.md; only rusty downloaded from Hugging Face.

The source basis is R38's `serve-ds41-flash.sh` at vLLM `66c29357`.
That launcher assumes four locally visible GPUs and enables PCIe transport,
so this kit renders the model-specific CLI directly with the existing
multi-node Spark contract. This is not a source patch or new image build.
The actual image already has the Spark source composition and PR 756.
See the [R38 source lock](../../glm53/r38-spark/source.lock.json).

Differences from the upstream workstation launcher are intentional: distributed
TP4, utilization 0.85 instead of 0.98, bounded initial context, four rather than
eight sequences, serial prefill with fixed compute sharing, explicit main/SWA
geometry, maximum reasoning, disabled processor cache, bounded InstantTensor
buffers, and no custom/PCIe all-reduce. These are candidate settings, not
claims of optimality. Upstream's K7 is retained for the first default render;
`DSPARK_TOKENS=0` is the target-only correctness control.
`B12X_DYNAMIC_DETERMINISTIC_OUTPUT=0|1` is a diagnostic passthrough added on
2026-09-16 for the nondeterminism investigation; it is not part of the
qualified profile and is unset unless given.

## Disk and model contract

Each node needs a complete local HF snapshot: 48 weight files totaling
510,296,708,312 bytes (475.25 GiB), plus metadata. The packed Engram tables
account for about 188.83 GiB overall, about 47.21 GiB per TP4 rank. Keeping
them on disk leaves roughly 71.60 GiB/rank of non-Engram file bytes; this is
not a runtime peak estimate. Packed layouts, draft, workspaces and UMA
headroom must be measured at boot. RAM Engram is not a capacity solution
on unified memory.

`model-manifest.json` comes from the pinned HF tree API and records every
weight SHA256, size and metadata Git blob identity. The host preflight checks
all files, symlink containment inside the HF cache, exact sizes, metadata
identity and the 48-shard index. A separate `--hash` verifies every weight byte.
Size checks alone do not prove payload identity. Any newly transferred replica
must pass the full hash gate before admission. Existing unrelated `.incomplete`
files are not grounds to delete anything. **Never clear the HF cache.**

The mount is read-only. Disk Engram reads checkpoint shards directly through
the B12X disk table; it does not put the full tables in a writable `/tmp` cache.
JIT and temporary files use a separate persistent runtime cache, default
`~/.cache/vllm-jj-ds41-tp4`, mounted at `/cache` and `/container-tmp`.
The image's fingerprinted JIT environment remains intact.

The in-container preflight checks the imported source paths, SM121, registered
vLLM native ops, coherent NCCL preload, B12X loader ABI and liburing linkage.
It then uses the actual native row reader with queue depth 128, registered
buffers and files, and O_DIRECT on a real checkpoint shard. Both byte planes
are compared to buffered reference reads at deliberately unaligned offsets.
This is an I/O gate, not Engram numerical or performance qualification. The
ten GPU disk-gather cases subsequently passed on GB10, including consumer
graph replay and large row addresses. See EXECUTION.md for their scope.

R38's Dockerfile already compiled and linkage-gated the loader in
`/opt/local-inference/r38-host-cache` via `verify_native.py loader`. The new
runtime cache is separate, so the first use can recompile the same source.
Do not describe that as the first-ever loader build or as previously executed
io_uring I/O. Stage the cold-cache build and disk probe before full weight load.

Default seccomp remains on. If io_uring setup, registration or a direct read
fails, capture the errno and
inspect the host sysctl and effective seccomp policy before changing anything.
Do not substitute `--privileged`, turn seccomp off, or fall back to RAM. The
optional `IO_URING_SECCOMP_PROFILE=/absolute/reviewed.json` accepts a deny-default
profile and records its SHA256 in the container environment. It is intended
only for a separately reviewed narrow addition of necessary io_uring syscalls
to the host's normal policy; the runner does not infer or generate that policy.

The live gate found that the current baseline returns ENOSYS for all three
io_uring syscalls. The reviewed `seccomp-io-uring.json` preserves every baseline
rule and adds only setup, enter and register. It passes the real disk-read gate
without added capabilities, host changes or disabling seccomp. For this rollout
set `IO_URING_SECCOMP_PROFILE=/home/jugs/git/ds41-r38/seccomp-io-uring.json` on
each runner invocation. Record the profile digest along with the runner digest.

## Local checks

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 spark/ds41/r38/test_runner.py
PYTHONDONTWRITEBYTECODE=1 python3 spark/ds41/r38/test_probe_conversations.py
shellcheck spark/ds41/r38/restart-profile.sh spark/ds41/r38/wait-ready.sh
PYTHONDONTWRITEBYTECODE=1 python3 spark/ds41/r38/repro-nondeterminism.py --help
shellcheck spark/ds41/r38/run-node.sh
DRY_RUN=1 NODE=dusty bash spark/ds41/r38/run-node.sh
DRY_RUN=1 NODE=toby bash spark/ds41/r38/run-node.sh
DRY_RUN=1 NODE=rusty bash spark/ds41/r38/run-node.sh
DRY_RUN=1 NODE=kirby bash spark/ds41/r38/run-node.sh
```

Dry runs make no external calls and create no directories. The host and
container both verify `runtime-files.sha256`; edit a runtime file only with
a corresponding manifest update and a fresh review. Hostname/rank mismatch,
wrong image, wrong topology, arbitrary extra arguments and unsupported
resource/transport overrides fail closed. `PYTHONOPTIMIZE` cannot remove the
checks because they use explicit exceptions, not Python `assert`.

`parse_cli.py` gates eight default-profile renders (four ranks, DSpark K0 and
K7) using the image's actual vLLM parser. It runs in the launch preflight, and
can also run alone in a CPU-only container during the approved preparation:

```bash
podman run --rm --pull never --network none \
  -v "$PWD/spark/ds41/r38:/opt/ds41:ro" \
  --entrypoint /opt/venv/bin/python \
  ea031e1d3d051033f077fc986bf6f8fce04cf9ab52483d5a719ba13114567fc5 \
  /opt/ds41/parse_cli.py
```

Local tests cover the gate control flow with mocks, not the native I/O or
parser execution. All four real-image preflights subsequently passed. The first
admission retains upstream `--jit-monitor-mode error`; a new-shape compile
after activation must be recorded as a JIT warmup failure, not misclassified
as bad model output. Also verify in the installed InstantTensor package that
its bounded buffer, concurrency and I/O-depth environment controls are read.

The K0 control exposed a startup warmup hole: the first 21-token prefill
compiled `_quantize_attention_inv_rope_to_tdg_kernel` after the strict monitor
was active, which terminated the engine. An explicitly untimed K0 correctness
control may set `JIT_MONITOR_MODE=warn`; all compilations remain logged. The
runner refuses this setting with speculation enabled. K7 qualification and
benchmarking retain error mode. Warning-mode success does not qualify cold
K0 startup with the strict monitor or establish warmup completeness.

## Execution, only after the four-node window is approved

1. Inventory current containers and save exact restart commands. Dusty/kirby
   ordinarily serve Qwen R32; rusty/toby serve DS4 Vision R38. Recheck live
   state before stopping either. Do not touch the four GLM nodes.
2. Locate existing DS4.1 replicas and R38 images before downloading. Verify
   local NVMe backing, free space, the snapshot hashes and exact image ID on
   all four. No build is required for this candidate. Stage this directory in
   a persistent location on each node, not tmpfs.
3. Review idle-memory headroom, `/dev/shm` ownership, both switched rails,
   the liburing disk-gather smoke and all effective image environment defaults.
   Do not unlink shared memory, kill foreign jobs or prune images automatically.
4. Stop old workers first, then each head, using graceful `podman stop -t 60`
   in that approved window. Retain old images and recorded relaunch commands.
   The new runner itself never stops, removes or replaces a container.
5. On each idle host, run `bash run-node.sh --preflight-only`. This checks
   host state without starting a container or CUDA workload. Running containers,
   existing candidate containers, active GPU processes or failed state probes
   block the launch. This candidate also requires at least 100 GiB available
   and at most 1 GiB used swap while idle. These are conservative admission
   thresholds, not a proven memory optimum. They only refuse a new launch;
   there is no automatic kill or cleanup guard. Record the later load-time
   memory minima separately, since passing an idle check cannot bound them.
6. Start toby, rusty and kirby workers first, then dusty, by running
   `bash run-node.sh` locally on each host. Follow all four logs and UMA usage
   actively. `podman run -d` is only a launch receipt, not readiness or success.
   Worker-first ordering is the operator sequence, not an automated orchestrator.
7. Require cold first-completion correctness before timing: repeated meaningful
   arithmetic/text outputs, tool round trip, image interpretation, parser and
   maximum-reasoning behavior. `/v1/models` alone is never admission. Run the
   corresponding target-only control and check DSpark acceptance, not merely
   that the speculative flag appears.
8. Extend semantic retrieval and context checks from the 131K admission
   envelope to native context only as measured KV and UMA headroom permit.
   Include repeated prefixes, multi-turn concurrent requests and mixed loads.
9. Use the existing `~/git/llm-inference-bench/run_bench.sh` standard grid,
   with source/runner/image/model metadata, warmup and acceptance-normalized
   engine steps. Do not modify the benchmark repository. Record raw timings,
   disk activity, page-cache pressure, per-rank KV, clock/thermal samples and
   startup/serving errors separately. No DS4.1 throughput is claimed yet.

Stopping a failed candidate and restoring either previous service remains an
explicit operational decision. There is no automatic rollback, cleanup timer
or host-reset action in this kit.

## Release context

New source-addressed main/beta snapshots were published September 16, but no
numbered R39 or ARM64 successor was found. They move to CUDA 13.4/PyTorch 2.14
and are not necessary for the existing R38 DS4.1 code path. See the
[snapshot check](../../../docs/release-snapshot-check-20260916.md) and
[DS4.1 source review](../../../docs/deepseek-v4.1-spark-tp4-source-review-20260916.md).

## Local review and validation, September 16

Claude reviewed the initial kit against the composed vLLM tree, B12X source,
R38 build recipe and existing runners, then reviewed the corrective delta.
The final source-review verdict was no blocking defects. The later real-image
parser, native disk I/O and initial model-serving checks are recorded in
EXECUTION.md; source review does not substitute for them or for the remaining
target-only, sustained-load and performance qualification.

Implemented from review: real registered O_DIRECT byte-parity probe, actual-image
parser gate, both-rail/GID validation, absolute first-hop HF-link rejection,
one-time idle memory admission, native KV dtype default, explicit master
environment and seccomp-path validation. The claim that the loader had never
been built was corrected against the R38 Dockerfile and `verify_native.py`.

Local results: 19 unittest methods pass normally and under `python -O`;
shellcheck and Bash syntax pass; every new Python file parses; all eight
runtime manifest entries match. Native I/O in the unit suite is mocked and
is not an executed io_uring result. Dry renders cover all four ranks and
have no external calls or directory creation.

Non-blocking review notes retained for preparation:

- The image parser gate covers default K0/K7 renders. Environment-overridden
  commands are parsed by vLLM at startup, not by that additional eight-case gate.
- Verify the secondary-rail address suffix assumption live before scheduling
  the window; an unexpected layout fails closed.
- A relative HF link leading to a second absolute link is not explicitly
  rejected by the host's lexical-ancestor check. Confirm links resolve inside
  the actual container mount before admission.
- A container preflight exception prints a traceback and exits. It never
  proceeds to serving, but does not share the host runner's exit-78 wrapper.

No node access, build, transfer, model launch, benchmark-repository edit,
commit or push was performed for this kit during authoring and review.
