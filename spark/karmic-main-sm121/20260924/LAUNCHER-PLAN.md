# DS4.1 upstream-based Spark launch proposal

Status: user approved Podman and bypassing the external cluster launcher.
User subsequently approved removing container limits and fixed KV bytes, with
utilization 0.85. Claude reviewed these revised settings. User approved stock
NVIDIA NCCL `v2.30.7-1` at `73cf112295c33aee2b895f329f592f2a9b4b0f97`
as an explicit substitution for the unidentified patched launcher library.
No launch script has been implemented and no services stopped for this candidate.

## Starting source

vLLM `1794dcf18454900263e0c66711af8ea4a1283ac1`:

- `scripts/serve-ds41-flash-dspark-tp4-rdma.sh`, SHA256
  `7d18de06239bcea97d1ce1da5917a99904b6483a94e7bb72b03986a4d9286b31`.
- Its delegated `scripts/serve-ds4-flash-dspark-tp4-rdma.sh`, SHA256
  `4d751080ff43452f3110736b44213d0b9368b0369781ead27bf32ee351409ddb`.

The previous R38 DS4.1 launcher is historical evidence, not this candidate's
starting recipe. The earlier proposal to carry its runtime settings is superseded.

## Preserve upstream starting settings

- TP4/DCP1; K7 DSpark, greedy draft, adaptive verification.
- 500,000 context. Upstream's 10 GiB KV and utilization 0.82 are superseded by
  the approved automatic KV sizing at utilization 0.85.
- Four sequences, 8,192 batched tokens, upstream capture-size list and graph mode.
- B12X loader/plugin, attention, linears and MoE; upstream FP8 KV argument.
- RoCEnante, its upstream thresholds and network settings unless explicitly approved otherwise.
- Disk Engram with resident scales enabled; two-image limit.
- Thinking enabled, reasoning effort high.
- Non-privileged container. Omit upstream container memory, memory-plus-swap,
  explicit shm size and PID limits per the user. Audit effective Podman defaults;
  absence of a command-line flag is not evidence of an unlimited resource.

The 524K historical input exceeds this initial envelope. No claim about that
failure can be made from a 500K run; expansion requires separate approval.

## Deployment adaptation and unresolved dependencies

1. Podman/CDI instead of Docker GPU flags, with the same model command.
2. Our image ID, image-contained source trees, venv, CUDA compatibility activation
   instead of host source/venv mounts. User declined substituting NCCL 2.31.2:
   use the explicitly approved stock NVIDIA 2.30.7 substitution before upgrades.
3. dusty/toby/rusty/kirby and their verified switched f0 fabric instead of upstream
   host addresses and NIC names; hostnames for workstation management.
4. Existing local checkpoint paths and persistent fingerprinted caches instead of
   upstream host paths. Keep the image's prebuilt RoCE proxy path visible.
5. Refuse existing candidate containers rather than removing them; graceful,
   worker-first teardown and retained production containers for rollback.
6. Carry the reviewed io_uring seccomp permissions, with the exact base/profile
   difference checked before choosing the upstream or existing policy.

Direct engine process through Bash (activating the image's CUDA compatibility
environment) and `--init`, not keepalive plus exec. Do not add unlimited memlock
or disable SELinux labeling merely by convention; establish necessity first.
Keep private IPC. Dusty's installed Podman documentation specifies `--shm-size=0`
for unlimited IPC memory and `--pids-limit=-1` for unlimited PIDs. Omitting those
options instead gives 64 MiB and 2,048 respectively. Omit memory and memory-swap
flags. This implements removal of limits rather than accepting tighter defaults.
Use one switched f0 interface for Gloo
and socket bootstrap and both f0 RDMA HCAs. Set upstream `VLLM_PLUGINS=b12x_loader`
and `NCCL_NET_PLUGIN=none`, leave NCCL_PROTO unset, and preserve merge-nics=1.
Those inherited-image overrides restore upstream settings rather than add tuning.

Claude's adapter review identified memlock as an effective-runtime requirement.
A clean base container on dusty, with no `--ulimit` override, reports unlimited
memlock, as does the host. Check all four nodes before serving; do not add a
redundant override. The adapter explicitly selects the mp executor (also the
upstream engine default for CUDA with multiple nodes). External-launcher Ray,
UCX, MPI and TP-socket environment scaffolding is not carried into the direct mp
process. NCCL/Gloo socket selection is explicit on the switched f0 interface.
Use upstream's byte-exact io_uring seccomp profile, SHA256
`823b66051a52180058aa04b4343eb29a9cc963c1cff306e5c81dd606324252dc`.

NCCL provenance limitation: the launcher names a patched library under Luke's local
`/home/luke/projects/nccl-2.30.7/`, but no source commit or patch manifest was found
in the inspected launcher history, project docs or publisher NCCL branches/tags.
Claude's independent read-only search reached the same limit. The verifiable
stock NVIDIA tag `v2.30.7-1` points to
`73cf112295c33aee2b895f329f592f2a9b4b0f97`; it is not proven equivalent to Luke's
library. The user approved testing this stock build. Record it as a native
addition, not pure native reuse. Do not silently use 2.31.2 or the
NGC-bundled 2.30.7. No external message has been posted.

Broader follow-up search found a concrete Spark patch lineage in existing
`spark-vllm-docker` history: commit `990a7b3837338cff52d37c123bced61f6747bdba`
builds `zyang-dev/nccl:dgxspark-3node-ring`. Its inspected head
`fab1850acd902672d79ca81c2e7fb8e1848c208c` is "Support 3 DGX Sparks in a ring
topology via CX-7" and `makefiles/version.mk` reports 2.29.7, not 2.30.7.
Recipe commit `8dcbfe39acb2360665d1960e649ea825acfd4d1d` switched to NVIDIA
`v2.30u1`; `c45b95edaf62a850732873b268351898258df056` later switched to NVIDIA
default branch. This is evidence that a Spark-specific patch reference exists
in our repositories, but does not identify Luke's local 2.30.7 library. Do not
claim that no relevant patch exists or transplant it without matching provenance.

Lifecycle audit correction: the checked local cluster-launcher HEAD
`87fde023ecb0ccacb4d3cc246b7acb6fb79ad3e5` has no `--rm` flag. It explicitly
removes named containers during cleanup, and removes stopped same-name containers
at startup. That is the retention incompatibility, not automatic `--rm` removal.
Claude's follow-up distinguishes newer upstream cluster-launcher revisions that
do use `--rm`; this plan bypasses that external launcher rather than porting it.
The vLLM launcher pair remains the source of model argv and environment. Test
those against the adaptation and check effective environment after Bash activation,
particularly the loader plugin and NCCL plugin values.

The adaptations above have been discussed with Claude and approved by the user.
Further changes require that same review and approval. A rendered argv/environment comparison must
expose every deviation before launch. The external cluster launcher checkout
has existing modifications and an unresolved unrelated file; leave it untouched.

## Qualification boundaries

GLM on sparky/buddy/rocky/lucky remains serving and untouched. Qwen and DS4 Vision
clients are paused, but current containers remain running until the approved
build/window sequence begins. Record current container IDs and mounts for rollback,
not the September 16 restoration inventory.

Build/source/native gates precede a real completion. Model correctness, repeated
cold retrieval, tool and conversation probes precede performance timing. Use the
existing benchmark harness without changing its separate repository. Failures
produce a reproducible issue draft; external posting requires approval.
Existing retrieval probes through 262K fit this starting envelope. Neither those
probes nor a successful 500K boot close the original 524K failure.
