# DeepSeek V4.1 Flash on four DGX Sparks: source review

Reviewed 2026-09-16. Read-only research against `master` at `e8e23de` and
the committed `spark` branch R38 kit. No node access, source-repository
fetches, builds, restarts or benchmark-repository changes were performed.

## Verdict

Four GB10 nodes are a credible target for the official checkpoint, with
Engram tables on local NVMe. This is not yet qualified on our fleet. Our
R38 Spark image already contains the same model-source pins as the newly
documented upstream DS4.1 R38 release. The immediate work is a dedicated
four-node launcher, disk-Engram admission and correctness qualification,
not an assumed requirement to build a newer model implementation.

The upstream R38 qualification is four RTX PRO 6000 GPUs on one host with
RAM Engram. It does not qualify SM121, distributed Engram or whole-model
SSD throughput. Its measured throughput must not be projected onto Spark.
Sources: [release](https://github.com/voipmonitor/rtx6kpro/blob/e8e23de4a7d72e247b2809f5cc416a78201c1ffd/models/deepseek-v4.1-flash.md),
[qualification](https://github.com/voipmonitor/rtx6kpro/blob/e8e23de4a7d72e247b2809f5cc416a78201c1ffd/models/deepseek-v4.1-flash/r38/qualification.json),
`git show spark:spark/glm53/r38-spark/source.lock.json`.

## Model and TP4 geometry

The official configuration has 64 target attention heads, eight output
groups, hidden size 5120, expert width 2304, 384 target experts and 128 draft
experts. Those dimensions divide by four; target attention uses 16 heads
and two output groups per rank. The single latent KV head is not a TP4
divisibility obstruction: JJ shards query heads and retains native latent
cache geometry. Its DS4.1 attention explicitly rejects context parallelism
other than one. TP4/DCP1 is the appropriate starting topology, not the
custom DCP2/DCP4 port from another serving lineage.
Sources: [official config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/fb2764a5cf321eaa5070ca8f9e892818f477c16d/config.json),
[JJ attention](https://github.com/local-inference-lab/vllm/blob/66c293578412417476f842c1da5805d3a3d959a8/vllm/models/deepseek_v4_1/attention.py#L278-L334).

The model natively supports text and images, embeds its DSpark draft, and
declares a 1,048,576-token position limit. The released R38 serving profile
is qualified at 131,072 tokens; native maximum is not an automatic admission
or correctness claim. Source: [model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
and the pinned config above.

## Capacity and Engram placement

Public HF metadata was queried at the upstream-qualified checkpoint
revision `fb2764a5cf321eaa5070ca8f9e892818f477c16d`, without downloading weights.

| Quantity | Exact or derived size |
| --- | ---: |
| 48 safetensors files, including headers | 510,296,708,312 bytes, 475.251 GiB |
| Index tensor payload | 510,286,023,000 bytes |
| Packed Engram tables plus scales | 202,758,032,400 bytes, 188.833 GiB |
| Engram per TP4 rank | approximately 47.208 GiB |
| Remaining files, before runtime replication/repacking | 286.418 GiB total, 71.604 GiB/rank average |
| All checkpoint files divided by four | 118.813 GiB/rank, before runtime/KV/OS |

The Engram calculation is `(384006168 + 384016682) * (256 + 8)` bytes.
B12X uses ceil-row sharding, 256 packed weight bytes and eight scale bytes
per row. File averages are not memory-admission estimates: replicated
weights, kernel layouts, compiler workspaces, graph pools, table staging and
KV add to them. On GB10, pinned host RAM and GPU allocations share the UMA
pool. Whole-table RAM placement therefore is not a viable stock TP4
starting profile. Use `table_memory=disk`; generic CPU model offload is not
a separate source of physical capacity.
Sources: [HF tree API](https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4.1-Flash/tree/fb2764a5cf321eaa5070ca8f9e892818f477c16d?recursive=true&limit=1000),
[checkpoint index](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/fb2764a5cf321eaa5070ca8f9e892818f477c16d/model.safetensors.index.json),
[B12X Engram shapes and sharding](https://github.com/local-inference-lab/b12x/blob/ce419b52681b7922bb0972d4b58b590a3fd005b2/b12x/sequence/engram/api.py),
[JJ mapped-host and disk allocation](https://github.com/local-inference-lab/vllm/blob/66c293578412417476f842c1da5805d3a3d959a8/vllm/models/deepseek_v4_1/common/engram.py#L334-L420).

JJ already implements native disk row gathering, graph-aware staging and
TP summation of local Engram contributions. Complete checkpoint copies on
each local NVMe are the simplest fidelity-first arrangement for our fleet.
Rank-local sparse table copies are a later storage optimization, not a
reason to transplant a different Engram implementation. Full local copies
need about 510.3 decimal GB of checkpoint space per node, in addition to the
image and compilation cache. Never remove the Hugging Face cache.
Source: the pinned B12X Engram API and JJ `common/engram.py` above.

## What our image already has, and what its launcher lacks

The committed Spark R38 lock and upstream DS4.1 R38 lock agree on:

- vLLM `66c293578412417476f842c1da5805d3a3d959a8`.
- B12X `ce419b52681b7922bb0972d4b58b590a3fd005b2`.
- LMCache `29bc5a2efde737c436b04499eb62cd1776cebeec`.

Spark additionally applies its reviewed overlay and PR756, and supplies
SM121/aarch64 native artifacts. Its committed dedicated launchers cover
GLM and Qwen, with DS4 Vision tooling separate. There is no dedicated
committed DS4.1 Spark runner in that kit. The source-level support is present;
the new model's full runtime path has not been qualified by our GLM or DS4
Vision tests. Sources: `git show spark:spark/glm53/r38-spark/source.lock.json`,
`git show spark:spark/glm53/r38-spark/Dockerfile`, and
[published lock](https://github.com/voipmonitor/rtx6kpro/blob/e8e23de4a7d72e247b2809f5cc416a78201c1ffd/models/deepseek-v4.1-flash/r38/source.lock).

Do not invoke upstream's workstation wrapper unchanged:

- It forces `VLLM_HOST_IP=127.0.0.1`, defaults to loopback interfaces,
  disables IB and enables PCIe all-reduce.
- Its source launcher requires at least TP_SIZE entries in
  `CUDA_VISIBLE_DEVICES`, so a one-GPU-per-node TP4 launch violates that
  single-host preflight.
- The image wrapper defaults to 0.95 utilization and 16 NCCL channels.
  Those are not qualified GB10 multi-node settings.

Source: pinned recipe
`69a86060a158bcdd14af182b0381be9e92649db3:recipes/glm53/serve-ds41-jovian.sh`
and [pinned source launcher](https://github.com/local-inference-lab/vllm/blob/66c293578412417476f842c1da5805d3a3d959a8/serve-ds41-flash.sh).

The multi-node adaptation should use our worker-first Podman pattern,
`mp`, four nodes, explicit rank/master address, and switched f0 200G RoCE.
It should retain native DS4.1 parsers, cache precision, DSpark and page
geometry, with runner-owned communication settings and per-rank memory
receipts. Start with the tested 131K envelope and measured headroom before
enlarging context. Disk io_uring access needs a bounded container preflight;
do not assume full privilege is necessary or copy workstation permissions
without checking.

## Same-hardware evidence outside JJ

[Tony/Kai's repository](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark)
documents an official-release-checkpoint TP4 boot on four Sparks, with
disk Engram, native draft, graphs, vision and tools. It reports 81.6 GiB
model memory per node and seven end-to-end checks passing. Its later
EXL3/abliterated speed runs are different model/recipe arms, not evidence
for our unmodified checkpoint. Its scripts use a separate vLLM branch
and runtime patches, so they are portability evidence rather than a
drop-in JJ launch contract.

[Aiden's recipe](https://aidenle.com/recipes/deepseek-v4-1-flash-4x-dgx-spark/)
independently describes four GB10 nodes over 200G RoCE with local disk
Engram. It uses a day-zero vLLM branch, custom patches and optional DCP
ports. Do not import its NFS dependency, root deployment, cache flushing,
explicit KV budget or its reasoning-name mapping into our runner by habit.
The useful lesson is that stock-weight TP4 with disk tables has been
demonstrated, not that its throughput or settings are qualified for us.

## Qualification boundary

No additional model-source rebuild is justified by this review alone.
First author and render-test the DS4.1-specific runner against the existing
frozen Spark image. Check the actual image for model registration, loaded
module identities, kernel availability, io_uring linkage/access and disk
Engram component tests before an approved four-node window. Model startup
must prove per-rank memory headroom and Engram placement, then a real cold
completion, repeated text/vision/tool tests, prefix-continuation checks,
DSpark acceptance, native-context progression and the standard benchmark.
Treat transport and disk performance as unqualified until that run.

The JJ source's DCP1 restriction, stock checkpoint disk capacity, missing
four-node launcher and lack of our DS4.1 runtime receipts are the current
gaps. There is no evidence here that a fifth node or a speculative new
quantization is required for first admission.

## Master update and post-R38 source developments

The latest local `master` and `upstream/master` both resolve to `e8e23de`.
Relative to the preceding local tip `2b76bbd`, only the September 16 daily
summary and its index were added. R38 remains the latest published community
image documented here. The summary's `local/ds41:r39luk` is a user's custom
image, not evidence of a published community R39 release. The broader
post-R38 documentation includes the Qwen QAD evaluations below.

Primary-source API checks on September 16 found JJ at `59fbf05008` and B12X
master at `0add93fbea`, beyond the frozen R38 pins. Relevant changes are:

- [B12X #383](https://github.com/local-inference-lab/b12x/pull/383), merged:
  fixes dtype conversion and compiled-program ownership in the new
  RoCEnante preparation flow. The author reports 67 distributed checks
  passed per rank on four GB10 nodes, plus a Qwen serving check. This is
  direct Spark evidence, but not an isolated throughput gain or a required
  fix for R38's older preparation path.
- [B12X #362](https://github.com/local-inference-lab/b12x/pull/362), merged:
  bounds manual MXFP8 weight-scale reads on padded N swizzle tiles. This
  is a correctness fix with sanitizer evidence, not a demonstrated cause
  of our Qwen prefill deficit or the earlier empirical allocation-touch
  workaround. The affected scale dimension and mechanism are distinct.
- [Qwen projection scratch isolation](https://github.com/local-inference-lab/vllm/commit/98cd717d61c940a1a8f36668e0fea418ede28600)
  and [QSA selector isolation](https://github.com/local-inference-lab/vllm/commit/200b6e23590dfe9bab7adf740cdd9facb979a4ed)
  prevent concurrent prepared operations from overwriting scratch. Their
  source reports TP4 graph-decode validation. They do not establish that
  our R38 prefill cost is fixed; our Qwen production choice remains R32.
- [vLLM #770](https://github.com/local-inference-lab/vllm/pull/770), merged:
  reclaims released profiling blocks before the final KV admission
  snapshot. [#771](https://github.com/local-inference-lab/vllm/pull/771) and
  [#772](https://github.com/local-inference-lab/vllm/pull/772), also merged,
  repair sparse-MLA and GLM target/draft startup preparation. They are
  safeguards for the newer preparation lifecycle, not measured Spark gains.
- Cross-repository dependency warning: #771 requires
  [B12X #379](https://github.com/local-inference-lab/b12x/pull/379), still
  open at review with head `59f0cad9e4aa9eab57b8773ed83c064be608d95a`.
  B12X master `0add93fbea` does not expose `plan_pooled_selection` in
  `b12x/attention/sparse_mla/api.py`. Blindly pairing current tips is not
  a complete GLM composition. [B12X #380](https://github.com/local-inference-lab/b12x/pull/380)
  is also open and addresses prepared PCIe two-shot row counts.
- [vLLM #752](https://github.com/local-inference-lab/vllm/commit/ba9289fe91e1ac38750d3a26e4519e16158c134f)
  and B12X #371 remove background disk-Engram prefetch after R38. Disk
  table support remains. Do not carry R38's optional prefetch field into a
  later source candidate without checking its config API.
- [vLLM #775](https://github.com/local-inference-lab/vllm/pull/775) adds
  shared source-addressed CUDA 13.4 wheel publication for JJ and beta.
  Its own status says live publication is pending deployment. This is
  build-infrastructure progress, not proof that a compatible aarch64/SM121
  image or reusable wheel is published for us.

The official checkpoint's
[encoding correction](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/commit/dba1be0a40aa45a94ad051997016db3960a90277)
preserves tool namespaces in schemas, prompts and parsed completions.
Only encoding code, documentation and tests change. It is not a proven
general cure for reported looping. R38 uses its own
`vllm/tokenizers/deepseek_v41_encoding.py`; updating a downloaded checkpoint
alone does not automatically update that vendored encoder. Include namespace
round trips and long agentic tool cycles in a DS4.1 qualification.

The new [Qwen AA-LCR report](https://github.com/voipmonitor/rtx6kpro/blob/e8e23de4a7d72e247b2809f5cc416a78201c1ffd/models/qwen38-flash-next/aa-lcr-nvfp4-vs-qad.md)
favors QAD 79.4% versus 77.5%, with a question-cluster interval of 0.0 to
3.8 percentage points. QAD used 7.84% more reasoning tokens and 6.11% more
completion tokens in this evaluation. It is a possible quality experiment,
not evidence of a latency improvement or our checkpoint's TP2 performance.
The [arithmetic report](https://github.com/voipmonitor/rtx6kpro/blob/e8e23de4a7d72e247b2809f5cc416a78201c1ffd/models/qwen38-flash-next/direct-arithmetic-stability-nvfp4-vs-qad.md)
shows that low reasoning has a much larger effect than changing between
these two checkpoints; QAD remains research-only as a deployment target.

This research created only this note. No source pins, launchers, serving
configuration, nodes or benchmark repositories were changed.
