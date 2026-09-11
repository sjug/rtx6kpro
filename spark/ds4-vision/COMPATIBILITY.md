# DeepSeek V4 Flash Vision on rusty/toby

Read-only compatibility review, 2026-09-10. This is source evidence, not a
claim that this checkpoint has already passed a GB10 deployment.

## Runtime decision

Use the existing **JJ R32 Spark image as the admission candidate**. A new
image build is not justified by the source review. Its frozen vLLM tree
`80a18accc688cdee2974f4c5e03e9416b2896087` contains the NVIDIA Vision model,
image preprocessing, DSpark draft integration and scheduler-reachable memory
profiling. The upstream R32 documentation explicitly supports DS4 text and
Vision in the shared composition, although its DS4 evidence is bounded TP2
RTX PRO 6000 evidence from R29, not a GB10 test. The Spark overlay changes
only CMake and two GLM files relative to upstream commit
`5576927057cf71b6ec61d120932338b333efa089`.
[Spark lock](../glm53/r32-spark/source.lock.json),
[shared-image specification at reviewed master](https://github.com/voipmonitor/rtx6kpro/blob/59f01d1/models/ds4-jovian-community-r29.md).

The candidate image is
`74e53e710bef141f6f68e722582569f9c6aa388bce405ad6f1423566a2300c9c`.
R32's Spark Dockerfile installs Qwen and GLM launchers, not a DS4 Vision
launcher. Author and gate a model-specific host launcher, retaining the
image's source-tree-first Python environment and coherent NCCL object.
Do not invoke the GLM image entrypoint for this model.
[R32 Dockerfile](../glm53/r32-spark/Dockerfile),
[build status](../glm53/r32-spark/qualification/BUILD-STATUS.md).

## Model identity and memory

The public Hugging Face API still resolves the model to
`6821d6ad3681a4b137b066b76094fa82ebd0a380`. Its 48 safetensors files total
**167,819,404,368 bytes, about 156.29 GiB**. Dividing by TP2 gives about
78.15 GiB per rank before replicated tensors, weight processing, activations,
graphs, vision storage and KV. This arithmetic is not a runtime allocation
receipt. The configuration advertises 1,048,576 positions, FP4 expert weights,
FP8 dense quantization, a 32-layer vision tower and three draft layers.
[HF metadata API](https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp?blobs=true),
[pinned config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/blob/6821d6ad3681a4b137b066b76094fa82ebd0a380/config.json).

R32 explicitly profiles one complete scheduler-budget DS4 prefill with real
attention and minimal temporary KV while vision encoder outputs remain
resident. Both old and v2 model runners contain this path. Keep multimodal
profiling enabled and let utilization-based admission size KV; do not copy
the RTX profile's 0.975 utilization onto GB10 UMA. Record available host
memory, per-rank weights, profiler peaks and admitted KV from this boot.
[old runner](https://github.com/local-inference-lab/vllm/blob/5576927057cf71b6ec61d120932338b333efa089/vllm/v1/worker/gpu_model_runner.py#L6685),
[v2 runner](https://github.com/local-inference-lab/vllm/blob/5576927057cf71b6ec61d120932338b333efa089/vllm/v1/worker/gpu/model_runner.py#L924).

## Launch contract

- TP2/DCP1, one rank per node, native multiprocess executor, worker first.
  No Ray. Preserve the pair's verified RoCE interfaces and image-owned NCCL
  and cache paths. Upstream's standalone Spark launcher demonstrates this
  topology but is not a drop-in container launcher: its defaults include K7,
  another HF cache, another model name and machine-specific addresses.
  [Spark launcher](https://github.com/local-inference-lab/vllm/blob/5576927057cf71b6ec61d120932338b333efa089/serve-ds4-flash-spark.sh).
- Fixed probabilistic **DSpark K3**, max sequences 4, scheduler budget 4096,
  graph cap 16, `FULL_AND_PIECEWISE`, FP8 KV and block size 256. The draft is
  contained in the same checkpoint. Do not carry 0731's K5 or invent a
  separate draft download. LMCache remains off for admission. Use only the
  served name `DeepSeek-V4-Flash-Vision-Exp`, not the old 0731 name or alias.
  [DS4 r9 profile](https://github.com/voipmonitor/rtx6kpro/blob/59f01d1/models/ds4-jovian-judgement-r9.md#serving-profiles),
  [official model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp).
- Require resolved architecture `DeepseekV4ForConditionalGeneration`.
  Although HF says `DeepseekV4ForCausalLM`, R32's config converter rewrites
  it when `vision_n_layers > 0`. Pin `deepseek_v4` tokenizer, tool and
  reasoning parsers, and keep the user's maximum reasoning default after
  bounded text, tool and image checks at that default.
  [architecture conversion](https://github.com/local-inference-lab/vllm/blob/5576927057cf71b6ec61d120932338b333efa089/vllm/transformers_utils/model_arch_config_convertor.py#L590).
- B12X sparse MLA and W4A8 experts are the shared DS4 path. Dense selection
  needs a runtime receipt. The shared RTX profile uses `b12x-a8-dglin`, while
  `b12x-a8` explicitly selects B12X dense. Do not assume that DGLIN means
  CUTLASS on this image: R32's platform code now admits capability family
  120 to DeepGEMM, but actual availability depends on installed native code
  and each projection. Record the selected dense and vision-attention
  classes rather than infer them from a backend nickname.
  [dense preference order](https://github.com/local-inference-lab/vllm/blob/5576927057cf71b6ec61d120932338b333efa089/vllm/model_executor/kernels/linear/__init__.py#L431),
  [DeepGEMM capability](https://github.com/local-inference-lab/vllm/blob/5576927057cf71b6ec61d120932338b333efa089/vllm/platforms/cuda.py#L675).

## Admission and qualification gates

Before stopping 0731, validate all 48 local shard targets, tokenizer and config
at the pinned revision; verify candidate image ID, source imports and CLI
render. Preserve the old container/image and HF snapshot as stopped rollback.
The existing Vision tests cover tokenizer identity, streamed/finalized weight
loading and rank-local routing; they are useful static gates but do not prove
serving output. [Vision regression file](https://github.com/local-inference-lab/vllm/blob/5576927057cf71b6ec61d120932338b333efa089/tests/models/test_deepseek_v4_vision.py).

After boot: first validate meaningful text, reasoning and complete tool round
trips; deterministic image content and image-plus-text answers; cold and
cached multimodal continuations; concurrent mixed-image/text requests; and
long-context retrieval at the configured limit. Record errors, allocation
pressure and speculative acceptance separately. Only then run the standard
`llm-inference-bench/run_bench.sh` grid with warm shapes and isolated traffic.
Compare steps, acceptance and output separately: changing checkpoint and K5
to K3 is not an isolated engine A/B. The RTX record includes a long-prefill
plus ten-image admission check and bounded concurrent Vision checks, so
neither text-only smoke nor `/v1/models` establishes the relevant memory
and correctness contract.
[r9 correctness/admission scope](https://github.com/voipmonitor/rtx6kpro/blob/59f01d1/models/ds4-jovian-judgement-r9.md#measured-results),
[shared-image concurrent Vision scope](https://github.com/voipmonitor/rtx6kpro/blob/59f01d1/models/ds4-jovian-community-r29.md#concurrent-vision-and-cache-transfers).

## Exact R32 launcher adaptation

The frozen tree also contains the self-contained `serve-ds4-flash.sh`, which
is preferable to the machine-specific `serve-ds4-flash-spark.sh`. No sourced
helper or `.venv` is needed when `PYTHON_BIN` names the image's actual Python.
For the approved first admission profile, pass the following environment
alongside the verified f1 pair network variables and image-owned cache/NCCL
environment:

```bash
PYTHON_BIN=/opt/venv/bin/python
MODEL=deepseek-ai/DeepSeek-V4-Flash-Vision-Exp
MODEL_REVISION=6821d6ad3681a4b137b066b76094fa82ebd0a380
SERVED_MODEL_NAME=DeepSeek-V4-Flash-Vision-Exp
DS4_MODEL_VARIANT=vision
MODE=dspark
DSPARK_TOKENS=3
DSPARK_DEPTH_MODE=fixed
DRAFT_SAMPLE_METHOD=probabilistic
REJECTION_SAMPLE_METHOD=standard
TP_SIZE=2
DCP_SIZE=1
MAX_NUM_SEQS=4
MAX_NUM_BATCHED_TOKENS=4096
MAX_MODEL_LEN=524288
MAX_CUDAGRAPH_CAPTURE_SIZE=16
GPU_MEMORY_UTILIZATION=0.85
KV_CACHE_DTYPE=fp8
BLOCK_SIZE=256
LOAD_FORMAT=instanttensor
INSTANTTENSOR_BACKEND=BUFFERED
LMCACHE_MODE=off
CUTE_DSL_ARCH=sm_121a
OMP_NUM_THREADS=2
NCCL_IB_DISABLE=0
NCCL_PROTO=LL,Simple
ALLREDUCE_MODE=nccl
VLLM_ENABLE_ROCE_ALLREDUCE=0
```

Resolve `PYTHON_BIN` in the image before adopting the path above. Explicitly
choose `BACKEND=b12x-a8` for the existing B12X-dense baseline or
`BACKEND=b12x-a8-dglin` for the published shared-image dense selection; record
which one is measured. `ALLREDUCE_MODE=nccl` disables custom all-reduce,
disables PCIe all-reduce and clears stale PCIe selector variables. Do not
introduce channel/protocol tuning as part of the checkpoint swap.

Invoke the source launcher with positional arguments, which it appends to
the generated vLLM command:

```bash
bash /opt/jovian-judgement/vllm/serve-ds4-flash.sh \
  --distributed-executor-backend mp \
  --nnodes 2 --node-rank "$rank" \
  --master-addr "$verified_head_pair_ip" --master-port "$port" \
  --speculative-config.revision=6821d6ad3681a4b137b066b76094fa82ebd0a380 \
  --default-chat-template-kwargs.reasoning_effort=max
# Worker rank adds --headless.
```

The target `--revision` is not inherited by DSpark's separate draft
`ModelConfig`; the explicit dotted revision above pins that configuration
as well. An immutable local `SPEC_MODEL_PATH` is an alternative.
[draft configuration](https://github.com/local-inference-lab/vllm/blob/5576927057cf71b6ec61d120932338b333efa089/vllm/config/speculative.py#L1268).

The script writes an earlier `reasoning_effort=high` argument, so require
the final effective value to be `max` in the CLI gate and boot config. Its
default prefix retention interval is 4096 and its compilation mode is
`FULL_AND_PIECEWISE`; these are separate from GLM/Qwen's recurrent-checkpoint
policy. It creates cache directories before exiting on `DRY_RUN=1`, so run
that render in a disposable CPU container rather than claiming a
side-effect-free direct host invocation. Verify both role renders before
the worker-first boot.
[frozen DS4 launcher](https://github.com/local-inference-lab/vllm/blob/5576927057cf71b6ec61d120932338b333efa089/serve-ds4-flash.sh).
