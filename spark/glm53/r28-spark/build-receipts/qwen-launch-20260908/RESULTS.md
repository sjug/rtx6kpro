# Qwen R28 startup and semantic admission, 2026-09-08

User authorized bringing Qwen up on dusty/kirby. Worker started first on kirby,
then head on dusty. GLM and nous were untouched. No benchmark or full R28
qualification battery ran in this startup task.

- Image: localhost/voipmonitor/vllm:jj-r28-spark-sm121
- Image ID: cd93d80b3f9547f70e1e4608cd42af7c4b0f2ff33d92d5e520913cc9bab2d8d1
- Runner SHA256: d00f73937267f4d095d39aa065b7b66b9ba09714244c453ea139cee4a2b31dda
- Checkpoint revision: c374e7e24b54f6cb0017d0c2e6d26823d2f2fb5d
- Model ID: Qwen3.8-Flash-Next-NVFP4-4p89
- Endpoint: http://dusty:8000/v1
- TP2, MTP3, aligned recurrent checkpoints, max length 262144, max sequences 4,
  batched-token budget 4096, utilization 0.85, FP8 KV, InstantTensor BUFFERED.
- CUDA graphs enabled, FULL_AND_PIECEWISE, capture sizes 1,2,4,8,16,24,32.
- PyNCCL over the existing f1 pair rails, NCCL_PROTO=LL,Simple. LMCache disabled.
- Head started 15:42:30 EDT; API initialized around 15:47 EDT.
- Engine-reported GPU KV capacity: 5194550 tokens at this context envelope.
  This is the hybrid-cache reported figure, not a change to max sequences.

The first non-thinking completion independently computed 17*23-58 as 333,
finish_reason=stop. The existing semantic-admission script then passed all six
cases in one run: low/medium/xhigh reasoning, non-thinking arithmetic, image
understanding, and a tool-call round trip. Responses and receipts are preserved
alongside startup logs and both container inspections.

The pair is left serving. These are startup and semantic smoke receipts only;
they do not establish long-context correctness, repeatability, R28 auto-policy
behavior, concurrency performance, or full release qualification.
