# Official NCCL 2.31.2 automatic-64 stack signature

This receipt discriminates the GLM DCP2 CKV-prefetch/indexer reproducer from NVIDIA/nccl#2353. It is a weight-free run on the idle `dusty`/`kirby` GB10 pair and did not touch either production deployment.

## Run identity

- Date: 2026-08-20
- Nodes: `dusty` rank 0, `kirby` rank 1
- Image: `infernal-invocation-r18-spark-sm121-vllmf560085-b12x75787c7-fi1ac6942-cu133-torch213-20260818`
- NCCL source: NVIDIA `v2.31.2-1`, commit `7b83616df3ae082a1f32bb74c27458bfe8153a13`
- NCCL library sha256: `1e9696ffeee4074005f99b9172c78803c5c92d10cb53f958550daaea2109aae8`
- Harness: `../microbench_glm_dcp2_ckv_prefetch.py`, receipts variant already staged as `/home/jugs/mb-dcp2-receipts.py` on both nodes
- Transport: dual 200 Gbit/s RoCE, `NCCL_PROTO=LL,Simple`
- Channel controls: unset; NCCL selected 64 channels on both communicators
- Implicit ordering: unset
- Started: 2026-08-20 09:09:42 EDT
- Stopped deliberately after stack capture: 2026-08-20 09:10:57 EDT, exit 143

The only diagnostic change was a five-line wrapper that called `prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY)` before executing the unchanged harness. Containers used the host PID namespace, `CAP_SYS_PTRACE`, and an unconfined seccomp profile. NCCL, tensor shapes, stream ordering, library loading, and transport variables were unchanged.

## Verdict

This is not the proxy-op-pool deadlock in NVIDIA/nccl#2353.

| Required #2353 probe | Rank 0 | Rank 1 |
|---|---|---|
| application in `ncclLocalOpAppend` / `uploadProxyOps` | absent | absent |
| proxy in `ncclProxyGetPostedOps` | absent | absent |

Instead, rank 1's application thread had completed enqueueing and was blocked in `cuCtxSynchronize_v2`. Rank 0 was likewise inside `libcuda` during final synchronization, although that stack did not fully unwind. Each rank had two communicator proxy-progress threads: one active in network/proxy progress and one spinning at `ncclProxyProgress` line 1003.

This matches the broader concurrent-communicator start-order/resource deadlock class in NVIDIA/nccl#592 and #1766. It does not justify porting #2353's flush-before-block patch.

## Artifacts

- `official2312-auto64-rank0.gdb.txt`: 424 lines, sha256 `99017b675b3c2b65a1fa6b2f2ea2845d95fb6eb25f6c702f693666609bee25b2`
- `official2312-auto64-rank1.gdb.txt`: 425 lines, sha256 `90c43951cf5bc40c71fbe26d200952b09961df1347c31405584ccaad4880a0fc`

The earlier 300-second hang and validated implicit-ordering/exact-four controls remain in `../nccl-6arm-results/receipts-20260819/`.

## Combined ordering and resource contract

The previously untested combination also passed against the same official library:

- `NCCL_LAUNCH_ORDER_IMPLICIT=1`
- `NCCL_MIN_CTAS=4`
- `NCCL_MAX_CTAS=4`

Both ranks exited zero and all three patterned-output cases validated. Rank 0 measured 2758.96, 3357.00, and 1478.11 microseconds/layer for the 8K/3072, 32K/3072, and 64K-tail/222 cases. This is a correctness/admission result from one run, not a performance qualification; serving throughput and the 121 GiB UMA footprint still require their own staging window.

The only runtime warnings were the already known `ibv_query_port_speed` `Protocol not supported` notices. Both rails carried the collectives successfully.

The exact deployed variable form was then tested separately, retaining `NCCL_MIN_NCHANNELS=4` and `NCCL_MAX_NCHANNELS=4` rather than substituting CTA variables. With `NCCL_LAUNCH_ORDER_IMPLICIT=1`, both ranks again exited zero and all outputs validated. Rank 0 measured 2763.42, 3204.95, and 1359.56 microseconds/layer. This is the candidate runner contract for serving qualification.
