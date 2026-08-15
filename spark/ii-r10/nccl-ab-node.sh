#!/usr/bin/env bash
# Cross-node NCCL A/B for the II r10 port: run the same collective sweep in
# the r33-spark image (NCCL 2.30.4) and the ii-r10 image (NCCL 2.31.2, Turin
# branch fb6f409 built for sm_121) over the dusty/kirby pair rails, before any
# serving qualification. Gate: 2.31.2 busbw within noise of 2.30.4 at every
# size; latency-band sizes (<=64KB) get 3 repeats.
#
# Usage: on kirby: ROLE=worker IMAGE=<image> ./nccl-ab-node.sh
#        on dusty: ROLE=head   IMAGE=<image> ./nccl-ab-node.sh
# Run once per image; results land in ~/nccl-ab-<tag>.json on the head node.
set -euo pipefail

ROLE=${ROLE:?set ROLE=head|worker}
IMAGE=${IMAGE:?set IMAGE}
TAG=${TAG:-$(sed -E 's/.*(r33-spark|ii-r10|infernal[^:]*)-.*/\1/;s/[^A-Za-z0-9._-]/-/g' <<<"$IMAGE" | cut -c1-40)}
MASTER_ADDR=${MASTER_ADDR:-10.11.1.1}
MASTER_PORT=${MASTER_PORT:-25100}
case "$ROLE" in
  head)   NODE_RANK=0 ;;
  worker) NODE_RANK=1 ;;
  *) echo "ROLE must be head or worker" >&2; exit 2 ;;
esac

cat > /tmp/nccl-ab-sweep.py <<'PY'
import json, os, time, torch, torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank(); world = dist.get_world_size()
dev = torch.device("cuda:0")
results = []
sizes = [4*2**10, 16*2**10, 64*2**10, 256*2**10, 2**20, 16*2**20, 64*2**20, 256*2**20]
for op in ("all_reduce", "all_gather"):
    for size in sizes:
        n = size // 2
        x = torch.randn(n, dtype=torch.float16, device=dev)
        if op == "all_gather":
            out = [torch.empty_like(x) for _ in range(world)]
        reps = 60 if size <= 64*2**10 else 15
        for _ in range(5):
            dist.all_reduce(x) if op == "all_reduce" else dist.all_gather(out, x)
        torch.cuda.synchronize()
        best = None
        for _ in range(3 if size <= 64*2**10 else 1):
            t0 = time.perf_counter()
            for _ in range(reps):
                dist.all_reduce(x) if op == "all_reduce" else dist.all_gather(out, x)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / reps
            best = dt if best is None or dt < best else best
        factor = 2*(world-1)/world if op == "all_reduce" else (world-1)/world
        busbw = size*factor/best/1e9
        if rank == 0:
            results.append({"op": op, "size": size, "us": round(best*1e6, 1),
                            "busbw_GBps": round(busbw, 2)})
            print(f"{op} {size>>10}KB: {best*1e6:.1f}us busbw={busbw:.2f}GB/s", flush=True)
if rank == 0:
    with open(os.environ.get("AB_OUT", "/tmp/nccl-ab.json"), "w") as f:
        json.dump({"nccl": torch.cuda.nccl.version(), "results": results}, f, indent=1)
dist.destroy_process_group()
PY

podman run --rm --name "nccl-ab-$ROLE" \
  --device nvidia.com/gpu=all \
  --device /dev/infiniband \
  --security-opt label=disable \
  --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /tmp/nccl-ab-sweep.py:/tmp/nccl-ab-sweep.py:ro \
  -v "$HOME:/host-home" \
  -e AB_OUT="/host-home/nccl-ab-${TAG}.json" \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 \
  -e NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}" \
  -e NCCL_IB_TC=106 \
  -e NCCL_SOCKET_IFNAME=enp1s0f1np1,enP2p1s0f1np1 \
  -e GLOO_SOCKET_IFNAME=enp1s0f1np1 \
  --entrypoint torchrun "$IMAGE" \
  --nnodes 2 --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
  --nproc-per-node 1 /tmp/nccl-ab-sweep.py
