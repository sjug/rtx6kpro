#!/usr/bin/env bash
# TP4 NCCL grid on the GLM cluster (runs ON sparky, rank0; workers via ssh).
# Rails and env mirror run-glm52-exl3-tp4-node.sh. Cells: image x proto x ch.
set -euo pipefail
exec > ~/nccl-tp4/grid.log 2>&1

R33=localhost/voipmonitor/vllm:gilded-gnosis-v20-r33-spark-sm121-vllm28e8eaf-b12x06db0f4-fi1ac6942-cu132-20260808
II=localhost/voipmonitor/vllm:infernal-invocation-r10-spark-sm121-vllmd650d5c-b12x5d648d9-fi1ac6942-cu133-torch213-20260813
OUT=~/nccl-tp4
mkdir -p "$OUT"; rm -f "$OUT"/GRID-DONE "$OUT"/GRID-FAIL

declare -A RANKS=( [buddy]=1 [rocky]=2 [lucky]=3 )

cat > /tmp/tp4-sweep.py <<'PY'
import json, os, time, torch, torch.distributed as dist
dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
dev = torch.device("cuda:0")
KB = 1024
sizes = [4*KB, 16*KB, 32*KB, 64*KB, 128*KB, 256*KB, 1024*KB, 16384*KB]
results = []
for op in ("all_reduce", "all_gather"):
    for size in sizes:
        x = torch.randn(size // 2, dtype=torch.float16, device=dev)
        out = [torch.empty_like(x) for _ in range(world)] if op == "all_gather" else None
        lat = size <= 256*KB
        reps = 100 if lat else 15
        for _ in range(10):
            dist.all_reduce(x) if op == "all_reduce" else dist.all_gather(out, x)
        torch.cuda.synchronize()
        best = None
        for _ in range(5 if lat else 2):
            t0 = time.perf_counter()
            for _ in range(reps):
                dist.all_reduce(x) if op == "all_reduce" else dist.all_gather(out, x)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / reps
            best = dt if best is None or dt < best else best
        if rank == 0:
            results.append({"op": op, "size": size, "us": round(best*1e6, 2)})
            print(f"{op} {size>>10}KB: {best*1e6:.2f}us", flush=True)
if rank == 0:
    json.dump({"world": world, "nccl": torch.cuda.nccl.version(),
               "env": {k: os.environ.get(k) for k in ("NCCL_PROTO","NCCL_MAX_NCHANNELS")},
               "results": results}, open(os.environ["AB_OUT"], "w"), indent=1)
dist.destroy_process_group()
PY
for w in buddy rocky lucky; do scp -q /tmp/tp4-sweep.py "$w:/tmp/tp4-sweep.py"; done

run_cell() {
  local img=$1 tag=$2 proto=$3 ch=$4
  local envs="-e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_TC=106 -e NCCL_SOCKET_IFNAME=enp1s0f0np0,enP2p1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0"
  [ "$proto" != default ] && envs+=" -e NCCL_PROTO=$proto"
  [ "$ch" != default ] && envs+=" -e NCCL_MAX_NCHANNELS=$ch -e NCCL_MIN_NCHANNELS=$ch"

  local wpids=()
  for w in buddy rocky lucky; do
    ssh "$w" "podman run --rm --name tp4grid --device nvidia.com/gpu=all \
      --device /dev/infiniband --security-opt label=disable --network host \
      --ipc host --ulimit memlock=-1 --ulimit stack=67108864 \
      -v /tmp/tp4-sweep.py:/tmp/s.py:ro $envs \
      --entrypoint torchrun $img --nnodes 4 --node-rank ${RANKS[$w]} \
      --master-addr 10.11.11.1 --master-port 25300 --nproc-per-node 1 /tmp/s.py \
      > /tmp/tp4grid.log 2>&1" &
    wpids+=($!)
  done
  podman run --rm --name tp4grid --device nvidia.com/gpu=all \
    --device /dev/infiniband --security-opt label=disable --network host \
    --ipc host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /tmp/tp4-sweep.py:/tmp/s.py:ro -v "$OUT:/out" -e AB_OUT="/out/${tag}.json" ${envs} \
    --entrypoint torchrun "$img" --nnodes 4 --node-rank 0 \
    --master-addr 10.11.11.1 --master-port 25300 --nproc-per-node 1 /tmp/s.py \
    > "$OUT/${tag}.log" 2>&1 || true
  for p in "${wpids[@]}"; do wait "$p" || true; done
  test -f "$OUT/${tag}.json" || { echo "cell ${tag} FAILED"; return 1; }
  echo "cell done: ${tag}"
}

fail=0
for pair in "r33:$R33" "ii:$II"; do
  name="${pair%%:*}"; img="${pair#*:}"
  for proto in default "LL,Simple"; do
    for ch in default 4 2; do
      ptag=$(tr ',' '-' <<<"$proto")
      run_cell "$img" "tp4-${name}-p${ptag}-c${ch}" "$proto" "$ch" || fail=1
    done
  done
done
if [ "$fail" = 0 ]; then touch "$OUT/GRID-DONE"; else touch "$OUT/GRID-FAIL"; fi
echo GRID-COMPLETE
