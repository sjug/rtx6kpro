#!/usr/bin/env bash
# NCCL tuning matrix driver. Runs ON dusty (head); orchestrates kirby (worker)
# over ssh. Phase-1 grid: image(nccl) x proto x channel-pin, latency-focused
# sweep per cell. Emits ~/nccl-matrix/<tag>.json per cell + summary at end.
set -euo pipefail

R33=localhost/voipmonitor/vllm:gilded-gnosis-v20-r33-spark-sm121-vllm28e8eaf-b12x06db0f4-fi1ac6942-cu132-20260808
II=localhost/voipmonitor/vllm:infernal-invocation-r10-spark-sm121-vllmd650d5c-b12x5d648d9-fi1ac6942-cu133-torch213-20260813
OUT=~/nccl-matrix
mkdir -p "$OUT"
rm -f "$OUT/MATRIX-DONE" "$OUT/MATRIX-FAIL"

cat > /tmp/nccl-matrix-sweep.py <<'PY'
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
        lat_band = size <= 256*KB
        reps = 100 if lat_band else 15
        for _ in range(10):
            dist.all_reduce(x) if op == "all_reduce" else dist.all_gather(out, x)
        torch.cuda.synchronize()
        best = None
        for _ in range(5 if lat_band else 2):
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
    json.dump({"nccl": torch.cuda.nccl.version(),
               "env": {k: os.environ.get(k) for k in
                       ("NCCL_PROTO", "NCCL_MAX_NCHANNELS", "NCCL_MIN_NCHANNELS")},
               "results": results}, open(os.environ["AB_OUT"], "w"), indent=1)
dist.destroy_process_group()
PY
scp -q /tmp/nccl-matrix-sweep.py 10.11.1.2:/tmp/nccl-matrix-sweep.py

run_cell() {
  local img=$1 tag=$2 proto=$3 ch=$4
  local envs=(-e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1
              -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_TC=106
              -e NCCL_SOCKET_IFNAME=enp1s0f1np1,enP2p1s0f1np1
              -e GLOO_SOCKET_IFNAME=enp1s0f1np1)
  [ "$proto" != default ] && envs+=(-e "NCCL_PROTO=$proto")
  if [ "$ch" != default ]; then
    envs+=(-e "NCCL_MAX_NCHANNELS=$ch" -e "NCCL_MIN_NCHANNELS=$ch")
  fi
  local envstr=""
  for e in "${envs[@]}"; do envstr+=" $e"; done

  ssh 10.11.1.2 "podman run --rm --name nccl-mx-w --device nvidia.com/gpu=all \
    --device /dev/infiniband --security-opt label=disable --network host \
    --ipc host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /tmp/nccl-matrix-sweep.py:/tmp/s.py:ro $envstr \
    --entrypoint torchrun $img --nnodes 2 --node-rank 1 \
    --master-addr 10.11.1.1 --master-port 25200 --nproc-per-node 1 /tmp/s.py \
    > /tmp/nccl-mx-w.log 2>&1" &
  local wpid=$!
  podman run --rm --name nccl-mx-h --device nvidia.com/gpu=all \
    --device /dev/infiniband --security-opt label=disable --network host \
    --ipc host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /tmp/nccl-matrix-sweep.py:/tmp/s.py:ro \
    -v "$OUT:/out" -e AB_OUT="/out/${tag}.json" ${envstr} \
    --entrypoint torchrun "$img" --nnodes 2 --node-rank 0 \
    --master-addr 10.11.1.1 --master-port 25200 --nproc-per-node 1 /tmp/s.py \
    > "$OUT/${tag}.log" 2>&1
  wait "$wpid" || true
  test -f "$OUT/${tag}.json" || { echo "cell ${tag} FAILED" >&2; return 1; }
  echo "cell done: ${tag}"
}

fail=0
for pair in "r33:$R33" "ii:$II"; do
  name="${pair%%:*}"; img="${pair#*:}"
  for proto in default "LL,Simple" "LL128,Simple" Simple; do
    for ch in default 4 2 1; do
      ptag=$(tr ',' '-' <<<"$proto")
      run_cell "$img" "${name}-p${ptag}-c${ch}" "$proto" "$ch" || fail=1
    done
  done
done

if [ "$fail" = 0 ]; then touch "$OUT/MATRIX-DONE"; else touch "$OUT/MATRIX-FAIL"; fi
echo "matrix complete (fail=$fail)"
