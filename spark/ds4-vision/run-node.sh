#!/usr/bin/env bash
set -euo pipefail
case ${DRY_RUN:-0} in 0|1) ;; *) exit 2;; esac
role=${ROLE:?ROLE=head or worker}
case "$role" in
  head) rank=0; expected_host=rusty; host_ip=10.11.1.1; role_args=();;
  worker) rank=1; expected_host=toby; host_ip=10.11.1.2; role_args=(--headless);;
  *) exit 2;;
esac
root=/home/jugs/git/ds4-vision
image=localhost/voipmonitor/vllm:jj-r32-spark-sm121
expected=74e53e710bef141f6f68e722582569f9c6aa388bce405ad6f1423566a2300c9c
name=ds4-vision-jj-r32-tp2
revision=6821d6ad3681a4b137b066b76094fa82ebd0a380
cache=/home/jugs/.cache/vllm-jj-ds4-vision
model=/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-Vision-Exp/snapshots/$revision
cmd=(podman run -d --pull never --name "$name"
  --device nvidia.com/gpu=all --device /dev/infiniband
  --security-opt label=disable --network host --ipc host --init
  --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=500000:500000
  -v /home/jugs/.cache/huggingface:/root/.cache/huggingface:ro
  -v "$cache:/cache:rw" -v "$cache/tmp:/container-tmp:rw"
  -v "$root/launch-in-container.sh:/opt/ds4-vision/launch-in-container.sh:ro"
  -v "$root/runtime-preflight.py:/opt/ds4-vision/runtime-preflight.py:ro"
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e TMPDIR=/container-tmp
  -e PYTHONUNBUFFERED=1 -e B12X_PRINT_COMPILE_PROGRESS=1
  -e MODEL="$model" -e MODEL_REVISION="$revision"
  -e SERVED_MODEL_NAME=DeepSeek-V4-Flash-Vision-Exp -e PORT=8000
  -e TP_SIZE=2 -e DCP_SIZE=1 -e DSPARK_TOKENS=3
  -e MAX_NUM_SEQS=4 -e MAX_MODEL_LEN=524288 -e MAX_NUM_BATCHED_TOKENS=4096
  -e MAX_CUDAGRAPH_CAPTURE_SIZE=16 -e 'CUDAGRAPH_CAPTURE_SIZES=1,2,4,8,12,16'
  -e GPU_MEMORY_UTILIZATION=0.85 -e BACKEND=b12x-a8-dglin
  -e KV_CACHE_DTYPE=fp8 -e BLOCK_SIZE=256 -e LOAD_FORMAT=instanttensor
  -e VLLM_USE_B12X_FP8_GEMM=0 -e VLLM_WORKER_MULTIPROC_METHOD=spawn
  -e CUTE_DSL_ARCH=sm_121a -e TORCH_CUDA_ARCH_LIST=12.1a
  -e VLLM_HOST_IP="$host_ip" -e NCCL_DEBUG=WARN -e NCCL_IB_DISABLE=0
  -e 'NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1' -e NCCL_IB_GID_INDEX=3
  -e NCCL_IB_TC=106 -e NCCL_IB_MERGE_NICS=0 -e NCCL_IB_SUBNET_AWARE_ROUTING=1
  -e 'NCCL_PROTO=LL,Simple' -e 'NCCL_SOCKET_IFNAME=enp1s0f1np1,enP2p1s0f1np1'
  -e GLOO_SOCKET_IFNAME=enp1s0f1np1
  --entrypoint /bin/bash "$image" /opt/ds4-vision/launch-in-container.sh
  --distributed-executor-backend mp --nnodes 2 --node-rank "$rank"
  --master-addr 10.11.1.1 --master-port 25000 "${role_args[@]}")
if [[ ${DRY_RUN:-0} == 1 ]]; then
  printf 'DRY-RUN:'; printf ' %q' "${cmd[@]}"; printf '\n'
  exit 0
fi
[[ $(hostname -s) == "$expected_host" && $(uname -m) == aarch64 ]] || exit 78
[[ $(podman image inspect "$image" --format '{{.Id}}') == "$expected" ]] || exit 78
python3 "$root/verify-model.py" --cache /home/jugs/.cache/huggingface \
  --manifest "$root/receipts/model-manifest.json"
(cd "$root" && sha256sum --check runtime-files.sha256)
if podman container exists "$name"; then
  echo 'Candidate container already exists; refusing replacement' >&2; exit 78
fi
running=$(podman ps --format '{{.Names}}') || {
  echo 'Cannot establish container state; refusing launch' >&2; exit 78
}
if [[ -n "$running" ]]; then
  echo 'Stop existing serving containers before launching the candidate' >&2; exit 78
fi
mkdir -p "$cache/tmp"
"${cmd[@]}"
