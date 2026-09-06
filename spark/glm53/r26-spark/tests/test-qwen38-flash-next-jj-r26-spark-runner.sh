#!/usr/bin/env bash
set -euo pipefail

runner=${1:-"$(dirname "$0")/../run-qwen38-flash-next-jj-r26-spark-tp2-node.sh"}
image=localhost/voipmonitor/vllm:glm53-jj-r26-spark-sm121-vllm59c9787-b12x00248b0-lmcachefe5442f-cu133-torch213-20260905-r1
test_home=$(mktemp -d)
trap 'rm -rf "${test_home}"' EXIT

render() {
  HOME="${test_home}" ROLE="$1" DRY_RUN=1 IMAGE="${image}" \
    NODE_RANK="$2" HOST_IP="$3" "${runner}"
}
require() { [[ "$1" == *"$2"* ]] || { echo "missing runner token: $2" >&2; exit 1; }; }
reject() { [[ "$1" != *"$2"* ]] || { echo "unexpected runner token: $2" >&2; exit 1; }; }

head_render=$(render head 0 10.11.1.1)
worker_render=$(render worker 1 10.11.1.2)
stop_render=$(HOME="${test_home}" ROLE=stop DRY_RUN=1 NODE_RANK=1 \
  HOST_IP=10.11.1.2 "${runner}")

for output in "${head_render}" "${worker_render}"; do
  require "${output}" "${image}"
  require "${output}" 'c374e7e24b54f6cb0017d0c2e6d26823d2f2fb5d'
  require "${output}" 'Qwen3.8-Flash-Next-NVFP4-4p89'
  require "${output}" '--nnodes 2'
  require "${output}" '--master-addr 10.11.1.1'
  require "${output}" 'TP=2'
  require "${output}" 'GPU_MEMORY_UTILIZATION=0.85'
  require "${output}" 'MAX_MODEL_LEN=262144'
  require "${output}" 'MAX_NUM_SEQS=4'
  require "${output}" 'MAX_NUM_BATCHED_TOKENS=4096'
  require "${output}" 'NUM_SPECULATIVE_TOKENS=3'
  require "${output}" 'LMCACHE_ENABLED=0'
  require "${output}" 'NCCL_IB_HCA=rocep1s0f1\,roceP2p1s0f1'
  require "${output}" 'NCCL_PROTO=LL\,Simple'
  require "${output}" 'NCCL_SOCKET_IFNAME=enp1s0f1np1\,enP2p1s0f1np1'
  require "${output}" 'GLOO_SOCKET_IFNAME=enp1s0f1np1'
  require "${output}" 'serve-qwen38-flash-next-jj-r26-spark.sh'
  reject "${output}" 'KV_CACHE_MEMORY_BYTES'
  reject "${output}" '--privileged'
  reject "${output}" 'seccomp=unconfined'
  reject "${output}" 'NCCL_MIN_NCHANNELS'
  reject "${output}" 'NCCL_MAX_NCHANNELS'
  reject "${output}" 'NCCL_LAUNCH_ORDER_IMPLICIT'
  reject "${output}" '192.168.2.'
  reject "${output}" 'ray'
done

require "${head_render}" '--node-rank 0'
require "${head_render}" 'VLLM_HOST_IP=10.11.1.1'
reject "${head_render}" '--headless'
require "${worker_render}" '--node-rank 1'
require "${worker_render}" 'VLLM_HOST_IP=10.11.1.2'
require "${worker_render}" '--headless'
require "${stop_render}" 'podman stop -t 60 qwen38-flash-next-nvfp4-jj-r26-tp2'

mtp0=$(HOME="${test_home}" ROLE=head DRY_RUN=1 IMAGE="${image}" \
  NODE_RANK=0 HOST_IP=10.11.1.1 NUM_SPECULATIVE_TOKENS=0 "${runner}")
require "${mtp0}" 'NUM_SPECULATIVE_TOKENS=0'

if HOME="${test_home}" ROLE=head DRY_RUN=1 IMAGE="${image}" \
  NODE_RANK=0 HOST_IP=10.11.1.1 TP_SIZE=4 "${runner}" >/dev/null 2>&1; then
  echo 'TP4 unexpectedly passed the TP2 runner' >&2
  exit 1
fi
if HOME="${test_home}" ROLE=head DRY_RUN=1 IMAGE="${image}" \
  NODE_RANK=0 HOST_IP=192.168.2.1 "${runner}" >/dev/null 2>&1; then
  echo 'public HOST_IP unexpectedly passed' >&2
  exit 1
fi
if [[ -e "${test_home}/.cache" ]]; then
  echo 'DRY_RUN created cache state' >&2
  exit 1
fi

echo 'Qwen3.8 Flash Next NVFP4 JJ r26 Spark TP2 runner: PASS'
