#!/usr/bin/env bash
set -euo pipefail

runner=${1:-"$(dirname "$0")/../run-glm53-flash-jj-r26-spark-tp4-node.sh"}
image=localhost/voipmonitor/vllm:glm53-jj-r26-spark-sm121-vllm59c9787-b12x00248b0-lmcachefe5442f-cu133-torch213-20260905-r1
test_home=$(mktemp -d)
trap 'rm -rf "${test_home}"' EXIT

render() {
  local role=$1 rank=$2 host_ip=$3
  HOME="${test_home}" ROLE="${role}" DRY_RUN=1 IMAGE="${image}" \
    NODE_RANK="${rank}" HOST_IP="${host_ip}" "${runner}"
}
require() {
  [[ "$1" == *"$2"* ]] || { echo "missing runner token: $2" >&2; exit 1; }
}
reject() {
  [[ "$1" != *"$2"* ]] || { echo "unexpected runner token: $2" >&2; exit 1; }
}

head_render=$(render head 0 10.11.11.1)
buddy_render=$(render worker 1 10.11.11.2)
rocky_render=$(render worker 2 10.11.11.4)
lucky_render=$(render worker 3 10.11.11.3)
stop_render=$(HOME="${test_home}" ROLE=stop DRY_RUN=1 NODE_RANK=2 \
  HOST_IP=10.11.11.4 "${runner}")

for output in "${head_render}" "${buddy_render}" "${rocky_render}" "${lucky_render}"; do
  require "${output}" "${image}"
  require "${output}" '46aaae8a82032f77100f2f03e9cc11b391df3b4d'
  require "${output}" '--nnodes 4'
  require "${output}" '--master-addr 10.11.11.1'
  require "${output}" 'TP=4'
  require "${output}" 'DCP=1'
  require "${output}" 'MAX_MODEL_LEN=1048576'
  require "${output}" 'MAX_NUM_SEQS=8'
  require "${output}" 'MAX_NUM_BATCHED_TOKENS=4096'
  require "${output}" 'MAX_CUDAGRAPH_CAPTURE_SIZE=32'
  require "${output}" 'CUDAGRAPH_CAPTURE_SIZES=1\ 2\ 4\ 8\ 12\ 16\ 24\ 32'
  require "${output}" 'PREFILL_SCHEDULE_INTERVAL=8'
  require "${output}" 'FAIRNESS_ENGINE=none'
  require "${output}" 'GLM53_KDA_DECODE_BACKEND=auto'
  require "${output}" 'GLM53_KDA_PREFILL_BACKEND=flashkda'
  require "${output}" 'SPECULATOR=mtp'
  require "${output}" 'MTP=3'
  require "${output}" 'NUM_SPECULATIVE_TOKENS=3'
  require "${output}" 'VLLM_GLM53_MTP_DRAFT_HEAD=bf16'
  require "${output}" 'LMCACHE_ENABLED=0'
  require "${output}" 'VLLM_ENABLE_PCIE_ALLREDUCE=0'
  require "${output}" 'VLLM_USE_B12X_PCIE_DMA=0'
  require "${output}" 'VLLM_ENABLE_ROCE_ALLREDUCE=1'
  require "${output}" 'VLLM_ROCE_ALLREDUCE_MAX_SIZE=2MB'
  require "${output}" 'VLLM_ROCE_ALLGATHER_MAX_SIZE=16MB'
  require "${output}" 'B12X_ROCE_SPIN_LIMIT=50000000'
  require "${output}" 'B12X_ROCE_CACHE_DIR=/opt/jovian-judgement/b12x-roce'
  require "${output}" 'B12X_ROCE_HCA=rocep1s0f0\,roceP2p1s0f0'
  require "${output}" 'B12X_ROCE_GID_INDEX=3'
  require "${output}" 'NCCL_IB_HCA=rocep1s0f0\,roceP2p1s0f0'
  require "${output}" 'NCCL_PROTO=LL\,Simple'
  require "${output}" 'NCCL_SOCKET_IFNAME=enp1s0f0np0\,enP2p1s0f0np0'
  require "${output}" 'GLOO_SOCKET_IFNAME=enp1s0f0np0'
  require "${output}" 'serve-glm53-flash-jj-r26-spark.sh'
  reject "${output}" 'KV_CACHE_MEMORY_BYTES'
  reject "${output}" '--disable-custom-all-reduce'
  reject "${output}" '--privileged'
  reject "${output}" 'seccomp=unconfined'
  reject "${output}" 'NCCL_MIN_NCHANNELS'
  reject "${output}" 'NCCL_MAX_NCHANNELS'
  reject "${output}" 'NCCL_LAUNCH_ORDER_IMPLICIT'
  reject "${output}" '10.11.1.'
  reject "${output}" '10.11.2.'
  reject "${output}" 'enp1s0f1np1'
  reject "${output}" 'rocep1s0f1'
  reject "${output}" '192.168.2.'
  reject "${output}" 'ray'
done

require "${head_render}" '--node-rank 0'
require "${head_render}" 'VLLM_HOST_IP=10.11.11.1'
reject "${head_render}" '--headless'
require "${buddy_render}" '--node-rank 1'
require "${buddy_render}" 'VLLM_HOST_IP=10.11.11.2'
require "${buddy_render}" '--headless'
require "${rocky_render}" '--node-rank 2'
require "${rocky_render}" 'VLLM_HOST_IP=10.11.11.4'
require "${rocky_render}" '--headless'
require "${lucky_render}" '--node-rank 3'
require "${lucky_render}" 'VLLM_HOST_IP=10.11.11.3'
require "${lucky_render}" '--headless'
require "${stop_render}" 'podman stop -t 60 glm53-flash-nvfp4-jj-r26-spark-tp4'

mtp0=$(
  HOME="${test_home}" ROLE=head DRY_RUN=1 IMAGE="${image}" \
    NODE_RANK=0 HOST_IP=10.11.11.1 MTP=0 \
    CUDAGRAPH_CAPTURE_SIZES='1 2 4 8' "${runner}"
)
require "${mtp0}" 'NUM_SPECULATIVE_TOKENS=0'
require "${mtp0}" 'MAX_CUDAGRAPH_CAPTURE_SIZE=8'

nvfp4_head=$(
  HOME="${test_home}" ROLE=head DRY_RUN=1 IMAGE="${image}" \
    NODE_RANK=0 HOST_IP=10.11.11.1 VLLM_GLM53_MTP_DRAFT_HEAD=nvfp4 "${runner}"
)
require "${nvfp4_head}" 'VLLM_GLM53_MTP_DRAFT_HEAD=nvfp4'

dflash=$(
  HOME="${test_home}" ROLE=head DRY_RUN=1 IMAGE="${image}" \
    NODE_RANK=0 HOST_IP=10.11.11.1 SPECULATOR=dflash2 \
    CUDAGRAPH_CAPTURE_SIZES='1 2 4 8 16 32 48 64' "${runner}"
)
require "${dflash}" 'NUM_SPECULATIVE_TOKENS=7'
require "${dflash}" 'MAX_CUDAGRAPH_CAPTURE_SIZE=64'

fairness=$(
  HOME="${test_home}" ROLE=head DRY_RUN=1 IMAGE="${image}" \
    NODE_RANK=0 HOST_IP=10.11.11.1 PREFILL_SCHEDULE_INTERVAL=1 \
    FAIRNESS_ENGINE=compute_share PREFILL_COMPUTE_SHARE=0.4 "${runner}"
)
require "${fairness}" 'FAIRNESS_ENGINE=compute_share'
require "${fairness}" 'PREFILL_COMPUTE_SHARE=0.4'

control=$(
  HOME="${test_home}" ROLE=head DRY_RUN=1 IMAGE="${image}" \
    NODE_RANK=0 HOST_IP=10.11.11.1 VLLM_ENABLE_ROCE_ALLREDUCE=0 "${runner}"
)
require "${control}" 'VLLM_ENABLE_ROCE_ALLREDUCE=0'
reject "${control}" '--disable-custom-all-reduce'

if HOME="${test_home}" ROLE=head DRY_RUN=1 IMAGE="${image}" \
  NODE_RANK=0 HOST_IP=10.11.11.1 TP_SIZE=2 "${runner}" >/dev/null 2>&1; then
  echo 'TP2 unexpectedly passed the TP4 runner' >&2
  exit 1
fi
if HOME="${test_home}" ROLE=head DRY_RUN=1 IMAGE="${image}" \
  NODE_RANK=0 HOST_IP=10.11.1.1 "${runner}" >/dev/null 2>&1; then
  echo 'back-to-back HOST_IP unexpectedly passed' >&2
  exit 1
fi
if HOME="${test_home}" ROLE=head DRY_RUN=1 IMAGE="${image}" \
  NODE_RANK=0 HOST_IP=10.11.11.1 MASTER_ADDR=192.168.2.1 \
  "${runner}" >/dev/null 2>&1; then
  echo 'public MASTER_ADDR unexpectedly passed' >&2
  exit 1
fi
if HOME="${test_home}" ROLE=head DRY_RUN=1 IMAGE="${image}" \
  NODE_RANK=0 HOST_IP=10.11.11.1 FAIRNESS_ENGINE=compute_share \
  PREFILL_COMPUTE_SHARE=0.4 "${runner}" >/dev/null 2>&1; then
  echo 'fairness with cadence 8 unexpectedly passed' >&2
  exit 1
fi
if HOME="${test_home}" ROLE=head DRY_RUN=1 IMAGE="${image}" \
  NODE_RANK=0 HOST_IP=10.11.11.1 VLLM_GLM53_MTP_DRAFT_HEAD=int4 \
  "${runner}" >/dev/null 2>&1; then
  echo 'invalid draft-head mode unexpectedly passed' >&2
  exit 1
fi
if [[ -e "${test_home}/.cache" ]]; then
  echo 'DRY_RUN created cache state' >&2
  exit 1
fi

echo 'GLM-5.3 Flash NVFP4 JJ r26 Spark TP4 runner: PASS'
