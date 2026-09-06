#!/usr/bin/env bash
set -euo pipefail

# Build the JJ r26 GLM and Qwen runtime for SM121 from the qualified r22 Spark
# image. R26 reuses the unchanged SM121 FlashKDA object, refreshes the exact
# vLLM and B12X source trees, and rebuilds R26 LMCache for 12.1a.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_root}"

container_engine=${CONTAINER_ENGINE:-podman}
release_date=${RELEASE_DATE:-20260905}
revision=${REVISION:-r1}
allow_dirty_build=${ALLOW_DIRTY_BUILD:-0}
skip_gpu_check=${SKIP_GPU_CHECK:-0}
push_image=${PUSH_IMAGE:-0}
release_status=research-only
lock=source.lock.json

case "${container_engine}" in podman) ;; *) echo 'CONTAINER_ENGINE must be podman.' >&2; exit 2 ;; esac
case "${revision}" in r[1-9]|r[1-9][0-9]*) ;; *) echo 'REVISION must use rN form.' >&2; exit 2 ;; esac
[[ "${release_date}" =~ ^[0-9]{8}$ ]] || { echo 'RELEASE_DATE must use YYYYMMDD.' >&2; exit 2; }
for value in allow_dirty_build skip_gpu_check push_image; do
  case "${!value}" in 0|1) ;; *) echo "${value} must be 0 or 1." >&2; exit 2 ;; esac
done
if ((allow_dirty_build == 1 && push_image == 1)); then
  echo 'PUSH_IMAGE=1 is forbidden when ALLOW_DIRTY_BUILD=1.' >&2
  exit 2
fi

test -f "${lock}"
vllm_refresh=$(jq -er '.vllm.refresh_patch' "${lock}")
vllm_paths=$(jq -er '.vllm.changed_paths' "${lock}")
b12x_refresh=$(jq -er '.b12x.refresh_patch' "${lock}")
b12x_paths=$(jq -er '.b12x.changed_paths' "${lock}")
for path in "${vllm_refresh}" "${vllm_paths}" "${b12x_refresh}" "${b12x_paths}"; do
  test -f "${path}" || { echo "Missing source artifact: ${path}" >&2; exit 1; }
done
verify_sha() {
  local expected=$1 path=$2
  printf '%s  %s\n' "${expected}" "${path}" | sha256sum -c - >/dev/null
}
verify_sha "$(jq -er '.vllm.refresh_patch_sha256' "${lock}")" "${vllm_refresh}"
verify_sha "$(jq -er '.vllm.changed_paths_sha256' "${lock}")" "${vllm_paths}"
verify_sha "$(jq -er '.b12x.refresh_patch_sha256' "${lock}")" "${b12x_refresh}"
verify_sha "$(jq -er '.b12x.changed_paths_sha256' "${lock}")" "${b12x_paths}"

base_image_locked=$(jq -er '.native_artifact_base.image' "${lock}")
base_image=${BASE_IMAGE:-${base_image_locked}}
base_image_id=$(jq -er '.native_artifact_base.image_id' "${lock}")
base_vllm_tree=$(jq -er '.native_artifact_base.vllm_spark_tree' "${lock}")
base_vllm_package=$(jq -er '.native_artifact_base.vllm_package_tree' "${lock}")
base_b12x_tree=$(jq -er '.native_artifact_base.b12x_tree' "${lock}")
base_b12x_package=$(jq -er '.native_artifact_base.b12x_package_tree' "${lock}")

vllm_commit=$(jq -er '.vllm.r26_integration_commit' "${lock}")
vllm_tree=$(jq -er '.vllm.r26_integration_tree' "${lock}")
vllm_r26_package=$(jq -er '.vllm.r26_package_tree' "${lock}")
vllm_package=$(jq -er '.vllm.spark_package_tree' "${lock}")
vllm_spark_tree=$(jq -er '.vllm.spark_tree' "${lock}")
vllm_refresh_sha=$(jq -er '.vllm.refresh_patch_sha256' "${lock}")
vllm_paths_sha=$(jq -er '.vllm.changed_paths_sha256' "${lock}")
flashkda_commit=$(jq -er '.vllm.flashkda_commit' "${lock}")
flashkda_sha=$(jq -er '.native_artifact_base.flashkda_sha256' "${lock}")

b12x_r26_commit=$(jq -er '.b12x.r26_integration_commit' "${lock}")
b12x_tree=$(jq -er '.b12x.r26_integration_tree' "${lock}")
b12x_package=$(jq -er '.b12x.r26_package_tree' "${lock}")
b12x_refresh_sha=$(jq -er '.b12x.refresh_patch_sha256' "${lock}")
b12x_paths_sha=$(jq -er '.b12x.changed_paths_sha256' "${lock}")
b12x_roce_api=$(jq -er '.b12x.rocenante_api_version' "${lock}")
b12x_proxy_abi=$(jq -er '.b12x.rocenante_proxy_abi' "${lock}")
b12x_roce_cache=$(jq -er '.b12x.rocenante_proxy_cache_dir' "${lock}")

lmcache_repo=$(jq -er '.lmcache.repository' "${lock}")
lmcache_r26=$(jq -er '.lmcache.r26_integration_commit' "${lock}")
lmcache_acquisition=$(jq -er '.lmcache.acquisition_commit' "${lock}")
lmcache_tree=$(jq -er '.lmcache.integration_tree' "${lock}")
lmcache_package=$(jq -er '.lmcache.package_tree' "${lock}")
lmcache_version=$(jq -er '.lmcache.version' "${lock}")

test "${base_image}" = "${base_image_locked}"
test "$(jq -er '.published_r26.digest' "${lock}")" = sha256:d0592ea9d73cac5aadb151a58bbb43cf7aff03829d46bb4f4ba7396aaef67c68
test "$(jq -er '.published_r26.source_lock_sha256' "${lock}")" = bced2847d40e650a145d7060d614bcf77de7df5d8a1d2db1a64c7bcaa338217a
test "${base_image_id}" = 907c1265f308ea87c77d98e0b787455af228a5dbdfa228b1c7653a48d3433e0a
test "${base_vllm_tree}" = 997f08f583f1439363939f592c54980d584bc4c7
test "${base_vllm_package}" = 4fbb1c257ac59e5e68450655ad4061d2c8a05e5c
test "${base_b12x_tree}" = 8ad308af020b9d176f7d1b3767c5404acf384ef3
test "${base_b12x_package}" = e1edb6d11c9b760d4a0d3fdf05f8b6646479d551
test "${vllm_commit}" = 7f53b30481b4110f293521d09f7af18f9e8f9d9e
test "${vllm_tree}" = c861b31da5cf527d96b5772a6709450b23c93a6f
test "${vllm_r26_package}" = 3d98f6ebfaabf06309c9a5d3a7fbd469747aa5e1
test "${vllm_package}" = 59c9787400c85d5d4547ef898c5dee41804e7088
test "${vllm_spark_tree}" = d4571e5ba189fcc05534ecae88dee763ffef4e35
test "${flashkda_commit}" = 3b225bf26bb8e218928a1fe14751cb48cf31d11b
test "${flashkda_sha}" = 484f88deea08f7e07fec6d1c275560cccbeeeb9928fcb047de0b25b4135f3116
test "${b12x_r26_commit}" = 60dbc57098a3abff60cbf6048bcf8784c117feaf
test "${b12x_tree}" = c20b6aab67ed791cc226ee91de23f7c45509f436
test "${b12x_package}" = 00248b09689830e55d8fbce8ee630600b63ef663
test "${b12x_roce_api}" = 1
test "${b12x_proxy_abi}" = 3
test "${b12x_roce_cache}" = /opt/jovian-judgement/b12x-roce
test "${lmcache_r26}" = 822330db1160d2847538d53f9e6f4eeee2404f4d
test "${lmcache_acquisition}" = 63919a2c6c310f9b34de7049b9b28b77fab13ca0
test "${lmcache_tree}" = 008ac3e09ae5917aa0849147480d7bd5b9f8b37a
test "${lmcache_package}" = fe5442fbf258accaa7f26d2bbb00d8b7b5c349ca
test "$(wc -l < "${vllm_paths}")" = "$(jq -er '.vllm.changed_paths_count' "${lock}")"
test "$(wc -l < "${b12x_paths}")" = "$(jq -er '.b12x.changed_paths_count' "${lock}")"
if rg -n -v '\.(py|sh|cmake|md)$' "${vllm_paths}"; then
  echo 'The r26 vLLM refresh contains an unclassified path.' >&2
  exit 1
fi
if rg -n -v '\.(py|c|md|json|json\.gz|txt|toml|sh)$' "${b12x_paths}"; then
  echo 'The r26 B12X refresh contains an unclassified path.' >&2
  exit 1
fi

source_lock_sha256=$(sha256sum "${lock}" | cut -d' ' -f1)
recipe_sha256="$({
  sha256sum Dockerfile.glm53-jj-r26-spark-sm121 \
    build-glm53-jj-r26-spark-sm121.sh \
    launchers/serve-glm53-flash-jj-r26-spark.sh \
    launchers/serve-qwen38-flash-next-jj-r26-spark.sh \
    run-glm53-flash-jj-r26-spark-tp4-node.sh \
    run-qwen38-flash-next-jj-r26-spark-tp2-node.sh \
    tests/verify_jj_r26_runtime.py \
    tests/verify_glm53_nvfp4_draft_head_sm121.py \
    tests/test-glm53-flash-jj-r26-spark-tp4-runner.sh \
    tests/test-qwen38-flash-next-jj-r26-spark-runner.sh \
    "${lock}" "${vllm_refresh}" "${vllm_paths}" \
    "${b12x_refresh}" "${b12x_paths}"
} | sha256sum | cut -d' ' -f1)"

worktree_dirty=0
if git -C "${repo_root}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  docker_commit=${DOCKER_COMMIT:-$(git -C "${repo_root}" rev-parse HEAD)}
  [[ -n "$(git -C "${repo_root}" status --porcelain --untracked-files=all -- .)" ]] && worktree_dirty=1
else
  docker_commit=${DOCKER_COMMIT:?DOCKER_COMMIT is required outside a Git worktree}
  worktree_dirty=1
fi
if ((worktree_dirty == 1 && allow_dirty_build == 0)); then
  echo 'Commit the r26 recipe or set ALLOW_DIRTY_BUILD=1 with an explicit dev image.' >&2
  exit 1
fi

vllm_version=${VLLM_PACKAGE_VERSION:-0.26.1rc0+glm53.jj.r26.spark.${revision}.vllm${vllm_package:0:7}.b12x${b12x_package:0:7}}
cache_fingerprint=cu133-torch213-jj-r26-sm121-vllm${vllm_package:0:10}-b12x${b12x_package:0:10}-flashkda${flashkda_commit:0:10}
image=${IMAGE:-localhost/voipmonitor/vllm:glm53-jj-r26-spark-sm121-vllm${vllm_package:0:7}-b12x${b12x_package:0:7}-lmcache${lmcache_package:0:7}-cu133-torch213-${release_date}-${revision}}
if ((worktree_dirty == 1)); then
  [[ -v IMAGE ]] || { echo 'A dirty build requires an explicit IMAGE.' >&2; exit 2; }
fi

if [[ "${PRINT_RELEASE_CONFIG:-0}" == 1 ]]; then
  printf 'base=%s\nbase_id=%s\nimage=%s\nstatus=%s\n' "${base_image}" "${base_image_id}" "${image}" "${release_status}"
  printf 'vllm_commit=%s\nvllm_tree=%s\nvllm_spark_tree=%s\nvllm_package=%s\n' "${vllm_commit}" "${vllm_tree}" "${vllm_spark_tree}" "${vllm_package}"
  printf 'flashkda_commit=%s\nflashkda_sha256=%s\n' "${flashkda_commit}" "${flashkda_sha}"
  printf 'b12x_r26=%s\nb12x_tree=%s\nb12x_package=%s\n' "${b12x_r26_commit}" "${b12x_tree}" "${b12x_package}"
  printf 'lmcache_r26=%s\nlmcache_acquisition=%s\nlmcache_tree=%s\nlmcache_default=disabled\n' "${lmcache_r26}" "${lmcache_acquisition}" "${lmcache_tree}"
  printf 'cache_fingerprint=%s\nrecipe_sha256=%s\nsource_lock_sha256=%s\n' "${cache_fingerprint}" "${recipe_sha256}" "${source_lock_sha256}"
  exit 0
fi

"${repo_root}/tests/test-glm53-flash-jj-r26-spark-tp4-runner.sh" \
  "${repo_root}/run-glm53-flash-jj-r26-spark-tp4-node.sh"
"${repo_root}/tests/test-qwen38-flash-next-jj-r26-spark-runner.sh" \
  "${repo_root}/run-qwen38-flash-next-jj-r26-spark-tp2-node.sh"
if ((worktree_dirty == 0)); then
  runner_home=$(mktemp -d)
  trap 'rm -rf "${runner_home}"' EXIT
  runner_default=$(HOME="${runner_home}" ROLE=head DRY_RUN=1 NODE_RANK=0 \
    HOST_IP=10.11.11.1 "${repo_root}/run-glm53-flash-jj-r26-spark-tp4-node.sh")
  grep -Fq -- "${image}" <<<"${runner_default}" || {
    echo 'The unoverridden runner does not point at the image being built.' >&2
    exit 1
  }
  qwen_runner_default=$(HOME="${runner_home}" ROLE=head DRY_RUN=1 NODE_RANK=0 \
    HOST_IP=10.11.1.1 "${repo_root}/run-qwen38-flash-next-jj-r26-spark-tp2-node.sh")
  grep -Fq -- "${image}" <<<"${qwen_runner_default}" || {
    echo 'The unoverridden Qwen runner does not point at the image being built.' >&2
    exit 1
  }
  rm -rf "${runner_home}"
  trap - EXIT
fi
test "$(uname -m)" = aarch64 || { echo 'JJ r26 Spark builds require aarch64.' >&2; exit 78; }
command -v "${container_engine}" >/dev/null
"${container_engine}" image exists "${base_image}" || { echo "Missing qualified r22 Spark base image: ${base_image}" >&2; exit 78; }
actual_base_id=$("${container_engine}" image inspect "${base_image}" --format '{{.Id}}')
test "${actual_base_id#sha256:}" = "${base_image_id}"

"${container_engine}" build --format docker --pull=false \
  --build-arg "BASE_IMAGE=${base_image}" \
  --build-arg "BASE_IMAGE_ID=${base_image_id}" \
  --build-arg "BASE_VLLM_SPARK_TREE=${base_vllm_tree}" \
  --build-arg "BASE_VLLM_PACKAGE_TREE=${base_vllm_package}" \
  --build-arg "BASE_B12X_TREE=${base_b12x_tree}" \
  --build-arg "BASE_B12X_PACKAGE_TREE=${base_b12x_package}" \
  --build-arg "VLLM_R26_COMMIT=${vllm_commit}" \
  --build-arg "VLLM_R26_TREE=${vllm_tree}" \
  --build-arg "VLLM_R26_PACKAGE_TREE=${vllm_r26_package}" \
  --build-arg "VLLM_PACKAGE_TREE=${vllm_package}" \
  --build-arg "VLLM_SPARK_TREE=${vllm_spark_tree}" \
  --build-arg "VLLM_REFRESH_PATCH_SHA256=${vllm_refresh_sha}" \
  --build-arg "VLLM_CHANGED_PATHS_SHA256=${vllm_paths_sha}" \
  --build-arg "VLLM_PACKAGE_VERSION=${vllm_version}" \
  --build-arg "FLASHKDA_COMMIT=${flashkda_commit}" \
  --build-arg "FLASHKDA_SHA256=${flashkda_sha}" \
  --build-arg "B12X_R26_COMMIT=${b12x_r26_commit}" \
  --build-arg "B12X_TREE=${b12x_tree}" \
  --build-arg "B12X_PACKAGE_TREE=${b12x_package}" \
  --build-arg "B12X_REFRESH_PATCH_SHA256=${b12x_refresh_sha}" \
  --build-arg "B12X_CHANGED_PATHS_SHA256=${b12x_paths_sha}" \
  --build-arg "B12X_ROCE_API_VERSION=${b12x_roce_api}" \
  --build-arg "B12X_ROCE_PROXY_ABI=${b12x_proxy_abi}" \
  --build-arg "LMCACHE_REPO=${lmcache_repo}" \
  --build-arg "LMCACHE_R26_COMMIT=${lmcache_r26}" \
  --build-arg "LMCACHE_ACQUISITION_COMMIT=${lmcache_acquisition}" \
  --build-arg "LMCACHE_TREE=${lmcache_tree}" \
  --build-arg "LMCACHE_PACKAGE_TREE=${lmcache_package}" \
  --build-arg "LMCACHE_VERSION=${lmcache_version}" \
  --build-arg "SOURCE_LOCK_SHA256=${source_lock_sha256}" \
  --build-arg 'RELEASE_NAME=glm53-jj-r26-spark-sm121' \
  --build-arg "RELEASE_DATE=${release_date}" \
  --build-arg "DOCKER_COMMIT=${docker_commit}" \
  --build-arg "RECIPE_SHA256=${recipe_sha256}" \
  --build-arg "RELEASE_STATUS=${release_status}" \
  --build-arg "CACHE_FINGERPRINT=${cache_fingerprint}" \
  --file Dockerfile.glm53-jj-r26-spark-sm121 \
  --tag "${image}" .

labels=$("${container_engine}" image inspect "${image}" --format '{{json .Config.Labels}}')
assert_label() {
  local key=$1 expected=$2
  jq -e --arg key "${key}" --arg expected "${expected}" '.[$key] == $expected' <<<"${labels}" >/dev/null || {
    echo "Label ${key} mismatch." >&2
    exit 1
  }
}
assert_label local-inference.status "${release_status}"
assert_label local-inference.scope glm53-qwen38-r26-spark-sm121-qualification
assert_label local-inference.runtime.native-artifact-base-id "${base_image_id}"
assert_label local-inference.vllm.r26.commit "${vllm_commit}"
assert_label local-inference.vllm.integration.tree "${vllm_tree}"
assert_label local-inference.vllm.r26.package.tree "${vllm_r26_package}"
assert_label local-inference.vllm.spark-overlay.tree "${vllm_spark_tree}"
assert_label local-inference.vllm.package.tree "${vllm_package}"
assert_label local-inference.flashkda.commit "${flashkda_commit}"
assert_label local-inference.flashkda.sha256 "${flashkda_sha}"
assert_label local-inference.flashkda.arch 12.1a
assert_label local-inference.b12x.r26.commit "${b12x_r26_commit}"
assert_label local-inference.b12x.integration.tree "${b12x_tree}"
assert_label local-inference.b12x.package.tree "${b12x_package}"
assert_label local-inference.b12x.rocenante.proxy-abi "${b12x_proxy_abi}"
assert_label local-inference.lmcache.r26.commit "${lmcache_r26}"
assert_label local-inference.lmcache.acquisition-commit "${lmcache_acquisition}"
assert_label local-inference.lmcache.integration.tree "${lmcache_tree}"
assert_label local-inference.lmcache.package.tree "${lmcache_package}"
assert_label local-inference.lmcache.default disabled

glm_launch=$("${container_engine}" run --rm -e DRY_RUN=1 "${image}")
grep -Fq -- '--revision 46aaae8a82032f77100f2f03e9cc11b391df3b4d' <<<"${glm_launch}"
grep -Fq -- 'num_speculative_tokens\":3' <<<"${glm_launch}"
grep -Fq -- 'kda_prefill_backend\":\"flashkda' <<<"${glm_launch}"
grep -Fq -- '--cudagraph-capture-sizes 1 2 4 8 12 16 24 32' <<<"${glm_launch}"
grep -Fq -- 'VLLM_GLM53_MTP_DRAFT_HEAD=bf16' <<<"${glm_launch}"
glm_nvfp4_launch=$("${container_engine}" run --rm -e DRY_RUN=1 \
  -e VLLM_GLM53_MTP_DRAFT_HEAD=nvfp4 "${image}")
grep -Fq -- 'VLLM_GLM53_MTP_DRAFT_HEAD=nvfp4' <<<"${glm_nvfp4_launch}"
if grep -Fq -- '--disable-custom-all-reduce' <<<"${glm_launch}"; then
  echo 'The GLM launcher unexpectedly disables custom all-reduce.' >&2
  exit 1
fi
qwen_launch=$("${container_engine}" run --rm \
  --entrypoint /usr/local/bin/serve-qwen38-flash-next-jj-r26-spark.sh \
  -e DRY_RUN=1 "${image}")
grep -Fq -- '--revision c374e7e24b54f6cb0017d0c2e6d26823d2f2fb5d' <<<"${qwen_launch}"
grep -Fq -- '--max-model-len 262144' <<<"${qwen_launch}"
grep -Fq -- '--max-num-seqs 4' <<<"${qwen_launch}"
grep -Fq -- '--block-size 16' <<<"${qwen_launch}"
grep -Fq -- 'num_speculative_tokens\":3' <<<"${qwen_launch}"
grep -Fq -- '--disable-custom-all-reduce' <<<"${qwen_launch}"
for setting in \
  VLLM_MXFP8_LM_HEAD=1 \
  VLLM_LM_HEAD_A16=1 \
  VLLM_MTP_NVFP4_LM_HEAD=1 \
  VLLM_QWEN3_8_FLASH_NEXT_OVERLAP=1 \
  VLLM_QWEN3_8_FLASH_NEXT_MTP_COMPACT=1 \
  VLLM_GDN_SPEC_DECODE_METADATA_FASTPATH=1; do
  grep -Fq -- "${setting}" <<<"${qwen_launch}"
done
if "${container_engine}" run --rm -e DRY_RUN=1 -e LMCACHE_ENABLED=1 "${image}"; then
  echo 'LMCache unexpectedly enabled before Spark qualification.' >&2
  exit 1
fi

set +e
hollow_output=$("${container_engine}" run --rm -e PYTHONPATH=/tmp "${image}" --help 2>&1)
hollow_status=$?
set -e
if [[ "${hollow_status}" != 78 || "${hollow_output}" != *'runtime preflight failed'* ]]; then
  printf 'Hollow-package preflight failed open (status %s).\n%s\n' "${hollow_status}" "${hollow_output}" >&2
  exit 1
fi
set +e
qwen_hollow_output=$("${container_engine}" run --rm -e PYTHONPATH=/tmp \
  --entrypoint /usr/local/bin/serve-qwen38-flash-next-jj-r26-spark.sh \
  "${image}" --help 2>&1)
qwen_hollow_status=$?
set -e
if [[ "${qwen_hollow_status}" != 78 || "${qwen_hollow_output}" != *'runtime source-path preflight failed'* ]]; then
  printf 'Qwen hollow-package preflight failed open (status %s).\n%s\n' \
    "${qwen_hollow_status}" "${qwen_hollow_output}" >&2
  exit 1
fi

# shellcheck disable=SC2016
"${container_engine}" run --rm --entrypoint /bin/bash -e CC=/bin/false "${image}" -c '
  set -euo pipefail
  test "${B12X_ROCE_CACHE_DIR}" = /opt/jovian-judgement/b12x-roce
  /opt/venv/bin/python -c "from b12x.comm import roce; assert roce.API_VERSION == 1"
  /opt/venv/bin/python -c "from b12x.comm.roce._proxy import load; assert load().roce_abi_version() == 3"
  test "$(find "${B12X_ROCE_CACHE_DIR}" -maxdepth 1 -type f -name "roce_proxy-*.so" | wc -l)" = 1
'
"${container_engine}" run --rm --entrypoint /opt/venv/bin/python "${image}" -m pytest -q \
  --confcutdir=/opt/jovian-judgement/vllm/tests/v1/worker \
  /opt/jovian-judgement/vllm/tests/v1/worker/test_b12x_roce_health.py

if ((skip_gpu_check == 0)); then
  "${container_engine}" run --rm --device nvidia.com/gpu=all --ipc=host \
    -e PYTHONUNBUFFERED=1 -e B12X_PRINT_COMPILE_PROGRESS=1 \
    --entrypoint /opt/venv/bin/python "${image}" \
    /opt/local-inference/verify_jj_r26_runtime.py \
    --vllm-version "${vllm_version}" \
    --vllm-spark-tree "${vllm_spark_tree}" \
    --vllm-subtree "${vllm_package}" \
    --b12x-tree "${b12x_tree}" \
    --b12x-subtree "${b12x_package}" \
    --lmcache-version "${lmcache_version}" \
    --source-lock-sha256 "${source_lock_sha256}"
  "${container_engine}" run --rm --device nvidia.com/gpu=all --ipc=host \
    -e PYTHONUNBUFFERED=1 --entrypoint /opt/venv/bin/python "${image}" -m pytest -s -vv \
    --confcutdir=/opt/jovian-judgement/vllm/tests/models \
    /opt/jovian-judgement/vllm/tests/models/test_glm5next_model.py \
    -k 'mtp_draft_head_mode or mtp_nvfp4_draft_head_capability'
  "${container_engine}" run --rm --device nvidia.com/gpu=all --ipc=host \
    -e PYTHONUNBUFFERED=1 -e B12X_PRINT_COMPILE_PROGRESS=1 \
    --entrypoint /opt/venv/bin/python "${image}" \
    /opt/local-inference/verify_glm53_nvfp4_draft_head_sm121.py
  "${container_engine}" run --rm --device nvidia.com/gpu=all --ipc=host \
    -e PYTHONUNBUFFERED=1 --entrypoint /opt/venv/bin/python "${image}" -m pytest -s -vv \
    --confcutdir=/opt/jovian-judgement/vllm/tests/models/kimi_k3 \
    /opt/jovian-judgement/vllm/tests/models/kimi_k3/test_kda.py \
    -k near_collinear_keys_remain_finite
  "${container_engine}" run --rm --device nvidia.com/gpu=all --ipc=host \
    -e PYTHONUNBUFFERED=1 -e B12X_PRINT_COMPILE_PROGRESS=1 \
    --entrypoint /opt/venv/bin/python "${image}" -m pytest -s -vv \
    --confcutdir=/opt/jovian-judgement/b12x/tests/gemm \
    /opt/jovian-judgement/b12x/tests/gemm/test_mxfp8_linear.py \
    -k persistent_ctas_complete_single_stage_epilogue_stores
elif ((push_image == 1)); then
  echo 'PUSH_IMAGE=1 requires the GPU runtime gate.' >&2
  exit 1
fi

if ((push_image == 1)); then "${container_engine}" push "${image}"; fi
"${container_engine}" image inspect "${image}" --format \
  'image={{.Id}} size={{.Size}} entrypoint={{json .Config.Entrypoint}}'
printf '%s\n' "${image}"
