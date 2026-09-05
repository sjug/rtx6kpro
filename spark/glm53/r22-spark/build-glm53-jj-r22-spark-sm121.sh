#!/usr/bin/env bash
set -euo pipefail

# Build the JJ r22 GLM and Qwen runtime for SM121 from the stock-r17 native
# base. The exact r22 source trees are installed, FlashKDA is rebuilt at the
# corrected r22 pin, and B12X carries only three explicitly locked newer
# commits: dual-rail RoCEnante striping, graph-collective sizing, and the
# persistent-epilogue store wait.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_root}"

container_engine=${CONTAINER_ENGINE:-podman}
release_date=${RELEASE_DATE:-20260904}
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

vllm_commit=$(jq -er '.vllm.r22_integration_commit' "${lock}")
vllm_tree=$(jq -er '.vllm.r22_integration_tree' "${lock}")
vllm_package=$(jq -er '.vllm.package_tree' "${lock}")
vllm_spark_tree=$(jq -er '.vllm.spark_tree' "${lock}")
vllm_refresh_sha=$(jq -er '.vllm.refresh_patch_sha256' "${lock}")
vllm_paths_sha=$(jq -er '.vllm.changed_paths_sha256' "${lock}")
flashkda_commit=$(jq -er '.vllm.flashkda_commit' "${lock}")
flashkda_tree=$(jq -er '.vllm.flashkda_tree' "${lock}")
flashkda_cutlass=$(jq -er '.vllm.flashkda_cutlass_commit' "${lock}")

b12x_r22_commit=$(jq -er '.b12x.r22_integration_commit' "${lock}")
b12x_r22_tree=$(jq -er '.b12x.r22_integration_tree' "${lock}")
b12x_r22_package=$(jq -er '.b12x.r22_package_tree' "${lock}")
b12x_selected=$(jq -er '.b12x.selected_commits | join(",")' "${lock}")
b12x_tree=$(jq -er '.b12x.integration_tree' "${lock}")
b12x_package=$(jq -er '.b12x.package_tree' "${lock}")
b12x_refresh_sha=$(jq -er '.b12x.refresh_patch_sha256' "${lock}")
b12x_paths_sha=$(jq -er '.b12x.changed_paths_sha256' "${lock}")
b12x_roce_api=$(jq -er '.b12x.rocenante_api_version' "${lock}")
b12x_proxy_abi=$(jq -er '.b12x.rocenante_proxy_abi' "${lock}")
b12x_roce_cache=$(jq -er '.b12x.rocenante_proxy_cache_dir' "${lock}")

lmcache_repo=$(jq -er '.lmcache.repository' "${lock}")
lmcache_published=$(jq -er '.lmcache.published_release_commit' "${lock}")
lmcache_acquisition=$(jq -er '.lmcache.acquisition_commit' "${lock}")
lmcache_tree=$(jq -er '.lmcache.tree' "${lock}")
lmcache_package=$(jq -er '.lmcache.package_tree' "${lock}")
lmcache_version=$(jq -er '.lmcache.version' "${lock}")

test "${base_image}" = "${base_image_locked}"
test "$(jq -er '.published_r22.digest' "${lock}")" = sha256:284784e685aa0377f1cf63a312a364fc884b02beb98949d6886624edbddb3806
test "$(jq -er '.published_r22.source_lock_sha256' "${lock}")" = 57789f330528d65c80f4ffac208beb63617b6c6ab1e077056b2a0fe992e997d8
test "${base_image_id}" = 1e2051dade83f41fa46ee7d1dd4432c1a3e2cf0ca50dd21b5b892b597a4fcb68
test "${base_vllm_tree}" = 6ba304782a948408612c2aa8e468ea51f6da398c
test "${base_vllm_package}" = bd8e2ab97002642a0512d1b80893d6a9a2f26a01
test "${base_b12x_tree}" = 3c30f0fd7c6c77bdaef6132439db8b51b81f958e
test "${base_b12x_package}" = 5ee3a5a8b79b1325f4b25709bcf17ca05f6f9e94
test "${vllm_commit}" = 70b3c1c7f1c76fcf0847fcbb4a0b8b5583b78d19
test "${vllm_tree}" = 89481110674c08be1759a9222c525a0be14ad52a
test "${vllm_package}" = 4fbb1c257ac59e5e68450655ad4061d2c8a05e5c
test "${vllm_spark_tree}" = 997f08f583f1439363939f592c54980d584bc4c7
test "${flashkda_commit}" = 3b225bf26bb8e218928a1fe14751cb48cf31d11b
test "${flashkda_tree}" = e8cf226562e56d0817462de76a614d32a83409ef
test "${flashkda_cutlass}" = 5c149f52a436782210263fb2f19b354443a61c6a
test "${b12x_r22_commit}" = 1e59a1fd09f782d302b1068b15c8a0bd66103894
test "${b12x_r22_tree}" = f322c804eec1c58a63bd4fe6e7901a95a678a575
test "${b12x_r22_package}" = aaa5f189acae0206d886553421f6e9044f4c458a
test "${b12x_selected}" = aa90a277a61f9ded46c0f504e37a955b7706659b,1a7e3ec286b0ff0b7c2aabee22dce08daab7e011,9ae41c5cb9935d740456479954b0089f80bd2ef2
test "${b12x_tree}" = 8ad308af020b9d176f7d1b3767c5404acf384ef3
test "${b12x_package}" = e1edb6d11c9b760d4a0d3fdf05f8b6646479d551
test "${b12x_roce_api}" = 1
test "${b12x_proxy_abi}" = 3
test "${b12x_roce_cache}" = /opt/jovian-judgement/b12x-roce
test "${lmcache_published}" = aefe3ab701ab7a835532e701be89f5055b13ec0f
test "${lmcache_acquisition}" = b13fa35eb2a1e35ba2cfd4277a0b93bf1cfd322b
test "${lmcache_tree}" = 683ab2c165a9aa0e2d1a1ab757af4a8b193688c5
test "${lmcache_package}" = 976a97f22c0497f34db089dc5f02a713dd0b5888
test "$(wc -l < "${vllm_paths}")" = "$(jq -er '.vllm.changed_paths_count' "${lock}")"
test "$(wc -l < "${b12x_paths}")" = "$(jq -er '.b12x.changed_paths_count' "${lock}")"
if rg -n -v '\.(py|sh|cmake)$' "${vllm_paths}"; then
  echo 'The r22 vLLM refresh contains an unclassified path.' >&2
  exit 1
fi
if rg -n -v '\.(py|c|md|json|json\.gz|txt|toml|sh)$' "${b12x_paths}"; then
  echo 'The r22 B12X refresh contains an unclassified path.' >&2
  exit 1
fi

source_lock_sha256=$(sha256sum "${lock}" | cut -d' ' -f1)
recipe_sha256="$({
  sha256sum Dockerfile.glm53-jj-r22-spark-sm121 \
    build-glm53-jj-r22-spark-sm121.sh \
    launchers/serve-glm53-flash-jj-r22-spark.sh \
    launchers/serve-qwen38-flash-next-jj-r22-spark.sh \
    run-glm53-flash-jj-r22-spark-tp4-node.sh \
    run-qwen38-flash-next-jj-r22-spark-tp2-node.sh \
    tests/verify_jj_r22_runtime.py \
    tests/test-glm53-flash-jj-r22-spark-tp4-runner.sh \
    tests/test-qwen38-flash-next-jj-r22-spark-runner.sh \
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
  echo 'Commit the r22 recipe or set ALLOW_DIRTY_BUILD=1 with an explicit dev image.' >&2
  exit 1
fi

vllm_version=${VLLM_PACKAGE_VERSION:-0.26.1rc0+glm53.jj.r22.spark.${revision}.vllm${vllm_package:0:7}.b12x${b12x_package:0:7}}
cache_fingerprint=cu133-torch213-jj-r22-sm121-vllm${vllm_package:0:10}-b12x${b12x_package:0:10}-flashkda${flashkda_commit:0:10}
image=${IMAGE:-localhost/voipmonitor/vllm:glm53-jj-r22-spark-sm121-vllm${vllm_package:0:7}-b12x${b12x_package:0:7}-lmcache${lmcache_package:0:7}-cu133-torch213-${release_date}-${revision}}
if ((worktree_dirty == 1)); then
  [[ -v IMAGE ]] || { echo 'A dirty build requires an explicit IMAGE.' >&2; exit 2; }
  case "${image##*:}" in *dev*|*test*|*scratch*) ;; *) echo "Unsafe dirty IMAGE: ${image}" >&2; exit 2 ;; esac
fi

if [[ "${PRINT_RELEASE_CONFIG:-0}" == 1 ]]; then
  printf 'base=%s\nbase_id=%s\nimage=%s\nstatus=%s\n' "${base_image}" "${base_image_id}" "${image}" "${release_status}"
  printf 'vllm_commit=%s\nvllm_tree=%s\nvllm_spark_tree=%s\nvllm_package=%s\n' "${vllm_commit}" "${vllm_tree}" "${vllm_spark_tree}" "${vllm_package}"
  printf 'flashkda_commit=%s\nflashkda_tree=%s\n' "${flashkda_commit}" "${flashkda_tree}"
  printf 'b12x_r22=%s\nb12x_selected=%s\nb12x_tree=%s\nb12x_package=%s\n' "${b12x_r22_commit}" "${b12x_selected}" "${b12x_tree}" "${b12x_package}"
  printf 'lmcache_published=%s\nlmcache_acquisition=%s\nlmcache_tree=%s\nlmcache_default=disabled\n' "${lmcache_published}" "${lmcache_acquisition}" "${lmcache_tree}"
  printf 'cache_fingerprint=%s\nrecipe_sha256=%s\nsource_lock_sha256=%s\n' "${cache_fingerprint}" "${recipe_sha256}" "${source_lock_sha256}"
  exit 0
fi

"${repo_root}/tests/test-glm53-flash-jj-r22-spark-tp4-runner.sh" \
  "${repo_root}/run-glm53-flash-jj-r22-spark-tp4-node.sh"
"${repo_root}/tests/test-qwen38-flash-next-jj-r22-spark-runner.sh" \
  "${repo_root}/run-qwen38-flash-next-jj-r22-spark-tp2-node.sh"
if ((worktree_dirty == 0)); then
  runner_home=$(mktemp -d)
  trap 'rm -rf "${runner_home}"' EXIT
  runner_default=$(HOME="${runner_home}" ROLE=head DRY_RUN=1 NODE_RANK=0 \
    HOST_IP=10.11.11.1 "${repo_root}/run-glm53-flash-jj-r22-spark-tp4-node.sh")
  grep -Fq -- "${image}" <<<"${runner_default}" || {
    echo 'The unoverridden runner does not point at the image being built.' >&2
    exit 1
  }
  qwen_runner_default=$(HOME="${runner_home}" ROLE=head DRY_RUN=1 NODE_RANK=0 \
    HOST_IP=10.11.1.1 "${repo_root}/run-qwen38-flash-next-jj-r22-spark-tp2-node.sh")
  grep -Fq -- "${image}" <<<"${qwen_runner_default}" || {
    echo 'The unoverridden Qwen runner does not point at the image being built.' >&2
    exit 1
  }
  rm -rf "${runner_home}"
  trap - EXIT
fi
test "$(uname -m)" = aarch64 || { echo 'JJ r22 Spark builds require aarch64.' >&2; exit 78; }
command -v "${container_engine}" >/dev/null
"${container_engine}" image exists "${base_image}" || { echo "Missing stock-r17 base image: ${base_image}" >&2; exit 78; }
actual_base_id=$("${container_engine}" image inspect "${base_image}" --format '{{.Id}}')
test "${actual_base_id#sha256:}" = "${base_image_id}"

"${container_engine}" build --format docker --pull=false \
  --build-arg "BASE_IMAGE=${base_image}" \
  --build-arg "BASE_IMAGE_ID=${base_image_id}" \
  --build-arg "BASE_VLLM_SPARK_TREE=${base_vllm_tree}" \
  --build-arg "BASE_VLLM_PACKAGE_TREE=${base_vllm_package}" \
  --build-arg "BASE_B12X_TREE=${base_b12x_tree}" \
  --build-arg "BASE_B12X_PACKAGE_TREE=${base_b12x_package}" \
  --build-arg "VLLM_R22_COMMIT=${vllm_commit}" \
  --build-arg "VLLM_R22_TREE=${vllm_tree}" \
  --build-arg "VLLM_PACKAGE_TREE=${vllm_package}" \
  --build-arg "VLLM_SPARK_TREE=${vllm_spark_tree}" \
  --build-arg "VLLM_REFRESH_PATCH_SHA256=${vllm_refresh_sha}" \
  --build-arg "VLLM_CHANGED_PATHS_SHA256=${vllm_paths_sha}" \
  --build-arg "VLLM_PACKAGE_VERSION=${vllm_version}" \
  --build-arg "FLASHKDA_REPO=https://github.com/vllm-project/FlashKDA.git" \
  --build-arg "FLASHKDA_COMMIT=${flashkda_commit}" \
  --build-arg "FLASHKDA_TREE=${flashkda_tree}" \
  --build-arg "FLASHKDA_CUTLASS_COMMIT=${flashkda_cutlass}" \
  --build-arg "B12X_R22_COMMIT=${b12x_r22_commit}" \
  --build-arg "B12X_R22_TREE=${b12x_r22_tree}" \
  --build-arg "B12X_R22_PACKAGE_TREE=${b12x_r22_package}" \
  --build-arg "B12X_SELECTED_COMMITS=${b12x_selected}" \
  --build-arg "B12X_TREE=${b12x_tree}" \
  --build-arg "B12X_PACKAGE_TREE=${b12x_package}" \
  --build-arg "B12X_REFRESH_PATCH_SHA256=${b12x_refresh_sha}" \
  --build-arg "B12X_CHANGED_PATHS_SHA256=${b12x_paths_sha}" \
  --build-arg "B12X_ROCE_API_VERSION=${b12x_roce_api}" \
  --build-arg "B12X_ROCE_PROXY_ABI=${b12x_proxy_abi}" \
  --build-arg "LMCACHE_REPO=${lmcache_repo}" \
  --build-arg "LMCACHE_PUBLISHED_COMMIT=${lmcache_published}" \
  --build-arg "LMCACHE_ACQUISITION_COMMIT=${lmcache_acquisition}" \
  --build-arg "LMCACHE_TREE=${lmcache_tree}" \
  --build-arg "LMCACHE_PACKAGE_TREE=${lmcache_package}" \
  --build-arg "LMCACHE_VERSION=${lmcache_version}" \
  --build-arg "SOURCE_LOCK_SHA256=${source_lock_sha256}" \
  --build-arg 'RELEASE_NAME=glm53-jj-r22-spark-sm121' \
  --build-arg "RELEASE_DATE=${release_date}" \
  --build-arg "DOCKER_COMMIT=${docker_commit}" \
  --build-arg "RECIPE_SHA256=${recipe_sha256}" \
  --build-arg "RELEASE_STATUS=${release_status}" \
  --build-arg "CACHE_FINGERPRINT=${cache_fingerprint}" \
  --file Dockerfile.glm53-jj-r22-spark-sm121 \
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
assert_label local-inference.scope glm53-qwen38-r22-spark-sm121-qualification
assert_label local-inference.runtime.native-artifact-base-id "${base_image_id}"
assert_label local-inference.vllm.r22.commit "${vllm_commit}"
assert_label local-inference.vllm.integration.tree "${vllm_tree}"
assert_label local-inference.vllm.spark-overlay.tree "${vllm_spark_tree}"
assert_label local-inference.vllm.package.tree "${vllm_package}"
assert_label local-inference.flashkda.commit "${flashkda_commit}"
assert_label local-inference.flashkda.tree "${flashkda_tree}"
assert_label local-inference.flashkda.arch 12.1a
assert_label local-inference.b12x.r22.commit "${b12x_r22_commit}"
assert_label local-inference.b12x.selected-commits "${b12x_selected}"
assert_label local-inference.b12x.integration.tree "${b12x_tree}"
assert_label local-inference.b12x.package.tree "${b12x_package}"
assert_label local-inference.b12x.rocenante.proxy-abi "${b12x_proxy_abi}"
assert_label local-inference.lmcache.published-commit "${lmcache_published}"
assert_label local-inference.lmcache.acquisition-commit "${lmcache_acquisition}"
assert_label local-inference.lmcache.integration.tree "${lmcache_tree}"
assert_label local-inference.lmcache.package.tree "${lmcache_package}"
assert_label local-inference.lmcache.default disabled

glm_launch=$("${container_engine}" run --rm -e DRY_RUN=1 "${image}")
grep -Fq -- '--revision 2e8b6daeb2be8716c06b54d746d12e581e551cca' <<<"${glm_launch}"
grep -Fq -- 'num_speculative_tokens\":3' <<<"${glm_launch}"
grep -Fq -- 'kda_prefill_backend\":\"flashkda' <<<"${glm_launch}"
grep -Fq -- '--cudagraph-capture-sizes 1 2 4 8 12 16 24 32' <<<"${glm_launch}"
if grep -Fq -- '--disable-custom-all-reduce' <<<"${glm_launch}"; then
  echo 'The GLM launcher unexpectedly disables custom all-reduce.' >&2
  exit 1
fi
qwen_launch=$("${container_engine}" run --rm \
  --entrypoint /usr/local/bin/serve-qwen38-flash-next-jj-r22-spark.sh \
  -e DRY_RUN=1 "${image}")
grep -Fq -- '--revision c374e7e24b54f6cb0017d0c2e6d26823d2f2fb5d' <<<"${qwen_launch}"
grep -Fq -- '--max-model-len 262144' <<<"${qwen_launch}"
grep -Fq -- '--max-num-seqs 4' <<<"${qwen_launch}"
grep -Fq -- '--block-size 16' <<<"${qwen_launch}"
grep -Fq -- 'num_speculative_tokens\":3' <<<"${qwen_launch}"
grep -Fq -- '--disable-custom-all-reduce' <<<"${qwen_launch}"
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
  --entrypoint /usr/local/bin/serve-qwen38-flash-next-jj-r22-spark.sh \
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
    /opt/local-inference/verify_jj_r22_runtime.py \
    --vllm-version "${vllm_version}" \
    --vllm-spark-tree "${vllm_spark_tree}" \
    --vllm-subtree "${vllm_package}" \
    --b12x-tree "${b12x_tree}" \
    --b12x-subtree "${b12x_package}" \
    --lmcache-version "${lmcache_version}" \
    --source-lock-sha256 "${source_lock_sha256}"
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
