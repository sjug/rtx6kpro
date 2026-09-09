#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "${BASH_SOURCE[0]}")"
# shellcheck source=./build_safety.sh
source ./build_safety.sh
require_bool DRY_RUN "${DRY_RUN:-0}"
require_bool ALLOW_DIRTY_BUILD "${ALLOW_DIRTY_BUILD:-0}"
image=localhost/voipmonitor/vllm:jj-r28-spark-sm121
base=ef669fa1cde3e99936c02575eca8f610990bb6c64dd1bbfcb04fd42dde87afae
lock=source.lock.json
value() { jq -er "$1" "$lock"; }
for name in vllm b12x lmcache; do
  for pair in 'patch patch_sha256' 'changed_paths changed_paths_sha256'; do
    read -r path_key sha_key <<<"$pair"
    printf '%s  %s\n' "$(value ".${name}.${sha_key}")" "$(value ".${name}.${path_key}")" | sha256sum -c -
  done
done
printf '%s  %s\n' "$(value '.lmcache.bundle_sha256')" "$(value '.lmcache.bundle')" | sha256sum -c -
echo '15a9a649559830822cb943ea0c3c6a644c8b69c9e54d7be3e265801851843932  upstream-r28.source.lock' | sha256sum -c -
python3 tests/test_build_contracts.py
bash tests/test-glm53-flash-jj-r28-spark-tp4-runner.sh
bash tests/test-qwen38-flash-next-jj-r28-spark-runner.sh
lock_sha=$(sha256sum "$lock" | cut -d' ' -f1)
# Outputs, caches and receipts cannot change recipe identity.
recipe_manifest=$(find . -type f \( -name '*.py' -o -name '*.sh' -o -name '*.patch' -o -name '*.txt' \
  -o -name '*.json' -o -name '*.lock' -o -name Dockerfile -o -name .dockerignore \) \
  ! -path './.compose/*' ! -path './build-receipts/*' ! -path './qualification/*' \
  ! -path './__pycache__/*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum)
recipe_sha=$(printf '%s\n' "$recipe_manifest" | sha256sum | cut -d' ' -f1)
# Standalone exports on the build host retain the exporting checkout's commit,
# but are always dirty: the input manifest identifies their actual recipe bytes.
if [[ -n ${RECIPE_COMMIT:-} ]]; then
  recipe_commit=$RECIPE_COMMIT
  [[ $recipe_commit =~ ^[0-9a-f]{40}$ ]] || { echo 'Invalid RECIPE_COMMIT' >&2; exit 2; }
  recipe_dirty=1
else
  recipe_commit=$(git rev-parse HEAD)
  recipe_dirty=0
  [[ -z $(git status --porcelain --untracked-files=all -- .) ]] || recipe_dirty=1
fi
glm_sha=$(sha256sum launchers/serve-glm53-flash-jj-r28-spark.sh | cut -d' ' -f1)
[[ $glm_sha == "$(value '.launchers["serve-glm53-flash-jj-r28-spark.sh"]')" ]]
version="0.26.1rc0+glm53.jj.r28.spark.vllm$(value '.vllm.package_tree' | cut -c1-7)"
label_args=()
for item in 'vllm.integration.tree vllm.upstream_tree' 'vllm.spark-overlay.tree vllm.tree' \
  'vllm.package.tree vllm.package_tree' 'b12x.integration.tree b12x.tree' \
  'b12x.package.tree b12x.package_tree' 'lmcache.integration.tree lmcache.tree' \
  'lmcache.package.tree lmcache.package_tree'; do
  read -r label key <<<"$item"
  label_args+=(--label "local-inference.${label}=$(value ".${key}")")
done
label_args+=(--label local-inference.b12x.rocenante.api-version=1
  --label local-inference.b12x.rocenante.proxy-abi=3
  --label local-inference.b12x.rocenante.proxy-cache-dir=/opt/jovian-judgement/b12x-roce)
if [[ ${DRY_RUN:-0} == 1 ]]; then
  printf 'image=%s\nbase=%s\nversion=%s\nlock=%s\nrecipe=%s\ndirty=%s\n' "$image" "$base" "$version" "$lock_sha" "$recipe_sha" "$recipe_dirty"
  exit 0
fi
[[ $(hostname -s) == dusty ]] || { echo 'This build is scoped to dusty.' >&2; exit 78; }
[[ $(uname -m) == aarch64 ]] || { echo 'SM121 native build requires aarch64.' >&2; exit 78; }
if [[ $recipe_dirty == 1 && ${ALLOW_DIRTY_BUILD:-0} != 1 ]]; then
  echo 'Uncommitted recipe: commit or explicitly set ALLOW_DIRTY_BUILD=1; dirty provenance will be recorded.' >&2
  exit 78
fi
require_idle_pair
[[ $(podman image inspect "$base" --format '{{.Id}}' | sed 's/^sha256://') == "$base" ]]
if podman image exists "$image"; then
  echo 'Qualified tag already exists; inspect before explicitly replacing it.' >&2; exit 78
fi
receipt="build-receipts/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$receipt"
cp source.lock.json "$receipt/source.lock.json"
printf '%s\n' "$recipe_manifest" > "$receipt/recipe.sha256"
printf 'base=%s\nrecipe_commit=%s\nrecipe_sha256=%s\nrecipe_dirty=%s\n' \
  "$base" "$recipe_commit" "$recipe_sha" "$recipe_dirty" > "$receipt/provenance.txt"
exec > >(tee -a "$receipt/build.log") 2>&1
finish_receipt() {
  local status=$?
  printf 'exit_status=%s\nfinished_utc=%s\n' "$status" "$(date -u +%FT%TZ)" > "$receipt/status.txt"
}
trap finish_receipt EXIT
common=(--pull=never --layers --jobs=1 --format docker -f Dockerfile --build-arg "BASE_IMAGE=$base")
# No serving tag until all gates pass. Failure retains only a receipt and ID.
podman build "${common[@]}" --target flashkda-builder --iidfile "$receipt/native.id" .
native_id=$(<"$receipt/native.id")
flash_sha=$(podman run --pull=never --rm --entrypoint sha256sum "$native_id" /artifacts/_flashkda_C.abi3.so | cut -d' ' -f1)
[[ $flash_sha =~ ^[0-9a-f]{64}$ ]]
printf 'image_id=%s\nflashkda_sha256=%s\nsource_lock_sha256=%s\n' \
  "$native_id" "$flash_sha" "$lock_sha" > "$receipt/native.txt"
podman build "${common[@]}" --target runtime --iidfile "$receipt/candidate.id" \
  --build-arg "VLLM_VERSION=$version" --build-arg "FLASHKDA_SHA256=$flash_sha" \
  --build-arg "GLM_LAUNCHER_SHA256=$glm_sha" --build-arg "RECIPE_DIRTY=$recipe_dirty" \
  --build-arg "CACHE_FINGERPRINT=$(value '.cache_fingerprint')" \
  --build-arg "SOURCE_LOCK_SHA256=$lock_sha" --build-arg "RECIPE_SHA256=$recipe_sha" \
  --build-arg "RECIPE_COMMIT=$recipe_commit" "${label_args[@]}" .
candidate=$(<"$receipt/candidate.id")
require_idle_pair
publish_after_gates "$candidate" "$image" bash tests/gate-image.sh "$candidate" "$version" "$flash_sha" "$lock_sha"
podman image inspect "$candidate" > "$receipt/image-inspect.json"
printf 'BUILD-OK image=%s id=%s flashkda=%s lock=%s\n' "$image" "$candidate" "$flash_sha" "$lock_sha" | tee "$receipt/BUILD-OK"
# No stop, serving, distribution, posting, or pushing is implied.
