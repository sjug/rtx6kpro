#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "${BASH_SOURCE[0]}")"
# shellcheck source-path=SCRIPTDIR
source ./build_safety.sh
require_bool DRY_RUN "${DRY_RUN:-0}"
require_bool ALLOW_DIRTY_BUILD "${ALLOW_DIRTY_BUILD:-0}"
image=localhost/voipmonitor/vllm:jj-r38-spark-sm121
value() { jq -er "$1" source.lock.json; }
base=$(value '.base_image_id')
python3 tests/test_build_contracts.py
python3 tests/test_native_reuse.py
python3 tests/test_gate_repairs.py
lock_sha=$(sha256sum source.lock.json | cut -d' ' -f1)
recipe_manifest=$(find . -type f \( -name '*.py' -o -name '*.sh' -o -name '*.patch' -o -name '*.txt' \
  -o -name '*.json' -o -name '*.lock' -o -name 'Dockerfile*' -o -name .dockerignore \) \
  ! -path './.compose/*' ! -path './build-receipts/*' ! -path '*/__pycache__/*' \
  -print0 | LC_ALL=C sort -z | xargs -0 sha256sum)
recipe_sha=$(printf '%s\n' "$recipe_manifest" | sha256sum | cut -d' ' -f1)
if [[ -n ${RECIPE_COMMIT:-} ]]; then
  recipe_commit=$RECIPE_COMMIT
  [[ $recipe_commit =~ ^[0-9a-f]{40}$ ]] || exit 2
  recipe_dirty=1
else
  recipe_commit=$(git rev-parse HEAD)
  recipe_dirty=0
  [[ -z $(git status --porcelain --untracked-files=all -- .) ]] || recipe_dirty=1
fi
glm_sha=$(value '.launchers["serve-glm53-flash-jj-r38-spark.sh"]')
qwen_sha=$(value '.launchers["serve-qwen38-flash-next-jj-r38-spark.sh"]')
if [[ ${DRY_RUN:-0} == 1 ]]; then
  printf 'image=%s\nbase=%s\nflashinfer=%s\nlock=%s\nrecipe=%s\ndirty=%s\n' \
    "$image" "$base" "${FLASHINFER_IMAGE:-required-after-component-build}" "$lock_sha" "$recipe_sha" "$recipe_dirty"
  exit 0
fi
[[ $(hostname -s) == rusty && $(uname -m) == aarch64 ]] || { echo 'Build is scoped to rusty/ARM64.' >&2; exit 78; }
[[ $recipe_dirty == 0 || ${ALLOW_DIRTY_BUILD:-0} == 1 ]] || { echo 'Explicit ALLOW_DIRTY_BUILD=1 required.' >&2; exit 78; }
require_idle_pair
[[ $(podman image inspect "$base" --format '{{.Id}}' | sed 's/^sha256://') == "$base" ]] || { echo 'Base image ID mismatch.' >&2; exit 78; }
[[ $(podman image inspect "$base" --format '{{index .Config.Labels "local-inference.runtime.source-lock.sha256"}}') == "$(value '.base_source_lock_sha256')" ]] || { echo 'Base source-lock label mismatch.' >&2; exit 78; }
: "${FLASHINFER_IMAGE:?Set the completed FlashInfer component image ID}"
fi_id=$(podman image inspect "$FLASHINFER_IMAGE" --format '{{.Id}}')
[[ ${fi_id#sha256:} =~ ^[0-9a-f]{64}$ ]] || { echo 'Invalid component image ID.' >&2; exit 78; }
[[ $(podman image inspect "$fi_id" --format '{{index .Config.Labels "local-inference.flashinfer.commit"}}') == "$(value '.flashinfer.commit')" ]] || { echo 'FlashInfer commit mismatch.' >&2; exit 78; }
[[ $(podman image inspect "$fi_id" --format '{{index .Config.Labels "local-inference.flashinfer.arch"}}') == 12.1a ]] || { echo 'FlashInfer architecture mismatch.' >&2; exit 78; }
[[ $(podman image inspect "$fi_id" --format '{{index .Config.Labels "local-inference.flashinfer.recipe.sha256"}}') == "$(value '.native_build_inputs["Dockerfile.flashinfer"]')" ]] || { echo 'FlashInfer recipe label differs from the lock.' >&2; exit 78; }
[[ $(podman image inspect "$fi_id" --format '{{index .Config.Labels "local-inference.flashinfer.build-script.sha256"}}') == "$(value '.native_build_inputs["build-flashinfer.sh"]')" ]] || { echo 'FlashInfer build-script label differs from the lock.' >&2; exit 78; }
if podman image exists "$image"; then
  echo 'Qualified tag already exists; no implicit replacement.' >&2; exit 78
fi
receipt="build-receipts/runtime-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$receipt"
retained_tag="localhost/voipmonitor/build-candidates:jj-r38-${receipt##*/}"
printf '%s\n' "$retained_tag" > "$receipt/retained-tag.txt"
cp source.lock.json "$receipt/source.lock.json"
printf '%s\n' "$recipe_manifest" > "$receipt/recipe.sha256"
printf 'base=%s\nflashinfer_image=%s\nrecipe_commit=%s\nrecipe_sha256=%s\nrecipe_dirty=%s\n' \
  "$base" "$fi_id" "$recipe_commit" "$recipe_sha" "$recipe_dirty" > "$receipt/provenance.txt"
exec > >(tee -a "$receipt/build.log") 2>&1
finish_receipt() {
  local status=$?
  printf 'exit_status=%s\nfinished_utc=%s\n' "$status" "$(date -u +%FT%TZ)" > "$receipt/status.txt"
}
trap finish_receipt EXIT
podman image inspect "$base" > "$receipt/base-image-inspect.json"
podman image inspect "$fi_id" > "$receipt/flashinfer-image-inspect.json"
mkdir -p "$receipt/flashinfer"
# Extract only the small, known receipt files, not arbitrary archive members.
for file in Dockerfile.flashinfer build-flashinfer.sh recipe.sha256 inputs.txt; do
  podman run --pull=never --rm --entrypoint cat "$fi_id" "/artifacts/flashinfer/build-inputs/$file" > "$receipt/flashinfer/$file"
done
for file in SHA256SUMS submodules.txt; do
  podman run --pull=never --rm --entrypoint cat "$fi_id" "/artifacts/flashinfer/$file" > "$receipt/flashinfer/$file"
done
python3 verify_component.py "$receipt/flashinfer" source.lock.json
labels=()
# Replace the entire immediate-native-base family together. The unmodified
# base inspection above preserves older ancestry without contradictory labels.
while IFS= read -r key; do
  case "$key" in
    local-inference.runtime.native-artifact-base-id|local-inference.runtime.native-artifact-base-image|local-inference.runtime.native-artifact-base-recipe-commit|local-inference.runtime.native-artifact-base-recipe-sha256) ;;
    *) labels+=(--unsetlabel "$key") ;;
  esac
done < <(jq -r '.[0].Config.Labels | keys[] | select(startswith("local-inference.runtime.native-artifact-base-"))' "$receipt/base-image-inspect.json")
labels+=(--label "local-inference.runtime.native-artifact-base-id=$base" \
  --label local-inference.runtime.native-artifact-base-image=localhost/voipmonitor/vllm:jj-r32-spark-sm121 \
  --label "local-inference.runtime.native-artifact-base-recipe-commit=$(jq -er '.[0].Config.Labels["org.opencontainers.image.revision"]' "$receipt/base-image-inspect.json")" \
  --label "local-inference.runtime.native-artifact-base-recipe-sha256=$(jq -er '.[0].Config.Labels["local-inference.recipe.sha256"]' "$receipt/base-image-inspect.json")")
for pair in 'vllm.integration.tree vllm.upstream_tree' 'vllm.spark-overlay.tree vllm.tree' \
  'vllm.package.tree vllm.package_tree' 'b12x.integration.tree b12x.tree' \
  'b12x.package.tree b12x.package_tree' 'lmcache.integration.tree lmcache.tree' \
  'lmcache.package.tree lmcache.package_tree' 'flashkda.sha256 flashkda.spark_extension_sha256' \
  'vllm.stable-native.sha256 stable_native_sha256'; do
  read -r label key <<<"$pair"
  labels+=(--label "local-inference.$label=$(value ".$key")")
done
for name in vllm b12x lmcache; do
  labels+=(--label "local-inference.$name.commit=$(value ".$name.commit")")
done
labels+=(--label "local-inference.vllm.package-version=$(value '.vllm.version')" \
  --label "local-inference.lmcache.version=$(value '.lmcache.version')" \
  --label local-inference.b12x.rocenante.api-version=1 \
  --label local-inference.b12x.rocenante.proxy-abi=3)
podman build --pull=never --layers --jobs=1 --format docker -f Dockerfile --target runtime --tag "$retained_tag" \
  --iidfile "$receipt/candidate.id" --build-arg "BASE_IMAGE=$base" --build-arg "FLASHINFER_IMAGE=$fi_id" \
  --build-arg "CACHE_FINGERPRINT=$(value '.cache_fingerprint')" \
  --build-arg "SOURCE_LOCK_SHA256=$lock_sha" --build-arg "RECIPE_SHA256=$recipe_sha" \
  --build-arg "RECIPE_COMMIT=$recipe_commit" --build-arg "RECIPE_DIRTY=$recipe_dirty" \
  --build-arg "BUILD_GLM_LAUNCHER_SHA256=$glm_sha" --build-arg "BUILD_QWEN_LAUNCHER_SHA256=$qwen_sha" "${labels[@]}" .
candidate=$(<"$receipt/candidate.id")
podman image inspect "$candidate" > "$receipt/image-inspect.json"
require_idle_pair
publish_after_gates "$candidate" "$image" bash tests/gate-image.sh "$candidate" "$lock_sha"
podman image inspect "$candidate" > "$receipt/qualified-image-inspect.json"
printf 'BUILD-OK image=%s id=%s lock=%s\n' "$image" "$candidate" "$lock_sha" | tee "$receipt/BUILD-OK"
# Build acceptance only. No serving launch or promotion follows implicitly.
