#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
[[ $(hostname -s) == dusty ]] || exit 78
exec 9>.native-build.lock
flock -n 9 || exit 78
idle=$(podman ps -q) || exit 78
peer=$(ssh -o BatchMode=yes -o ConnectTimeout=10 kirby 'podman ps -q') || exit 78
[[ -z $idle && -z $peer ]] || { echo 'Pair must be idle'; exit 78; }
receipt="build-receipts/serving-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$receipt"
exec > >(tee "$receipt/build.log") 2>&1
trap 'printf "exit_status=%s\n" "$?" > "$receipt/status.txt"' EXIT
qwen=$(sha256sum launchers/serve-qwen38-flash-next-karmic-spark.sh | cut -d' ' -f1)
glm=$(sha256sum launchers/serve-glm53-flash-karmic-spark.sh | cut -d' ' -f1)
recipe=$(sha256sum Dockerfile.serving | cut -d' ' -f1)
# The runtime image ID includes all native objects, dependencies and tracked sources.
fingerprint=$(printf '%s' 627e4cf62a02a73a5f01aea3fc8194b9f567a2e475ecf0cec82de091d148ec14 | sha256sum | cut -d' ' -f1)
cp Dockerfile.serving build-serving.sh verify_runtime.py "$receipt/"
cp -a launchers "$receipt/"
tag="localhost/voipmonitor/build-components:karmic-${receipt##*/}"
podman build --pull=never --format docker --layers --tag "$tag" --iidfile "$receipt/image.id" \
  -f Dockerfile.serving --build-arg "QWEN_LAUNCHER_INPUT_SHA256=$qwen" \
  --build-arg "GLM_LAUNCHER_INPUT_SHA256=$glm" --build-arg "CACHE_INPUT_SHA256=$fingerprint" \
  --build-arg "RECIPE_INPUT_SHA256=$recipe" .
image=$(<"$receipt/image.id")
podman image inspect "$image" > "$receipt/image-inspect.json"
podman run --rm --device nvidia.com/gpu=all --security-opt label=disable --ipc=host \
  -e PYTHONUNBUFFERED=1 -e CC=/bin/false -v "$PWD:/gate:ro" "$image" \
  python /gate/verify_runtime.py | tee "$receipt/native-smoke.log"
for model in qwen38-flash-next glm53-flash; do
  podman run --rm -e DRY_RUN=1 "$image" bash "/usr/local/bin/serve-$model-karmic-spark.sh" \
    > "$receipt/$model-render.txt"
  grep -q -- '--recurrent-checkpoint-policy aligned' "$receipt/$model-render.txt" || {
    echo "Aligned policy missing from $model"; exit 78;
  }
done
printf 'PACKAGING-GATES-PASS id=%s\n' "$image" | tee "$receipt/PACKAGING-OK"
# Retention tag only. The release gate must also accept kernel and CLI receipts.
