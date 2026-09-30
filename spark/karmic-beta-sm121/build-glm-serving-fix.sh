#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
[[ $(hostname -s) == sparky ]] || exit 78
idle=$(podman ps -q) || exit 78
[[ -z $idle ]] || exit 78
out=build-receipts/glm-serving-fix-20260921
[[ ! -e $out/build.log ]] || exit 78
mkdir -p "$out"
exec > >(tee "$out/build.log") 2>&1
trap 'printf "exit_status=%s\n" "$?" > "$out/status.txt"' EXIT
digest=$(sha256sum launchers/serve-glm53-flash-karmic-spark.sh | cut -d' ' -f1)
cp Dockerfile.glm-serving-fix build-glm-serving-fix.sh verify_runtime.py "$out/"
cp -a launchers "$out/"
podman build --pull=never --format docker --layers --tag localhost/voipmonitor/build-components:karmic-glm-launcher \
  --iidfile "$out/image.id" -f Dockerfile.glm-serving-fix --build-arg "GLM_REPAIRED_SHA256=$digest" .
image=$(<"$out/image.id")
podman image inspect "$image" > "$out/image-inspect.json"
podman run --rm --device nvidia.com/gpu=all --security-opt label=disable --ipc=host \
  -e CC=/bin/false -e PYTHONUNBUFFERED=1 -v "$PWD:/gate:ro" "$image" \
  python /gate/verify_runtime.py | tee "$out/native-smoke.log"
# Unlike DRY_RUN, --help passes through the live self-hash/import/proxy preflight.
podman run --rm --device nvidia.com/gpu=all --security-opt label=disable --ipc=host \
  -e CC=/bin/false "$image" bash /usr/local/bin/serve-glm53-flash-karmic-spark.sh --help \
  > "$out/live-preflight.log" 2>&1
grep -q 'usage: vllm serve' "$out/live-preflight.log"
podman tag "$image" localhost/voipmonitor/vllm:karmic-spark-sm121
printf 'GLM-SERVING-REPAIR-PASS image=%s\n' "$image" | tee "$out/PACKAGING-OK"
