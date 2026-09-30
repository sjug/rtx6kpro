#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
python3 preflight.py
case ${DRY_RUN:-0} in
  1) echo 'DRY-RUN: pinned native reuse plus approved stock NCCL; idle pair; parent and DS41 gates; tag only after gates'; exit 0 ;;
  0) ;;
  *) echo 'DRY_RUN must be 0 or 1' >&2; exit 78 ;;
esac
[[ $(hostname -s) == dusty ]] || { echo 'Build requires dusty' >&2; exit 78; }
exec 9>../.build.lock
flock -n 9 || { echo 'Another build owns the pair lock' >&2; exit 78; }
idle_pair() {
  local head worker
  head=$(podman ps -q) || { echo 'Dusty probe failed' >&2; return 78; }
  worker=$(ssh -o BatchMode=yes -o ConnectTimeout=10 kirby 'podman ps -q') || {
    echo 'Kirby probe failed' >&2; return 78;
  }
  [[ -z $head && -z $worker ]] || { echo 'Pair is serving; refuse build' >&2; return 78; }
}
idle_pair
receipt="$PWD/receipts/runtime-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$receipt"
exec > >(tee "$receipt/build.log") 2>&1
finish() { local result=$?; printf 'exit_status=%s\n' "$result" > "$receipt/status.txt"; }
trap finish EXIT
python3 preflight.py --values > "$receipt/values.txt"
python3 preflight.py --manifest > "$receipt/recipe-inputs.json"
readarray -t values < "$receipt/values.txt"
[[ ${#values[@]} == 8 ]] || { echo 'Malformed build values' >&2; exit 78; }
cp build.lock.json runtime.lock.json nccl.lock.json Dockerfile install.py build.sh "$receipt/"
podman image exists "${values[6]}" || { echo 'Pinned NCCL component missing' >&2; exit 78; }
podman build --pull=never --network=none --format docker --layers \
  --tag "localhost/voipmonitor/build-components:${receipt##*/}" --iidfile "$receipt/image.id" \
  --build-arg "BUILD_LOCK_INPUT=${values[0]}" --build-arg "SOURCE_LOCK_INPUT=${values[1]}" \
  --build-arg "CACHE_INPUT=${values[2]}" --build-arg "VLLM_TREE_INPUT=${values[3]}" \
  --build-arg "B12X_TREE_INPUT=${values[4]}" --build-arg "RECIPE_INPUT=${values[5]}" \
  --build-arg "NCCL_COMPONENT_IMAGE=${values[6]}" --build-arg "NCCL_LIBRARY_INPUT=${values[7]}" .
image=$(<"$receipt/image.id")
podman image inspect "$image" > "$receipt/image-inspect.json"
idle_pair
bash ../20260922/gate.sh "$image" "$receipt"
podman run --rm --pull=never --device nvidia.com/gpu=all --ipc=private --shm-size=0 --pids-limit=-1 \
  --security-opt "seccomp=$PWD/seccomp-io-uring.json" \
  -e PYTHONUNBUFFERED=1 -e PYTHONOPTIMIZE=0 -e B12X_PRINT_COMPILE_PROGRESS=1 \
  -e LD_PRELOAD=/opt/nccl-2.30.7/lib/libnccl.so.2.30.7 \
  -e VLLM_NCCL_SO_PATH=/opt/nccl-2.30.7/lib/libnccl.so.2.30.7 \
  -e VLLM_PLUGINS=b12x_loader -e NCCL_NET_PLUGIN=none \
  -v "$PWD:/ds41-gate:ro" --entrypoint /bin/bash "$image" -c \
  'export LD_LIBRARY_PATH=/opt/nccl-2.30.7/lib:${LD_LIBRARY_PATH:-}; exec /opt/venv/bin/python /ds41-gate/gate_ds41.py' \
  2>&1 | tee "$receipt/ds41-native-cli.log"
podman run --rm --pull=never --device nvidia.com/gpu=all --ipc=private --shm-size=0 --pids-limit=-1 \
  --security-opt "seccomp=$PWD/seccomp-io-uring.json" \
  -e PYTHONUNBUFFERED=1 -e PYTHONOPTIMIZE=0 -e B12X_PRINT_COMPILE_PROGRESS=1 \
  -e LD_PRELOAD=/opt/nccl-2.30.7/lib/libnccl.so.2.30.7 \
  -e VLLM_NCCL_SO_PATH=/opt/nccl-2.30.7/lib/libnccl.so.2.30.7 \
  -e VLLM_PLUGINS=b12x_loader -e NCCL_NET_PLUGIN=none \
  -v "$PWD/../20260922:/gate:ro" -v "$PWD:/ds41-gate:ro" --entrypoint /bin/bash "$image" -c \
  'export LD_LIBRARY_PATH=/opt/nccl-2.30.7/lib:${LD_LIBRARY_PATH:-}; exec /opt/venv/bin/python /ds41-gate/gate_engram.py' \
  2>&1 | tee "$receipt/ds41-engram.log"
podman tag "$image" localhost/voipmonitor/vllm:karmic-main-ds41-spark-sm121
printf 'BUILD-OK image=%s model_qualification=pending\n' "$image" | tee "$receipt/BUILD-OK"
