#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# SM121 foundation tree: base image, shared contracts, gates and the dusty/kirby pair lock.
FOUNDATION=$(cd ../../karmic-main-sm121/20260922 && pwd)
python3 preflight.py
case ${DRY_RUN:-0} in
  1) echo 'DRY-RUN: offline source refresh, CUTLASS 4.7.1 and ARM64/SM121 FlashInfer rebuild; idle dusty/kirby; foundation, PR865, beta and compiler gates; tag only after gates'; exit 0 ;;
  0) ;;
  *) echo 'DRY_RUN must be 0 or 1' >&2; exit 78 ;;
esac
[[ $(hostname -s) == dusty ]] || { echo 'Build requires dusty' >&2; exit 78; }
[[ $(uname -m) == aarch64 ]] || { echo 'Build requires ARM64' >&2; exit 78; }
[[ $(nvidia-smi --query-gpu=name --format=csv,noheader) == *GB10* ]] || { echo 'Build requires a verified GB10 GPU' >&2; exit 78; }
exec 9>"$FOUNDATION/../.build.lock"
flock -n 9 || { echo 'Another build owns the pair lock' >&2; exit 78; }
idle_pair() {
  local head worker head_gpu worker_gpu
  head=$(podman ps -q) || { echo 'Dusty probe failed' >&2; return 78; }
  worker=$(ssh -o BatchMode=yes -o ConnectTimeout=10 kirby 'podman ps -q') || {
    echo 'Kirby probe failed' >&2; return 78;
  }
  [[ -z $head && -z $worker ]] || { echo 'Pair is serving; refuse build' >&2; return 78; }
  head_gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader) || return 78
  worker_gpu=$(ssh -o BatchMode=yes -o ConnectTimeout=10 kirby \
    'nvidia-smi --query-compute-apps=pid --format=csv,noheader') || return 78
  [[ -z $head_gpu && -z $worker_gpu ]] || { echo 'Pair GPUs are occupied; refuse build' >&2; return 78; }
}
idle_pair
podman image exists 1a7a8acff71044e3b8bf97dacb7617cd1bdb50d0ab2fc2e35364a4d4dc165dcc || {
  echo 'Pinned base image is absent on dusty' >&2; exit 78;
}
for foundation in \
  nvcr.io/nvidia/pytorch@sha256:237ecf9ac7373daf91b31bb4f86651ce1ce57b676366ed435aa1aba61dad81d5 \
  ghcr.io/astral-sh/uv@sha256:2f0a1c36ce6323e61bc81d39e692319669c03e385b592d670b6ba36dc884e3aa; do
  podman image exists "$foundation" || { echo "Pinned component foundation absent: $foundation" >&2; exit 78; }
done
receipt="$PWD/receipts/build-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$receipt"
exec > >(tee "$receipt/build.log") 2>&1
finish() { local result=$?; printf 'exit_status=%s\n' "$result" > "$receipt/status.txt"; }
trap finish EXIT
python3 preflight.py --values > "$receipt/values.txt"
readarray -t values < "$receipt/values.txt"
[[ ${#values[@]} == 6 ]] || { echo 'Malformed preflight values' >&2; exit 78; }
python3 preflight.py --manifest > "$receipt/recipe-inputs.json"
cp build.lock.json Dockerfile install.py build.sh "$receipt/"
cp inputs.lock.json compiler-arm64.lock publication.json "$receipt/"
podman build --pull=never --network=none --format docker --layers \
  --tag "localhost/voipmonitor/build-components:${receipt##*/}" --iidfile "$receipt/image.id" \
  --build-arg "BUILD_LOCK_INPUT=${values[0]}" --build-arg "SOURCE_LOCK_INPUT=${values[1]}" \
  --build-arg "CACHE_INPUT=${values[2]}" --build-arg "VLLM_TREE_INPUT=${values[3]}" \
  --build-arg "B12X_TREE_INPUT=${values[4]}" --build-arg "RECIPE_INPUT=${values[5]}" .
image=$(<"$receipt/image.id")
podman image inspect "$image" > "$receipt/image-inspect.json"
podman run --rm --pull=never --entrypoint /bin/cat "$image" \
  /opt/karmic-beta-refresh/dependency-upgrade.json > "$receipt/dependency-upgrade.json"
idle_pair
bash "$FOUNDATION/gate.sh" "$image" "$receipt"
gate() {
  local log=$1 script=$2
  podman run --rm --pull=never --device nvidia.com/gpu=all --security-opt label=disable --ipc=host \
    -e PYTHONUNBUFFERED=1 -e PYTHONOPTIMIZE=0 -e B12X_PRINT_COMPILE_PROGRESS=1 \
    -v "$FOUNDATION:/gate:ro" -v "$PWD:/kit:ro" "$image" /opt/venv/bin/python "$script" 2>&1 | tee "$receipt/$log.log"
}
gate qsa865 /gate/qsa865/gate.py
gate karmic-beta /kit/gate_beta.py
grep -q '^KARMIC-BETA-GATE-PASS ' "$receipt/karmic-beta.log" || { echo 'Beta gate did not pass' >&2; exit 1; }
idle_pair
gate compiler-471 /kit/gate_compiler.py
grep -q '^KARMIC-COMPILER-471-GATE-PASS ' "$receipt/compiler-471.log" || { echo 'Compiler gate did not pass' >&2; exit 1; }
gate flashinfer-kernels /kit/gate_flashinfer.py
grep -q '^FLASHINFER-KERNEL-GATE-PASS ' "$receipt/flashinfer-kernels.log" || { echo 'FlashInfer kernel gate did not pass' >&2; exit 1; }
idle_pair
podman tag "$image" localhost/voipmonitor/vllm:karmic-beta-20261001-spark-sm121
printf 'BUILD-OK image=%s model_qualification=pending\n' "$image" | tee "$receipt/BUILD-OK"
