#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
[[ $(hostname -s) == dusty ]] || exit 78
image=f30dc6d9a2a6f6fc0ac9ff8cddb04d9f631f4a87b48ca7cac69254802fe83233
tag=localhost/voipmonitor/vllm:karmic-spark-sm121
gates=build-receipts/runtime-20260921T002742Z-1655697
grep -q 'KARMIC-RUNTIME-NATIVE-SMOKE-PASS' "$gates/native-smoke.log"
grep -q 'FLASHKDA-KARMIC-GATE-PASS collected=12 passed=12' "$gates/flashkda-poisoned-gate.log"
grep -Eq '^Karmic-REGRESSION-PASS .+: 235 passed$' "$gates/regressions.log"
grep -Eq '^Karmic-REGRESSION-PASS .*test_mamba_sparse_cleanup.py: 7 passed$' "$gates/regressions.log"
grep -Eq '^Karmic-REGRESSION-PASS .+: 6 passed$' "$gates/regressions-continuation.log"
grep -Eq '^Karmic-REGRESSION-PASS .*incomplete_tail: 1 passed$' "$gates/regressions-continuation.log"
grep -Eq '^Karmic-REGRESSION-PASS .*test_single_type_kv_cache_manager.py: 7 passed$' "$gates/regressions-continuation.log"
grep -q 'GLM Karmic NVFP4 proposal head SM121: PASS' "$gates/regressions-continuation.log"
for count in 2 5 19; do grep -qx "Karmic REQUIRED PASS: $count cases" "$gates/worker-gates.log"; done
for model in qwen38-flash-next glm53-flash; do
  grep -q "KARMIC-LAUNCH-ARGUMENTS-PASS $model" "$gates/launch-arguments.log"
done
route=$(ip route get 10.11.11.8)
[[ $route == *'dev enp1s0f0np0 src 10.11.11.7'* ]] || { echo "Wrong bulk route: $route"; exit 78; }
ssh_args=(-o BatchMode=yes -o ConnectTimeout=10 -o Compression=no -c aes128-gcm@openssh.com)
[[ $(ssh "${ssh_args[@]}" 10.11.11.8 hostname -s) == kirby ]] || exit 78
out="$PWD/build-receipts/image-transfer"
mkdir -p "$out"
exec > >(tee -a "$out/transfer.log") 2>&1
trap 'printf "exit_status=%s\n" "$?" > "$out/status.txt"' EXIT
podman tag "$image" "$tag"
podman image inspect "$image" > "$out/source-inspect.json"
archive="$out/karmic-spark-sm121.docker.tar"
if [[ -e $archive ]]; then echo 'Archive exists; review before retry'; exit 78; fi
podman save --format docker-archive --output "$archive" "$tag"
ssh "${ssh_args[@]}" 10.11.11.8 "mkdir -p '$out'"
rsync -a --whole-file --partial --info=progress2 \
  -e 'ssh -o BatchMode=yes -o Compression=no -c aes128-gcm@openssh.com' \
  "$archive" "10.11.11.8:$archive"
ssh "${ssh_args[@]}" 10.11.11.8 "podman load -i '$archive'"
actual=$(ssh "${ssh_args[@]}" 10.11.11.8 "podman image inspect '$tag' --format '{{.Id}}'")
[[ ${actual#sha256:} == "$image" ]] || { echo "Receiver identity mismatch: $actual"; exit 78; }
printf 'TRANSFER-VERIFIED image=%s receiver=kirby route=10.11.11.7-to-10.11.11.8 format=docker-archive\n' "$image" | tee "$out/TRANSFER-OK"
