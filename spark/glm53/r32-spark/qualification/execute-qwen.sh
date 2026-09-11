#!/usr/bin/env bash
# Authorized R32 Qwen chain. No GLM/DS4 transition and no auto-policy arm.
set -euo pipefail
base=$(cd "$(dirname "$0")" && pwd)
recipe=$(dirname "$base")
remote=/home/jugs/git/bld-jj-r32-spark
receipt=$remote/build-receipts/20260910T144935Z-1388331
[[ ! -e "$base/execution.log" ]] || { echo 'Execution receipt exists; inspect before retry'; exit 78; }
exec > >(tee "$base/execution.log") 2>&1
finish() {
  local status=$?
  printf 'exit_status=%s\nfinished=%s\n' "$status" "$(date -Is)" > "$base/execution-status.txt"
}
trap finish EXIT
echo "WAIT-BUILD $(date -Is) receipt=$receipt"
while :; do
  status=$(ssh -n -o BatchMode=yes -o ConnectTimeout=10 dusty \
    "if test -f '$receipt/status.txt'; then cat '$receipt/status.txt'; elif systemctl --user is-active --quiet jj-r32-build.service; then echo RUNNING; else exit 78; fi")
  if [[ $status == RUNNING ]]; then
    sleep 30
    continue
  fi
  grep -qx 'exit_status=0' <<<"$status"
  break
done
export EXPECTED_IMAGE_ID
EXPECTED_IMAGE_ID=$(ssh -n -o BatchMode=yes dusty \
  'podman image inspect localhost/voipmonitor/vllm:jj-r32-spark-sm121 --format "{{.Id}}"')
[[ $EXPECTED_IMAGE_ID =~ ^[0-9a-f]{64}$ ]]
lock=$(sha256sum "$recipe/source.lock.json" | cut -d' ' -f1)
labels=$(ssh -n -o BatchMode=yes dusty \
  'podman image inspect localhost/voipmonitor/vllm:jj-r32-spark-sm121 --format "{{json .Config.Labels}}"')
actual=$(jq -er '."local-inference.runtime.source-lock.sha256"' <<<"$labels")
[[ $actual == "$lock" ]]
printf 'image_id=%s\nsource_lock_sha256=%s\n' "$EXPECTED_IMAGE_ID" "$lock" > "$base/candidate.txt"
mkdir -p "$recipe/build-receipts/$(basename "$receipt")"
rsync -a -e 'ssh -o BatchMode=yes -o Compression=no -c aes128-gcm@openssh.com' \
  "dusty:$receipt/" "$recipe/build-receipts/$(basename "$receipt")/"
grep -Fq "$EXPECTED_IMAGE_ID" "$recipe/build-receipts/$(basename "$receipt")/BUILD-OK"
echo "BUILD-VERIFIED $(date -Is) image=$EXPECTED_IMAGE_ID"
ssh -n -o BatchMode=yes dusty \
  "EXPECTED_IMAGE_ID=$EXPECTED_IMAGE_ID bash '$remote/qualification/distribute-qwen-image.sh'" \
  | tee "$base/transfer.log"
bash "$base/launch-qwen.sh" initial
bash "$base/qualify-qwen.sh" initial
bash "$base/benchmark-qwen.sh"
bash "$base/launch-qwen.sh" aligned-mtp0
bash "$base/qualify-qwen.sh" aligned-mtp0
bash "$base/launch-qwen.sh" aligned-mtp3-restored
echo "QWEN-R32-ALIGNED-BATTERY-COMPLETE $(date -Is)"
echo 'Aligned MTP3 left serving on dusty/kirby after a correct first completion.'
echo 'Benchmark review and any matched repeat remain. GLM, DS4, and auto-policy controls untouched.'
