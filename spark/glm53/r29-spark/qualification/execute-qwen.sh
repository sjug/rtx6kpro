#!/usr/bin/env bash
# Authorized R29 chain. Any failure stops here; never touches GLM or DS4.
set -euo pipefail
base=$(cd "$(dirname "$0")" && pwd)
recipe=$(dirname "$base")
remote=/home/jugs/git/bld-jj-r29-spark
[[ ! -e "$base/execution.log" ]] || { echo 'Execution receipt exists; inspect before retry'; exit 78; }
exec > >(tee "$base/execution.log") 2>&1
finish() {
  local status=$?
  printf 'exit_status=%s\nfinished=%s\n' "$status" "$(date -Is)" > "$base/execution-status.txt"
}
trap finish EXIT
echo "WAIT-BUILD $(date -Is)"
receipt=$(ssh -n -o BatchMode=yes -o ConnectTimeout=10 dusty \
  'find /home/jugs/git/bld-jj-r29-spark/build-receipts -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1')
[[ $receipt =~ ^/home/jugs/git/bld-jj-r29-spark/build-receipts/[0-9]{8}T[0-9]{6}Z-[0-9]+$ ]]
echo "BUILD-RECEIPT $receipt"
while :; do
  status=$(ssh -n -o BatchMode=yes -o ConnectTimeout=10 dusty \
    "if test -f '$receipt/status.txt'; then cat '$receipt/status.txt'; elif systemctl --user is-active --quiet jj-r29-build.service; then echo RUNNING; else exit 78; fi")
  if [[ $status == RUNNING ]]; then
    sleep 30
    continue
  fi
  grep -qx 'exit_status=0' <<<"$status"
  break
done
export EXPECTED_IMAGE_ID
EXPECTED_IMAGE_ID=$(ssh -n -o BatchMode=yes dusty \
  'podman image inspect localhost/voipmonitor/vllm:jj-r29-spark-sm121 --format "{{.Id}}"')
[[ $EXPECTED_IMAGE_ID =~ ^[0-9a-f]{64}$ ]]
lock=$(sha256sum "$recipe/source.lock.json" | cut -d' ' -f1)
actual=$(ssh -n -o BatchMode=yes dusty \
  'podman image inspect localhost/voipmonitor/vllm:jj-r29-spark-sm121 --format "{{index .Config.Labels \"local-inference.runtime.source-lock.sha256\"}}"')
[[ $actual == "$lock" ]]
printf 'image_id=%s\nsource_lock_sha256=%s\n' "$EXPECTED_IMAGE_ID" "$lock" > "$base/candidate.txt"
rsync -a -e 'ssh -o Compression=no -c aes128-gcm@openssh.com' \
  "dusty:$remote/build-receipts/" "$recipe/build-receipts/"
grep -Fq "$EXPECTED_IMAGE_ID" "$recipe/build-receipts/$(basename "$receipt")/BUILD-OK"
echo "BUILD-VERIFIED $(date -Is) $EXPECTED_IMAGE_ID"
ssh -n -o BatchMode=yes dusty \
  "EXPECTED_IMAGE_ID=$EXPECTED_IMAGE_ID bash '$remote/qualification/distribute-qwen-image.sh'" \
  | tee "$base/transfer.log"
bash "$base/launch-qwen.sh" initial
bash "$base/qualify-qwen.sh" initial
bash "$base/benchmark-qwen.sh"
results=/home/jugs/git/llm-inference-bench/results/runs/qwen3.8-flash-next/nvfp4-4p89
baseline=$results/2026-09-jj-r28-vs-r27/throughput/20260908T160143-0400__jj-r28-aligned-sm121-tp2-mtp3__r01.json
mapfile -t candidates < <(find "$results/2026-09-jj-r29-vs-r28/throughput" -maxdepth 1 -type f -name '*__jj-r29-aligned-sm121-tp2-mtp3__r01.json')
[[ ${#candidates[@]} == 1 ]]
python3 "$base/compare-qwen-grids.py" "$baseline" "${candidates[0]}" > "$base/r28-vs-r29.json"
bash "$base/launch-qwen.sh" aligned-mtp0
bash "$base/qualify-qwen.sh" aligned-mtp0
bash "$base/auto-control.sh"
bash "$base/launch-qwen.sh" aligned-mtp3-restored
bash "$base/qualify-qwen.sh" aligned-mtp3-restored
echo "QWEN-R29-QUALIFICATION-COMPLETE $(date -Is)"
echo 'Aligned MTP3 left on dusty/kirby. GLM and DS4 untouched. Results require review before any promotion claim.'
