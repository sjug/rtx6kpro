#!/usr/bin/env bash
set -euo pipefail
[[ $(hostname -s) == rusty ]] || exit 78
root=/home/jugs/git/ds4-vision
repo=models--deepseek-ai--DeepSeek-V4-Flash-Vision-Exp
cache=/home/jugs/.cache/huggingface
deadline=$((SECONDS + 7200))
while systemctl --user is-active --quiet ds4-vision-model-stage.service; do
  (( SECONDS < deadline )) || exit 124
  sleep 10
done
grep -q '^MODEL-STAGED revision=6821d6ad3681a4b137b066b76094fa82ebd0a380' "$root/receipts/download.log"
route=$(ip route get 10.11.11.6)
[[ $route == *'dev enp1s0f0np0 src 10.11.11.5'* ]] || exit 78
transport='ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o HostKeyAlias=toby -o Compression=no -c aes128-gcm@openssh.com'
# shellcheck disable=SC2086
[[ $($transport 10.11.11.6 hostname -s) == toby ]] || exit 78
printf 'MODEL-TRANSFER route=%s\n' "$route"
# shellcheck disable=SC2086
$transport 10.11.11.6 "mkdir -p '$cache/hub/$repo' '$root/receipts'"
# Four independent blob lanes use several ARM crypto cores without compression
# or reserialization. Preserve HF relative snapshot symlinks, never --delete.
for lane in 0 1 2 3; do
  find "$cache/hub/$repo/blobs" -maxdepth 1 -type f ! -name '*.incomplete' \
    -printf 'blobs/%f\n' | sort | awk -v lane="$lane" 'NR % 4 == lane' \
    > "$root/receipts/transfer-lane-$lane.txt"
done
pids=()
for lane in 0 1 2 3; do
  rsync -a --whole-file --inplace --partial --info=progress2 \
    --files-from="$root/receipts/transfer-lane-$lane.txt" -e "$transport" \
    "$cache/hub/$repo/" "10.11.11.6:$cache/hub/$repo/" \
    > "$root/receipts/transfer-lane-$lane.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
((status == 0)) || exit 1
rsync -a --exclude=blobs/ -e "$transport" \
  "$cache/hub/$repo/" "10.11.11.6:$cache/hub/$repo/"
rsync -a -e "$transport" "$root/receipts/model-manifest.json" "10.11.11.6:$root/receipts/"
# shellcheck disable=SC2086
$transport 10.11.11.6 "python3 '$root/verify-model.py' --cache '$cache' --manifest '$root/receipts/model-manifest.json' --hash"
printf 'MODEL-TRANSFER-PASS %s\n' "$(date -Is)"
