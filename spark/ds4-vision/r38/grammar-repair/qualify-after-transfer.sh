#!/usr/bin/env bash
# Reviewed driver handoff: proceed immediately after verified distribution.
set -euo pipefail
cd "$(dirname "$0")"
expected=$(jq -r '.[0].Id' receipts/build/image-inspect.json)
deadline=$(( $(date +%s) + 2700 ))
while :; do
  state=$(ssh -n -o BatchMode=yes -o ConnectTimeout=10 rusty \
    'systemctl --user show ds4-r38p-transfer-docker.service -p ActiveState --value')
  case $state in
    active|activating) (( $(date +%s) < deadline )) || exit 1; sleep 10;;
    inactive) break;;
    *) echo "Transfer failed: $state"; exit 1;;
  esac
done
ssh -n -o BatchMode=yes rusty \
  "test \"\$(systemctl --user show ds4-r38p-transfer-docker.service -p ExecMainStatus --value)\" = 0 && grep -qx 'TRANSFER-OK toby $expected' /home/jugs/git/ds4-vision-r38/grammar-repair/receipts/transfer/TRANSFER-OK"
rsync -a rusty:/home/jugs/git/ds4-vision-r38/grammar-repair/receipts/transfer/ receipts/transfer/
export DS4_R38P_APPROVED=1
exec bash "$PWD/execute.sh"
