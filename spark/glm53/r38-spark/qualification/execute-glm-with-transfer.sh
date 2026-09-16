#!/usr/bin/env bash
# Explicitly authorized independent GLM rollout from idle rusty.
set -euo pipefail
[[ ${GLM_CUTOVER_APPROVED:-0} == 1 ]] || exit 78
base=$(cd "$(dirname "$0")" && pwd)
remote=/home/jugs/git/bld-jj-r38-spark
out=${GLM_RECEIPT_DIR:?new receipt directory required}
mkdir -p "$out"
[[ ! -e $out/distribution.log ]] || exit 78
exec > >(tee "$out/distribution.log") 2>&1
ssh -n -o BatchMode=yes rusty \
  "systemd-run --user --unit=jj-r38-transfer-glm --working-directory='$remote' /bin/bash qualification/distribute-image.sh glm"
while :; do
  state=$(ssh -n -o BatchMode=yes -o ConnectTimeout=10 rusty 'systemctl --user show jj-r38-transfer-glm.service -p ActiveState --value')
  case $state in
    active|activating) sleep 10;;
    inactive) break;;
    *) ssh -n -o BatchMode=yes rusty 'journalctl --user -u jj-r38-transfer-glm.service -n 40 --no-pager'; exit 1;;
  esac
done
status=$(ssh -n -o BatchMode=yes rusty 'systemctl --user show jj-r38-transfer-glm.service -p ExecMainStatus --value')
[[ $status == 0 ]] || exit 1
ssh -n -o BatchMode=yes rusty 'journalctl --user -u jj-r38-transfer-glm.service --no-pager' > "$out/transfer-journal.log"
rsync -a --include='/transfer-glm-*/' --include='/transfer-glm-*/***' --exclude='*' \
  "rusty:$remote/qualification/" "$out/transfer-receipts/"
echo "TRANSFER-COMPLETE $(date -Is)"
bash "$base/execute-glm.sh"
