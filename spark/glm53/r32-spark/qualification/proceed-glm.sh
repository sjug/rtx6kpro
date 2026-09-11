#!/usr/bin/env bash
# Approved transfer, cutover, qualification and comparison in one continuation.
set -euo pipefail
[[ ${GLM_CUTOVER_APPROVED:-0} == 1 ]] || exit 78
base=$(cd "$(dirname "$0")" && pwd)
export EXPECTED_IMAGE_ID=74e53e710bef141f6f68e722582569f9c6aa388bce405ad6f1423566a2300c9c
export GLM_RECEIPT_DIR=$base/glm-20260910
export PYTHONUNBUFFERED=1
mkdir -p "$GLM_RECEIPT_DIR"
[[ ! -e $GLM_RECEIPT_DIR/workflow.log ]] || exit 78
exec > >(tee "$GLM_RECEIPT_DIR/workflow.log") 2>&1
finish() { local status=$?; printf 'exit_status=%s\nfinished=%s\n' "$status" "$(date -Is)" > "$GLM_RECEIPT_DIR/workflow-status.txt"; }
trap finish EXIT
echo "GLM-WORKFLOW-START $(date -Is)"
ssh -n -o BatchMode=yes -o ConnectTimeout=10 dusty \
  "EXPECTED_IMAGE_ID=$EXPECTED_IMAGE_ID ARCHIVE_SHA256=3bffabe501a07bd0640e7eebeb85d872963bfa6da914ee47c3cd735b8e2dbde0 TRANSFER_RECEIPT_DIR=/home/jugs/git/bld-jj-r32-spark/qualification/glm-transfer-20260910 bash /home/jugs/git/bld-jj-r32-spark/qualification/distribute-glm.sh" \
  | tee "$GLM_RECEIPT_DIR/transfer.log"
bash "$base/execute-glm.sh"
jq '{summary,prefill}' "$GLM_RECEIPT_DIR/r29-vs-r32.json"
echo "GLM-WORKFLOW-AND-COMPARISON-COMPLETE $(date -Is)"
