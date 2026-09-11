#!/usr/bin/env bash
# Authorized rebuild continuation. Restore tested Qwen, then test corrected GLM.
set -euo pipefail
base=$(cd "$(dirname "$0")" && pwd)
root=/home/jugs/git/bld-jj-r29-spark
receipt=$root/build-receipts/20260909T173029Z-765802
export GLM_RECEIPT_DIR=$base/glm-metadata-rebuild-20260909
mkdir -p "$GLM_RECEIPT_DIR"
[[ ! -e $GLM_RECEIPT_DIR/continuation.log ]] || exit 78
exec > >(tee "$GLM_RECEIPT_DIR/continuation.log") 2>&1
finish() { local code=$?; printf 'exit_status=%s\nfinished=%s\n' "$code" "$(date -Is)" > "$GLM_RECEIPT_DIR/continuation-status.txt"; }
trap finish EXIT
while :; do
  status=$(ssh -n -o BatchMode=yes -o ConnectTimeout=10 dusty "if test -f '$receipt/status.txt'; then cat '$receipt/status.txt'; elif systemctl --user is-active --quiet jj-r29-metadata-rebuild.service; then echo RUNNING; else exit 78; fi")
  [[ $status != RUNNING ]] && break
  sleep 30
done
echo "BUILD-FINISHED $(date -Is) $status"
# These stopped containers pin the already qualified Qwen image by ID, not tag.
for node in kirby dusty; do
  actual=$(ssh -n -o BatchMode=yes "$node" 'podman inspect qwen38-flash-next-nvfp4-jj-r29-tp2 --format "{{.Image}}"')
  [[ $actual == ee996ef8e531eb6e9c3208ebe30b141dc41702da1466e6ab2c47daa666d5694f ]] || exit 78
  ssh -n -o BatchMode=yes "$node" 'podman start qwen38-flash-next-nvfp4-jj-r29-tp2'
done
deadline=$(( $(date +%s) + 1200 ))
ready=0
while (( $(date +%s) < deadline )); do
  if curl --fail --silent --show-error --connect-timeout 3 --max-time 60 http://dusty:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"Qwen3.8-Flash-Next-NVFP4-4p89","messages":[{"role":"user","content":"Calculate 17 times 23 minus 58. Reply with only the resulting integer."}],"chat_template_kwargs":{"enable_thinking":false},"temperature":0,"max_tokens":32}' \
    -o "$GLM_RECEIPT_DIR/qwen-restored.json"; then
    jq -e '.choices[0] | .finish_reason=="stop" and (.message.content|gsub("^\\s+|\\s+$";""))=="333"' "$GLM_RECEIPT_DIR/qwen-restored.json"
    ready=1; break
  fi
  sleep 10
done
[[ $ready == 1 ]] || exit 78
echo "QWEN-RESTORED $(date -Is)"
grep -qx 'exit_status=0' <<< "$status"
export EXPECTED_IMAGE_ID
EXPECTED_IMAGE_ID=$(ssh -n dusty "cat '$receipt/candidate.id'")
EXPECTED_IMAGE_ID=${EXPECTED_IMAGE_ID#sha256:}
[[ $EXPECTED_IMAGE_ID =~ ^[0-9a-f]{64}$ ]] || exit 78
ssh -n dusty "grep -F '$EXPECTED_IMAGE_ID' '$receipt/BUILD-OK'"
lock=$(sha256sum "$base/../source.lock.json" | cut -d' ' -f1)
actual=$(ssh -n dusty "podman image inspect '$EXPECTED_IMAGE_ID' --format '{{index .Config.Labels \"local-inference.runtime.source-lock.sha256\"}}'")
[[ $actual == "$lock" ]] || exit 78
rsync -a "dusty:$receipt/" "$base/../build-receipts/$(basename "$receipt")/"
archive=$root/jj-r29-$EXPECTED_IMAGE_ID.docker.tar
ssh -n dusty "test ! -e '$archive' && podman save --format docker-archive --output '$archive' localhost/voipmonitor/vllm:jj-r29-spark-sm121"
digest=$(ssh -n dusty "sha256sum '$archive'")
digest=${digest%% *}
[[ $digest =~ ^[0-9a-f]{64}$ ]] || exit 78
printf 'image=%s\narchive_sha256=%s\n' "$EXPECTED_IMAGE_ID" "$digest" > "$GLM_RECEIPT_DIR/candidate.txt"
ssh -n dusty "EXPECTED_IMAGE_ID=$EXPECTED_IMAGE_ID ARCHIVE_SHA256=$digest bash '$root/qualification/distribute-glm.sh'"
bash "$base/execute-glm.sh"
