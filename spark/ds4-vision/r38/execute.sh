#!/usr/bin/env bash
# Authorized DSv4 Vision Exp qualification on rusty/toby, with no model change.
set -euo pipefail
[[ ${DS4_R38_APPROVED:-0} == 1 ]] || exit 78
base=$(cd "$(dirname "$0")" && pwd)
remote=/home/jugs/git/ds4-vision-r38
out=$base/receipts/qualification
[[ ! -e $out/execution.log ]] || exit 78
mkdir -p "$out"
exec > >(tee "$out/execution.log") 2>&1
telemetry=()
finish() {
  local status=$?
  for pid in "${telemetry[@]}"; do kill "$pid" 2>/dev/null || true; done
  printf 'exit_status=%s\nfinished=%s\n' "$status" "$(date -Is)" > "$out/status.txt"
}
trap finish EXIT
expected=ea031e1d3d051033f077fc986bf6f8fce04cf9ab52483d5a719ba13114567fc5
name=ds4-vision-jj-r38-tp2
deadline=$(( $(date +%s) + 1200 ))
while :; do
  state=$(ssh -n -o BatchMode=yes -o ConnectTimeout=10 rusty 'systemctl --user show ds4-r38-image-stage.service -p ActiveState --value')
  case $state in
    active|activating) (( $(date +%s) < deadline )) || exit 1; sleep 10;;
    inactive) break;;
    *) exit 1;;
  esac
done
status=$(ssh -n -o BatchMode=yes rusty 'systemctl --user show ds4-r38-image-stage.service -p ExecMainStatus --value')
[[ $status == 0 ]] || exit 1
rsync -a --include='/transfer-*/' --include='/transfer-*/***' --exclude='*' \
  "rusty:$remote/receipts/" "$base/receipts/"
for node in rusty toby; do
  ssh -n -o BatchMode=yes "$node" "podman image inspect localhost/voipmonitor/vllm:jj-r38-spark-sm121" > "$out/$node-image.json"
  jq -e --arg id "$expected" '.[0].Id==$id' "$out/$node-image.json"
  ssh -n -o BatchMode=yes "$node" "cd '$remote' && sha256sum -c runtime-files.sha256 && python3 verify-model.py --cache /home/jugs/.cache/huggingface --manifest receipts/model-manifest.json"
  running=$(ssh -n -o BatchMode=yes "$node" "podman ps --format '{{.Names}}'")
  [[ -z $running ]] || { echo "Unexpected workload on $node; refusing launch"; exit 78; }
done
for node in toby rusty; do
  role=worker; [[ $node != rusty ]] || role='head'
  ssh -n -o BatchMode=yes "$node" "ROLE=$role bash '$remote/run-node.sh'"
done
for node in rusty toby; do
  ssh -n -o BatchMode=yes "$node" "podman logs --timestamps -f '$name'" > "$out/$node-live.log" 2>&1 & telemetry+=("$!")
  ssh -n -o BatchMode=yes "$node" 'nvidia-smi --query-gpu=timestamp,clocks.sm,clocks_event_reasons.active,temperature.gpu,power.draw --format=csv -l 1' > "$out/$node-gpu.csv" 2>&1 & telemetry+=("$!")
done
deadline=$(( $(date +%s) + 1200 )); ready=0
while (( $(date +%s) < deadline )); do
  for node in rusty toby; do
    running=$(ssh -n -o BatchMode=yes -o ConnectTimeout=5 "$node" "podman inspect '$name' --format '{{.State.Running}} {{.State.OOMKilled}}'")
    [[ $running == 'true false' ]] || exit 1
  done
  if curl --fail --silent --show-error --connect-timeout 3 --max-time 90 http://rusty:8000/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{"model":"DeepSeek-V4-Flash-Vision-Exp","messages":[{"role":"user","content":"Calculate 17 times 23 minus 58. Reply with only the resulting integer."}],"chat_template_kwargs":{"thinking":false},"temperature":0,"max_tokens":64}' > "$out/first-completion.json"; then
    jq -e '.choices[0] | .finish_reason=="stop" and (.message.content|gsub("^\\s+|\\s+$";""))=="333"' "$out/first-completion.json"
    ready=1; break
  fi
  sleep 10
done
[[ $ready == 1 ]] || exit 1
curl --fail --silent --show-error --max-time 10 http://rusty:8000/v1/models > "$out/models.json"
jq -e '[.data[].id]==["DeepSeek-V4-Flash-Vision-Exp"]' "$out/models.json"
for node in rusty toby; do
  ssh -n -o BatchMode=yes "$node" "podman inspect '$name'" > "$out/$node-container.json"
  jq -e --arg id "$expected" '.[0] | (.Image|ltrimstr("sha256:"))==$id and .State.Running and (.State.OOMKilled|not)' "$out/$node-container.json"
done
echo "FIRST-COMPLETION-PASS $(date -Is)"
python3 -u "$base/../qualify.py" --out "$out/semantic" --long
for node in rusty toby; do
  ssh -n -o BatchMode=yes "$node" "podman logs --timestamps '$name'" > "$out/$node-before-grid.log" 2>&1
done
bash "$base/benchmark.sh"
