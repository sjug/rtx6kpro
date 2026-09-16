#!/usr/bin/env bash
# Fresh matched R32 control and R38 return, only after the two R38 grids finish.
set -euo pipefail
base=$(cd "$(dirname "$0")" && pwd)
[[ ${QWEN_CONTROL_APPROVED:-0} == 1 ]] || exit 78
grep -qx 'exit_status=0' "$base/execution-status.txt"
[[ ! -e $base/contrast-execution.log ]] || exit 78
exec > >(tee "$base/contrast-execution.log") 2>&1
finish() {
  local status=$?
  if [[ -n ${telemetry_pid:-} ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  printf 'exit_status=%s\nfinished=%s\n' "$status" "$(date -Is)" > "$base/contrast-status.txt"
}
trap finish EXIT
r32_id=74e53e710bef141f6f68e722582569f9c6aa388bce405ad6f1423566a2300c9c
r38_id=ea031e1d3d051033f077fc986bf6f8fce04cf9ab52483d5a719ba13114567fc5
for node in dusty kirby; do
  for release in r32 r38; do
    expected=$r32_id; [[ $release != r38 ]] || expected=$r38_id
    actual=$(ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$node" "podman inspect qwen38-flash-next-nvfp4-jj-$release-tp2 --format '{{.Image}}'")
    [[ ${actual#sha256:} == "$expected" ]] || exit 78
  done
done
bash "$base/sample-qwen.sh" "$base/contrast-telemetry" & telemetry_pid=$!
for release in r32 r38; do
  if [[ $release == r32 ]]; then
    previous=r38; profile=contrast-r32; export EXPECTED_IMAGE_ID=$r32_id; repetition=1
  else
    previous=r32; profile=contrast-r38-return; export EXPECTED_IMAGE_ID=$r38_id; repetition=3
  fi
  out=$base/$profile
  mkdir -p "$out"
  [[ ! -e $out/first-completion.json ]] || exit 78
  for node in kirby dusty; do
    ssh -n -o BatchMode=yes "$node" "podman stop -t 60 qwen38-flash-next-nvfp4-jj-$previous-tp2"
    ssh -n -o BatchMode=yes "$node" "podman logs --timestamps qwen38-flash-next-nvfp4-jj-$previous-tp2" > "$out/$node-previous.log" 2>&1
  done
  for node in kirby dusty; do
    ssh -n -o BatchMode=yes "$node" "podman start qwen38-flash-next-nvfp4-jj-$release-tp2"
  done
  deadline=$(( $(date +%s) + 1200 )); ready=0
  while (( $(date +%s) < deadline )); do
    for node in dusty kirby; do
      running=$(ssh -n -o BatchMode=yes -o ConnectTimeout=5 "$node" "podman inspect qwen38-flash-next-nvfp4-jj-$release-tp2 --format '{{.State.Running}}'")
      [[ $running == true ]] || exit 1
    done
    if curl --fail --silent --show-error --connect-timeout 3 --max-time 60 http://dusty:8000/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{"model":"Qwen3.8-Flash-Next","messages":[{"role":"user","content":"Calculate 17 times 23 minus 58. Reply with only the resulting integer."}],"chat_template_kwargs":{"enable_thinking":false},"temperature":0,"max_tokens":32}' > "$out/first-completion.json"; then
      jq -e '.choices[0] | .finish_reason=="stop" and (.message.content|gsub("^\\s+|\\s+$";""))=="333"' "$out/first-completion.json"
      ready=1; break
    fi
    sleep 10
  done
  [[ $ready == 1 ]] || exit 1
  for node in dusty kirby; do
    ssh -n -o BatchMode=yes "$node" "podman inspect qwen38-flash-next-nvfp4-jj-$release-tp2" > "$out/$node-container.json"
    jq -e --arg id "$EXPECTED_IMAGE_ID" '.[0] | (.Image|ltrimstr("sha256:"))==$id and
      (.Config.Env|index("NUM_SPECULATIVE_TOKENS=3")!=null) and
      (.Args|index("--recurrent-checkpoint-policy") as $i | $i!=null and .[$i+1]=="aligned")' "$out/$node-container.json"
    ssh -n -o BatchMode=yes "$node" "podman logs --timestamps qwen38-flash-next-nvfp4-jj-$release-tp2" > "$out/$node-boot.log" 2>&1
  done
  echo "CONTROL-READY $profile $(date -Is)"
  QWEN_RECEIPT_ROOT=$base RELEASE_LABEL=$release bash "$base/qualify-qwen.sh" "$profile"
  REPETITION=$repetition bash "$base/benchmark-qwen-control.sh" "$profile"
done
echo "FRESH-CONTROL-COMPLETE $(date -Is)"
echo 'R38 MTP3 remains serving as the candidate; compare receipts before GLM.'
