#!/usr/bin/env bash
# Restore the preserved qualified pair only after the comparison has finished.
set -euo pipefail
[[ ${RESTORE_QUALIFIED_R32:-0} == 1 ]] || exit 78
base=$(cd "$(dirname "$0")" && pwd)
grep -qx 'exit_status=0' "$base/contrast-status.txt"
out=$base/restore-r32
[[ ! -e $out/execution.log ]] || exit 78
mkdir -p "$out"
exec > >(tee "$out/execution.log") 2>&1
finish() {
  local status=$?
  printf 'exit_status=%s\nfinished=%s\n' "$status" "$(date -Is)" > "$out/status.txt"
}
trap finish EXIT
qualified=74e53e710bef141f6f68e722582569f9c6aa388bce405ad6f1423566a2300c9c
candidate=ea031e1d3d051033f077fc986bf6f8fce04cf9ab52483d5a719ba13114567fc5
old=qwen38-flash-next-nvfp4-jj-r32-tp2
new=qwen38-flash-next-nvfp4-jj-r38-tp2
for node in dusty kirby; do
  for release in r32 r38; do
    expected=$qualified; [[ $release != r38 ]] || expected=$candidate
    ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$node" \
      "podman inspect qwen38-flash-next-nvfp4-jj-$release-tp2" > "$out/$node-$release-before.json"
    jq -e --arg id "$expected" --arg release "$release" \
      '.[0] | (.Image|ltrimstr("sha256:"))==$id and
       (.State.Running == ($release=="r38"))' "$out/$node-$release-before.json"
  done
done
for node in kirby dusty; do
  ssh -n -o BatchMode=yes "$node" "podman stop -t 60 '$new'"
  ssh -n -o BatchMode=yes "$node" "podman logs --timestamps '$new'" > "$out/$node-r38.log" 2>&1
done
for node in kirby dusty; do
  ssh -n -o BatchMode=yes "$node" "podman start '$old'"
done
deadline=$(( $(date +%s) + 1200 )); ready=0
while (( $(date +%s) < deadline )); do
  for node in dusty kirby; do
    running=$(ssh -n -o BatchMode=yes -o ConnectTimeout=5 "$node" "podman inspect '$old' --format '{{.State.Running}}'")
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
  ssh -n -o BatchMode=yes "$node" "podman inspect '$old'" > "$out/$node-final.json"
  jq -e --arg id "$qualified" '.[0] | (.Image|ltrimstr("sha256:"))==$id and .State.Running and
    (.State.OOMKilled|not) and (.Config.Env|index("NUM_SPECULATIVE_TOKENS=3")!=null) and
    (.Args|index("--recurrent-checkpoint-policy") as $i|$i!=null and .[$i+1]=="aligned")' "$out/$node-final.json"
  ssh -n -o BatchMode=yes "$node" "podman logs --timestamps '$old'" > "$out/$node-r32.log" 2>&1
done
echo "R32-RESTORED $(date -Is)"
