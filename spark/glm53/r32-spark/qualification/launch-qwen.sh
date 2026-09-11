#!/usr/bin/env bash
# Only dusty/kirby Qwen. Preserve R29 rollback containers and all model/JIT caches.
set -euo pipefail
profile=${1:?initial|aligned-mtp0|auto-mtp3|aligned-mtp3-restored}
case "$profile" in
  initial|aligned-mtp3-restored) policy=aligned; mtp=3;;
  aligned-mtp0) policy=aligned; mtp=0;;
  auto-mtp3) policy=auto; mtp=3;;
  *) exit 2;;
esac
image=${EXPECTED_IMAGE_ID:?Pin the gated R32 image ID}
[[ "$image" =~ ^[0-9a-f]{64}$ ]] || exit 2
base=$(cd "$(dirname "$0")" && pwd)
out="$base/qwen-20260910/$profile"
mkdir -p "$out"
[[ ! -e "$out/launch.log" ]] || { echo 'Receipt already exists; inspect before repeating.' >&2; exit 78; }
exec > >(tee "$out/launch.log") 2>&1
runner=/home/jugs/git/bld-jj-r32-spark/run-qwen38-flash-next-jj-r32-spark-tp2-node.sh
name=qwen38-flash-next-nvfp4-jj-r32-tp2
echo "START $profile $(date -Is) image=$image"
if [[ $profile != initial ]]; then
  for host in kirby dusty; do
    ssh -n -o BatchMode=yes "$host" "podman logs '$name'" > "$out/$host-previous.log" 2>&1
    ssh -n -o BatchMode=yes "$host" "ROLE=stop bash '$runner'"
  done
fi
ssh -n -o BatchMode=yes kirby "EXPECTED_IMAGE_ID=$image RECURRENT_CHECKPOINT_POLICY=$policy NUM_SPECULATIVE_TOKENS=$mtp ROLE=worker bash '$runner'"
ssh -n -o BatchMode=yes dusty "EXPECTED_IMAGE_ID=$image RECURRENT_CHECKPOINT_POLICY=$policy NUM_SPECULATIVE_TOKENS=$mtp ROLE=head bash '$runner'"
deadline=$(( $(date +%s) + 1200 ))
ready=0
while (( $(date +%s) < deadline )); do
  for host in dusty kirby; do
    running=$(ssh -n -o BatchMode=yes -o ConnectTimeout=5 "$host" "podman inspect '$name' --format '{{.State.Running}}'")
    [[ $running == true ]] || { echo "$host container stopped; inspect before retry"; exit 1; }
  done
  if curl --fail-with-body --silent --show-error --connect-timeout 3 --max-time 60 http://dusty:8000/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{"model":"Qwen3.8-Flash-Next-NVFP4-4p89","messages":[{"role":"user","content":"Calculate 17 times 23 minus 58. Reply with only the resulting integer."}],"chat_template_kwargs":{"enable_thinking":false},"temperature":0,"max_tokens":32}' \
      -o "$out/first-completion.json"; then
    jq -e '.choices[0] | .finish_reason == "stop" and (.message.content | gsub("^\\s+|\\s+$"; "")) == "333"' "$out/first-completion.json"
    ready=1
    break
  fi
  sleep 10
done
[[ $ready == 1 ]] || { echo 'No correct completion within the bounded startup window'; exit 1; }
for host in dusty kirby; do
  ssh -n -o BatchMode=yes "$host" "podman inspect '$name'" > "$out/$host-container.json"
  ssh -n -o BatchMode=yes "$host" "podman logs '$name'" > "$out/$host-boot.log" 2>&1
  jq -e --arg image "$image" --arg policy "$policy" --arg mtp "$mtp" '
    .[0] | (.Image | ltrimstr("sha256:")) == $image and .State.Running and
    (.Args | index("--recurrent-checkpoint-policy") as $i | $i != null and .[$i+1] == $policy) and
    (.Config.Env | index("NUM_SPECULATIVE_TOKENS="+$mtp) != null)
  ' "$out/$host-container.json"
done
echo "PROFILE-READY $profile $(date -Is)"
