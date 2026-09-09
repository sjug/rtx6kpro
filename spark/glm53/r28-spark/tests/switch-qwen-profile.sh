#!/usr/bin/env bash
# Authorized Qwen-only qualification controls. Preserve HF and JIT caches.
set -euo pipefail
profile=${1:?aligned-mtp0|auto-mtp3|aligned-mtp3-restored}
case $profile in
  aligned-mtp0) policy=aligned; mtp=0;;
  auto-mtp3) policy=auto; mtp=3;;
  aligned-mtp3-restored) policy=aligned; mtp=3;;
  *) exit 2;;
esac
repo=/home/jugs/git/rtx6kpro
out=$repo/spark/glm53/r28-spark/qualification/qwen-20260908/$profile
mkdir -p "$out"
exec > >(tee "$out/launch.log") 2>&1
runner=/home/jugs/git/bld-jj-r28-spark/run-qwen38-flash-next-jj-r28-spark-tp2-node.sh
image=cd93d80b3f9547f70e1e4608cd42af7c4b0f2ff33d92d5e520913cc9bab2d8d1
name=qwen38-flash-next-nvfp4-jj-r28-tp2
echo "START $profile $(date -Is)"
for host in dusty kirby; do
  ssh -o BatchMode=yes "$host" "podman logs '$name'" > "$out/$host-previous.log" 2>&1
done
ssh -o BatchMode=yes kirby "ROLE=stop bash '$runner'"
ssh -o BatchMode=yes dusty "ROLE=stop bash '$runner'"
ssh -o BatchMode=yes kirby "EXPECTED_IMAGE_ID=$image RECURRENT_CHECKPOINT_POLICY=$policy NUM_SPECULATIVE_TOKENS=$mtp ROLE=worker bash '$runner'"
ssh -o BatchMode=yes dusty "EXPECTED_IMAGE_ID=$image RECURRENT_CHECKPOINT_POLICY=$policy NUM_SPECULATIVE_TOKENS=$mtp ROLE=head bash '$runner'"
deadline=$(( $(date +%s) + 1200 ))
ready=0
while (( $(date +%s) < deadline )); do
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
[[ $ready == 1 ]] || { echo 'No correct completion within the 20-minute startup window'; exit 1; }
for host in dusty kirby; do
  ssh -o BatchMode=yes "$host" "podman inspect '$name'" > "$out/$host-container.json"
  jq -e --arg image "$image" --arg policy "$policy" --arg mtp "$mtp" '
    .[0] | (.Image | ltrimstr("sha256:")) == $image and .State.Running and
    (.Args | index("--recurrent-checkpoint-policy") as $i | $i != null and .[$i+1] == $policy) and
    (.Config.Env | index("NUM_SPECULATIVE_TOKENS="+$mtp) != null)
  ' "$out/$host-container.json"
done
echo "PROFILE-READY $profile $(date -Is)"
