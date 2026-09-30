#!/usr/bin/env bash
# Controlled qualification restart. Retain each stopped candidate and all R32 containers.
set -euo pipefail
tokens=${1:?Usage: restart-qwen.sh 0|3 retained-suffix}
suffix=${2:?Specify the completed arm name}
[[ $tokens == 0 || $tokens == 3 ]] || exit 2
[[ $suffix =~ ^[a-z0-9-]+$ ]] || exit 2
name=qwen38-flash-next-nvfp4-karmic-tp2
image=f30dc6d9a2a6f6fc0ac9ff8cddb04d9f631f4a87b48ca7cac69254802fe83233
remote=/home/jugs/git/bld-jj-r38-spark/karmic-beta-sm121
recovery_hint() {
  status=$?
  if (( status != 0 )); then
    echo "Restart failed. Inspect both nodes before recovery; do not start a second head."
    echo "Inspect: ssh NODE podman ps -a; retained arm is $name-$suffix."
    echo 'For recovery, stop any current head/worker first; restore the chosen matching pair worker-first.'
    echo 'R32 containers remain untouched: qwen38-flash-next-nvfp4-jj-r32-tp2.'
  fi
}
trap recovery_hint EXIT
base=$(cd "$(dirname "$0")" && pwd)
out="$base/qualification/restarts/$suffix"
[[ ! -e $out ]] || { echo 'Restart receipt exists'; exit 78; }
mkdir -p "$out"
for node in kirby dusty; do
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$node" "podman inspect '$name'" > "$out/$node-before.json"
  jq -e --arg image "$image" '.[0] | .State.Running and .Image==$image' "$out/$node-before.json" >/dev/null
  if ssh -n -o BatchMode=yes "$node" "podman container exists '$name-$suffix'"; then
    echo "Retained container already exists on $node"; exit 78
  fi
done
for node in kirby dusty; do
  ssh -n -o BatchMode=yes "$node" "podman logs --timestamps '$name'" > "$out/$node-final.log" 2>&1
  ssh -n -o BatchMode=yes "$node" "podman stop -t 60 '$name' && podman rename '$name' '$name-$suffix'"
done
for node in kirby dusty; do
  role=worker
  [[ $node != dusty ]] || role='head'
  ssh -n -o BatchMode=yes "$node" \
    "cd '$remote' && ROLE=$role NUM_SPECULATIVE_TOKENS=$tokens EXPECTED_IMAGE_ID=$image bash run-qwen-tp2-node.sh"
done
