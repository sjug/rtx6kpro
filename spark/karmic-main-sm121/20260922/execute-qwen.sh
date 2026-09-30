#!/usr/bin/env bash
# One approved pair window: cold startup, correctness, then the existing grid.
set -euo pipefail
base=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$base/../../.." && pwd)
image=${EXPECTED_IMAGE_ID:?pin built image ID without sha256 prefix}
remote=${REMOTE_KIT:?staged absolute kit path}
[[ $image =~ ^[0-9a-f]{64}$ && $remote =~ ^/home/jugs/git/[a-zA-Z0-9_./-]+$ && $remote != *..* ]] || exit 78
name=qwen38-flash-next-nvfp4-karmic-main-tp2
export QUAL_ARM=${QUAL_ARM:-baseline}
hc_env=''
runner=run-qwen-tp2-node.sh
case $QUAL_ARM in
  baseline) suffix='' ;;
  hc-off) suffix=-hc-off; hc_env='VLLM_QWEN3_8_FLASH_NEXT_HC_TP=0'; runner=run-qwen-hc-diagnostic.sh ;;
  hc-on-return) suffix=-hc-on-return; hc_env='VLLM_QWEN3_8_FLASH_NEXT_HC_TP=1'; runner=run-qwen-hc-diagnostic.sh ;;
  *) echo 'Unknown qualification arm'; exit 78 ;;
esac
window="$base/qualification/window$suffix"
out="$base/qualification/qwen-mtp3$suffix"
[[ ! -e $window && ! -e $out ]] || { echo 'Window receipt already exists'; exit 78; }
for node in dusty kirby; do
  running=$(ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$node" 'podman ps -q') || exit 78
  [[ -z $running ]] || { echo "$node must be idle"; exit 78; }
  actual=$(ssh -n -o BatchMode=yes "$node" 'podman image inspect localhost/voipmonitor/vllm:karmic-main-spark-sm121 --format "{{.Id}}"')
  [[ ${actual#sha256:} == "$image" ]] || { echo "$node image mismatch"; exit 78; }
done
mkdir -p "$window"
exec > >(tee "$window/execution.log") 2>&1
since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf '%s\n' "$since" > "$window/start.txt"
observers=()
finish() {
  local result=$?
  trap - EXIT
  for pid in "${observers[@]}"; do kill "$pid" 2>/dev/null || true; done
  for node in dusty kirby; do
    ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$node" "podman logs --timestamps '$name'" \
      > "$window/$node-final.log" 2>&1 || true
    ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$node" \
      "journalctl -k --since '$since' --no-pager -o short-iso" > "$window/$node-kernel.log" 2>&1 || true
  done
  printf 'exit_status=%s\n' "$result" > "$window/status.txt"
  exit "$result"
}
trap finish EXIT
for node in dusty kirby; do
  bash "$repo/spark/jj-main-sm121/observe-qwen.sh" "$node" > "$window/$node-telemetry.log" 2>&1 &
  observers+=("$!")
done
for node in kirby dusty; do
  role=worker
  [[ $node != dusty ]] || role='head'
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$node" \
    "cd '$remote' && $hc_env ROLE=$role EXPECTED_IMAGE_ID=$image bash '$runner'"
  ssh -n -o BatchMode=yes "$node" "podman inspect '$name'" > "$window/$node-container.json"
  if [[ $QUAL_ARM == hc-off ]]; then
    jq -e '.[0].Config.Env | index("VLLM_QWEN3_8_FLASH_NEXT_HC_TP=0") != null' \
      "$window/$node-container.json" >/dev/null
  fi
done
python3 -u "$base/verify-hc-boot.py" --arm "$QUAL_ARM" --out "$window"
python3 -u "$base/qualify.py" --out "$out"
for node in dusty kirby; do cp "$window/$node-container.json" "$out/"; done
python3 -u "$repo/spark/glm53/r38-spark/qualification/probes/probe-qwen-head-of-line.py" \
  --policy aligned --receipt-file "$out/head-of-line.jsonl"
python3 -u "$repo/spark/glm53/r38-spark/qualification/probes/probe-concurrency-identical-vs-distinct.py" \
  | tee "$out/identical-vs-distinct.log"
bash "$base/benchmark-qwen.sh"
echo 'QWEN-KARMIC-MAIN-EXECUTION-COMPLETE: review correctness, performance and health before promotion'
