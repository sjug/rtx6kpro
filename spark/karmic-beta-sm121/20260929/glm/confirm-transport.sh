#!/usr/bin/env bash
# Transport confirmation (user-approved 2026-09-29): same beta image and profile as the stopped
# attempt, only NCCL changed to LL,Simple on both rails. R38 is stopped worker first and retained.
set -euo pipefail
kit=$(cd "$(dirname "$0")" && pwd)
out=$kit/../qualification/glm-transport
remote=/home/jugs/git/bld-jj-r38-spark/karmic-beta-sm121/20260929/glm
old=glm53-flash-nvfp4-jj-r38-spark-tp4
new=glm53-flash-nvfp4-karmic-beta-20260929-tp4
kept=$new-ll1rail
image=500ae05b98da0658c1a5e1820387f96f2121c5659bd7954ad1c5861f20934f05
nodes=(sparky buddy rocky lucky)
observers=()
trap 'status=$?; for p in "${observers[@]}"; do kill "$p" 2>/dev/null || true; done; echo "EXIT status=$status $(date -Is)"' EXIT
[[ ! -e $out/confirm.log ]] || exit 78
mkdir -p "$out"
exec > >(tee "$out/confirm.log") 2>&1
started=$(date -Is)
echo "START $started"
curl -fsS --max-time 10 http://sparky:8000/metrics > "$out/preflight.metrics"
python3 - "$out/preflight.metrics" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
for m in ('num_requests_running', 'num_requests_waiting'):
    v = re.findall(r'^vllm:' + m + r'\{[^\n]*\} ([\d.e+-]+)$', text, re.M)
    if not v or any(float(x) for x in v):
        raise SystemExit('GLM is not verified idle: ' + m)
PY
for node in "${nodes[@]}"; do
  ssh -n -o BatchMode=yes "$node" "if podman container exists '$new'; then ! podman container exists '$kept' && podman rename '$new' '$kept'; else podman container exists '$kept'; fi"
  scp -q "$kit/run-glm-tp4-node.sh" "$node:$remote/"
  role=worker; [[ $node != sparky ]] || role=head
  ssh -n -o BatchMode=yes "$node" "ROLE=$role DRY_RUN=1 NCCL_DEBUG=INFO bash '$remote/run-glm-tp4-node.sh'" > "$out/$node-render.txt"
  grep -qF 'NCCL_PROTO=LL\,Simple' "$out/$node-render.txt"
  grep -qF 'NCCL_IB_HCA=rocep1s0f0\,roceP2p1s0f0' "$out/$node-render.txt"
done
echo "STOP-R38 $(date -Is)"
for node in buddy rocky lucky sparky; do
  ssh -n -o BatchMode=yes "$node" "podman logs --timestamps '$old' > ~/logs/$old-$node-\$(date -u +%Y%m%dT%H%M%SZ).log 2>&1; podman stop -t 60 '$old'"
done
for node in "${nodes[@]}"; do
  [[ -z $(ssh -n -o BatchMode=yes "$node" 'podman ps -q') ]] || { echo "$node not idle after stop"; exit 78; }
done
echo "START-BETA $(date -Is)"
for node in buddy rocky lucky sparky; do
  role=worker; [[ $node != sparky ]] || role=head
  ssh -n -o BatchMode=yes "$node" "ROLE=$role NCCL_DEBUG=INFO EXPECTED_IMAGE_ID=$image bash '$remote/run-glm-tp4-node.sh'"
done
for node in "${nodes[@]}"; do
  ssh -n -o BatchMode=yes "$node" 'nvidia-smi --query-gpu=timestamp,clocks.sm,utilization.gpu,power.draw --format=csv -l 1' > "$out/$node-gpu.log" 2>&1 & observers+=("$!")
done
deadline=$((SECONDS + 1800))
until curl -fsS --max-time 90 http://sparky:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.3-Flash","messages":[{"role":"user","content":"What is 17 * 23 - 58? Reply with the number only."}],"max_tokens":128,"temperature":0,"reasoning_effort":"low"}' > "$out/first-completion.json" 2>/dev/null \
  && jq -e '.choices[0] | .finish_reason=="stop" and (.message.content|gsub("^\\s+|\\s+$";""))=="333"' "$out/first-completion.json" >/dev/null; do
  for node in "${nodes[@]}"; do
    [[ $(ssh -n -o BatchMode=yes "$node" "podman inspect '$new' --format '{{.State.Running}}'") == true ]] || { echo "$node container stopped"; exit 1; }
  done
  ((SECONDS < deadline)) || { echo 'startup timeout'; exit 1; }
  sleep 20
done
echo "READY $(date -Is)"
for node in "${nodes[@]}"; do
  ssh -n -o BatchMode=yes "$node" "podman inspect '$new'" > "$out/$node-container.json"
  jq -e --arg id "$image" '.[0] | .Image==$id and (.Config.Env|index("NCCL_PROTO=LL,Simple")!=null) and (.Config.Env|index("NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0")!=null)' "$out/$node-container.json" >/dev/null
  ssh -n -o BatchMode=yes "$node" "podman logs '$new' 2>&1 | grep -E 'NCCL INFO (NET/IB|Using network|Channel 0[01]/|comm .* nRanks|NCCL_PROTO|Connected all)' | head -60" > "$out/$node-nccl-info.txt" || true
done
echo "WARMUP $(date -Is)"
python3 "$kit/probe-fresh-prefill.py" "$out/warmup.jsonl" 4096
echo "PROBE $(date -Is)"
python3 "$kit/probe-fresh-prefill.py" "$out/fresh-prefill.jsonl" 8192 8192 131072 131072
echo "CONFIRM-PROBE-COMPLETE $(date -Is)"
