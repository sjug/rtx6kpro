#!/usr/bin/env bash
# User-approved two-boot prefill attribution (2026-09-30): one fresh 16K prefill profiled with the
# torch profiler on R38 (matched defaults) and on the production beta profile. Beta boots last and
# stays serving (the profiler is idle unless /start_profile is called). Nothing is removed: the
# current production beta and the earlier R38 comparison container are renamed and retained.
set -euo pipefail
kit=$(cd "$(dirname "$0")" && pwd)
out=$kit/../qualification/glm-prefill-profile
remote=/home/jugs/git/bld-jj-r38-spark/karmic-beta-sm121/20260929/glm
base=http://sparky:8000
beta=glm53-flash-nvfp4-karmic-beta-20260929-tp4
r38m=glm53-flash-nvfp4-jj-r38-spark-tp4-matched-defaults
beta_image=500ae05b98da0658c1a5e1820387f96f2121c5659bd7954ad1c5861f20934f05
r38_image=ea031e1d3d051033f077fc986bf6f8fce04cf9ab52483d5a719ba13114567fc5
stamp=20260930
host_cache=/home/jugs/.cache/vllm-jj-glm53-flash-tp4
nodes=(sparky buddy rocky lucky)
[[ ! -e $out ]] || { echo "receipt directory exists: $out"; exit 78; }
mkdir -p "$out"
exec > >(tee "$out/profile.log") 2>&1
trap 'status=$?; echo "EXIT status=$status $(date -Is)"' EXIT
echo "START $(date -Is)"

idle() {
  curl -fsS --max-time 10 "$base/metrics" | python3 -c '
import re, sys
text = sys.stdin.read()
for m in ("num_requests_running", "num_requests_waiting"):
    v = re.findall(r"^vllm:" + m + r"\{[^\n]*\} ([\d.e+-]+)$", text, re.M)
    if not v or any(float(x) for x in v):
        raise SystemExit("endpoint not idle: " + m)'
}
stop_all() {
  for node in buddy rocky lucky sparky; do
    ssh -n -o BatchMode=yes "$node" "podman logs --timestamps '$1' > ~/logs/$1-$node-\$(date -u +%Y%m%dT%H%M%SZ).log 2>&1; podman stop -t 60 '$1' >/dev/null"
  done
  for node in "${nodes[@]}"; do
    [[ -z $(ssh -n -o BatchMode=yes "$node" 'podman ps -q') ]] || { echo "$node not idle after stopping $1"; exit 78; }
  done
}
start_all() {  # $1 runner, $2 image, $3 profiler dir
  for node in buddy rocky lucky sparky; do
    role=worker; [[ $node != sparky ]] || role=head
    ssh -n -o BatchMode=yes "$node" "ROLE=$role PROFILER_DIR=$3 EXPECTED_IMAGE_ID=$2 bash '$remote/$1'"
  done
}
wait_ready() {
  local deadline=$((SECONDS + 1800))
  until curl -fsS --max-time 90 "$base/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"GLM-5.3-Flash","messages":[{"role":"user","content":"What is 17 * 23 - 58? Reply with the number only."}],"max_tokens":128,"temperature":0,"reasoning_effort":"low"}' > "$out/$2-first-completion.json" 2>/dev/null \
    && jq -e '.choices[0] | .finish_reason=="stop" and (.message.content|gsub("^\\s+|\\s+$";""))=="333"' "$out/$2-first-completion.json" >/dev/null; do
    for node in "${nodes[@]}"; do
      [[ $(ssh -n -o BatchMode=yes "$node" "podman inspect '$1' --format '{{.State.Running}}'") == true ]] || { echo "$node $1 stopped"; exit 1; }
    done
    ((SECONDS < deadline)) || { echo "startup timeout $1"; exit 1; }
    sleep 20
  done
  echo "READY $2 $(date -Is)"
}
capture() {  # $1 container, $2 arm, $3 image
  for node in "${nodes[@]}"; do
    ssh -n -o BatchMode=yes "$node" "podman inspect '$1'" > "$out/$2-$node-container.json"
    jq -e --arg id "$3" '.[0] | (.Image|ltrimstr("sha256:"))==$id and (.Args|index("--profiler-config")!=null)' "$out/$2-$node-container.json" >/dev/null \
      || { echo "$node $2: image or profiler flag missing"; exit 1; }
  done
  idle
  echo "== $2 warmup and timed fresh prefills $(date -Is) =="
  python3 "$kit/probe-fresh-prefill.py" "$out/$2-timed.jsonl" 4096 16384 16384 16384
  idle
  echo "== $2 profiled fresh 16K $(date -Is) =="
  curl -fsS --max-time 60 -X POST "$base/start_profile" >/dev/null
  python3 "$kit/probe-fresh-prefill.py" "$out/$2-profiled.jsonl" 16384
  curl -fsS --max-time 900 -X POST "$base/stop_profile" >/dev/null
  echo "PROFILE-STOPPED $2 $(date -Is)"
  # Traces are written asynchronously; wait until every rank's directory is non-empty and stable.
  for node in "${nodes[@]}"; do
    dir=$host_cache/glm-prefill-profile-$stamp/$2
    state() { ssh -n -o BatchMode=yes "$node" "echo \$(du -sb '$dir' 2>/dev/null | cut -f1) \$(find '$dir' -name '*.json.gz' 2>/dev/null | wc -l)"; }
    stable=0
    for _ in $(seq 60); do
      a=$(state); sleep 10; b=$(state)
      if [[ $a == "$b" && ${b##* } -ge 1 ]]; then stable=1; break; fi
    done
    ((stable)) || { echo "$node $2: traces not written or not stable"; exit 1; }
    mkdir -p "$out/$2/$node"
    scp -q -r "$node:$dir/." "$out/$2/$node/"
    echo "COPIED $2 $node $(find "$out/$2/$node" -name '*.json.gz' | wc -l) traces $(du -sh "$out/$2/$node" | cut -f1)"
  done
}

idle
for node in "${nodes[@]}"; do
  scp -q "$kit/run-glm-tp4-node.sh" "$kit/run-glm-r38-matched-defaults-node.sh" "$node:$remote/"
  ssh -n -o BatchMode=yes "$node" "! podman container exists '$beta-noprof-$stamp' && ! podman container exists '$r38m-grid-$stamp' && podman container exists '$r38m'"
  ssh -n -o BatchMode=yes "$node" "[ ! -e '$host_cache/glm-prefill-profile-$stamp' ]"
done
echo "STOP-BETA $(date -Is)"
stop_all "$beta"
for node in "${nodes[@]}"; do
  ssh -n -o BatchMode=yes "$node" "podman rename '$beta' '$beta-noprof-$stamp' && podman rename '$r38m' '$r38m-grid-$stamp'"
done

echo "START-R38M-PROFILED $(date -Is)"
start_all run-glm-r38-matched-defaults-node.sh "$r38_image" "/cache/glm-prefill-profile-$stamp/r38m"
wait_ready "$r38m" r38m
capture "$r38m" r38m "$r38_image"
stop_all "$r38m"

echo "START-BETA-PROFILED $(date -Is)"
start_all run-glm-tp4-node.sh "$beta_image" "/cache/glm-prefill-profile-$stamp/beta"
wait_ready "$beta" beta
capture "$beta" beta "$beta_image"
echo "PROFILE-WINDOW-COMPLETE $(date -Is)"
