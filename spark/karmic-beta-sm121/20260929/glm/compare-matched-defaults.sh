#!/usr/bin/env bash
# User-approved window (2026-09-30): R38 and Karmic beta benchmarked on identical request defaults
# (reasoning_effort max, clear_thinking false, temperature 1.0, top_p 0.95) and 4 slots.
# Order: R38 comparison boot, then beta, which stays serving. Production R38 and the
# 2026-09-29 beta (effort high) containers are stopped and retained, never removed.
set -euo pipefail
kit=$(cd "$(dirname "$0")" && pwd)
repo=/home/jugs/git/rtx6kpro
export RESULTS_REPO=$repo  # run_bench records under rtx6kpro/runs, as in qualify-glm.sh
out=$kit/../qualification/glm-matched-defaults
remote=/home/jugs/git/bld-jj-r38-spark/karmic-beta-sm121/20260929/glm
base=http://sparky:8000
model=GLM-5.3-Flash
beta=glm53-flash-nvfp4-karmic-beta-20260929-tp4
beta_high=$beta-effort-high
r38m=glm53-flash-nvfp4-jj-r38-spark-tp4-matched-defaults
beta_image=500ae05b98da0658c1a5e1820387f96f2121c5659bd7954ad1c5861f20934f05
r38_image=ea031e1d3d051033f077fc986bf6f8fce04cf9ab52483d5a719ba13114567fc5
nodes=(sparky buddy rocky lucky)
resume=${RESUME:-}  # empty, or r38m-grid to continue after the r38m semantic step (receipts kept)
case $resume in ''|r38m-grid) ;; *) echo "unknown RESUME=$resume"; exit 2 ;; esac
if [[ -z $resume ]]; then
  [[ ! -e $out ]] || { echo "receipt directory exists: $out"; exit 78; }
  mkdir -p "$out"
fi
exec > >(tee -a "$out/window.log") 2>&1
observers=()
trap 'status=$?; for p in "${observers[@]}"; do kill "$p" 2>/dev/null || true; done; echo "EXIT status=$status $(date -Is)"' EXIT
echo "START $(date -Is)"

idle() {
  curl -fsS --max-time 10 "$base/metrics" > "$out/$1.metrics"
  python3 - "$out/$1.metrics" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
for m in ('num_requests_running', 'num_requests_waiting'):
    v = re.findall(r'^vllm:' + m + r'\{[^\n]*\} ([\d.e+-]+)$', text, re.M)
    if not v or any(float(x) for x in v):
        raise SystemExit('endpoint not idle: ' + m)
PY
}
stop_all() {  # $1 container name; worker first, logs archived, retained
  for node in buddy rocky lucky sparky; do
    ssh -n -o BatchMode=yes "$node" "podman logs --timestamps '$1' > ~/logs/$1-$node-\$(date -u +%Y%m%dT%H%M%SZ).log 2>&1; podman stop -t 60 '$1' >/dev/null"
  done
  for node in "${nodes[@]}"; do
    [[ -z $(ssh -n -o BatchMode=yes "$node" 'podman ps -q') ]] || { echo "$node not idle after stopping $1"; exit 78; }
  done
}
start_all() {  # $1 runner basename, $2 image id
  for node in buddy rocky lucky sparky; do
    role=worker; [[ $node != sparky ]] || role=head
    ssh -n -o BatchMode=yes "$node" "ROLE=$role EXPECTED_IMAGE_ID=$2 bash '$remote/$1'"
  done
}
wait_ready() {  # $1 container name, $2 arm label
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
arm() {  # $1 container, $2 image id, $3 arm label
  for node in "${nodes[@]}"; do
    ssh -n -o BatchMode=yes "$node" "podman inspect '$1'" > "$out/$3-$node-container.json"
    jq -e --arg id "$2" '.[0] | (.Image|ltrimstr("sha256:"))==$id and .State.Running and
      (.Config.Env|index("NCCL_PROTO=LL,Simple")!=null) and (.Config.Env|index("MAX_NUM_SEQS=4")!=null) and
      (.Args|index("--override-generation-config") as $i|$i!=null and .[$i+1]=="{\"temperature\":1.0,\"top_p\":0.95}") and
      (.Args|index("--default-chat-template-kwargs") as $i|$i!=null and .[$i+1]=="{\"reasoning_effort\":\"max\",\"clear_thinking\":false}")' \
      "$out/$3-$node-container.json" >/dev/null || { echo "$node $3 contract mismatch"; exit 1; }
  done
  ssh -n -o BatchMode=yes sparky "podman logs '$1' 2>&1 | grep -E 'non-default args|Default vLLM sampling parameters'" > "$out/$3-args.txt" || true
  grep -q "'top_p': 0.95" "$out/$3-args.txt" || { echo "$3: top_p 0.95 not in effective args"; exit 1; }
  for node in "${nodes[@]}"; do
    ssh -n -o BatchMode=yes "$node" 'nvidia-smi --query-gpu=timestamp,clocks.sm,utilization.gpu,power.draw,clocks_throttle_reasons.active --format=csv -l 1' > "$out/$3-$node-gpu.log" 2>&1 & observers+=("$!")
  done
  if [[ ${4:-} != skip-semantic ]]; then
    echo "== $3 semantic $(date -Is) =="
    if ! python3 "$repo/spark/glm53/verify-semantic-admission.py" --base-url "$base" --model "$model" --runs 1 --receipt-file "$out/$3-semantic.jsonl"; then
      # Performance window: a semantic failure is recorded with the rate-based tool diagnostic
      # (user rule 2026-09-29, baseline 85/100) and judged afterwards; the grid still runs.
      echo "SEMANTIC-FAILED $3; running fixed tool-format diagnostic"
      python3 "$repo/spark/karmic-main-sm121/20260922/glm-defaults/probe_fixed_tool_response.py" --arm "$3-matched-defaults" \
        --corpus "$repo/spark/karmic-main-sm121/20260922/glm-defaults/qualification/tool-response-corpus-20260924.json" \
        --output "$out/$3-fixed-tool-response.jsonl" --trials 100 | tail -2
    fi
  fi
  idle "$3-pre-grid"
  echo '5c79b9760a2381b4b5233f5bbc8f1f279841b46f596dc718a127eaa8eea3e4f2  /home/jugs/git/llm-inference-bench/run_bench.sh' | sha256sum -c -
  echo '2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3  /home/jugs/git/llm-inference-bench/llm_decode_bench.py' | sha256sum -c -
  echo "== $3 grid $(date -Is) =="
  HOST=$base MODEL=$model MODEL_FAMILY=glm-5.3-flash MODEL_VARIANT=nvfp4 \
    CAMPAIGN=2026-09-karmic-beta-sm121-qualification VARIANT="$3-matched-defaults-max-topp095-seq4-tp4-mtp3" CONCURRENCY=1,2,4 \
    /home/jugs/git/llm-inference-bench/run_bench.sh --duration 30 --max-total-tokens 6412288 --display-mode plain --calibration-cache "$out/$3-token-calibration.json" \
    --metadata image_id="$2" --metadata checkpoint_revision=46aaae8a82032f77100f2f03e9cc11b391df3b4d \
    --metadata request_defaults=effort-max_temp1.0_topp0.95 \
    --metadata harness_sha256=2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3 < <(printf 'n\n') > "$out/$3-benchmark.log" 2>&1
  raw=$(sed -n 's/^Results saved to: //p' "$out/$3-benchmark.log" | tail -n 1)
  [[ -f $raw ]] || { echo "$3: missing raw benchmark receipt"; exit 1; }
  echo "$raw" > "$out/$3-raw-path.txt"
  python3 "$repo/spark/glm53/r38-spark/qualification/validate-grid.py" "$raw" > "$out/$3-grid-validation.json"
  wait_ready "$1" "$3-post-grid"
  for p in "${observers[@]}"; do kill "$p" 2>/dev/null || true; done; observers=()
  echo "GRID-DONE $3 $(date -Is)"
}

if [[ $resume == r38m-grid ]]; then
  echo "RESUME r38m-grid $(date -Is)"
  arm "$r38m" "$r38_image" r38m skip-semantic
else
idle preflight
for node in "${nodes[@]}"; do
  scp -q "$kit/run-glm-tp4-node.sh" "$kit/run-glm-r38-matched-defaults-node.sh" "$node:$remote/"
  ssh -n -o BatchMode=yes "$node" "! podman container exists '$r38m' && ! podman container exists '$beta_high'"
done
echo "STOP-BETA-HIGH $(date -Is)"
stop_all "$beta"
for node in "${nodes[@]}"; do ssh -n -o BatchMode=yes "$node" "podman rename '$beta' '$beta_high'"; done

echo "START-R38-MATCHED $(date -Is)"
start_all run-glm-r38-matched-defaults-node.sh "$r38_image"
wait_ready "$r38m" r38m
arm "$r38m" "$r38_image" r38m
fi
stop_all "$r38m"

echo "START-BETA-MAX $(date -Is)"
start_all run-glm-tp4-node.sh "$beta_image"
wait_ready "$beta" beta
arm "$beta" "$beta_image" beta

python3 "$repo/spark/karmic-beta-sm121/compare-grids.py" "$(cat "$out/r38m-raw-path.txt")" "$(cat "$out/beta-raw-path.txt")" > "$out/r38m-vs-beta.json"
echo "WINDOW-COMPLETE $(date -Is)"
