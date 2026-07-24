#!/usr/bin/env bash
# v18p4 Grid48: fan out image, redeploy GLM cluster, verify Grid48 arms on
# every rank, sweep from workstation. Run AFTER the rusty build completes.
set -uo pipefail

TAG=voipmonitor/vllm:gilded-gnosis-v18p4-grid48-spark-sm121-vllmdf7a0b7-b12x6b10833-fi801d57a-cu132-20260718
LTAG=localhost/$TAG
OLD_NAME=glm52-v18-tp4
NEW_NAME=glm52-v18p4-grid48-tp4
BENCH_DIR="$HOME/git/llm-inference-bench"
CIPHER=aes128-gcm@openssh.com

step() { echo "=== [$(date +%H:%M:%S)] $*"; }

# --- Phase A: fan-out ------------------------------------------------------
IMGID=$(ssh rusty "podman images --format '{{.ID}}' $LTAG" < /dev/null)
[[ -n "$IMGID" ]] || { echo "ABORT: image not on rusty"; exit 1; }
step "fan-out of $IMGID: rusty -> toby-ib (200G) and rusty -> sparky (workstation relay), parallel"
ssh rusty "podman save $LTAG | ssh -c $CIPHER toby-ib 'podman load'" < /dev/null &
P1=$!
ssh rusty "podman save $LTAG" < /dev/null | ssh sparky 'podman load' &
P2=$!
wait $P1 || { echo "ABORT: rusty->toby transfer failed"; exit 1; }
wait $P2 || { echo "ABORT: rusty->sparky transfer failed"; exit 1; }

# 200G switch fabric is 10.11.11.0/24 (sparky .1, buddy .2, lucky .3,
# rocky .4); the *-ib hostnames are the back-to-back pair links only.
step "cluster 200G-switch fan from sparky: -> buddy(.2), lucky(.3), rocky(.4) (parallel)"
declare -A PEER_IP=([buddy]=10.11.11.2 [lucky]=10.11.11.3 [rocky]=10.11.11.4)
for peer in buddy lucky rocky; do
  ssh sparky "podman save $LTAG | ssh -c $CIPHER -o StrictHostKeyChecking=accept-new ${PEER_IP[$peer]} 'podman load'" < /dev/null &
  eval "P_${peer}=\$!"
done
wait "$P_buddy" || { echo "ABORT: sparky->buddy transfer failed"; exit 1; }
wait "$P_lucky" || { echo "ABORT: sparky->lucky transfer failed"; exit 1; }
wait "$P_rocky" || { echo "ABORT: sparky->rocky transfer failed"; exit 1; }

for n in toby sparky buddy rocky lucky; do
  got=$(ssh "$n" "podman images --format '{{.ID}}' $LTAG" < /dev/null)
  [[ "$got" == "$IMGID" ]] || { echo "ABORT: image ID mismatch on $n: '$got'"; exit 1; }
  echo "    $n: $got OK"
done
step "ALL_TRANSFERS_VERIFIED"

# --- Phase B: check for foreign activity, stage runner ---------------------
for n in sparky buddy rocky lucky; do
  act=$(ssh "$n" 'who | grep -v "^jugs.*(10\.99\." ; tmux ls 2>/dev/null' < /dev/null)
  [[ -z "$act" ]] || echo "    NOTE: activity on $n: $act"
done

for n in sparky buddy rocky lucky; do
  ssh "$n" 'mkdir -p ~/v18.4' < /dev/null
  scp -q ~/v18.4-local/run-glm52-v18.4-grid48-hybrid-tp4-node.sh "$n":~/v18.4/ \
    || { echo "ABORT: stage failed on $n"; exit 1; }
  ssh "$n" 'chmod +x ~/v18.4/run-glm52-v18.4-grid48-hybrid-tp4-node.sh' < /dev/null
done
step "runner staged on all 4 nodes"

# --- Phase C: graceful v18 teardown (workers first, head last) -------------
for n in buddy rocky lucky sparky; do
  ssh "$n" "if podman container exists $OLD_NAME 2>/dev/null; then mkdir -p ~/logs; podman logs $OLD_NAME > ~/logs/$OLD_NAME-pre-v18p4-\$(date +%Y%m%d_%H%M%S).log 2>&1; podman stop -t 60 $OLD_NAME >/dev/null; podman rm $OLD_NAME >/dev/null; fi; ! podman container exists $OLD_NAME 2>/dev/null" < /dev/null \
    || { echo "ABORT: teardown failed on $n"; exit 1; }
  echo "    $n: v18 container archived+removed"
done

# also ensure no stale new-name container anywhere
for n in sparky buddy rocky lucky; do
  ssh "$n" "! podman container exists $NEW_NAME 2>/dev/null || { podman stop -t 60 $NEW_NAME; podman rm $NEW_NAME; }" < /dev/null
done

# --- Phase D: launch v18p4 (workers first, then head) ----------------------
LAUNCH_ENV="MTP=3 MAX_MODEL_LEN=262144 GPU_MEM=0.86"
for n in buddy rocky lucky; do
  ssh "$n" "cd ~/v18.4 && $LAUNCH_ENV ./run-glm52-v18.4-grid48-hybrid-tp4-node.sh" < /dev/null \
    || { echo "ABORT: worker launch failed on $n"; exit 1; }
done
sleep 5
ssh sparky "cd ~/v18.4 && $LAUNCH_ENV ./run-glm52-v18.4-grid48-hybrid-tp4-node.sh" < /dev/null \
  || { echo "ABORT: head launch failed on sparky"; exit 1; }
step "all 4 ranks launched"

# --- Phase E: health + Grid48 proof ----------------------------------------
deadline=$((SECONDS + 2400))
healthy=0
while (( SECONDS < deadline )); do
  if curl -sf -m 5 http://sparky:8000/v1/models > /dev/null 2>&1; then healthy=1; break; fi
  for n in sparky buddy rocky lucky; do
    state=$(ssh "$n" "podman inspect --format '{{.State.Status}}' $NEW_NAME 2>/dev/null" < /dev/null 2>/dev/null)
    [[ "$state" == "running" ]] || { echo "ABORT: $n state='$state' during boot"; exit 1; }
  done
  sleep 30
done
(( healthy )) || { echo "ABORT: health timeout"; exit 1; }
step "endpoint healthy"

armed=0
for n in sparky buddy rocky lucky; do
  if ssh "$n" "podman logs $NEW_NAME 2>&1 | grep -aq 'armed exact TP4 Grid48 one-grid decode'" < /dev/null; then
    echo "    $n: ARMED Grid48"
    armed=$((armed+1))
  else
    reason=$(ssh "$n" "podman logs $NEW_NAME 2>&1 | grep -a 'mapped one-grid decode unavailable' | tail -1" < /dev/null)
    echo "    $n: NOT ARMED ${reason:+— $reason}"
  fi
done
step "armed ranks: $armed/4"

# trigger a decode so the executing path fires (cc1 + MTP3 -> m=4)
curl -sf -m 120 http://sparky:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.2-MXFP8-NVFP4-NF3-Hybrid","prompt":"The three laws of thermodynamics are","max_tokens":64}' > /dev/null \
  || echo "    WARN: probe completion failed"
sleep 3
exec_count=0
for n in sparky buddy rocky lucky; do
  if ssh "$n" "podman logs $NEW_NAME 2>&1 | grep -aq 'executing TP4 Grid48 one-grid decode'" < /dev/null; then
    echo "    $n: EXECUTING Grid48"
    exec_count=$((exec_count+1))
  else
    echo "    $n: no executing line"
  fi
done
step "executing ranks: $exec_count/4"
if (( armed < 4 || exec_count < 4 )); then
  echo "GRID48_PROOF_INCOMPLETE (armed=$armed executing=$exec_count) — continuing to KV extraction but bench is only meaningful at 4/4"
fi

kv=$(ssh sparky "podman logs $NEW_NAME 2>&1 | grep -ao 'GPU KV cache size: [0-9,]* tokens' | tail -1" < /dev/null | grep -o '[0-9,]*' | tr -d ,)
step "KV capacity: ${kv:-unknown} tokens (v18 was 400576)"

# --- Phase F: settle + sweep -----------------------------------------------
step "settling 300s"
sleep 300
budget=400576
if [[ -n "$kv" && "$kv" -lt 400576 ]]; then budget=$kv; echo "    NOTE: budget reduced to $budget"; fi
step "sweep from workstation vs sparky (budget $budget)"
( cd "$BENCH_DIR" && HOST=http://sparky:8000 MODEL=GLM-5.2-MXFP8-NVFP4-NF3-Hybrid \
  CONCURRENCY=1,2,3,4,8 LABEL=glm52-v18p4-grid48-spark-tp4-dcp1-mtp3-gpu086-len256k \
  ./run_bench.sh --contexts 0,16384,32768,65536,131072 --duration 30 --max-total-tokens "$budget" ) \
  || { echo "BENCH FAILED (server left running)"; exit 1; }
step "GRID48_PIPELINE_COMPLETE"
