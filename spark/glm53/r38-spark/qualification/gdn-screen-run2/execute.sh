#!/usr/bin/env bash
# Authorized Qwen-only GDN backend screen. Never touches GLM or DS4 nodes.
set -euo pipefail
[[ ${QWEN_GDN_SCREEN_APPROVED:-0} == 1 ]] || exit 78
base=$(cd "$(dirname "$0")" && pwd)
repo=/home/jugs/git/rtx6kpro
bench=/home/jugs/git/llm-inference-bench
campaign=2026-09-r38-qwen-gdn-prefill-screen
results=$bench/results/runs/qwen3.8-flash-next/nvfp4-4p89/$campaign/prefill
remote=/home/jugs/git/bld-jj-r38-spark/experiments/gdn-screen
image=ea031e1d3d051033f077fc986bf6f8fce04cf9ab52483d5a719ba13114567fc5
qualified=74e53e710bef141f6f68e722582569f9c6aa388bce405ad6f1423566a2300c9c
r32=qwen38-flash-next-nvfp4-jj-r32-tp2
[[ ! -e $base/execution.log ]] || { echo 'Existing execution receipt; inspect first'; exit 78; }
exec > >(tee "$base/execution.log") 2>&1
export PYTHONUNBUFFERED=1
active=
changed=0
telemetry_pid=

capture() {
  local out=$1 name=$2 node
  for node in dusty kirby; do
    ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$node" "podman inspect '$name'" > "$out/$node-container.json"
    ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$node" "podman logs --timestamps --since '$boot_start' '$name'" > "$out/$node.log" 2>&1
  done
}
ready() {
  local out=$1 name=$2 deadline node running
  deadline=$(( $(date +%s) + 1200 ))
  while (( $(date +%s) < deadline )); do
    for node in dusty kirby; do
      running=$(ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$node" "podman inspect '$name' --format '{{.State.Running}}'")
      [[ $running == true ]] || return 1
    done
    if curl --fail --silent --show-error --connect-timeout 3 --max-time 60 http://dusty:8000/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{"model":"Qwen3.8-Flash-Next","messages":[{"role":"user","content":"Calculate 17 times 23 minus 58. Reply with only the resulting integer."}],"chat_template_kwargs":{"enable_thinking":false},"temperature":0,"max_tokens":32}' > "$out/first-completion.json"; then
      jq -e '.choices[0] | .finish_reason=="stop" and (.message.content|gsub("^\\s+|\\s+$";""))=="333"' "$out/first-completion.json" || return 1
      return 0
    fi
    sleep 10
  done
  return 1
}
stop_pair() {
  local name=$1 node
  for node in kirby dusty; do
    ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$node" "podman stop -t 60 '$name'"
  done
}
restore() {
  local out=$base/restore-r32 node
  mkdir -p "$out"
  if [[ -n $active ]]; then stop_pair "$active" || return 1; fi
  boot_start=$(date -u +%FT%TZ)
  for node in kirby dusty; do
    ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$node" "podman start '$r32'" || return 1
  done
  ready "$out" "$r32" || return 1
  capture "$out" "$r32" || return 1
  for node in dusty kirby; do
    jq -e --arg id "$qualified" '.[0] | .State.Running and (.Image|ltrimstr("sha256:"))==$id' "$out/$node-container.json" || return 1
  done
  echo "R32-RESTORED $(date -Is)"
}
finish() {
  local status=$? restore_status=0
  trap - EXIT
  set +e
  if (( changed )); then restore; restore_status=$?; fi
  if [[ -n $telemetry_pid ]]; then kill "$telemetry_pid"; wait "$telemetry_pid"; fi
  printf 'test_exit_status=%s\nrestore_exit_status=%s\nfinished=%s\n' "$status" "$restore_status" "$(date -Is)" > "$base/status.txt"
  (( restore_status == 0 )) || exit 1
  exit "$status"
}
trap finish EXIT

echo "START $(date -Is)"
echo '2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3  /home/jugs/git/llm-inference-bench/llm_decode_bench.py' | sha256sum -c -
echo '5c79b9760a2381b4b5233f5bbc8f1f279841b46f596dc718a127eaa8eea3e4f2  /home/jugs/git/llm-inference-bench/run_bench.sh' | sha256sum -c -
cmp "$base/campaign.yaml" "$bench/results/runs/qwen3.8-flash-next/nvfp4-4p89/$campaign/campaign.yaml"
for node in dusty kirby; do
  ssh -n -o BatchMode=yes "$node" "podman inspect '$r32'" > "$base/$node-original-r32.json"
  jq -e --arg id "$qualified" '.[0] | .State.Running and (.Image|ltrimstr("sha256:"))==$id' "$base/$node-original-r32.json"
  actual=$(ssh -n -o BatchMode=yes "$node" "podman image inspect localhost/voipmonitor/vllm:jj-r38-spark-sm121 --format '{{.Id}}'")
  [[ ${actual#sha256:} == "$image" ]] || exit 78
done
curl --fail --silent --show-error --max-time 10 http://dusty:8000/metrics > "$base/before.metrics"
awk '/^vllm:num_requests_(running|waiting)\{/ {seen++; if ($NF != 0) bad=1} END {exit (seen<2 || bad)}' "$base/before.metrics"
bash "$base/../sample-qwen.sh" "$base/telemetry" & telemetry_pid=$!
changed=1
stop_pair "$r32"

for turn in 1 2 3 4; do
  arm=b12x; (( turn % 2 == 1 )) || arm=flashinfer
  boot=$(( (turn+1)/2 ))
  out=$base/$arm-boot$boot
  mkdir -p "$out"
  name=qwen38-flash-next-r38-gdn-screen-$arm
  boot_start=$(date -u +%FT%TZ)
  printf 'arm=%s\nboot=%s\nstarted=%s\n' "$arm" "$boot" "$boot_start" > "$out/boot.txt"
  active=$name
  for node in kirby dusty; do
    if (( boot == 1 )); then
      role=worker; [[ $node != dusty ]] || role='head'
      ssh -n -o BatchMode=yes "$node" "GDN_SCREEN_ARM=$arm EXPECTED_IMAGE_ID=$image ROLE=$role bash '$remote/runner.sh'"
    else
      ssh -n -o BatchMode=yes "$node" "podman start '$name'"
    fi
  done
  if ! ready "$out" "$name"; then capture "$out" "$name"; exit 1; fi
  capture "$out" "$name"
  for node in dusty kirby; do
    jq -e --arg id "$image" --arg arm "$arm" '.[0] | .State.Running and (.Image|ltrimstr("sha256:"))==$id and (.Config.Env|index("GDN_SCREEN_ARM="+$arm)!=null) and (.Config.Env|index("NUM_SPECULATIVE_TOKENS=3")!=null)' "$out/$node-container.json"
    if [[ $arm == b12x ]]; then
      grep -q 'Using b12x CuTeDSL GDN prefill kernel' "$out/$node.log"
      grep -q 'GDN decode kernel: b12x' "$out/$node.log"
    else
      grep -q 'Using FlashInfer GDN prefill kernel' "$out/$node.log"
      grep -Eq 'GDN decode kernel: (cuda|triton)' "$out/$node.log"
    fi
  done
  echo "CORRECTNESS-START $arm boot=$boot $(date -Is)"
  common=(--base-url http://dusty:8000 --model Qwen3.8-Flash-Next)
  python3 "$repo/spark/qwen38-flash-next/verify-semantic-admission.py" "${common[@]}" --runs 1 --receipt-file "$out/semantic.jsonl" > "$out/semantic.log" 2>&1
  python3 "$repo/spark/qwen38-flash-next/verify-native-context.py" "${common[@]}" --needle 739184 --lengths 2848 2849 131072 --max-tokens 64 --receipt-file "$out/needle.jsonl" > "$out/needle.log" 2>&1
  echo "CORRECTNESS-PASS $arm boot=$boot $(date -Is)"
  for pass in warmup measured; do
    repetition=$((2*boot-1)); [[ $pass != measured ]] || repetition=$((2*boot))
    echo "BENCH-START $arm boot=$boot $pass $(date -Is)"
    printf 'n\n' | HOST=http://dusty:8000 MODEL=Qwen3.8-Flash-Next \
      MODEL_FAMILY=qwen3.8-flash-next MODEL_VARIANT=nvfp4-4p89 \
      CAMPAIGN=$campaign VARIANT=r38-$arm WORKLOAD=prefill CONCURRENCY=1 REPETITION=$repetition \
      "$bench/run_bench.sh" --prefill-only --prefill-contexts 8k,16k,32k,64k,128k \
      --prefill-duration 10 --prefill-metric auto \
      --metadata "image_id=$image" --metadata "gdn_screen_arm=$arm" \
      --metadata "boot_repetition=$boot" --metadata "measurement_role=$pass" \
      --metadata "launcher_sha256=$(sha256sum "$base/launcher.sh" | cut -d' ' -f1)" \
      --metadata checkpoint_revision=c374e7e24b54f6cb0017d0c2e6d26823d2f2fb5d \
      --metadata llm_decode_bench_sha256=2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3 \
      --metadata run_bench_sha256=5c79b9760a2381b4b5233f5bbc8f1f279841b46f596dc718a127eaa8eea3e4f2 \
      > "$out/benchmark-$pass.log" 2>&1
    printf -v rep '%02d' "$repetition"
    mapfile -t paths < <(find "$results" -maxdepth 1 -name "*__r38-${arm}__r$rep.json")
    [[ ${#paths[@]} == 1 ]] || exit 78
    python3 "$base/validate-prefill.py" "${paths[0]}" > "$out/validation-$pass.json"
    echo "BENCH-PASS $arm boot=$boot $pass $(date -Is)"
  done
  capture "$out" "$name"
  stop_pair "$name"
  active=
done
echo "SCREEN-COMPLETE $(date -Is)"
