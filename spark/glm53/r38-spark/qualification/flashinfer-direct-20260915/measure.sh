#!/usr/bin/env bash
# FlashInfer only. Never changes a server or writes into the benchmark repo.
set -euo pipefail
base=$(cd "$(dirname "$0")" && pwd)
repo=/home/jugs/git/rtx6kpro
bench=/home/jugs/git/llm-inference-bench
export RESULTS_REPO="$repo" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HOST=http://dusty:8000 MODEL=Qwen3.8-Flash-Next CONCURRENCY=1
export MODEL_FAMILY=qwen3.8-flash-next MODEL_VARIANT=nvfp4-4p89
export CAMPAIGN=2026-09-r38-qwen-flashinfer-direct WORKLOAD=prefill
export VARIANT=r38-flashinfer
image=ea031e1d3d051033f077fc986bf6f8fce04cf9ab52483d5a719ba13114567fc5
output_dir=$repo/runs/$MODEL_FAMILY/$MODEL_VARIANT/$CAMPAIGN/$WORKLOAD
[[ -f "$base/correctness-passed.txt" ]] || exit 78
[[ ! -e "$base/measurement-started.txt" ]] || exit 78
echo '5c79b9760a2381b4b5233f5bbc8f1f279841b46f596dc718a127eaa8eea3e4f2  /home/jugs/git/llm-inference-bench/run_bench.sh' | sha256sum -c -
echo '2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3  /home/jugs/git/llm-inference-bench/llm_decode_bench.py' | sha256sum -c -
date -Is > "$base/measurement-started.txt"
for REPETITION in 1 2; do
  export REPETITION
  measurement_role=warmup
  [[ $REPETITION != 2 ]] || measurement_role=measured
  echo "PREFILL-START $measurement_role $(date -Is)"
  printf 'n\n' | "$bench/run_bench.sh" \
    --prefill-only --prefill-contexts 8k,16k,32k,64k,128k \
    --prefill-duration 10 --prefill-metric auto --display-mode plain \
    --calibration-cache "$base/token-calibration.json" \
    --metadata "image_id=$image" --metadata gdn_screen_arm=flashinfer \
    --metadata "measurement_role=$measurement_role" --metadata boot_repetition=1 \
    --metadata checkpoint_revision=c374e7e24b54f6cb0017d0c2e6d26823d2f2fb5d \
    --metadata launcher_sha256=1caa57eece7fd7470a3ad07190bb72507fc56e89ed7075b825030ba46ed99af0 \
    --metadata run_bench_sha256=5c79b9760a2381b4b5233f5bbc8f1f279841b46f596dc718a127eaa8eea3e4f2 \
    --metadata llm_decode_bench_sha256=2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3 \
    > "$base/prefill-$measurement_role.log" 2>&1
  printf -v suffix '%02d' "$REPETITION"
  mapfile -t records < <(find "$output_dir" -maxdepth 1 -name "*__r38-flashinfer__r$suffix.json")
  [[ ${#records[@]} == 1 ]] || exit 78
  python3 - "${records[0]}" <<'PY'
import json
import sys
from pathlib import Path
record = json.loads(Path(sys.argv[1]).read_text())
prefill = record.get("prefill_summary", record.get("prefill", {}))
if set(prefill) != {"8192", "16384", "32768", "65536", "131072"}:
    raise SystemExit("Missing prefill cells")
for context, row in prefill.items():
    server = row["server_validation"]
    if row["samples"] < 1 or row["tok_per_sec"] <= 0:
        raise SystemExit(f"Invalid client result: {context}")
    if server["cached_tokens"] != 0 or server["invalid_reason"] or server["samples"] < 1 or server["tok_per_sec"] <= 0:
        raise SystemExit(f"Invalid server result: {context}: {server}")
    print(context, row["tok_per_sec"], server["tok_per_sec"], row["samples"])
print("PREFILL-RECEIPT-PASS", sys.argv[1])
PY
  echo "PREFILL-END $measurement_role $(date -Is)"
done
