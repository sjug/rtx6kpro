#!/usr/bin/env bash
# Standard harness, unchanged source. Run only after this boot's output gates.
set -euo pipefail
root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
kit="$root/spark/ds41/r38"
receipts="$kit/receipts/20260916"
harness=/home/jugs/git/llm-inference-bench
[[ $(sha256sum "$harness/run_bench.sh" | cut -d' ' -f1) == 5c79b9760a2381b4b5233f5bbc8f1f279841b46f596dc718a127eaa8eea3e4f2 ]] || { echo 'Harness wrapper changed' >&2; exit 78; }
[[ $(sha256sum "$harness/llm_decode_bench.py" | cut -d' ' -f1) == 2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3 ]] || { echo 'Harness source changed' >&2; exit 78; }
grep -Fxq DS41-QUALIFICATION-PASS "$receipts/admission-k7-1m-u80.log"
grep -Fxq DS41-CONVERSATIONS-PASS "$receipts/conversations-k7-1m-u80.log"
for host in dusty toby rusty kirby; do
  identity=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" \
    "podman inspect ds41-flash-jj-r38-tp4 --format '{{.Image}} {{.State.Running}} {{index .Config.Labels \"local-inference.ds41.kit.sha256\"}}'")
  [[ "$identity" == 'ea031e1d3d051033f077fc986bf6f8fce04cf9ab52483d5a719ba13114567fc5 true 03ebcfe84135e768c4ae78dbd5472e8ea552217f6c90e76fd395df058170e3f3' ]] || { echo "Wrong identity on $host: $identity" >&2; exit 78; }
done
export RESULTS_REPO="$root" HOST=http://dusty:8000 MODEL=DeepSeek-V4.1-Flash
export MODEL_FAMILY=deepseek-v4.1-flash MODEL_VARIANT=official
export CAMPAIGN=2026-09-r38-sm121-tp4-disk-engram
export VARIANT=r38-tp4-disk-dspark7-native1m-u80 CONCURRENCY=1,2,4
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# Decline self-update: this checkout contains user-owned metadata changes.
printf 'n\n' | bash "$harness/run_bench.sh" \
  --display-mode plain --no-hw-monitor \
  --calibration-cache "$receipts/benchmark-calibration.json" \
  --duration 30 --contexts 0,16k,32k,64k,128k --max-tokens 8192 \
  --kv-budget 11682954 --dcp-size 1 \
  --metadata image_id=ea031e1d3d051033f077fc986bf6f8fce04cf9ab52483d5a719ba13114567fc5 \
  --metadata checkpoint_revision=fb2764a5cf321eaa5070ca8f9e892818f477c16d \
  --metadata kit_sha256=03ebcfe84135e768c4ae78dbd5472e8ea552217f6c90e76fd395df058170e3f3 \
  --metadata benchmark_source_sha256=2c447f16840e30e12335053ace433a1c6f00b4df280dac3987ae172a3a99b7c3 \
  --metadata nodes=dusty,toby,rusty,kirby \
  --metadata max_model_len=1048576 --metadata gpu_memory_utilization=0.80 \
  --metadata dspark_tokens=7 --metadata jit_monitor_mode=error \
  --metadata max_num_seqs=4 --metadata max_num_batched_tokens=4096 \
  --metadata engram_table_memory=disk --metadata reasoning_effort=max \
  --metadata boot_receipts=spark/ds41/r38/receipts/20260916/k7-1m-boot \
  --metadata remote_telemetry=ds41-observer.service
