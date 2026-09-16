#!/usr/bin/env bash
# Read-only telemetry. The parent owns and stops these SSH processes.
set -euo pipefail
out=${1:?receipt directory}
mkdir -p "$out"
pids=()
cleanup() { for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM
for node in dusty kirby; do
  ssh -n -o BatchMode=yes "$node" 'nvidia-smi --query-gpu=timestamp,clocks.sm,clocks_throttle_reasons.active,temperature.gpu,power.draw --format=csv -l 1' > "$out/$node-gpu.csv" 2>&1 & pids+=("$!")
  ssh -n -o BatchMode=yes "$node" 'while :; do date -Is; awk "/MemAvailable:|SwapFree:/{print}" /proc/meminfo; sleep 5; done' > "$out/$node-memory.log" 2>&1 & pids+=("$!")
done
wait
