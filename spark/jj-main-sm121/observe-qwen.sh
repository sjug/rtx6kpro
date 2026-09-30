#!/usr/bin/env bash
# Read-only bounded observer. Run one per node while qualification runs.
set -euo pipefail
node=${1:?dusty or kirby}
case $node in dusty|kirby) ;; *) exit 2;; esac
ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$node" '
  deadline=$((SECONDS+10800))
  while (( SECONDS < deadline )); do
    date -u +%Y-%m-%dT%H:%M:%SZ
    nvidia-smi --query-gpu=clocks.sm,clocks_throttle_reasons.active,temperature.gpu,power.draw --format=csv,noheader
    grep -E "^(MemAvailable|MemFree|Cached|SwapFree):" /proc/meminfo
    grep -E "^(allocstall|compact_stall|pgscan_direct)" /proc/vmstat
    sleep 1
  done'
