#!/usr/bin/env bash
# Read-only telemetry, bounded even if the workstation disconnects.
set -euo pipefail
deadline=$((SECONDS + 14400))
while (( SECONDS < deadline )); do
  date -u +%FT%TZ
  nvidia-smi --query-gpu=clocks.sm,clocks_throttle_reasons.active,temperature.gpu,power.draw --format=csv,noheader
  awk '/^(MemAvailable|MemFree|Cached|SwapFree|SwapTotal):/' /proc/meminfo
  awk '/^(allocstall|compact_stall|pgscan_direct)/' /proc/vmstat
  cat /proc/buddyinfo
  sleep 1
done
