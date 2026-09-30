#!/usr/bin/env bash
# Read-only one-second telemetry; lifecycle is owned by the qualification driver.
set -euo pipefail
name=ds41-flash-karmic-main-tp4
while [[ $(podman inspect "$name" --format '{{.State.Running}}') == true ]]; do
  date -u +%FT%TZ
  awk '/MemAvailable:|SwapTotal:|SwapFree:|Dirty:|Writeback:/ {print}' /proc/meminfo
  nvidia-smi --query-gpu=clocks.sm,clocks_event_reasons.active,temperature.gpu,power.draw --format=csv,noheader,nounits
  awk '$3 == "nvme0n1" {print "DISKSTAT", $0}' /proc/diskstats
  awk '/^(allocstall|compact_stall|pgscan_direct)/ {print "VMSTAT", $0}' /proc/vmstat
  sleep 1
done
