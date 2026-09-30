#!/usr/bin/env bash
# Read-only 1 Hz boot/qualification telemetry. Never stops a container.
set -euo pipefail
while [[ $(podman inspect ds41-flash-jj-r38-tp4 --format '{{.State.Running}}') == true ]]; do
  date -u +%FT%TZ
  awk '/MemAvailable:|SwapTotal:|SwapFree:|Dirty:|Writeback:/ {print}' /proc/meminfo
  nvidia-smi --query-gpu=clocks.sm,clocks_event_reasons.active,temperature.gpu,power.draw --format=csv,noheader,nounits
  awk '$3 == "nvme0n1" {print "DISKSTAT", $0}' /proc/diskstats
  sleep 1
done
