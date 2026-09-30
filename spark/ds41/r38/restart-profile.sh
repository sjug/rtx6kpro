#!/usr/bin/env bash
# Diagnostic restart of the four DS4.1 nodes under one admitted profile.
# Provenance: added 2026-09-16 for the approved 524K nondeterminism bisect.
# Stops workers first, then the head, with graceful `podman stop -t 60`; renames
# the stopped candidate for retention; launches workers first, then the head,
# through the unchanged node runner. Never removes a container or touches GLM.
set -euo pipefail
usage() { echo "usage: $0 stop-and-retain RETAIN_SUFFIX | launch PROFILE_ENV... " >&2; exit 2; }
[[ $# -ge 2 ]] || usage
mode=$1; shift
name=ds41-flash-jj-r38-tp4
workers=(toby rusty kirby); head=dusty
receipts=$(dirname "$0")/receipts/20260916/restarts
mkdir -p "$receipts"
stamp=$(date +%Y%m%dT%H%M%S)
run() { # host, command
  local host=$1; shift
  local out="$receipts/$stamp-$mode-$host.log"
  echo "== $host: $*" | tee -a "$out"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" "$@" 2>&1 | tee -a "$out"
}
case $mode in
  stop-and-retain)
    suffix=$1
    for host in "${workers[@]}" "$head"; do
      run "$host" "podman inspect $name --format '{{.State.Status}} {{.State.OOMKilled}} {{.Id}}'"
      run "$host" "podman stop -t 60 $name"
      run "$host" "podman rename $name $name-$suffix && podman ps -a --filter name=$name --format '{{.Names}} {{.Status}}'"
      run "$host" "nvidia-smi --query-compute-apps=pid --format=csv,noheader; grep -E 'MemAvailable|SwapFree' /proc/meminfo"
    done
    echo STOP_AND_RETAIN_DONE ;;
  launch)
    env_string=""
    for kv in "$@"; do
      [[ $kv =~ ^[A-Z0-9_]+=[A-Za-z0-9./_-]+$ ]] || { echo "bad env $kv" >&2; exit 2; }
      env_string+="$kv "
    done
    env_string+="IO_URING_SECCOMP_PROFILE=/home/jugs/git/ds41-r38/seccomp-io-uring.json"
    for host in "${workers[@]}"; do
      run "$host" "cd /home/jugs/git/ds41-r38 && $env_string bash run-node.sh"
    done
    run "$head" "cd /home/jugs/git/ds41-r38 && $env_string bash run-node.sh"
    echo LAUNCH_DONE ;;
  *) usage ;;
esac
