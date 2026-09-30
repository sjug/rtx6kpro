#!/usr/bin/env bash
# Serial build continuation only. Never launches serving or publishes a serving tag.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
receipt=${1:?Pass the exact active FlashInfer receipt directory}
start=${2:-nccl}
case "$start" in nccl|vllm|lmcache|instanttensor) ;; *) exit 78 ;; esac
case "$receipt" in
  build-receipts/flashinfer-*) ;;
  *) echo 'Expected a FlashInfer receipt in this workspace.' >&2; exit 78 ;;
esac
[[ $(realpath "$receipt") == "$(pwd)"/build-receipts/flashinfer-* ]] || exit 78
while systemctl --user is-active --quiet karmic-flashinfer-build.service; do
  sleep 30
done
[[ -f "$receipt/status.txt" && -f "$receipt/BUILD-OK" ]] || {
  echo 'FlashInfer did not complete; stopping continuation.' >&2; exit 78;
}
grep -qx 'exit_status=0' "$receipt/status.txt" || exit 78
started=0
for component in nccl vllm lmcache instanttensor; do
  if [[ $component == "$start" ]]; then started=1; fi
  ((started)) || continue
  printf 'Starting native component %s at %s\n' "$component" "$(date -u +%FT%TZ)"
  bash ./build-component.sh "$component"
done
echo 'NATIVE_COMPONENT_BUILDS_FINISHED; runtime assembly and qualification are still pending.'
