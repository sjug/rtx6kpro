#!/usr/bin/env bash
# Durable, bounded continuation. Does not run two builds concurrently.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
component=${1:?Pass the exact FlashInfer receipt directory}
[[ $component == build-receipts/flashinfer-* && -d $component ]] || exit 2
deadline=$((SECONDS + 14400))
echo "Waiting for the already-running FlashInfer build: $component"
while [[ ! -f $component/status.txt ]]; do
  ((SECONDS < deadline)) || { echo 'Component wait exceeded four hours.' >&2; exit 124; }
  state=$(systemctl --user show jj-r37-flashinfer-build.service -p ActiveState --value)
  [[ $state == active || $state == activating || $state == deactivating ]] || {
    echo "Component is $state without a completion receipt; refusing continuation." >&2; exit 78;
  }
  sleep 15
done
grep -Fxq 'exit_status=0' "$component/status.txt"
[[ -s $component/BUILD-OK && -s $component/image.id ]]
export FLASHINFER_IMAGE
FLASHINFER_IMAGE=$(<"$component/image.id")
[[ ${FLASHINFER_IMAGE#sha256:} =~ ^[0-9a-f]{64}$ ]]
# Never replace recipe files between stages. Runtime assembly verifies the
# component's embedded inputs against this kit before accepting its wheels.
echo "FlashInfer completed at $FLASHINFER_IMAGE; starting runtime assembly and gates."
exec bash build.sh
