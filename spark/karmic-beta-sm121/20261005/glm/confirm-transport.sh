#!/usr/bin/env bash
# Inspect and probe the running candidate; never stop or replace another deployment.
set -euo pipefail
kit=$(cd "$(dirname "$0")" && pwd)
out=${GLM_TRANSPORT_RECEIPT_DIR:-$kit/../qualification/glm-transport}
image=${EXPECTED_IMAGE_ID:?built image ID required}
[[ $image =~ ^[0-9a-f]{64}$ && ! -e $out/confirm.log ]] || exit 78
name=glm53-flash-nvfp4-karmic-beta-20261005-tp4
mkdir -p "$out"
exec > >(tee "$out/confirm.log") 2>&1
for node in sparky buddy rocky lucky; do
  ssh -n -o BatchMode=yes "$node" "podman inspect '$name'" > "$out/$node-container.json"
  jq -e --arg id "$image" '.[0] | (.Image|ltrimstr("sha256:"))==$id and .State.Running==true and (.Config.Env|index("NCCL_PROTO=LL,Simple")!=null) and (.Config.Env|index("NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0")!=null)' "$out/$node-container.json" >/dev/null
  ssh -n -o BatchMode=yes "$node" "podman logs '$name' 2>&1" > "$out/$node-boot.log"
done
grep -q 'B12X_ROCENANTE' "$out/sparky-boot.log"
grep -q 'PYNCCL' "$out/sparky-boot.log"
python3 "$kit/probe-fresh-prefill.py" "$out/fresh-prefill.jsonl" 8192 8192 131072 131072
echo "CONFIRM-PROBE-COMPLETE $(date -Is)"
