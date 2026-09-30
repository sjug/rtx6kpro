#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
[[ $(hostname -s) == rusty ]] || exit 78
idle() {
  local node running
  for node in rusty toby; do
    if [[ $node == rusty ]]; then
      running=$(podman ps -q) || return 78
    else
      running=$(ssh -n -o BatchMode=yes -o ConnectTimeout=10 \
        -o UserKnownHostsFile=/home/jugs/git/ds4-vision-r38/transfer-known-hosts \
        "$node" 'podman ps -q') || {
      echo "Cannot establish idle state: $node" >&2; return 78;
      }
    fi
    [[ -z $running ]] || { echo "$node is busy" >&2; return 78; }
  done
}
idle
grep -qx 'R38-GRAMMAR-SEMANTIC-RED-CONFIRMED: 12 failures, 20 passes' receipts/baseline-verdict.txt
mkdir -p receipts/build
[[ ! -e receipts/build/image.id ]] || { echo 'Existing build receipt; refusing overwrite'; exit 78; }
exec > >(tee receipts/build/build.log) 2>&1
trap 'echo "Build/gate failed at line $LINENO; candidate retained, serving tag not published" >&2' ERR
base=ea031e1d3d051033f077fc986bf6f8fce04cf9ab52483d5a719ba13114567fc5
[[ $(podman image inspect "$base" --format '{{.Id}}') == "$base" ]] || exit 78
sha256sum Dockerfile install.py source.lock.json upstream.patch build.sh gate.sh > receipts/build/inputs.sha256
lock=$(sha256sum source.lock.json | cut -d' ' -f1)
recipe=$(sha256sum receipts/build/inputs.sha256 | cut -d' ' -f1)
podman build --format docker --pull=never --network=none --iidfile receipts/build/image.id \
  --build-arg GRAMMAR_LOCK_SHA256="$lock" --build-arg GRAMMAR_RECIPE_SHA256="$recipe" \
  -t localhost/voipmonitor/vllm:jj-r38p-spark-sm121-build -f Dockerfile .
image=$(cat receipts/build/image.id)
podman image inspect "$image" > receipts/build/image-inspect.json
[[ $(podman image inspect "$image" --format '{{.ManifestType}}') == application/vnd.docker.distribution.manifest.v2+json ]] || exit 78
idle
bash gate.sh "$image"
podman tag "$image" localhost/voipmonitor/vllm:jj-r38p-spark-sm121
echo BUILD-OK | tee receipts/build/BUILD-OK
