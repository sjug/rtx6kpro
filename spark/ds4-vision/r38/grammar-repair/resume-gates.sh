#!/usr/bin/env bash
# Resume the identical built candidate after staging missing offline fixtures.
set -euo pipefail
cd "$(dirname "$0")"
[[ $(hostname -s) == rusty ]] || exit 78
running=$(podman ps -q)
[[ -z $running ]] || exit 78
image=$(cat receipts/build/image.id)
[[ $(podman image inspect "$image" --format '{{.Id}}') == "$(jq -r '.[0].Id' receipts/build/image-inspect.json)" ]] || exit 78
[[ ! -e receipts/build/regressions-missing-opt-fixture.log ]] || exit 78
mv receipts/build/regressions.log receipts/build/regressions-missing-opt-fixture.log
mv receipts/build/native.log receipts/build/native-first.log
bash gate.sh "$image"
podman tag "$image" localhost/voipmonitor/vllm:jj-r38p-spark-sm121
echo BUILD-OK | tee receipts/build/BUILD-OK
