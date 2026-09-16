#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
python3 tests/test_build_contracts.py
[[ $(hostname -s) == rusty && $(uname -m) == aarch64 ]] || exit 78
idle=$(podman ps -q) || exit 78
[[ -z "$idle" ]] || { echo 'Rusty is not idle; refusing build.' >&2; exit 78; }
peer=$(ssh -o BatchMode=yes -o ConnectTimeout=10 toby 'podman ps -q') || exit 78
[[ -z "$peer" ]] || { echo 'Toby is not idle; refusing build.' >&2; exit 78; }
base=74e53e710bef141f6f68e722582569f9c6aa388bce405ad6f1423566a2300c9c
[[ $(podman image inspect "$base" --format '{{.Id}}') == "$base" ]]
receipt="build-receipts/flashinfer-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$receipt"
exec > >(tee -a "$receipt/build.log") 2>&1
finish_receipt() {
  local status=$?
  printf 'exit_status=%s\nfinished_utc=%s\n' "$status" "$(date -u +%FT%TZ)" > "$receipt/status.txt"
}
trap finish_receipt EXIT
sha256sum Dockerfile.flashinfer build-flashinfer.sh > "$receipt/recipe.sha256"
# Preserve the actual policy as well as its digest when artifacts are edited
# during or after a long build. Existing receipts are never rewritten.
cp Dockerfile.flashinfer build-flashinfer.sh "$receipt/"
printf 'base=%s\narch=12.1a\ncommit=803c4664f4771ddc418f20a57f752469a237a825\n' "$base" > "$receipt/inputs.txt"
podman build --pull=never --format docker --layers --jobs=1 \
    -f Dockerfile.flashinfer --iidfile "$receipt/image.id" .
candidate=$(<"$receipt/image.id")
podman image inspect "$candidate" > "$receipt/image-inspect.json"
printf 'FLASHINFER-BUILD-OK id=%s\n' "$candidate" | tee "$receipt/BUILD-OK"
