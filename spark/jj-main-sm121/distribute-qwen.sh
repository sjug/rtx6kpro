#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
[[ $(hostname -s) == dusty ]] || exit 78
image=6633e678fee74f5e1060290a01df812d34d10edcf30e34dd1c9bf7db88260ef3
tag=localhost/voipmonitor/vllm:jj-main-spark-sm121
gate=build-receipts/cache-agreement-20260922T031658Z-3132975
grep -qx "BUILD-GATES-PASS id=sha256:$image; model qualification pending" "$gate/BUILD-OK"
idle=$(podman ps -q) || exit 78
peer=$(ssh -o BatchMode=yes -o ConnectTimeout=10 kirby 'podman ps -q') || exit 78
[[ -z $idle && -z $peer ]] || { echo 'Pair must be stopped before image save/load'; exit 78; }
route=$(ip route get 10.11.11.8)
[[ $route == *'dev enp1s0f0np0 src 10.11.11.7'* ]] || { echo "Wrong bulk route: $route"; exit 78; }
ssh_args=(-o BatchMode=yes -o ConnectTimeout=10 -o Compression=no -c aes128-gcm@openssh.com)
[[ $(ssh "${ssh_args[@]}" 10.11.11.8 hostname -s) == kirby ]] || exit 78
for node in dusty 10.11.11.8; do
  if [[ $node == dusty ]]; then
    free=$(df -B1 --output=avail /home/jugs | tail -1) || exit 78
  else
    free=$(ssh "${ssh_args[@]}" "$node" "df -B1 --output=avail /home/jugs | tail -1") || exit 78
  fi
  (( free > 110000000000 )) || { echo "Insufficient disk: $node $free"; exit 78; }
done
out="$PWD/build-receipts/qwen-transfer-cache-agreement"
mkdir -p "$out"
exec > >(tee -a "$out/transfer.log") 2>&1
trap 'printf "exit_status=%s\n" "$?" > "$out/status.txt"' EXIT
podman tag "$image" "$tag"
podman image inspect "$image" > "$out/source-inspect.json"
archive="$out/jj-main-spark-sm121.docker.tar"
[[ ! -e $archive ]] || { echo 'Archive exists; inspect before retry'; exit 78; }
podman save --format docker-archive --output "$archive" "$tag"
ssh "${ssh_args[@]}" 10.11.11.8 "mkdir -p '$out'"
rsync -a --whole-file --partial --info=progress2 \
  -e 'ssh -o BatchMode=yes -o Compression=no -c aes128-gcm@openssh.com' \
  "$archive" "10.11.11.8:$archive"
ssh "${ssh_args[@]}" 10.11.11.8 "podman load -i '$archive'"
ssh "${ssh_args[@]}" 10.11.11.8 "podman image inspect '$tag'" > "$out/receiver-inspect.json"
python3 - "$out" "$image" <<'PY'
import json
import sys
from pathlib import Path
out = Path(sys.argv[1])
a, b = [json.loads((out / name).read_text())[0] for name in ('source-inspect.json', 'receiver-inspect.json')]
for key in ('Id', 'ManifestType', 'RootFS', 'Labels'):
    if a[key] != b[key]:
        raise RuntimeError(f'Transfer changed {key}')
if a['Id'].removeprefix('sha256:') != sys.argv[2]:
    raise RuntimeError('Wrong image')
PY
printf 'TRANSFER-VERIFIED image=%s receiver=kirby format=docker-archive route=10.11.11.7-to-10.11.11.8\n' "$image" | tee "$out/TRANSFER-OK"
