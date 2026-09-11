#!/usr/bin/env bash
# Run on dusty only after BUILD-OK. Exact Docker archive, switched 200G, no compression.
# Fixed, locally validated paths are intentionally expanded before SSH.
# shellcheck disable=SC2029
set -euo pipefail
[[ $(hostname -s) == dusty ]] || exit 78
root=/home/jugs/git/bld-jj-r29-spark
image=localhost/voipmonitor/vllm:jj-r29-spark-sm121
expected=${EXPECTED_IMAGE_ID:?Pin the gated R29 image ID}
[[ $expected =~ ^[0-9a-f]{64}$ ]] || exit 2
[[ $(podman image inspect "$image" --format '{{.Id}}') == "$expected" ]] || exit 78
route=$(ip route get 10.11.11.8)
[[ $route == *'dev enp1s0f0np0 src 10.11.11.7'* ]] || exit 78
ssh_args=(-n -o BatchMode=yes -o StrictHostKeyChecking=yes -o HostKeyAlias=kirby
  -o Compression=no -c aes128-gcm@openssh.com)
[[ $(ssh "${ssh_args[@]}" 10.11.11.8 hostname -s) == kirby ]] || exit 78
archive=$root/jj-r29-spark-sm121.docker.tar
[[ ! -e "$archive" ]] || { echo 'Archive exists; verify it before an explicit retry'; exit 78; }
podman save --format docker-archive --output "$archive" "$image"
sha256sum "$archive" > "$archive.sha256"
digest=$(cut -d' ' -f1 "$archive.sha256")
transport='ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o HostKeyAlias=kirby -o Compression=no -c aes128-gcm@openssh.com'
ssh "${ssh_args[@]}" 10.11.11.8 "mkdir -p '$root'"
rsync -a --exclude=.compose --exclude=build-receipts --exclude='*.tar' \
  --exclude='*.tar.sha256' --exclude=__pycache__ --exclude=inputs \
  -e "$transport" "$root/" "10.11.11.8:$root/"
rsync --whole-file --inplace --partial --info=progress2 -e "$transport" \
  "$archive" "10.11.11.8:$archive"
received=$(ssh "${ssh_args[@]}" 10.11.11.8 "sha256sum '$archive'")
[[ ${received%% *} == "$digest" ]] || exit 78
ssh "${ssh_args[@]}" 10.11.11.8 "podman load --input '$archive'"
actual=$(ssh "${ssh_args[@]}" 10.11.11.8 "podman image inspect '$image' --format '{{.Id}}'")
[[ $actual == "$expected" ]] || { echo 'Image ID mismatch after load'; exit 78; }
printf 'TRANSFER-OK image=%s sha256=%s route=%s\n' "$expected" "$digest" "$route"
