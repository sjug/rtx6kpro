#!/usr/bin/env bash
# Docker archive, no conversion/compression, switched 200G only.
set -euo pipefail
cd "$(dirname "$0")"
[[ $(hostname -s) == rusty && -f receipts/build/BUILD-OK ]] || exit 78
image=localhost/voipmonitor/vllm:jj-r38p-spark-sm121
expected=$(jq -r '.[0].Id' receipts/build/image-inspect.json)
[[ $(podman image inspect "$image" --format '{{.Id}}') == "$expected" ]] || exit 78
route=$(ip route get 10.11.11.6)
[[ $route == *'dev enp1s0f0np0 src 10.11.11.5'* ]] || exit 78
transport='ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=/home/jugs/git/ds4-vision-r38/transfer-known-hosts -o HostKeyAlias=toby -o Compression=no -c aes128-gcm@openssh.com'
read -r -a remote <<< "$transport"
[[ $("${remote[@]}" 10.11.11.6 hostname -s) == toby ]] || exit 78
for node in local remote; do
  if [[ $node == local ]]; then
    running=$(podman ps -q); available=$(df -B1 --output=avail . | tail -1)
  else
    running=$("${remote[@]}" 10.11.11.6 'podman ps -q')
    available=$("${remote[@]}" 10.11.11.6 'df -B1 --output=avail /home/jugs | tail -1')
  fi
  [[ -z $running && $available -gt 100000000000 ]] || { echo "$node busy or insufficient disk"; exit 78; }
done
mkdir -p receipts/transfer
exec > >(tee receipts/transfer/transfer.log) 2>&1
echo "$route"
archive=$PWD/jj-r38p-spark-sm121.docker-archive
[[ ! -e $archive ]] || { echo 'Archive exists; refusing overwrite'; exit 78; }
podman save --format docker-archive --output "$archive" "$image"
sha256sum "$archive" > receipts/transfer/archive.sha256
digest=$(cut -d' ' -f1 receipts/transfer/archive.sha256)
"${remote[@]}" 10.11.11.6 "mkdir -p '$PWD'"
rsync --whole-file --inplace --partial --info=progress2 -e "$transport" "$archive" "10.11.11.6:$archive"
received=$("${remote[@]}" 10.11.11.6 "sha256sum '$archive'")
[[ ${received%% *} == "$digest" ]] || exit 78
"${remote[@]}" 10.11.11.6 "podman load --input '$archive'"
"${remote[@]}" 10.11.11.6 "podman image inspect '$image'" > receipts/transfer/toby-image.json
jq -e --arg id "$expected" '.[0].Id==$id' receipts/transfer/toby-image.json
diff <(jq -S '.[0]|{Id,ManifestType,RootFS,Config,History}' receipts/build/image-inspect.json) \
     <(jq -S '.[0]|{Id,ManifestType,RootFS,Config,History}' receipts/transfer/toby-image.json)
echo "TRANSFER-OK toby $expected" | tee receipts/transfer/TRANSFER-OK
