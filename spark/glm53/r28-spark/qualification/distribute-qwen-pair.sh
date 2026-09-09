#!/usr/bin/env bash
set -euo pipefail
[[ $(hostname -s) == dusty ]]
root=/home/jugs/git/bld-jj-r28-spark
image=localhost/voipmonitor/vllm:jj-r28-spark-sm121
expected=cd93d80b3f9547f70e1e4608cd42af7c4b0f2ff33d92d5e520913cc9bab2d8d1
archive=${root}/jj-r28-spark-sm121.docker.tar
mkdir -p "${root}/qualification"
exec > >(tee -a "${root}/qualification/distribution.log") 2>&1
[[ $(podman image inspect "$image" --format '{{.Id}}') == "$expected" ]]
route=$(ip route get 10.11.11.8)
[[ $route == *'dev enp1s0f0np0 src 10.11.11.7'* ]]
printf '%s\n' "$route"
ssh -o BatchMode=yes kirby 'mkdir -p /home/jugs/git/bld-jj-r28-spark/qualification'
[[ ! -e "$archive" ]] || { echo 'Archive exists; inspect before replacing.' >&2; exit 78; }
podman image inspect "$image" > "${root}/qualification/image-source.json"
podman save --format docker-archive --output "$archive" "$image"
digest=$(sha256sum "$archive" | cut -d' ' -f1)
printf '%s  %s\n' "$digest" "$archive" | tee "${root}/qualification/archive.sha256"
rsync --whole-file --inplace --partial --no-compress -a \
  -e 'ssh -o BatchMode=yes -o Compression=no -c aes128-gcm@openssh.com' \
  "$archive" "${root}/run-qwen38-flash-next-jj-r28-spark-tp2-node.sh" \
  10.11.11.8:/home/jugs/git/bld-jj-r28-spark/
received=$(ssh -o BatchMode=yes kirby "sha256sum '$archive'" | cut -d' ' -f1)
[[ $received == "$digest" ]]
ssh -o BatchMode=yes kirby "podman load --input '$archive'"
actual=$(ssh -o BatchMode=yes kirby "podman image inspect '$image' --format '{{.Id}}'")
[[ $actual == "$expected" ]]
ssh -o BatchMode=yes kirby "podman image inspect '$image'" > "${root}/qualification/image-kirby.json"
jq -S '.[0] | {Id, ManifestType, RootFS, Config}' "${root}/qualification/image-source.json" > "${root}/qualification/source-identity.json"
jq -S '.[0] | {Id, ManifestType, RootFS, Config}' "${root}/qualification/image-kirby.json" > "${root}/qualification/kirby-identity.json"
cmp "${root}/qualification/source-identity.json" "${root}/qualification/kirby-identity.json"
echo "DISTRIBUTION-OK image=$image id=$actual archive_sha256=$digest"
