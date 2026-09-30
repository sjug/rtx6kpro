#!/usr/bin/env bash
# Same Docker-archive transport as the parent, using only the switched 200G link.
set -euo pipefail
cd "$(dirname "$0")"
receipt=${1:?relative build receipt required}
[[ $(hostname -s) == dusty && $receipt =~ ^receipts/build-[0-9TZ]+-[0-9]+$ ]] || exit 78
grep -qx 'KARMIC-MAIN-BUILD-GATES-PASS' "$receipt/GATES-OK"
grep -q '^BUILD-OK image=' "$receipt/BUILD-OK"
image=$(<"$receipt/image.id")
image=${image#sha256:}
tag=localhost/voipmonitor/vllm:karmic-beta-20260929-spark-sm121
[[ $(podman image inspect "$tag" --format '{{.Id}}') == "$image" ]] || exit 78
head=$(podman ps -q) || exit 78
worker=$(ssh -o BatchMode=yes -o ConnectTimeout=10 kirby 'podman ps -q') || exit 78
[[ -z $head && -z $worker ]] || { echo 'Pair must be idle for import'; exit 78; }
route=$(ip route get 10.11.11.8)
[[ $route == *'dev enp1s0f0np0 src 10.11.11.7'* ]] || { echo "Invalid bulk route: $route"; exit 78; }
ssh_args=(-o BatchMode=yes -o ConnectTimeout=10 -o Compression=no -c aes128-gcm@openssh.com)
[[ $(ssh "${ssh_args[@]}" 10.11.11.8 hostname -s) == kirby ]] || exit 78
local_free=$(df -PB1 "$PWD" | awk 'NR==2 {print $4}')
peer_free=$(ssh "${ssh_args[@]}" 10.11.11.8 "df -PB1 /home/jugs | awk 'NR==2 {print \$4}'")
(( local_free > 107374182400 && peer_free > 107374182400 )) || exit 78
archive="$PWD/$receipt/qwen-image.docker.tar"
[[ ! -e $archive ]] || { echo 'Refuse archive overwrite'; exit 78; }
podman save --format docker-archive --output "$archive" "$tag"
sha256sum "$archive" > "$receipt/archive.sha256"
ssh "${ssh_args[@]}" 10.11.11.8 "mkdir -p '$PWD/$receipt'"
rsync -a --whole-file --partial --info=progress2 -e 'ssh -o BatchMode=yes -o Compression=no -c aes128-gcm@openssh.com' "$archive" "10.11.11.8:$archive"
expected=$(cut -d' ' -f1 "$receipt/archive.sha256")
actual=$(ssh "${ssh_args[@]}" 10.11.11.8 "sha256sum '$archive'" | cut -d' ' -f1)
[[ $actual == "$expected" ]] || { echo 'Archive digest mismatch'; exit 78; }
ssh "${ssh_args[@]}" 10.11.11.8 "podman load -i '$archive'"
actual=$(ssh "${ssh_args[@]}" 10.11.11.8 "podman image inspect '$tag' --format '{{.Id}}'")
[[ ${actual#sha256:} == "$image" ]] || { echo 'Image ID mismatch'; exit 78; }
printf 'TRANSFER-OK image=%s route=10.11.11.7-to-10.11.11.8\n' "$image" | tee "$receipt/TRANSFER-OK"
