#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
receipt=${1:?build receipt required}
[[ $(hostname -s) == dusty ]] || exit 78
grep -qx 'KARMIC-MAIN-BUILD-GATES-PASS' "$receipt/GATES-OK"
grep -q '^BUILD-OK image=' "$receipt/BUILD-OK"
image=$(<"$receipt/image.id")
image=${image#sha256:}
tag=localhost/voipmonitor/vllm:karmic-main-spark-sm121
[[ $(podman image inspect "$tag" --format '{{.Id}}') == "$image" ]] || exit 78
head=$(podman ps -q) || exit 78
peer=$(ssh -o BatchMode=yes -o ConnectTimeout=10 kirby 'podman ps -q') || exit 78
[[ -z $head && -z $peer ]] || { echo 'Transfer/load requires idle Qwen pair'; exit 78; }
route=$(ip route get 10.11.11.8)
[[ $route == *'dev enp1s0f0np0 src 10.11.11.7'* ]] || { echo "Wrong bulk route: $route"; exit 78; }
ssh_args=(-o BatchMode=yes -o ConnectTimeout=10 -o Compression=no -c aes128-gcm@openssh.com)
[[ $(ssh "${ssh_args[@]}" 10.11.11.8 hostname -s) == kirby ]] || exit 78
free_local=$(df -PB1 "$PWD" | awk 'NR==2 {print $4}')
free_peer=$(ssh "${ssh_args[@]}" 10.11.11.8 "df -PB1 /home/jugs | awk 'NR==2 {print \$4}'")
(( free_local > 107374182400 && free_peer > 107374182400 )) || { echo 'Need 100 GiB free per node'; exit 78; }
archive="$PWD/$receipt/qwen-image.docker.tar"
[[ $receipt != /* && $receipt != *..* && ! -e $archive ]] || { echo 'Require new relative receipt archive'; exit 78; }
podman save --format docker-archive --output "$archive" "$tag"
sha256sum "$archive" > "$receipt/archive.sha256"
ssh "${ssh_args[@]}" 10.11.11.8 "mkdir -p '$PWD/$receipt'"
rsync -a --whole-file --partial --info=progress2 \
  -e 'ssh -o BatchMode=yes -o Compression=no -c aes128-gcm@openssh.com' \
  "$archive" "10.11.11.8:$archive"
expected=$(cut -d' ' -f1 "$receipt/archive.sha256")
actual=$(ssh "${ssh_args[@]}" 10.11.11.8 "sha256sum '$archive'" | cut -d' ' -f1)
[[ $actual == "$expected" ]] || { echo 'Archive digest mismatch'; exit 78; }
ssh "${ssh_args[@]}" 10.11.11.8 "podman load -i '$archive'"
actual=$(ssh "${ssh_args[@]}" 10.11.11.8 "podman image inspect '$tag' --format '{{.Id}}'")
[[ ${actual#sha256:} == "$image" ]] || { echo 'Receiver image mismatch'; exit 78; }
printf 'TRANSFER-OK image=%s route=10.11.11.7-to-10.11.11.8\n' "$image" | tee "$receipt/TRANSFER-OK"
