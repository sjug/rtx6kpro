#!/usr/bin/env bash
# Existing archive only. Receiver must be idle. No image conversion or compression.
set -euo pipefail
[[ $(hostname -s) == dusty ]] || exit 78
archive=/home/jugs/git/bld-jj-r38-spark/karmic-main-sm121/20260922/build-receipts/qsa865-20260923T173427Z-925019/qwen-image.docker.tar
digest=615f9c36faa6ed6d8af3dec1e73570806b1f33a1c00145448b83f94d26e2d292
image=1a7a8acff71044e3b8bf97dacb7617cd1bdb50d0ab2fc2e35364a4d4dc165dcc
target=${1:?one GLM receiver required}
case $target in
  sparky) ip=10.11.11.1;; buddy) ip=10.11.11.2;; rocky) ip=10.11.11.4;; lucky) ip=10.11.11.3;;
  *) exit 78;;
esac
route=$(ip route get "$ip")
[[ $route == *'dev enp1s0f0np0 src 10.11.11.7'* ]] || { echo "Wrong fabric route: $route"; exit 78; }
ssh_args=(-o BatchMode=yes -o ConnectTimeout=10 -o Compression=no -c aes128-gcm@openssh.com)
[[ $(ssh "${ssh_args[@]}" "$ip" hostname -s) == "$target" ]] || exit 78
destination=/home/jugs/git/bld-jj-r38-spark/karmic-main-sm121/20260922/glm-defaults
ssh "${ssh_args[@]}" "$ip" bash -s -- "$destination" <<'REMOTE'
set -euo pipefail
idle=$(podman ps -q) || exit 78
[[ -z $idle ]] || { echo 'Receiver is serving'; exit 78; }
mkdir -p "$1"
storage=$(podman info --format '{{.Store.GraphRoot}}')
for path in "$1" "$storage"; do
  available=$(df -PB1 "$path" | awk 'NR==2 {print $4}')
  (( available > 107374182400 )) || { echo 'Less than 100 GiB disk headroom'; exit 78; }
done
REMOTE
rsync -a --whole-file --partial --info=progress2 \
  -e 'ssh -o BatchMode=yes -o Compression=no -c aes128-gcm@openssh.com' \
  "$archive" "$ip:$destination/candidate.docker.tar"
ssh "${ssh_args[@]}" "$ip" bash -s -- "$destination/candidate.docker.tar" "$digest" "$image" <<'REMOTE'
set -euo pipefail
archive=$1; digest=$2; image=$3
printf '%s  %s\n' "$digest" "$archive" | sha256sum -c -
idle=$(podman ps -q) || exit 78
[[ -z $idle ]] || exit 78
podman load -i "$archive"
actual=$(podman image inspect "$image" --format '{{.Id}}')
[[ ${actual#sha256:} == "$image" ]] || exit 78
echo "TRANSFER-VERIFIED $image"
# Exact disposable archive only, source archive and imported image retained.
rm -- "$archive"
echo 'Removed verified receiver archive; source archive retained on dusty.'
REMOTE
