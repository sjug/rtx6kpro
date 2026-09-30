#!/usr/bin/env bash
# Run on dusty after measured Qwen cells finish. Reuse the verified Docker archive.
set -euo pipefail
cd "$(dirname "$0")"
[[ $(hostname -s) == dusty ]] || exit 78
window=${QUALIFICATION_WINDOW:?Set glm or ds4 only for the authorized idle cluster window}
case $window in glm|ds4) ;; *) exit 2;; esac
for target in "$@"; do
  case "$window:$target" in glm:sparky|glm:buddy|glm:rocky|glm:lucky|ds4:rusty|ds4:toby) ;;
    *) echo 'Targets must belong to exactly the selected qualification cluster'; exit 78;;
  esac
done
image=f30dc6d9a2a6f6fc0ac9ff8cddb04d9f631f4a87b48ca7cac69254802fe83233
tag=localhost/voipmonitor/vllm:karmic-spark-sm121
archive="$PWD/build-receipts/image-transfer/karmic-spark-sm121.docker.tar"
grep -Fq "TRANSFER-VERIFIED image=$image receiver=kirby" build-receipts/image-transfer/TRANSFER-OK
[[ -f $archive && $(stat -c %s "$archive") == 36848983552 ]] || { echo 'Archive missing or size changed'; exit 78; }
[[ $(podman image inspect "$tag" --format '{{.Id}}') == "$image" ]] || exit 78
out="$PWD/build-receipts/cluster-transfer"
mkdir -p "$out"
exec > >(tee -a "$out/transfer.log") 2>&1
ssh_args=(-o BatchMode=yes -o ConnectTimeout=10 -o Compression=no -c aes128-gcm@openssh.com)
for target in "$@"; do
  case $target in
    sparky) ip=10.11.11.1;; buddy) ip=10.11.11.2;;
    rocky) ip=10.11.11.4;; lucky) ip=10.11.11.3;;
    rusty) ip=10.11.11.5;; toby) ip=10.11.11.6;;
    *) echo "Unsupported receiver $target"; exit 2;;
  esac
  route=$(ip route get "$ip")
  [[ $route == *'dev enp1s0f0np0 src 10.11.11.7'* ]] || { echo "Wrong bulk route: $route"; exit 78; }
  [[ $(ssh "${ssh_args[@]}" "$ip" hostname -s) == "$target" ]] || { echo 'Receiver ownership mismatch'; exit 78; }
  receiver_dir="/home/jugs/git/bld-jj-r38-spark/karmic-beta-sm121/build-receipts/image-transfer"
  receiver_archive="$receiver_dir/karmic-spark-sm121.docker.tar"
  ssh "${ssh_args[@]}" "$ip" bash -s -- "$receiver_dir" 36848983552 <<'PREFLIGHT'
set -euo pipefail
idle=$(podman ps -q) || exit 78
[[ -z $idle ]] || { echo 'Receiver is serving; stop this cluster in its authorized window first'; exit 78; }
directory=$1
archive_bytes=$2
storage=$(podman info --format '{{.Store.GraphRoot}}')
mkdir -p "$directory"
archive_free=$(df -PB1 "$directory" | awk 'NR==2 {print $4}')
storage_free=$(df -PB1 "$storage" | awk 'NR==2 {print $4}')
required=$((archive_bytes + 10 * 1024 * 1024 * 1024))
if [[ $(stat -c %d "$directory") == $(stat -c %d "$storage") ]]; then
  required=$((2 * archive_bytes + 10 * 1024 * 1024 * 1024))
fi
(( archive_free >= required && storage_free >= required )) || {
  echo "Insufficient disk: archive=$archive_free storage=$storage_free required=$required"; exit 78;
}
PREFLIGHT
  rsync -a --whole-file --partial --info=progress2 \
    -e 'ssh -o BatchMode=yes -o Compression=no -c aes128-gcm@openssh.com' \
    "$archive" "$ip:$receiver_archive"
  ssh "${ssh_args[@]}" "$ip" "set -e; idle=\$(podman ps -q) || exit 78; test -z \"\$idle\" || exit 78; podman load -i '$receiver_archive'"
  ssh "${ssh_args[@]}" "$ip" "podman image inspect '$tag'" > "$out/$target-inspect.json"
  jq -e --arg image "$image" '.[0].Id==$image' "$out/$target-inspect.json" >/dev/null
  printf 'TRANSFER-VERIFIED image=%s receiver=%s route=10.11.11.7-to-%s format=docker-archive\n' \
    "$image" "$target" "$ip" | tee "$out/$target-TRANSFER-OK"
  ssh "${ssh_args[@]}" "$ip" "rm -- '$receiver_archive'"
  echo "Removed verified receiver archive on $target; source archive retained on dusty."
done
