#!/usr/bin/env bash
# Run on dusty. Existing Docker archive only, switched 200G, no conversion.
set -euo pipefail
[[ $(hostname -s) == dusty ]] || exit 78
root=/home/jugs/git/bld-jj-r32-spark
expected=${EXPECTED_IMAGE_ID:?gated image ID required}
digest=${ARCHIVE_SHA256:?archive digest required}
[[ $expected =~ ^[0-9a-f]{64}$ && $digest =~ ^[0-9a-f]{64}$ ]] || exit 78
archive=$root/jj-r32-spark-sm121.docker.tar
image=localhost/voipmonitor/vllm:jj-r32-spark-sm121
[[ $(sha256sum "$archive" | cut -d' ' -f1) == "$digest" ]] || exit 78
hosts=(sparky buddy rocky lucky)
ips=(10.11.11.1 10.11.11.2 10.11.11.4 10.11.11.3)
receipts=${TRANSFER_RECEIPT_DIR:-$root/qualification/glm-transfer-$(date +%Y%m%dT%H%M%S)}
mkdir -p "$receipts"
transfer_one() {
  local host=$1 ip=$2 route transport received actual
  local -a ssh_args
  echo "TRANSFER-START $host $(date -Is)"
  route=$(ip route get "$ip")
  [[ $route == *'dev enp1s0f0np0 src 10.11.11.7'* ]] || exit 78
  printf 'route=%s\n' "$route"
  transport="ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o HostKeyAlias=$host -o Compression=no -c aes128-gcm@openssh.com"
  read -r -a ssh_args <<< "$transport"
  [[ $("${ssh_args[@]}" "$ip" hostname -s) == "$host" ]] || exit 78
  "${ssh_args[@]}" "$ip" "mkdir -p '$root'"
  rsync -a --exclude=.compose --exclude=build-receipts --exclude='*.tar' --exclude='*.tar.sha256' --exclude=__pycache__ --exclude=inputs --exclude=qualification -e "$transport" "$root/" "$ip:$root/"
  rsync --whole-file --inplace --partial --info=progress2 -e "$transport" "$archive" "$ip:$archive"
  received=$("${ssh_args[@]}" "$ip" "sha256sum '$archive'")
  [[ ${received%% *} == "$digest" ]] || exit 78
  "${ssh_args[@]}" "$ip" "podman load --input '$archive'"
  actual=$("${ssh_args[@]}" "$ip" "podman image inspect '$image' --format '{{.Id}}'")
  [[ $actual == "$expected" ]] || exit 78
  echo "TRANSFER-OK $host $expected $(date -Is)"
}
pids=()
for i in 0 1 2 3; do
  transfer_one "${hosts[$i]}" "${ips[$i]}" > "$receipts/${hosts[$i]}.log" 2>&1 &
  pids+=("$!")
done
status=0
for i in 0 1 2 3; do
  if wait "${pids[$i]}"; then
    tail -1 "$receipts/${hosts[$i]}.log"
  else
    echo "TRANSFER-FAILED ${hosts[$i]}: $receipts/${hosts[$i]}.log"
    tail -10 "$receipts/${hosts[$i]}.log"
    status=1
  fi
done
exit "$status"
