#!/usr/bin/env bash
set -euo pipefail
[[ $(hostname -s) == dusty ]] || exit 78
image=localhost/voipmonitor/vllm:jj-r32-spark-sm121
expected=74e53e710bef141f6f68e722582569f9c6aa388bce405ad6f1423566a2300c9c
archive=/home/jugs/git/bld-jj-r32-spark/jj-r32-spark-sm121.docker.tar
digest=3bffabe501a07bd0640e7eebeb85d872963bfa6da914ee47c3cd735b8e2dbde0
[[ $(podman image inspect "$image" --format '{{.Id}}') == "$expected" ]] || exit 78
[[ $(sha256sum "$archive" | cut -d' ' -f1) == "$digest" ]] || exit 78
transfer() {
  local node=$1 address=$2 route transport got
  route=$(ip route get "$address")
  [[ $route == *'dev enp1s0f0np0 src 10.11.11.7'* ]] || return 78
  transport="ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o HostKeyAlias=$node -o Compression=no -c aes128-gcm@openssh.com"
  # shellcheck disable=SC2086
  [[ $($transport "$address" hostname -s) == "$node" ]] || return 78
  printf 'TRANSFER node=%s route=%s\n' "$node" "$route"
  # shellcheck disable=SC2086
  $transport "$address" 'mkdir -p /home/jugs/git/ds4-vision/receipts'
  rsync --whole-file --inplace --partial --info=progress2 -e "$transport" \
    "$archive" "$address:/home/jugs/git/ds4-vision/jj-r32-spark-sm121.docker.tar"
  # shellcheck disable=SC2086
  got=$($transport "$address" 'sha256sum /home/jugs/git/ds4-vision/jj-r32-spark-sm121.docker.tar')
  [[ ${got%% *} == "$digest" ]] || return 78
  # shellcheck disable=SC2086
  $transport "$address" 'podman load --input /home/jugs/git/ds4-vision/jj-r32-spark-sm121.docker.tar'
  # shellcheck disable=SC2086
  got=$($transport "$address" "podman image inspect '$image' --format '{{.Id}}'")
  [[ $got == "$expected" ]] || return 78
  printf 'IMAGE-STAGED node=%s image=%s archive_sha256=%s\n' "$node" "$got" "$digest"
}
transfer rusty 10.11.11.5 & first=$!
transfer toby 10.11.11.6 & second=$!
status=0
wait "$first" || status=1
wait "$second" || status=1
exit "$status"
