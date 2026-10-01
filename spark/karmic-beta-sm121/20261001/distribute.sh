#!/usr/bin/env bash
# Transfer/load a gated candidate while serving; no container or GPU action.
# SSH commands intentionally expand locally validated paths and IDs.
# shellcheck disable=SC2029
set -euo pipefail
cd "$(dirname "$0")"
receipt=${1:?relative build receipt required}
[[ $(hostname -s) == dusty && $receipt =~ ^receipts/build-[0-9TZ]+-[0-9]+$ ]] || exit 78
case ${TARGET_GROUP:-qwen} in
  qwen) nodes=(kirby);; ds4-vision) nodes=(rusty toby);;
  glm) nodes=(sparky buddy rocky lucky);; all) nodes=(kirby rusty toby sparky buddy rocky lucky);;
  *) echo 'Invalid TARGET_GROUP'; exit 78;;
esac
declare -A addresses=([sparky]=10.11.11.1 [buddy]=10.11.11.2 [lucky]=10.11.11.3 [rocky]=10.11.11.4 [rusty]=10.11.11.5 [toby]=10.11.11.6 [kirby]=10.11.11.8)
remote=/home/jugs/git/bld-jj-r38-spark/karmic-beta-sm121/20261001
tag=localhost/voipmonitor/vllm:karmic-beta-20261001-spark-sm121
grep -qx 'KARMIC-MAIN-BUILD-GATES-PASS' "$receipt/GATES-OK"
grep -q '^BUILD-OK image=' "$receipt/BUILD-OK"
grep -q 'FLASHINFER-KERNEL-GATE-PASS' "$receipt/flashinfer-kernels.log"
image=$(<"$receipt/image.id"); image=${image#sha256:}
[[ $image =~ ^[0-9a-f]{64}$ && $(podman image inspect "$tag" --format '{{.Id}}') == "$image" ]] || exit 78
ssh_args=(-o BatchMode=yes -o ConnectTimeout=10 -o Compression=no -c aes128-gcm@openssh.com)
(( $(df -PB1 "$PWD" | awk 'NR==2 {print $4}') > 107374182400 )) || exit 78
for node in "${nodes[@]}"; do
  addr=${addresses[$node]}
  route=$(ip route get "$addr")
  [[ $route == *'dev enp1s0f0np0 src 10.11.11.7'* ]] || { echo "Invalid bulk route: $route"; exit 78; }
  [[ $(ssh "${ssh_args[@]}" "$addr" hostname -s) == "$node" ]] || exit 78
  free=$(ssh "${ssh_args[@]}" "$addr" "df -PB1 /home/jugs | awk 'NR==2 {print \$4}'")
  (( free > 107374182400 )) || exit 78
done
archive="$PWD/$receipt/candidate.docker.tar"
[[ ! -e $archive ]] || { echo 'Refuse archive overwrite'; exit 78; }
podman save --format docker-archive --output "$archive" "$tag"
sha256sum "$archive" > "$receipt/archive.sha256"
expected=$(cut -d' ' -f1 "$receipt/archive.sha256")
# Stage the same runner kit on the build head, which execute.sh also checks.
if [[ $PWD != "$remote" ]]; then
  mkdir -p "$remote"
  rsync -a --whole-file --exclude inputs/ --exclude receipts/ --exclude qualification/ --exclude refresh.tar --exclude __pycache__/ ./ "$remote/"
fi
for node in "${nodes[@]}"; do
  addr=${addresses[$node]}
  ssh "${ssh_args[@]}" "$addr" "mkdir -p '$remote/$receipt'"
  rsync -a --whole-file --exclude inputs/ --exclude receipts/ --exclude qualification/ --exclude refresh.tar --exclude __pycache__/ \
    -e 'ssh -o BatchMode=yes -o Compression=no -c aes128-gcm@openssh.com' ./ "$addr:$remote/"
  rsync -a --whole-file --partial --info=progress2 -e 'ssh -o BatchMode=yes -o Compression=no -c aes128-gcm@openssh.com' "$archive" "$addr:$remote/$receipt/candidate.docker.tar"
  actual=$(ssh "${ssh_args[@]}" "$addr" "sha256sum '$remote/$receipt/candidate.docker.tar'" | cut -d' ' -f1)
  [[ $actual == "$expected" ]] || exit 78
  ssh "${ssh_args[@]}" "$addr" "podman load -i '$remote/$receipt/candidate.docker.tar'"
  actual=$(ssh "${ssh_args[@]}" "$addr" "podman image inspect '$tag' --format '{{.Id}}'")
  [[ ${actual#sha256:} == "$image" ]] || exit 78
  printf 'TRANSFER-OK node=%s image=%s\n' "$node" "$image" | tee -a "$receipt/TRANSFER-OK"
done
