#!/usr/bin/env bash
# Existing Docker archive, switched 200G only, no compression or conversion.
set -euo pipefail
case $(hostname -s) in
  rusty) source_ip=10.11.11.5;;
  dusty) source_ip=10.11.11.7;;
  *) exit 78;;
esac
root=/home/jugs/git/ds4-vision-r38
image=localhost/voipmonitor/vllm:jj-r38-spark-sm121
expected=ea031e1d3d051033f077fc986bf6f8fce04cf9ab52483d5a719ba13114567fc5
archive=/home/jugs/git/bld-jj-r38-spark/jj-r38-spark-sm121.docker.tar
case ${1:?toby} in
  toby) hosts=(toby); ips=(10.11.11.6);;
  *) exit 2;;
esac
cd "$root"
[[ $(podman image inspect "$image" --format '{{.Id}}') == "$expected" ]] || exit 78
[[ -s "$archive" ]] || exit 78
if [[ ! -e $archive.sha256 ]]; then sha256sum "$archive" > "$archive.sha256"; fi
sha256sum -c "$archive.sha256"
digest=$(cut -d' ' -f1 "$archive.sha256")
receipt=$root/receipts/transfer-$1-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$receipt"
podman image inspect "$image" > "$receipt/source-image.json"
storage=$(podman info --format '{{.Store.GraphRoot}}')
cp "$storage/overlay-images/$expected/manifest" "$receipt/source-manifest.json"
transfer_one() {
  local host=$1 ip=$2 route transport received
  local -a remote
  route=$(ip route get "$ip")
  [[ $route == *"dev enp1s0f0np0 src $source_ip"* ]] || exit 78
  printf 'START %s %s\nroute=%s\n' "$host" "$(date -Is)" "$route"
  transport="ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=$root/transfer-known-hosts -o HostKeyAlias=$host -o Compression=no -c aes128-gcm@openssh.com"
  read -r -a remote <<< "$transport"
  [[ $("${remote[@]}" "$ip" hostname -s) == "$host" ]] || exit 78
  "${remote[@]}" "$ip" "mkdir -p '$root' '/home/jugs/git/bld-jj-r38-spark'"
  rsync --whole-file --inplace --partial --info=progress2 -e "$transport" "$archive" "$ip:$archive"
  received=$("${remote[@]}" "$ip" "sha256sum '$archive'")
  [[ ${received%% *} == "$digest" ]] || exit 78
  "${remote[@]}" "$ip" "podman load --input '$archive'"
  "${remote[@]}" "$ip" "podman image inspect '$image'" > "$receipt/$host-image.json"
  "${remote[@]}" "$ip" "cat '/home/jugs/.local/share/containers/storage/overlay-images/$expected/manifest'" > "$receipt/$host-manifest.json"
  jq -e --arg id "$expected" '.[0].Id == $id' "$receipt/$host-image.json"
  diff <(jq -S '.[0]|{Id,ManifestType,RootFS,Config,History}' "$receipt/source-image.json") \
       <(jq -S '.[0]|{Id,ManifestType,RootFS,Config,History}' "$receipt/$host-image.json")
  # Docker-archive loading re-labels raw layer descriptors as tar.gzip in
  # this Podman version, without changing their digest, size or image ID.
  # Save both raw manifests and permit only that demonstrated media-type alias.
  for manifest in "$receipt/source-manifest.json" "$receipt/$host-manifest.json"; do
    jq -e 'all(.layers[]; .mediaType=="application/vnd.docker.image.rootfs.diff.tar" or .mediaType=="application/vnd.docker.image.rootfs.diff.tar.gzip")' "$manifest" >/dev/null
  done
  diff <(jq -S '.layers |= map(.mediaType="application/vnd.docker.image.rootfs.diff.tar")' "$receipt/source-manifest.json") \
       <(jq -S '.layers |= map(.mediaType="application/vnd.docker.image.rootfs.diff.tar")' "$receipt/$host-manifest.json")
  echo "TRANSFER-OK $host $expected $(date -Is)"
}
pids=()
for i in "${!hosts[@]}"; do
  transfer_one "${hosts[$i]}" "${ips[$i]}" > "$receipt/${hosts[$i]}.log" 2>&1 & pids+=("$!")
done
status=0
for i in "${!hosts[@]}"; do
  if wait "${pids[$i]}"; then tail -1 "$receipt/${hosts[$i]}.log";
  else tail -15 "$receipt/${hosts[$i]}.log"; status=1; fi
done
printf 'exit_status=%s\nfinished=%s\n' "$status" "$(date -Is)" > "$receipt/status.txt"
exit "$status"
