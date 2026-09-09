#!/usr/bin/env bash
# Run on sparky with a short-lived forwarded workstation SSH agent.
# All archive bytes use the switched 200G fabric, never management addresses.
set -euo pipefail
[[ $(hostname -s) == sparky ]] || exit 78
root=/home/jugs/git/bld-jj-r28-spark
archive=jj-r28-spark-sm121.docker.tar
expected=4b915899489f238a6f2e0f578b6d5d0dfdad0cf00818f7f99219a60d460f23b4
transport="ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=$root/transfer-known-hosts -o Compression=no -c aes128-gcm@openssh.com"
for ip in 10.11.11.8 10.11.11.2 10.11.11.3 10.11.11.4; do
  route=$(ip route get "$ip")
  [[ $route == *'dev enp1s0f0np0 src 10.11.11.1'* ]] || exit 78
done
rsync --whole-file --inplace --partial --info=progress2 -e "$transport" \
  "10.11.11.8:$root/$archive" "$root/$archive"
[[ $(sha256sum "$root/$archive" | cut -d' ' -f1) == "$expected" ]] || exit 78
echo "SPARKY-ARCHIVE-VERIFIED $(date -Is)"
pids=()
for ip in 10.11.11.2 10.11.11.4 10.11.11.3; do
  (
    rsync --whole-file --inplace --partial --info=progress2 -e "$transport" \
      "$root/$archive" "$ip:$root/$archive"
    # Split the fixed transport string into argv, not user-controlled input.
    read -r -a ssh_args <<< "$transport"
    digest=$("${ssh_args[@]}" "$ip" "sha256sum '$root/$archive'")
    [[ ${digest%% *} == "$expected" ]] || exit 78
    echo "ARCHIVE-VERIFIED $ip $(date -Is)"
  ) > "$root/transfer-$ip.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ $failed == 0 ]] || exit 78
echo "ALL-ARCHIVES-VERIFIED $(date -Is)"
