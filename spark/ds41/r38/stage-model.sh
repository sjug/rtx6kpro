#!/usr/bin/env bash
# Run on rusty. Copy only completed HF blobs, preserving cache symlinks.
set -euo pipefail
[[ $(hostname -s) == rusty ]] || exit 78
root=$(cd -- "$(dirname -- "$0")" && pwd)
source_cache=/home/jugs/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash
destination_parent=/home/jugs/.cache/huggingface/hub/
[[ -d $source_cache && -r $root/transfer-known-hosts ]] || exit 78
hosts=(dusty toby kirby)
ips=(10.11.11.7 10.11.11.6 10.11.11.8)
for index in 0 1 2; do
  route=$(ip -j route get "${ips[$index]}")
  python3 -c 'import json,sys; r=json.loads(sys.argv[1])[0]; sys.exit(0 if r.get("dev")=="enp1s0f0np0" and r.get("prefsrc")=="10.11.11.5" else 78)' "$route"
done
pass=0
while :; do
  state=$(systemctl --user show ds41-checkpoint-download.service -p ActiveState --value)
  status=$(systemctl --user show ds41-checkpoint-download.service -p ExecMainStatus --value)
  case $state in
    active|activating) final=0 ;;
    inactive) [[ $status == 0 ]] || exit 1; final=1 ;;
    *) echo "Download failed or missing: state=$state status=$status" >&2; exit 1 ;;
  esac
  pass=$((pass + 1))
  printf 'COPY_PASS %s final=%s time=%s\n' "$pass" "$final" "$(date -Is)"
  jobs=()
  for index in 0 1 2; do
    host=${hosts[$index]}
    ip=${ips[$index]}
    transport="ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=yes -o UserKnownHostsFile=$root/transfer-known-hosts -o HostKeyAlias=$host -o Compression=no -c aes128-gcm@openssh.com"
    rsync -a --whole-file --inplace --partial --stats --exclude='*.incomplete' \
      -e "$transport" "$source_cache" "$ip:$destination_parent" &
    jobs+=("$!")
  done
  failed=0
  for job in "${jobs[@]}"; do wait "$job" || failed=1; done
  [[ $failed == 0 ]] || exit 1
  [[ $final == 1 ]] && break
  sleep 5
done
printf 'FANOUT_COMPLETE %s\n' "$(date -Is)"
