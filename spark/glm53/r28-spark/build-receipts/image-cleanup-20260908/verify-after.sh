#!/usr/bin/env bash
set -euo pipefail
root=/home/jugs/git/bld-jj-r28-spark/qualification/image-cleanup-20260908
host=$(hostname -s)
plan=$root/$host-plan.json
exec > >(tee "$root/verification.log") 2>&1
podman images -a --format json > "$root/images-after.json"
podman ps -a --external --format json > "$root/containers-after.json"
podman volume ls --format json > "$root/volumes-after.json"
podman system df > "$root/storage-after.txt"
df -B1 /home/jugs > "$root/filesystem-after.txt"
jq -e --slurpfile plan "$plan" '
  (map({key:.Id,value:.}) | from_entries) as $after |
  all($plan[0].retain[]; . as $keep |
    $after[$keep.Id] != null and
    (($keep.Names - ($after[$keep.Id].Names // [])) | length) == 0)
' "$root/images-after.json"
jq -S '[.[] | {Id,ImageID,Names}] | sort_by(.Id)' "$root/containers-before.json" > "$root/containers-before-stable.json"
jq -S '[.[] | {Id,ImageID,Names}] | sort_by(.Id)' "$root/containers-after.json" > "$root/containers-after-stable.json"
cmp "$root/containers-before-stable.json" "$root/containers-after-stable.json"
jq -S 'sort_by(.Name)' "$root/volumes-before.json" > "$root/volumes-before-stable.json"
jq -S 'sort_by(.Name)' "$root/volumes-after.json" > "$root/volumes-after-stable.json"
cmp "$root/volumes-before-stable.json" "$root/volumes-after-stable.json"
echo RETENTION-VERIFIED
cat "$root/storage-before.txt" "$root/storage-after.txt"
