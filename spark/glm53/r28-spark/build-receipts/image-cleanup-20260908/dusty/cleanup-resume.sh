#!/usr/bin/env bash
set -euo pipefail
root=/home/jugs/git/bld-jj-r28-spark/qualification/image-cleanup-20260908
host=$(hostname -s)
[[ $host == dusty || $host == kirby ]]
plan=$root/$host-plan.json
[[ $(jq -r .host "$plan") == "$host" ]]
# Hard stop if any protected candidate or rollback is absent from retention
# or appears in the deletion list, regardless of similar old release names.
for protected in cd93d80b3f95 ef669fa1cde3 bb9cb676e464 276f00868134 69f83ec7efff; do
  jq -e --arg id "$protected" '
    any(.retain[]; .Id | startswith($id)) and
    all(.candidates[]; (.Id | startswith($id)) | not)
  ' "$plan" >/dev/null
done
exec > >(tee "$root/cleanup-resume.log") 2>&1
[[ $host == dusty && -s $root/images-before.json && ! -e $root/images-after.json ]]
running=$(podman ps -q)
[[ -z $running ]] || { echo 'Serving containers present; refusing cleanup'; exit 78; }
if ps -eo comm= | grep -Eq '^(buildah|podman)$'; then
  echo 'Build or Podman process present; refusing cleanup'; exit 78
fi
jq -r '.candidates[] | .Id as $id | .Names[] | [$id,.] | @tsv' "$plan" > "$root/remove.tsv"
while IFS=$'\t' read -r expected ref; do
  if podman image exists "$ref"; then :; else
    status=$?; [[ $status == 1 ]] || exit "$status"; continue
  fi
  actual=$(podman image inspect "$ref" --format '{{.Id}}')
  [[ $actual == "$expected" ]] || { echo "Identity changed: $ref"; exit 78; }
done < "$root/remove.tsv"
while IFS=$'\t' read -r expected ref; do
  if podman image exists "$ref"; then :; else
    status=$?; [[ $status == 1 ]] || exit "$status"; continue
  fi
  printf 'REMOVE %s %s\n' "$expected" "$ref"
  podman image rm --no-prune "$ref"
done < "$root/remove.tsv"
# No --all, --external, or --build-cache. --force only skips the prune prompt.
podman image prune --force
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
echo CLEANUP-OK
cat "$root/storage-before.txt" "$root/storage-after.txt"
