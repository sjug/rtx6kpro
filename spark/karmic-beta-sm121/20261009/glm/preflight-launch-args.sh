#!/usr/bin/env bash
# In-image parse preflight for a GLM runner: render the launcher's vllm command with the runner's
# environment (DRY_RUN inside the image, no GPU work) and check every flag against `vllm serve --help=all`.
# Usage: preflight-launch-args.sh <runner> [node]   (node defaults to sparky; runs the image there)
set -euo pipefail
kit=$(cd "$(dirname "$0")" && pwd)
runner=${1:?runner script}; node=${2:-sparky}
tag=localhost/voipmonitor/vllm:karmic-beta-20261009-spark-sm121
# Runner dry-run on the head rank gives the exact podman command; turn it into an in-image launcher dry-run.
podman_cmd=$(DRY_RUN=1 ROLE=head NODE_RANK=0 HOST_IP=10.11.11.1 EXPECTED_IMAGE_ID=0 MODEL_REVISION=${MODEL_REVISION:-0000000000000000000000000000000000000000} bash "$kit/$runner" | sed -n 's/^DRY-RUN://p')
[[ -n $podman_cmd ]] || { echo 'runner dry-run produced no command'; exit 1; }
envs=$(printf '%s\n' "$podman_cmd" | tr ' ' '\n' | sed -n 's/^-e$/-e/p' >/dev/null; eval "set -- $podman_cmd"; while (($#)); do case $1 in -e) printf -- '-e %q ' "$2"; shift 2;; *) shift;; esac; done)
launcher_args=$(eval "set -- $podman_cmd"; while (($#)); do [[ $1 == "$tag" ]] && { shift; printf '%q ' "$@"; break; }; shift; done)
remote_cmd="podman run --rm --security-opt label=disable -e DRY_RUN=1 $envs --entrypoint /usr/local/bin/serve-glm53-flash-karmic-spark.sh $tag $launcher_args"
rendered=$(ssh -n -o BatchMode=yes "$node" "$remote_cmd" 2>&1 | sed -n 's/^GLM-5.3-Flash Karmic beta Spark launch: //p')
[[ -n $rendered ]] || { echo 'launcher dry-run produced no command'; ssh -n -o BatchMode=yes "$node" "$remote_cmd" 2>&1 | tail -5; exit 1; }
help=$(ssh -n -o BatchMode=yes "$node" "timeout 300 podman run --rm --device nvidia.com/gpu=all --security-opt label=disable --entrypoint /opt/venv/bin/vllm $tag serve --help=all 2>&1")
missing=0
for flag in $(printf '%s\n' "$rendered" | tr ' ' '\n' | grep -E '^--[a-z]' | sed 's/=.*//' | sort -u); do
  grep -q -E -- "(^|[ ,])${flag}([ ,=]|$)" <<<"$help" || { echo "UNKNOWN FLAG: $flag"; missing=1; }
done
for flag in --fairness-engine; do grep -q -- " $flag " <<<" $rendered " && { echo "STALE FLAG PRESENT: $flag"; missing=1; }; done
printf '%s\n' "$rendered" | tr ' ' '\n' | grep -c -- '^--' | sed 's/^/flags checked: /'
(( missing == 0 )) && echo LAUNCH-ARGS-PREFLIGHT-PASS || { echo LAUNCH-ARGS-PREFLIGHT-FAIL; exit 1; }
