#!/usr/bin/env bash
# No GPU required. Assert lock, host file, label, runtime ENV and installed file.
set -euo pipefail
image=${1:?image ID required}
launcher=${2:?launcher path required}
expected=${3:?lock digest required}
[[ -f $launcher && -r $launcher && -x $launcher ]] || {
  echo 'Mounted GLM launcher must be a readable executable file' >&2
  exit 78
}
[[ $(sha256sum "$launcher" | cut -d' ' -f1) == "$expected" ]]
[[ $(podman image inspect "$image" --format '{{index .Config.Labels "local-inference.launcher.glm.sha256"}}') == "$expected" ]]
podman run --pull=never --rm --entrypoint bash "$image" -c '
  set -euo pipefail
  [[ $GLM_LAUNCHER_SHA256 == "$1" ]]
  [[ $(sha256sum /usr/local/bin/serve-glm53-flash-jj-r38-spark.sh | cut -d" " -f1) == "$1" ]]
' bash "$expected"
