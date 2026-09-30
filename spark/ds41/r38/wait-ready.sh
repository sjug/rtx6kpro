#!/usr/bin/env bash
# Wait for the head container to report startup, then require a real completion.
# Provenance: added 2026-09-16 for the approved diagnostic restarts. /v1/models is
# never used as liveness. Prints READY or FAILED with the observed reason.
set -euo pipefail
name=ds41-flash-jj-r38-tp4
deadline=$(( $(date +%s) + ${1:-1200} ))
while :; do
  status=$(ssh -o BatchMode=yes -o ConnectTimeout=10 dusty "podman inspect $name --format '{{.State.Status}}'" 2>/dev/null || echo missing)
  if [[ $status != running ]]; then echo "FAILED head container status: $status"; exit 1; fi
  if ssh -o BatchMode=yes dusty "podman logs $name 2>&1 | grep -q 'Application startup complete'"; then break; fi
  if ssh -o BatchMode=yes dusty "podman logs $name 2>&1" | grep -qE 'EngineDeadError|Traceback \(most recent'; then
    echo "FAILED engine error in head log"; exit 1
  fi
  (( $(date +%s) < deadline )) || { echo "FAILED startup timeout"; exit 1; }
  sleep 15
done
body='{"model":"DeepSeek-V4.1-Flash","messages":[{"role":"user","content":"Reply with only the integer 7."}],"temperature":0,"max_tokens":8,"chat_template_kwargs":{"thinking":false}}'
out=$(curl -s --max-time 900 -H 'Content-Type: application/json' -d "$body" http://dusty:8000/v1/chat/completions)
echo "$out" | cut -c1-400
python3 - "$out" <<'EOF'
import json, sys
r = json.loads(sys.argv[1])
c = r["choices"][0]
assert c["finish_reason"] == "stop" and c["message"]["content"].strip() == "7", c
print("READY")
EOF
