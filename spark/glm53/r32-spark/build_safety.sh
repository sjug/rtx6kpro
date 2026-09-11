#!/usr/bin/env bash
# Sourced by the build and mocked by the local regression suite.
require_idle_pair() {
  local containers
  if ! containers=$(podman ps -q); then
    echo 'Cannot establish dusty idle state; refusing build.' >&2; return 78
  fi
  [[ -z "$containers" ]] || { echo 'dusty is not idle.' >&2; return 78; }
  if ! containers=$(ssh -o BatchMode=yes -o ConnectTimeout=10 kirby 'podman ps -q'); then
    echo 'Cannot establish kirby idle state; refusing build.' >&2; return 78
  fi
  [[ -z "$containers" ]] || { echo 'kirby is not idle.' >&2; return 78; }
}

require_bool() {
  case "$2" in 0|1) ;; *) echo "$1 must be literal 0 or 1" >&2; return 2 ;; esac
}

publish_after_gates() {
  local candidate=$1 final=$2; shift 2
  "$@" || return "$?"
  podman tag "$candidate" "$final"
}
