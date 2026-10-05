#!/usr/bin/env bash
# Rebuild the qualification harness pinned by the October 4 production baseline (llm_decode_bench.py
# cc9bb06a: llm-inference-bench 24fec94b (origin/main) plus the local metadata/chat-template-kwargs/coding-peak diff,
# run_bench.sh 5c79b976) into harness/, so later changes to ~/git/llm-inference-bench cannot drift it.
set -euo pipefail
kit=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
src=/home/jugs/git/llm-inference-bench
out=$kit/harness
mkdir -p "$out"
git -C "$src" show 24fec94b380decb249cab49e5034d04608197f14:llm_decode_bench.py > "$out/llm_decode_bench.py.orig"
patch -s -o "$out/llm_decode_bench.py" "$out/llm_decode_bench.py.orig" "$kit/harness-cc9bb06a.patch"
rm "$out/llm_decode_bench.py.orig"
install -m 755 "$kit/harness-run_bench.5c79b976" "$out/run_bench.sh"
ln -sfn "$src/.venv" "$out/.venv"
(cd "$out" && sha256sum -c - <<'SUMS'
cc9bb06a4f1142d2bf08afbcfc0e57d8a465a0bd180eee74e276dea70a82071c  llm_decode_bench.py
5c79b9760a2381b4b5233f5bbc8f1f279841b46f596dc718a127eaa8eea3e4f2  run_bench.sh
SUMS
) >&2
printf '%s\n' "$out"
