#!/usr/bin/env bash
# Verify claude-dsa-topk-tiebreak.patch against the pinned B12X source without
# touching the ~/git/b12x checkout: extract the blob with `git show` into a
# scratch directory, apply with `git apply --check`, byte-compile the result and
# compare its digest with the value recorded in CLAUDE-TOPK-TIEBREAK.md.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
pin=a7d7d29b2ef8869086e0ceaa787321f17544e3c9
path=b12x/attention/dsa_indexer/tiled_topk.py
patch=$here/claude-dsa-topk-tiebreak.patch
expected_patch=7d77a7fb6f2abdf6967621c445147fa968782f295fb3c6c4cf605ba536764ec0
expected_result=f2d0fadae632f4b2af233ca2c53122e7fbcb78a92dc5d16479e541330cf9052d
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/$(dirname "$path")"
git -C "${B12X_CHECKOUT:-$HOME/git/b12x}" show "$pin:$path" > "$work/$path"
printf 'PATCH-SHA256 %s\n' "$(sha256sum "$patch" | cut -d' ' -f1)"
(cd "$work" && git apply --check --whitespace=error "$patch" && git apply --whitespace=error "$patch")
python3 -m py_compile "$work/$path"
result=$(sha256sum "$work/$path" | cut -d' ' -f1)
printf 'RESULT-SHA256 %s\n' "$result"
if [[ $(sha256sum "$patch" | cut -d' ' -f1) != "$expected_patch" ]]; then
  echo 'patch digest differs from the reviewed value' >&2; exit 78
fi
if [[ $result != "$expected_result" ]]; then
  echo 'patched source digest differs from the reviewed value' >&2; exit 78
fi
grep -q 'row_topk_v11_stable_index_tiebreak' "$work/$path"
grep -q 'tiled_topk_v2_stable_index_tiebreak' "$work/$path"
grep -q '"attention.indexer.row_topk",$' "$work/$path"
grep -q '^def _exact_overflow_fallback_stable($' "$work/$path"
# The shared `_exact_overflow_fallback` (imported by fused_indexer.py, an
# untested consumer) must stay byte-identical to the pinned blob.
python3 - "$work/$path" <<'PY'
import subprocess, sys
patched = open(sys.argv[1]).read()
pin = subprocess.check_output(['git', '-C', __import__('os').environ.get('B12X_CHECKOUT', __import__('os').path.expanduser('~/git/b12x')),
                               'show', 'a7d7d29b2ef8869086e0ceaa787321f17544e3c9:b12x/attention/dsa_indexer/tiled_topk.py'], text=True)
def fn(src):
    i = src.index('\n@cute.jit\ndef _exact_overflow_fallback(\n')
    return src[i:src.index('\n\n\n', i + 1)]
if fn(pin) != fn(patched):
    sys.exit('shared _exact_overflow_fallback changed; fused_indexer.py consumer would be affected')
print('SHARED-FALLBACK-UNCHANGED')
PY
echo CLAUDE-TOPK-PATCH-CHECK-PASS
