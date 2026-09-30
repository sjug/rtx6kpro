#!/usr/bin/env python3
"""Adapt the previously executed R38 battery; do not change the serving tests."""
from pathlib import Path

root = Path(__file__).resolve().parent
old = root.parent / 'ds4-vision/r38'
dest = root / 'ds4-vision'
image = 'f30dc6d9a2a6f6fc0ac9ff8cddb04d9f631f4a87b48ca7cac69254802fe83233'
for name in ('parse-cli.py', 'test-render.py'):
    (dest / name).write_text((old / name).read_text().replace('jj-r38', 'karmic'))
for name in ('execute.sh', 'benchmark.sh'):
    text = (old / name).read_text()
    if name == 'execute.sh':
        start = text.index('deadline=$(( $(date +%s) + 1200 ))')
        end = text.index('for node in rusty toby; do', start)
        text = text[:start] + text[end:]
    text = text.replace('ea031e1d3d051033f077fc986bf6f8fce04cf9ab52483d5a719ba13114567fc5', image)
    text = text.replace('jj-r38', 'karmic').replace('DS4_R38_APPROVED', 'DS4_KARMIC_APPROVED')
    text = text.replace('DS4-R38-', 'DS4-KARMIC-')
    text = text.replace('remote=/home/jugs/git/ds4-vision-r38',
                        'remote=/home/jugs/git/ds4-vision-r38/karmic-beta-sm121')
    text = text.replace('out=$base/receipts/qualification', 'out=$base/../qualification/ds4-vision')
    text = text.replace('"$base/../qualify.py"', '"$base/../../ds4-vision/qualify.py"')
    text = text.replace('2026-09-karmic-vs-r32', '2026-09-karmic-sm121-qualification')
    text = text.replace('/home/jugs/git/llm-inference-bench/results/runs/', '/home/jugs/git/rtx6kpro/runs/')
    text = text.replace('set -euo pipefail', 'set -euo pipefail\nexport RESULTS_REPO=/home/jugs/git/rtx6kpro PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1\nstarted=$(date -Is)', 1)
    text = text.replace('  /home/jugs/git/llm-inference-bench/run_bench.sh --duration',
                        "  /home/jugs/git/llm-inference-bench/run_bench.sh --calibration-cache \"$out/token-calibration.json\" --display-mode plain --duration")
    # Explicitly decline an upgrade; never edit the benchmark checkout.
    text = text.replace('HOST=http://rusty:8000 MODEL=', "printf 'n\\n' | env HOST=http://rusty:8000 MODEL=")
    text = text.replace('  local status=$?\n', '''  local status=$?
  for node in rusty toby; do
    ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$node" "journalctl -k --since '$started' --no-pager" > "$out/$node-kernel.log" 2>&1 || true
  done
''')
    marker='done\ndeadline=$(( $(date +%s) + 1200 )); ready=0'
    if name == 'execute.sh':
        text = text.replace(marker, '''  ssh -n -o BatchMode=yes "$node" 'while :; do date -Is; grep -E "MemAvailable|MemFree|SwapFree|Cached:" /proc/meminfo; cat /proc/buddyinfo; grep -E "^(allocstall|compact_stall|pgscan_direct|pswpout|pswpin)" /proc/vmstat; sleep 5; done' > "$out/$node-memory.log" 2>&1 & telemetry+=("$!")
done
deadline=$(( $(date +%s) + 1200 )); ready=0''')
    (dest / name).write_text(text)
