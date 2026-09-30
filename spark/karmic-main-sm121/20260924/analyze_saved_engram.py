"""Resume CPU-only analysis of completed replay captures; never send inference."""
import concurrent.futures
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'receipts/replay-engram-trace-requests'
REMOTE = '/home/jugs/git/ds41-r38/karmic-main-20260924'
NAME = 'ds41-flash-karmic-main-tp4'


def analyze(node):
    subprocess.run(['scp', str(ROOT / 'analyze_attention_trace.py'), f'{node}:{REMOTE}/'], check=True)
    for length, count in ((129, 4), (160, 2), (192, 3)):
        target = OUT / f'{node}-{length}-engram-analysis.json'
        if target.exists():
            raise RuntimeError(f'Refusing to overwrite {target}')
        paths = ' '.join(f'/cache/ds41-attention-trace/replay-engram-{length}-{r}-{node}.pt' for r in range(count))
        command = (f'podman exec {NAME} /opt/venv/bin/python '
                   f'/opt/ds41-adapter/analyze_attention_trace.py --engram-only {paths}')
        raw = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', node, command], text=True, timeout=900)
        data = json.loads(raw)
        target.write_text(raw)
        print('ANALYZED', node, length, len(data), flush=True)


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(analyze, ('dusty', 'toby', 'rusty', 'kirby')))
print('SAVED-ENGRAM-ANALYSIS-COMPLETE', flush=True)
