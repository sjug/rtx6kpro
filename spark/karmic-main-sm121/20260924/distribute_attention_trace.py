"""Stage the canonical trace build receipt and reuse ordinary archive transport."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
REMOTE = '/home/jugs/git/ds41-r38/karmic-main-20260924'
receipt = Path(json.loads((ROOT / 'receipts/attention-trace-build-receipt.json').read_text())['directory'])
subprocess.run(['scp', str(ROOT / 'distribute_determinism.py'), f'dusty:{REMOTE}/'], check=True)
subprocess.run(['scp', str(receipt / 'BUILD-OK'), f'dusty:{REMOTE}/attention-trace-build-ok'], check=True)
subprocess.run(['ssh', '-o', 'BatchMode=yes', 'dusty',
                f'cd {REMOTE} && python3 distribute_determinism.py '
                '--receipt attention-trace-build-ok --prefix attention-trace '
                '--tag localhost/voipmonitor/build-components:ds41-attention-trace'], check=True)
for node in ('toby', 'rusty', 'kirby'):
    subprocess.run(['scp', f'dusty:{REMOTE}/receipts/attention-trace-{node}-image.json',
                    str(receipt / f'{node}-inspect.json')], check=True)
subprocess.run(['scp', f'dusty:{REMOTE}/receipts/attention-trace-archive.json', str(receipt)], check=True)
print('TRACE-DISTRIBUTION-COMPLETE', flush=True)
