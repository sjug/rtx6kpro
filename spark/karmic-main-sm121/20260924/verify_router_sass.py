"""Retain compiled fence evidence, not a claim of full-model correctness."""
import hashlib
import json
import re
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent


def main():
    evidence = {}
    for arm, directory in (
        ('baseline', 'router-isolation-baseline-20260925T212450Z'),
        ('fenced', 'router-isolation-fenced-20260925T212904Z'),
    ):
        folder = ROOT / 'receipts' / directory
        cubin = folder / 'router.cubin'
        text = subprocess.check_output(['/opt/cuda/bin/cuobjdump', '--dump-sass', str(cubin)], text=True)
        (folder / 'router.sass').write_text(text)
        lines = [line for line in text.splitlines() if re.match(r'\s*/\*[0-9a-f]+\*/\s+', line)]
        barriers = [i for i, line in enumerate(lines) if 'MEMBAR.ALL.CTA' in line]
        if len(barriers) != (1 if arm == 'fenced' else 0):
            raise RuntimeError('Unexpected compiled CTA fence count: ' + arm)
        if barriers:
            index = barriers[0]
            if 'LDSM.' not in lines[index - 1] or not any('SYNCS.ARRIVE' in line for line in lines[index + 1:index + 4]):
                raise RuntimeError('Fence not between last observed load and release instruction')
        evidence[arm] = {
            'cubin_sha256': hashlib.sha256(cubin.read_bytes()).hexdigest(),
            'sass_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'cta_fences': len(barriers),
            'fence_context': [lines[max(0, i - 2):i + 4] for i in barriers],
        }
    evidence['scope'] = 'Instruction-order evidence only; full-model causal test remains required'
    output = ROOT / 'receipts/router-stage-release-sass.json'
    output.write_text(json.dumps(evidence, indent=2) + '\n')
    print('ROUTER-SASS-PASS', output, flush=True)


if __name__ == '__main__':
    main()
