"""Static gate for the measured unroll-four dense kernel's release ordering.

This checks fence placement, not hardware execution or a full control-flow
proof. The captured-input replay is the independent runtime correctness gate.
"""
import argparse
from pathlib import Path
import re


def verify(text, expected=5):
    instructions = []
    for line in text.splitlines():
        match = re.match(r'\s*/\*[0-9a-f]+\*/\s+(.*?)\s*;', line)
        if match:
            instructions.append(match.group(1))
    arrivals = [i for i, op in enumerate(instructions) if 'SYNCS.ARRIVE.TRANS64.A1T0 ' in op]
    if len(arrivals) != expected:
        raise RuntimeError(f'Expected {expected} consumer arrivals, found {len(arrivals)}')
    previous = -1
    for index in arrivals:
        window = instructions[previous + 1:index]
        fences = [i for i, op in enumerate(window) if op == 'MEMBAR.ALL.CTA']
        if not fences:
            raise RuntimeError(f'No unconditional CTA fence before arrival {index}')
        if any(re.search(r'\bLDS(?:M)?\.', op) for op in window[fences[-1] + 1:]):
            raise RuntimeError(f'Shared load after fence before arrival {index}')
        previous = index
    return len(arrivals)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sass', type=Path)
    args = parser.parse_args()
    print('DENSE-RELEASE-SASS-PASS', verify(args.sass.read_text()), flush=True)
