"""Read-only lexical inventory, not a claim of runtime reachability or dtype."""
import hashlib
import json
from pathlib import Path
import re


ROOTS = (
    '/opt/jovian-judgement/vllm/vllm/models/deepseek_v4_1',
    '/opt/jovian-judgement/vllm/vllm/models/deepseek_v4',
    '/opt/jovian-judgement/vllm/vllm/v1/worker',
    '/opt/jovian-judgement/b12x/b12x',
)
PATTERN = re.compile(r'(?:torch\.(?:mm|bmm|matmul|addmm)|(?:F|functional)\.linear|'
                     r'allow_bf16_reduced_precision_reduction|\.matmul\(| @ )')


def main():
    files = {}
    missing = []
    for root in ROOTS:
        base = Path(root)
        if not base.is_dir():
            missing.append(root)
            continue
        for path in sorted(base.rglob('*.py')):
            raw = path.read_bytes()
            lines = raw.decode().splitlines()
            hits = [{'line': i+1, 'context': lines[max(0, i-3):i+4]}
                    for i, line in enumerate(lines) if PATTERN.search(line)]
            if hits:
                files[str(path)] = {'sha256': hashlib.sha256(raw).hexdigest(), 'hits': hits}
    print(json.dumps({'scope': __doc__, 'roots': ROOTS, 'missing_roots': missing,
                      'files': files}, indent=2))


if __name__ == '__main__':
    main()
