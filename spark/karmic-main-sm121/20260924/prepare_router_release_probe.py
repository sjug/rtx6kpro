"""Generate a one-site router stage-release diagnostic from the frozen source.

This is an isolated test arm, not a qualified repair or a serving mutation.
"""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
REVISION = 'a7d7d29b2ef8869086e0ceaa787321f17544e3c9'
PATH = 'b12x/gemm/bf16_gemv/_prefill.py'


def transform(source):
    needle = '                load_pipeline.consumer_release(consumer_state)\n'
    if source.count(needle) != 1:
        raise RuntimeError('Expected exactly one router consumer-release site')
    return source.replace(needle, '                cute.arch.fence_acq_rel_cta()\n' + needle)


def main():
    source = subprocess.check_output(
        ['git', '-C', '/home/jugs/git/b12x', 'show', f'{REVISION}:{PATH}'], text=True)
    updated = transform(source)
    patch = ''.join(difflib.unified_diff(source.splitlines(keepends=True),
                                       updated.splitlines(keepends=True),
                                       fromfile='a/' + PATH, tofile='b/' + PATH))
    outputs = {'router-prefill-release-before.py': updated,
               'router-prefill-release-before.patch': patch}
    for name, data in outputs.items():
        target = ROOT / name
        if target.exists() and target.read_text() != data:
            raise RuntimeError('Refusing to replace a different diagnostic: ' + name)
    identity = {'base_revision': REVISION, 'path': PATH,
                'base_sha256': hashlib.sha256(source.encode()).hexdigest(),
                'source_sha256': hashlib.sha256(updated.encode()).hexdigest(),
                'patch_sha256': hashlib.sha256(patch.encode()).hexdigest(),
                'release_sites': 1,
                'scope': 'isolated diagnostic only; requires causal replay and full-model qualification'}
    for name, data in outputs.items():
        (ROOT / name).write_text(data)
    (ROOT / 'router-prefill-release-before.json').write_text(json.dumps(identity, indent=2) + '\n')
    print(json.dumps(identity), flush=True)


if __name__ == '__main__':
    main()
