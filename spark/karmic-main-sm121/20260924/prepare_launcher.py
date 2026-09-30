#!/usr/bin/env python3
"""Render the pinned upstream shell arrays without executing host operations."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMMIT = '1794dcf18454900263e0c66711af8ea4a1283ac1'
SOURCES = {
    'serve-ds41-flash-dspark-tp4-rdma.sh': '7d18de06239bcea97d1ce1da5917a99904b6483a94e7bb72b03986a4d9286b31',
    'serve-ds4-flash-dspark-tp4-rdma.sh': '4d751080ff43452f3110736b44213d0b9368b0369781ead27bf32ee351409ddb',
}


def main():
    files = {}
    for name, digest in SOURCES.items():
        data = subprocess.check_output(['git', '-C', str(Path.home() / 'git/vllm'),
                                        'show', f'{COMMIT}:scripts/{name}'])
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError(f'Upstream launcher changed: {name}')
        files[name] = data.decode()
    base = files['serve-ds4-flash-dspark-tp4-rdma.sh']
    wrapper = files['serve-ds41-flash-dspark-tp4-rdma.sh']
    # Only assignments and command-array construction are evaluated. All host
    # checks, SSH, sync, mkdir, Docker and external-launcher execution are excluded.
    script = 'set -euo pipefail\nSCRIPT_DIR=/upstream/scripts\n'
    script += wrapper[wrapper.index('export MODEL_ID='):wrapper.index('\nexec ')]
    script += '\n' + base[base.index('VLLM_ROOT='):base.index('\nsync_code=0')]
    script += '\nmount_args=\ncheck_only=0\ndetach=0\nvllm_args=()\n'
    script += base[base.index('cluster_args=('):base.index('\nif ((check_only));')]
    script += '\n' + base[base.index('max_cudagraph_capture_size='):base.index('\nif ((NUM_SPECULATIVE_TOKENS > 0)); then\n  spec_summary=')]
    script += '\npython3 -c \'import json,sys; a=sys.argv[1:]; i=a.index("__MODEL_ARGV__"); print(json.dumps({"cluster":a[:i],"model":a[i+1:]}))\' "${cluster_args[@]}" __MODEL_ARGV__ "${vllm_command[@]}"\n'
    # No inherited user tuning variables may influence the frozen reference.
    env = {'PATH': os.environ['PATH'], 'HOME': '/upstream',
           'PYTHON_BIN': '/usr/bin/python3', 'VLLM_ROOT': '/upstream/vllm'}
    reference = json.loads(subprocess.check_output(['bash'], input=script, text=True, env=env))
    seccomp = subprocess.check_output(['git', '-C', str(Path.home() / 'git/vllm'),
                                      'show', f'{COMMIT}:scripts/seccomp/spark-io-uring.json'])
    seccomp_sha = hashlib.sha256(seccomp).hexdigest()
    if seccomp_sha != '823b66051a52180058aa04b4343eb29a9cc963c1cff306e5c81dd606324252dc':
        raise RuntimeError('Upstream seccomp profile changed')
    (ROOT / 'seccomp-io-uring.json').write_bytes(seccomp)
    model_manifest = (ROOT.parents[1] / 'ds41/r38/model-manifest.json').read_bytes()
    if hashlib.sha256(model_manifest).hexdigest() != '8493bdf24a904c63a941f32c219b479665b5c1739c14c4530c2aeeb7473c88cf':
        raise RuntimeError('Previously verified checkpoint inventory changed')
    (ROOT / 'model-manifest.json').write_bytes(model_manifest)
    row = {'vllm_commit': COMMIT, 'sources': SOURCES, 'reference': reference,
           'seccomp_sha256': seccomp_sha}
    (ROOT / 'upstream-launch.json').write_text(json.dumps(row, indent=2, sort_keys=True) + '\n')
    print('UPSTREAM-LAUNCH-RENDER-PASS')


if __name__ == '__main__':
    main()
