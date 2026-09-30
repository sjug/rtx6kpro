"""Read compilation provenance from the four diagnostic hosts, disassemble locally."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

from claude_router_artifact_provenance import compare, sass, sha256_file

ROOT = Path(__file__).resolve().parent


def main():
    out = ROOT / 'receipts/router-b1-actual-artifacts-20260925'
    out.mkdir(exist_ok=False)
    source = (ROOT / 'claude_router_artifact_provenance.py').read_text()
    (out / 'helper-sha256').write_text(hashlib.sha256(source.encode()).hexdigest() + '\n')
    code = ('import sys; ns={"__name__":"router_collector", "__file__":"router_collector.py"}; '
            'exec(compile(sys.stdin.read(), "router_collector.py", "exec"), ns); '
            'print(ns["json"].dumps(ns["collect"](sys.argv[1])))')
    results = {}
    for node in ('dusty', 'toby', 'rusty', 'kirby'):
        reports = {}
        for arm, fingerprint in (
            ('parent', 'ds41-engram-b3e4f0ada602fab6f3ce'),
            ('candidate', 'ds41-router-d8af8395f4328ee6df88'),
        ):
            cache = '/home/jugs/.cache/vllm-jj-ds41-tp4/jit/' + fingerprint + '/b12x'
            raw = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', node,
                shlex.join(['python3', '-c', code, cache])], input=source, text=True, timeout=90)
            report = json.loads(raw)
            (out / f'{node}-{arm}.json').write_text(json.dumps(report, indent=2) + '\n')
            reports[arm] = report
        comparison = compare(reports['parent'], reports['candidate'])
        (out / f'{node}-comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
        if comparison['verdict'] != 'pass':
            raise RuntimeError(f'{node}: provenance comparison failed: {comparison["problems"]}')
        result = {}
        for arm, report in reports.items():
            entry = report['router_artifacts'][0]
            obj = out / f'{node}-{arm}.o'
            subprocess.run(['scp', node + ':' + entry['object'], str(obj)], check=True, timeout=90)
            if sha256_file(obj) != entry['object_sha256']:
                raise RuntimeError('Object changed during copy')
            result[arm] = sass(obj, out / f'{node}-{arm}-sass', '/opt/cuda/bin/cuobjdump',
                               'baseline' if arm == 'parent' else 'fenced')
            if not result[arm]['pass']:
                raise RuntimeError('Actual router SASS did not pass: ' + node + '/' + arm)
        results[node] = result
        print('ROUTER-ACTUAL-ARTIFACT-PASS', node, flush=True)
    (out / 'result.json').write_text(json.dumps(results, indent=2) + '\n')
    print('ROUTER-ON-DISK-PROVENANCE-PASS', out, flush=True)


if __name__ == '__main__':
    main()
