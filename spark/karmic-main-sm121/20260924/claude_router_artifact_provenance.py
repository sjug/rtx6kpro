#!/usr/bin/env python3
"""On-disk provenance of the B12X router projection program (gemm.bf16_prefill, 384x5120 -> fp32).

Read-only with respect to the caches it inspects. Standard library only for
`collect` and `compare`, so `collect` can run on a node host with plain
python3 (no torch import, no podman exec) against the bind-mounted cache,
e.g. ~/.cache/vllm-jj-ds41-tp4/jit/<fingerprint>/b12x. Its only write is --out.

Why not the selection cache: a cached selection record's `programs` field is
written only when that choice is measured (b12x a7d7d29b session.py
cache_pending), so on a cache hit it names the program of the boot that first
measured it. The program a boot actually installs is recomputed at install
time and is a cute compile-cache key:
    <B12X_COMPILE_CACHE_DIR>/<key[:2]>/<key>.o  plus  <key>.json (compile_manifest.v3)
whose key covers package fingerprint, toolchain, device UUID, compile options
and compile environment (compiler.py 1644-1680, 2046-2050).

  collect  scan one cache dir, select router manifests by kernel_id and the
           tensor facts (weight exact 384x5120, output float32), verify the
           object's sha256/size against the manifest, the manifest name
           against cache_key, the fingerprint against cache_payload, and
           summarize fingerprints of all manifests in that dir.
  sass     extract the embedded CUDA ELF with the existing
           extract_dense_cubin.py, dump SASS with cuobjdump, classify every
           SYNCS.ARRIVE (consumer release = .A1T0 single arrive; others are
           producer expect-tx arrives) and, per consumer release, whether a
           MEMBAR.ALL.CTA lies between the nearest preceding LDSM and it.
           --expect fenced|baseline applies verify_router_sass.py's rule plus
           "every consumer release fenced / none fenced".
  compare  a parent and a candidate collect report (same node): exactly one
           router artifact each, different package fingerprints, one
           fingerprint across the candidate dir, same toolchain, options,
           device UUID and non-path compile environment.

Scope: this proves which object a cache key resolves to on disk and what
that object contains. It is not a trace of the module a process loaded.
Exit 0 pass, 1 check failed, 2 usage.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
KERNEL_ID = 'gemm.bf16_prefill'
WEIGHT_DIMS = (384, 5120)
MANIFEST_NAME = re.compile(r'^[0-9a-f]{64}\.json$')
INSTRUCTION = re.compile(r'^\s*/\*([0-9a-f]+)\*/\s+(.*?)\s*;?\s*(/\*.*)?$')


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_facts(node, found=None):
    """All ["tensor", name, dtype, rank, dims, strides, device, align, layout] facts in a spec."""
    found = {} if found is None else found
    if isinstance(node, list):
        if len(node) >= 5 and node[0] == 'tensor' and isinstance(node[1], str):
            found.setdefault(node[1], node)
        for item in node:
            tensor_facts(item, found)
    elif isinstance(node, dict):
        for item in node.values():
            tensor_facts(item, found)
    return found


def dims_of(fact):
    return [tuple(d[1:3]) if isinstance(d, list) and len(d) >= 3 and d[0] == 'dim' else ('?', d) for d in fact[4]]


def router_match(manifest):
    """(is_router, reason). Fails closed when the spec cannot be read."""
    if manifest.get('kernel_id') != KERNEL_ID:
        return False, 'kernel_id'
    try:
        spec = json.loads(manifest.get('compile_spec_json') or 'null')
    except json.JSONDecodeError:
        return False, 'unparsed compile_spec_json'
    facts = tensor_facts(spec)
    weight, output, source = facts.get('weight'), facts.get('output'), facts.get('source')
    if not (weight and output and source):
        return False, 'tensor facts absent'
    if dims_of(weight) != [('exact', WEIGHT_DIMS[0]), ('exact', WEIGHT_DIMS[1])]:
        return False, 'weight dims ' + str(dims_of(weight))
    if 'float32' not in str(output[2]) or 'bfloat16' not in str(weight[2]) or 'bfloat16' not in str(source[2]):
        return False, 'dtypes'
    if dims_of(source)[1:] != [('exact', WEIGHT_DIMS[1])] or dims_of(source)[0][0] != 'dynamic':
        return False, 'source dims ' + str(dims_of(source))
    return True, 'router'


def device_uuid(manifest):
    value = (manifest.get('semantic_payload') or {}).get('device_uuid')
    return value[1] if isinstance(value, list) and len(value) == 2 else value


def collect(cache_dir):
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        raise SystemExit(f'not a directory: {cache_dir}')
    manifests, errors, fingerprints, routers, others = 0, [], {}, [], []
    for path in sorted(cache_dir.rglob('*.json')):
        if not MANIFEST_NAME.match(path.name) or path.parent.name != path.name[:2]:
            continue
        try:
            manifest = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            errors.append({'manifest': str(path), 'error': f'unreadable: {error}'})
            continue
        if manifest.get('schema') != 'b12x._lib.compile_manifest.v3':
            continue
        manifests += 1
        fp = manifest.get('package_fingerprint')
        fingerprints[fp] = fingerprints.get(fp, 0) + 1
        is_router, reason = router_match(manifest)
        if manifest.get('kernel_id') == KERNEL_ID and not is_router:
            others.append({'manifest': str(path), 'reason': reason})
        if not is_router:
            continue
        key = manifest.get('cache_key')
        obj = path.with_suffix('.o')
        entry = {'manifest': str(path), 'manifest_sha256': sha256_file(path), 'cache_key': key,
                 'object': str(obj), 'package_fingerprint': fp, 'device_uuid': device_uuid(manifest),
                 'target': manifest.get('target'), 'toolchain': manifest.get('toolchain'),
                 'compile_options': manifest.get('compile_options'),
                 'compile_environment': manifest.get('compile_environment'),
                 'compile_spec_hash': manifest.get('compile_spec_hash'),
                 'launch_metadata': manifest.get('launch_metadata'),
                 'manifest_mtime': os.stat(path).st_mtime}
        problems = []
        if path.stem != key:
            problems.append('manifest name differs from cache_key')
        payload = manifest.get('cache_payload')
        if not isinstance(payload, list) or len(payload) < 3 or payload[2] != fp:
            problems.append('package_fingerprint differs from cache_payload[2]')
        if not obj.is_file():
            problems.append('object missing')
        else:
            actual = sha256_file(obj)
            entry.update(object_sha256=actual, object_bytes=obj.stat().st_size, object_mtime=os.stat(obj).st_mtime)
            if actual != manifest.get('object_sha256'):
                problems.append('object sha256 differs from manifest')
            if obj.stat().st_size != manifest.get('object_bytes'):
                problems.append('object size differs from manifest')
        entry['problems'] = problems
        routers.append(entry)
    return {'cache_dir': str(cache_dir), 'manifests': manifests, 'package_fingerprints': fingerprints,
            'router_artifacts': routers, 'other_prefill_manifests': others, 'errors': errors,
            'scope': 'on-disk provenance only; not process-level loaded-module tracing'}


def _path_key(name):
    return name.endswith('_DIR') or name in ('XDG_CACHE_HOME', 'VLLM_CACHE_ROOT')


def compare(parent, candidate):
    problems = []
    for label, report in (('parent', parent), ('candidate', candidate)):
        if report['errors']:
            problems.append(f'{label}: unreadable manifests')
        if len(report['router_artifacts']) != 1:
            problems.append(f'{label}: expected exactly one router artifact, found {len(report["router_artifacts"])}')
        for entry in report['router_artifacts']:
            problems.extend(f'{label}: {p}' for p in entry['problems'])
    if len(candidate['package_fingerprints']) != 1:
        problems.append('candidate cache holds more than one package fingerprint')
    result = {'problems': problems}
    if len(parent['router_artifacts']) == 1 and len(candidate['router_artifacts']) == 1:
        a, b = parent['router_artifacts'][0], candidate['router_artifacts'][0]
        env = lambda e: {k: v for k, v in (e['compile_environment'] or []) if not _path_key(k)}
        result['pair'] = {
            'parent': {k: a.get(k) for k in ('cache_key', 'object_sha256', 'package_fingerprint', 'device_uuid')},
            'candidate': {k: b.get(k) for k in ('cache_key', 'object_sha256', 'package_fingerprint', 'device_uuid')},
            'environment_differences': sorted(set(env(a).items()) ^ set(env(b).items()))}
        if a['package_fingerprint'] == b['package_fingerprint']:
            problems.append('parent and candidate share a package fingerprint')
        if a.get('object_sha256') == b.get('object_sha256'):
            problems.append('parent and candidate objects are byte-identical')
        for field in ('toolchain', 'compile_options', 'device_uuid', 'target', 'compile_spec_hash'):
            if a.get(field) != b.get(field):
                problems.append(f'{field} differs between parent and candidate')
        if result['pair']['environment_differences']:
            problems.append('non-path compile environment differs')
    result['verdict'] = 'pass' if not problems else 'fail'
    return result


def parse_sass(text):
    lines = []
    for raw in text.splitlines():
        m = INSTRUCTION.match(raw)
        if m and m.group(2) and not m.group(2).startswith('/*'):
            lines.append((m.group(1), m.group(2)))
    return lines


def analyze_sass(text, window=8):
    """Classify SYNCS.ARRIVE sites. A consumer release is the single-count arrive without a
    transaction count (.A1T0, lane-predicated, on the empty barrier); the TRANS64 arrives with a
    register count are the producer's expect-tx arrives. For consumer releases, report the
    instructions since the nearest preceding LDSM within `window` and whether a MEMBAR.ALL.CTA
    lies between them."""
    lines = parse_sass(text)
    membars = [i for i, (_, op) in enumerate(lines) if 'MEMBAR.ALL.CTA' in op]
    sites = []
    for i, (address, op) in enumerate(lines):
        if 'SYNCS.ARRIVE' not in op:
            continue
        kind = 'consumer-release' if '.A1T0' in op else 'producer-or-other'
        site = {'address': address, 'instruction': op, 'kind': kind}
        if kind == 'consumer-release':
            ldsm = next((j for j in range(i - 1, max(-1, i - 1 - window), -1) if 'LDSM' in lines[j][1]), None)
            site.update(last_ldsm_address=None if ldsm is None else lines[ldsm][0],
                        membar_between_ldsm_and_arrive=ldsm is not None and any(ldsm < m < i for m in membars),
                        context=[f'{a} {o}' for a, o in lines[max(0, i - window):i + 1]])
        sites.append(site)
    adjacent = [m for m in membars if m > 0 and 'LDSM' in lines[m - 1][1]
                and any('SYNCS.ARRIVE' in op for _, op in lines[m + 1:m + 4])]
    releases = [site for site in sites if site['kind'] == 'consumer-release']
    return {'instructions': len(lines), 'cta_fences': len(membars), 'arrive_sites': sites,
            'consumer_release_sites': len(releases),
            'consumer_releases_fenced': sum(site['membar_between_ldsm_and_arrive'] for site in releases),
            'fence_ldsm_then_arrive': len(adjacent)}


def sass_verdict(analysis, expect):
    """Same rule as verify_router_sass.py: fenced = exactly one CTA fence directly after an LDSM
    with a SYNCS.ARRIVE within three instructions; baseline = no CTA fence.
    Additionally every consumer release site must be fenced (fenced) or unfenced (baseline)."""
    releases, fenced = analysis['consumer_release_sites'], analysis['consumer_releases_fenced']
    if releases == 0:
        return False
    if expect == 'baseline':
        return analysis['cta_fences'] == 0 and fenced == 0
    return analysis['cta_fences'] == 1 and analysis['fence_ldsm_then_arrive'] == 1 and fenced == releases


def sass(obj, out_dir, cuobjdump, expect):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    cubin = out_dir / 'router.cubin'
    subprocess.run([sys.executable, str(HERE / 'extract_dense_cubin.py'), str(obj), str(cubin)], check=True,
                   capture_output=True, text=True)
    text = subprocess.check_output([cuobjdump, '--dump-sass', str(cubin)], text=True)
    (out_dir / 'router.sass').write_text(text)
    analysis = analyze_sass(text)
    report = {'object': str(obj), 'object_sha256': sha256_file(obj), 'cubin_sha256': sha256_file(cubin),
              'sass_sha256': hashlib.sha256(text.encode()).hexdigest(), 'cuobjdump': cuobjdump,
              'expect': expect, **analysis,
              'scope': 'instruction order of the on-disk object; not causal and not a loaded-module trace'}
    if expect:
        report['pass'] = sass_verdict(analysis, expect)
    (out_dir / 'sass-report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='mode', required=True)
    c = sub.add_parser('collect')
    c.add_argument('--cache-dir', type=Path, required=True)
    c.add_argument('--out', type=Path, required=True)
    s = sub.add_parser('sass')
    s.add_argument('--object', type=Path, required=True)
    s.add_argument('--out-dir', type=Path, required=True)
    s.add_argument('--cuobjdump', default=shutil.which('cuobjdump') or '/opt/cuda/bin/cuobjdump')
    s.add_argument('--expect', choices=('fenced', 'baseline'))
    p = sub.add_parser('compare')
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--out', type=Path)
    a = parser.parse_args(argv)
    if a.mode == 'collect':
        cache = a.cache_dir.resolve()
        if a.out.resolve().is_relative_to(cache):
            parser.error('--out must not be inside the inspected cache')
        report = collect(cache)
        a.out.write_text(json.dumps(report, indent=2) + '\n')
        ok = len(report['router_artifacts']) == 1 and not report['errors'] and \
            not report['router_artifacts'][0]['problems']
        print(json.dumps({'router_artifacts': [{k: e.get(k) for k in ('cache_key', 'object_sha256', 'package_fingerprint',
                                                                       'problems')} for e in report['router_artifacts']],
                          'package_fingerprints': report['package_fingerprints'], 'ok': ok}, indent=2))
        return 0 if ok else 1
    if a.mode == 'sass':
        report = sass(a.object, a.out_dir, a.cuobjdump, a.expect)
        print(json.dumps({k: report[k] for k in ('object_sha256', 'cubin_sha256', 'cta_fences', 'fence_ldsm_then_arrive',
                                                 'consumer_release_sites', 'consumer_releases_fenced', 'expect')} |
                         {'arrive_sites': len(report['arrive_sites']), 'pass': report.get('pass')}, indent=2))
        return 0 if report.get('pass', True) else 1
    result = compare(json.loads(a.parent.read_text()), json.loads(a.candidate.read_text()))
    if a.out:
        a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return 0 if result['verdict'] == 'pass' else 1


if __name__ == '__main__':
    sys.exit(main())
