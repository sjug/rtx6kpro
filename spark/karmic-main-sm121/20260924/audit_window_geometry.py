"""CPU-only audit of the NCCL geometry window arms on idle dusty; never gates on old numerical baselines.

Inputs are four receipt directories under receipts/: the geometry 8192 and 4096 arms
(summary.json nccl_geometry == tree-simple-1ch) and the corrected non-geometry
baselines of the same grids (nccl_geometry absent). The geometry arm changed the
reduction order of the PyNCCL-dispatched TP all-reduces (prefill-sized calls above the
RoCEnante limit); RoCEnante-eligible small calls are unchanged. Its tensors are a
separate numerical arm, so this audit REQUIRES only the structural checks that hold
for any valid capture and REPORTS every numerical comparison:

  required   per geometry receipt: driver decision-row audit verdict, complete window captures,
             per-rank self check (reference packing 128/128 rows, slot/position checks, same-boot
             sibling consistency) and cross-rank replicated equality (claude-window-compare)
  reported   per grid: geometry versus baseline (compare_window_geometry pair: window per-row
             profile with relaxed kit identity, decision-row fields from compare_window_baseline)
             and across the two geometry grids (compare_window_geometry grids), summarized in
             window-geometry-summary.json of the 4096 receipt; equality is never a pass condition

Conventions follow claude_run_decision_capture.phase_audit and compare_window_receipts:
analysis files are staged into each geometry receipt's remote directory with sha256
verification, every container runs --network=none with read-only receipt mounts and a
memory bound, stdout/stderr land beside the receipt, and idle() is checked first.
Cautions: NCCL builds its tree per boot (cross-boot equality is not predicted), splits chunks
across two complementary trees (partial cross-grid invariance is possible), and only warns on
unsupported settings (the driver preflight, not this audit, decides the geometry was in effect).
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from compare_window_geometry import GEOMETRY, validate_geometry_pair, validate_geometry_grids  # noqa: E402

NODES = ('dusty', 'toby', 'rusty', 'kirby')
REMOTE = '/home/jugs/git/ds41-r38/karmic-main-20260924'
STAGED = ('claude-window-compare.py', 'claude_decision_row_audit.py', 'claude-pinned-b12x-compressed_reference.py',
          'claude-ds41-attention-config.json', 'compare_window_geometry.py', 'compare_window_baseline.py',
          'compare_decision_grids.py')
WINDOW = 128


def ssh(node, command, timeout=300):
    return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', node, command], text=True, timeout=timeout)


def load_receipt(path, *, geometry, root=ROOT):
    """summary.json, window-captures.json and (geometry only) the decision-row audit verdict."""
    path = Path(path).resolve()
    if path.parent != (root / 'receipts').resolve():
        raise ValueError('Receipt outside this kit: ' + str(path))
    summary = json.loads((path / 'summary.json').read_text())
    if summary['remote_dir'] != REMOTE + '/receipts/' + path.name:
        raise ValueError('Unexpected remote path: ' + path.name)
    expected = GEOMETRY if geometry else None
    if summary.get('nccl_geometry') != expected:
        raise ValueError(f'{path.name}: nccl_geometry is {summary.get("nccl_geometry")!r}, expected {expected!r}')
    windows = json.loads((path / 'window-captures.json').read_text())
    if set(windows) != set(NODES) or any(v['problems'] for v in windows.values()):
        raise ValueError('Window capture failures: ' + path.name)
    if set(summary['captures']) != set(NODES) or any(summary['capture_problems'].values()):
        raise ValueError('Decision-row capture failures: ' + path.name)
    if geometry:
        verdict = json.loads((path / 'audit.json').read_text())['verdict']
        if verdict != 'within-conformance-envelope':
            raise ValueError(f'{path.name}: decision-row audit verdict {verdict!r}')
    return {'path': path, 'summary': summary, 'windows': windows}


def window_file(receipt, node):
    return Path(receipt['windows'][node]['container_file']).name


def container(image, mounts, argv, memory):
    command = ['podman', 'run', '--rm', '--pull=never', '--network=none', '--memory=' + memory]
    for target, source in mounts.items():
        command += ['-v', f'{source}:{target}:ro']
    return command + ['--entrypoint', '/opt/venv/bin/python', image, *argv]


def plan(geometry8192, geometry4096, baseline8192, baseline4096, memory='24g'):
    """Every container command, keyed by (receipt name, output stem). All mounts read-only."""
    validate_geometry_pair(geometry8192['summary'], baseline8192['summary'])
    validate_geometry_pair(geometry4096['summary'], baseline4096['summary'])
    validate_geometry_grids(geometry8192['summary'], geometry4096['summary'])
    image = geometry8192['summary']['image_id']
    commands = {}
    for receipt, baseline in ((geometry8192, baseline8192), (geometry4096, baseline4096)):
        remote, base_remote = receipt['summary']['remote_dir'], baseline['summary']['remote_dir']
        mounts = {'/gate': remote, '/base': base_remote}
        name = receipt['path'].name
        for node in NODES:
            window = '/gate/captures/' + window_file(receipt, node)
            decision = '/gate/captures/' + receipt['summary']['captures'][node]
            commands[(name, f'window-self-{node}')] = container(image, mounts, [
                '/gate/claude-window-compare.py', 'compare', window, window,
                '--decision-left', decision, '--decision-right', decision], memory)
            commands[(name, f'window-geometry-pair-{node}')] = container(image, mounts, [
                '/gate/compare_window_geometry.py', 'pair', '--geometry-window', window,
                '--baseline-window', '/base/captures/' + window_file(baseline, node),
                '--geometry-decision', decision,
                '--baseline-decision', '/base/captures/' + baseline['summary']['captures'][node]], memory)
        commands[(name, 'window-ranks')] = container(image, mounts, [
            '/gate/claude-window-compare.py', 'ranks',
            *['/gate/captures/' + window_file(receipt, n) for n in NODES]], memory)
    left = geometry8192['summary']['remote_dir']
    name = geometry4096['path'].name
    for node in NODES:
        commands[(name, f'window-geometry-grids-{node}')] = container(image, {
            '/gate': geometry4096['summary']['remote_dir'], '/left': left}, [
            '/gate/compare_window_geometry.py', 'grids',
            '--left', '/left/captures/' + window_file(geometry8192, node),
            '--right', '/gate/captures/' + window_file(geometry4096, node)], memory)
    return image, commands


def validate_within(reports, ranks):
    """Structural checks required of any valid window capture: sibling, packing, slots, replicated ranks."""
    if set(reports) != set(NODES):
        raise ValueError('Incomplete rank self checks')
    if ranks.get('replicated_all_equal') is not True:
        raise ValueError('Replicated window tensors differ across ranks')
    for node in NODES:
        report = reports[node]
        for layer in ('0', '1'):
            if report['decision_consistency_left'][layer]['all_equal'] is not True:
                raise ValueError('Sibling capture differs: ' + node)
            within = report['within_left'][layer]
            if within['packed']['rows_bit_equal'] != WINDOW or within['packed']['rows_not_in_gathered_records']:
                raise ValueError('Packed rows fail the reference encoder: ' + node)
            for field in ('write_slots_equal_last_row_window', 'write_offsets_equal_position_offsets',
                          'write_slots_ascending'):
                if within['slots'][field] is not True:
                    raise ValueError('Window slot check failed: ' + node)


def summarize(pairs8192, pairs4096, grids):
    """Descriptive digest of the numerical comparisons; nothing here is a pass condition."""
    def digest(report):
        window = report['window']
        out = {'first_differing_boundary': window['first_differing_boundary'],
               'layer0_wo_reduced_changed_rows': window['layers']['0']['boundaries']['wo_reduced']['changed_row_count'],
               'layer0_wo_reduced_relative_l2': window['layers']['0']['boundaries']['wo_reduced']['relative_l2_to_left'],
               'kit_equal': window['kit_sha256']['equal']}
        if 'decision_row' in report:
            out['decision_row_first_unequal_layer'] = report['decision_row']['first_unequal_layer']
            out['decision_row_equal_observations'] = report['decision_row']['equal_observations']
        return out
    return {'scope': 'descriptive digest of a numerical arm; equality is reported, never required',
            'prediction': 'layer-0 wo_reduced equal across the two geometry grids within this boot pair; partial '
                          'invariance possible (dual-tree offsets); no cross-boot equality predicted',
            'geometry_vs_baseline': {'8192': {n: digest(pairs8192[n]) for n in NODES},
                                     '4096': {n: digest(pairs4096[n]) for n in NODES}},
            'geometry_8192_vs_4096': {n: digest(grids[n]) for n in NODES},
            'layer0_wo_reduced_equal_across_geometry_grids_all_ranks':
                all(grids[n]['window']['layers']['0']['boundaries']['wo_reduced']['equal_values'] for n in NODES)}


def stage(receipt, files, run=subprocess.run, remote_hash=None):
    """Copy the analysis files into the geometry receipt's remote directory and verify each sha256."""
    remote = receipt['summary']['remote_dir']
    remote_hash = remote_hash or (lambda path: ssh('dusty', 'sha256sum ' + shlex.quote(path)).split()[0])
    for name, digest in files.items():
        run(['scp', str(ROOT / name), f'dusty:{remote}/{name}'], check=True)
        if remote_hash(remote + '/' + name) != digest:
            raise RuntimeError('Staged analysis file differs: ' + name)


def execute(geometry8192, geometry4096, baseline8192, baseline4096, *, memory='24g', idle=None,
            run=subprocess.run, stage_files=stage):
    receipts = {r['path'].name: r for r in (geometry8192, geometry4096)}
    image, commands = plan(geometry8192, geometry4096, baseline8192, baseline4096, memory)
    files = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in STAGED}
    if idle is not None:
        idle()
    for receipt in (geometry8192, geometry4096):
        stage_files(receipt, files)
        with (receipt['path'] / 'window-geometry-invocation.json').open('x') as handle:
            json.dump({'image_id': image, 'files': files,
                       'commands': {stem: cmd for (name, stem), cmd in commands.items() if name == receipt['path'].name}},
                      handle, indent=2)
    for (name, stem), command in commands.items():
        out = receipts[name]['path']
        with (out / f'{stem}.json').open('x') as stdout, (out / f'{stem}.stderr').open('x') as stderr:
            code = run(['ssh', '-o', 'BatchMode=yes', 'dusty', shlex.join(command)], stdout=stdout, stderr=stderr).returncode
        print(name, stem, 'exit', code, flush=True)
        if code:
            raise RuntimeError(f'{name}: {stem} failed; preserve and inspect')
    loaded = lambda r, stem: json.loads((r['path'] / f'{stem}.json').read_text())
    for receipt in (geometry8192, geometry4096):
        validate_within({n: loaded(receipt, f'window-self-{n}') for n in NODES}, loaded(receipt, 'window-ranks'))
    summary = summarize({n: loaded(geometry8192, f'window-geometry-pair-{n}') for n in NODES},
                        {n: loaded(geometry4096, f'window-geometry-pair-{n}') for n in NODES},
                        {n: loaded(geometry4096, f'window-geometry-grids-{n}') for n in NODES})
    with (geometry4096['path'] / 'window-geometry-summary.json').open('x') as handle:
        json.dump(summary, handle, indent=2)
    print('WINDOW-GEOMETRY-AUDIT-COMPLETE', json.dumps(summary['geometry_8192_vs_4096']), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--geometry-8192', type=Path, required=True)
    parser.add_argument('--geometry-4096', type=Path, required=True)
    parser.add_argument('--baseline-8192', type=Path, required=True)
    parser.add_argument('--baseline-4096', type=Path, required=True)
    parser.add_argument('--memory', default='24g', help='podman --memory bound for every CPU container')
    a = parser.parse_args(argv)
    from build_activation_trace import idle
    execute(load_receipt(a.geometry_8192, geometry=True), load_receipt(a.geometry_4096, geometry=True),
            load_receipt(a.baseline_8192, geometry=False), load_receipt(a.baseline_4096, geometry=False),
            memory=a.memory, idle=idle)


if __name__ == '__main__':
    main()
