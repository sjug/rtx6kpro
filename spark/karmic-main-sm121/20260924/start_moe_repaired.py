"""Start the gated MoE declaration repair and run its matched short control."""
import datetime
import argparse
import json
from pathlib import Path
import subprocess
import sys

from runtime import kit_digest

ROOT = Path(__file__).resolve().parent
REMOTE = '/home/jugs/git/ds41-r38/karmic-main-20260924'
NODES = ('toby', 'rusty', 'kirby', 'dusty')
NAME = 'ds41-flash-karmic-main-tp4'
parser = argparse.ArgumentParser()
parser.add_argument('--arm', choices=('moe-control-repaired', 'combined-determinism', 'combined-no-turbo', 'combined-prefetch-off', 'dense-release', 'stale-probe', 'ring-fix', 'engram-timing', 'engram-repair', 'activation-trace', 'moe-seams', 'moe-capture', 'router-fence', 'decision-row', 'window', 'indexer', 'precision', 'precision-release', 'ratio1-diag', 'selection-replay', 'selection-transplant', 'mhc-expanded-capture', 'engram-fault'), default='moe-control-repaired')
parser.add_argument('--lengths', default='256,513,514,1024,16384')
parser.add_argument('--cuda-module-loading', choices=('LAZY', 'EAGER'))
parser.add_argument('--ops-trace', action='store_true', help='After the short repro, collect the reviewed diagnostic operator traces')
parser.add_argument('--trace-prefix', default='ops')
parser.add_argument('--trace-pairs')
parser.add_argument('--dense-replay', action='store_true')
parser.add_argument('--decision-row-blocks', choices=('80927', '81389', '80022'),
                    help='Approved diagnostic-only KV geometry, not a serving default')
parser.add_argument('--prefill-threshold', choices=('8192', '7936', '4096'),
                    help='Approved router-image diagnostic, paired with 81389 KV blocks')
parser.add_argument('--nccl-geometry', choices=('tree-simple-1ch',),
                    help='User-approved NCCL geometry diagnostic (PROPOSED-NCCL-GEOMETRY-20260926.md): '
                         'matched window-capture arms only; changes every TP all-reduce order, never serving')
parser.add_argument('--nccl-arm', choices=('standard-upstream',),
                    help='Precision image only: ordinary launch NCCL settings on the matched profile, '
                         'recorded as an explicit arm; never combined with --nccl-geometry')
parser.add_argument('--capture-selections', choices=('passing', 'mhc-expanded8192'),
                    help='mhc-expanded-capture arm only (required there): the passing replay root, or the '
                         'expanded-only transplant root')
parser.add_argument('--transplant-set', choices=('275', 'mhc8192', 'mhc-pre8192', 'mhc-expanded8192'), default='275',
                    help='selection-transplant arm only: the frozen 275-record set, only the two mHC 8192 records, '
                         'or exactly one of them (mhc.pre or mhc.pre.expanded)')
args = parser.parse_args()
matched_capture = args.arm in ('decision-row', 'window', 'indexer', 'precision') and args.decision_row_blocks == '81389'
if args.arm in ('window', 'indexer', 'precision') and not matched_capture:
    parser.error('Window capture requires the approved 81389-block matched profile')
if args.nccl_geometry and (args.arm not in ('window', 'indexer', 'precision') or not matched_capture
                           or args.prefill_threshold not in ('8192', '4096') or args.ops_trace):
    parser.error('NCCL geometry requires the matched window-capture profile with an explicit 8192/4096 threshold')
if args.nccl_arm and (args.arm != 'precision' or not matched_capture or args.nccl_geometry
                      or args.prefill_threshold not in ('8192', '4096') or args.ops_trace):
    parser.error('The standard NCCL arm requires the matched precision profile without --nccl-geometry')
if args.arm == 'precision' and not (args.nccl_geometry or args.nccl_arm):
    parser.error('Precision requires an explicit NCCL arm: --nccl-geometry tree-simple-1ch or --nccl-arm standard-upstream')
if matched_capture and (args.prefill_threshold not in ('8192', '4096') or args.ops_trace):
    parser.error('Matched capture requires threshold8192/4096 without operator tracing')
if args.prefill_threshold and not matched_capture and args.arm not in ('selection-replay', 'selection-transplant', 'mhc-expanded-capture') and (args.arm != 'router-fence' or args.decision_row_blocks
                                                     or args.ops_trace or args.prefill_threshold == '8192'):
    parser.error('Chunking threshold requires clean router-fence or the reviewed matched capture')
if args.decision_row_blocks and args.arm not in ('decision-row', 'window', 'indexer', 'precision', 'precision-release', 'ratio1-diag', 'selection-replay', 'selection-transplant', 'mhc-expanded-capture'):
    parser.error('The 80927 block pin is restricted to the decision-row diagnostic')
if args.decision_row_blocks == '80022' and args.arm not in ('precision-release', 'ratio1-diag'):
    parser.error('The 80022-block pin is restricted to the ratio-1 compute-mode diagnostic sequence')
if (args.arm == 'mhc-expanded-capture') != (args.capture_selections is not None):
    parser.error('--capture-selections is required by, and only valid for, the mhc-expanded-capture arm')
if args.transplant_set != '275' and args.arm != 'selection-transplant':
    parser.error('--transplant-set applies only to the selection-transplant arm')
if args.arm in ('selection-replay', 'selection-transplant', 'mhc-expanded-capture') and (args.decision_row_blocks != '81389' or args.prefill_threshold != '8192'
                                       or args.nccl_geometry or args.nccl_arm or args.ops_trace or args.cuda_module_loading):
    parser.error('Selection replay requires --decision-row-blocks 81389 --prefill-threshold 8192 and nothing else')
if args.arm == 'ratio1-diag' and args.decision_row_blocks != '80022':
    parser.error('The ratio-1 diagnostic requires the 80022-block pin')
if args.arm in ('precision-release', 'ratio1-diag') and (args.decision_row_blocks not in (None, '80022') or args.prefill_threshold or args.nccl_geometry
                                        or args.nccl_arm or args.ops_trace or args.cuda_module_loading):
    parser.error('The precision release and ratio-1 diagnostic run the normal profile (only the 80022 KV pin): no chunking, NCCL or trace controls')
if args.cuda_module_loading and args.arm != 'engram-timing':
    parser.error('Loading-policy control is restricted to the timing diagnostic')
if (args.trace_pairs or args.dense_replay) and not args.ops_trace:
    parser.error('Trace options require --ops-trace')
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
since = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
out = ROOT / 'receipts' / f'{args.arm}-{stamp}'
if args.prefill_threshold:
    out = ROOT / 'receipts' / f'{args.arm}-chunk{args.prefill_threshold}-blocks81389-{stamp}'
if args.nccl_geometry:
    out = ROOT / 'receipts' / f'{args.arm}-geometry-{args.nccl_geometry}-chunk{args.prefill_threshold}-blocks81389-{stamp}'
if args.nccl_arm:
    out = ROOT / 'receipts' / f'{args.arm}-nccl-{args.nccl_arm}-chunk{args.prefill_threshold}-blocks81389-{stamp}'
if args.decision_row_blocks == '80022':
    out = ROOT / 'receipts' / f'{args.arm}-blocks80022-{stamp}'
if args.arm == 'mhc-expanded-capture':
    out = ROOT / 'receipts' / f'{args.arm}-{args.capture_selections}-chunk8192-blocks81389-{stamp}'
if args.arm == 'selection-transplant' and args.transplant_set != '275':
    out = ROOT / 'receipts' / f'{args.arm}-{args.transplant_set}-chunk8192-blocks81389-{stamp}'
out.mkdir(exist_ok=False)
print('RECEIPTS', out, flush=True)
(out / 'launch-options.json').write_text(json.dumps(vars(args), indent=2) + '\n')
digest = kit_digest()
if args.arm in ('dense-release', 'stale-probe', 'ring-fix', 'engram-timing', 'engram-repair', 'activation-trace', 'moe-seams', 'moe-capture', 'router-fence', 'decision-row', 'window', 'indexer', 'precision', 'precision-release', 'ratio1-diag', 'selection-replay', 'selection-transplant', 'mhc-expanded-capture', 'engram-fault'):
    pin = json.loads((ROOT / 'candidate.json').read_text())
    kind = {'dense-release': 'dense-release-fence', 'stale-probe': 'stale-state-probe',
            'ring-fix': 'compressor-ring-mapping-fix',
            'engram-timing': 'engram-host-timing',
            'engram-repair': 'engram-prequeued-native-io-fix',
            'activation-trace': 'activation-digest',
            'moe-seams': 'activation-digest-moe-seams',
            'moe-capture': 'activation-digest-moe-capture',
            'router-fence': 'router-stage-release-candidate',
            'decision-row': 'decision-row-capture',
            'window': 'window-capture',
            'indexer': 'indexer-capture',
            'precision': 'precision-capture',
            'precision-release': 'precision-release-candidate',
            'ratio1-diag': 'ratio1-bf16-diagnostic',
            'selection-replay': 'precision-release-candidate',
            'selection-transplant': 'precision-release-candidate',
            'mhc-expanded-capture': 'mhc-expanded-capture',
            'engram-fault': 'engram-fault-inject'}[args.arm]
    if pin.get('diagnostic', {}).get('kind') != kind or 'diagnostic_overlay' in pin or args.ops_trace:
        raise RuntimeError('Wrong image or overlay for ' + args.arm)
files = list(json.loads((ROOT / 'runtime-files.json').read_text())) + ['runtime-files.json', 'observe.sh']
for node in NODES:
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', node, 'podman ps -q'],
                            text=True, capture_output=True, check=True, timeout=30)
    if result.stdout.strip():
        raise RuntimeError(f'{node} is busy')
for node in NODES:
    subprocess.run(['scp', *[str(ROOT / p) for p in files], f'{node}:{REMOTE}/'], check=True)

def launch_command(node):
    overrides = 'B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1'
    if getattr(args, 'prefill_threshold', None):
        overrides += ' DS41_PREFILL_THRESHOLD=' + args.prefill_threshold
        if args.arm == 'router-fence':
            overrides += ' DS41_CHUNKING_BLOCKS=81389'
    if args.arm == 'selection-replay':
        overrides += ' DS41_SELECTION_REPLAY=standard8192-233036Z'
    if args.arm == 'selection-transplant':
        overrides += ' DS41_SELECTION_REPLAY=' + {'275': 'transplant275-release-into-233036Z',
                                                  'mhc8192': 'transplant2-mhc8192-release-into-233036Z',
                                                  'mhc-pre8192': 'transplant1-mhc-pre8192-release-into-233036Z',
                                                  'mhc-expanded8192': 'transplant1-mhc-expanded8192-release-into-233036Z',
                                                  }[args.transplant_set]
    if args.arm == 'mhc-expanded-capture':
        overrides += ' DS41_SELECTION_REPLAY=' + {'passing': 'standard8192-233036Z',
                                                  'mhc-expanded8192': 'transplant1-mhc-expanded8192-release-into-233036Z',
                                                  }[args.capture_selections]
    if args.decision_row_blocks:
        overrides += ' DS41_DECISION_ROW_BLOCKS=' + args.decision_row_blocks
    if getattr(args, 'nccl_geometry', None):
        overrides += ' DS41_NCCL_GEOMETRY=' + args.nccl_geometry
    if getattr(args, 'nccl_arm', None):
        overrides += ' DS41_NCCL_ARM=' + args.nccl_arm
    if args.cuda_module_loading:
        overrides += ' CUDA_MODULE_LOADING=' + args.cuda_module_loading
    if args.arm in ('combined-no-turbo', 'dense-release', 'stale-probe', 'ring-fix', 'engram-timing', 'engram-repair', 'activation-trace', 'moe-seams', 'moe-capture', 'router-fence', 'decision-row', 'window', 'indexer', 'precision', 'precision-release', 'ratio1-diag', 'selection-replay', 'selection-transplant', 'mhc-expanded-capture', 'engram-fault'):
        overrides += ' B12X_DENSE_SPLITK_TURBO=0'
    if args.arm == 'combined-prefetch-off':
        overrides += ' VLLM_DS41_L2_PREFETCH=0'
    # Clear remote-shell overrides before applying the explicit repaired profile.
    clean = 'env -u VLLM_DS41_L2_PREFETCH -u B12X_DENSE_SPLITK_TURBO' if args.arm in ('dense-release', 'stale-probe', 'ring-fix', 'engram-timing', 'engram-repair', 'activation-trace', 'moe-seams', 'moe-capture', 'router-fence', 'decision-row', 'window', 'indexer', 'precision', 'precision-release', 'ratio1-diag', 'selection-replay', 'selection-transplant', 'mhc-expanded-capture', 'engram-fault') else 'env'
    clean += ' -u DS41_DECISION_ROW_BLOCKS'
    clean += ' -u DS41_PREFILL_THRESHOLD -u DS41_CHUNKING_BLOCKS -u DS41_NCCL_GEOMETRY -u DS41_NCCL_ARM -u DS41_SELECTION_REPLAY'
    return f'cd {REMOTE} && {clean} {overrides} python3 run_node.py --node {node}'

for node in NODES:
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', node, launch_command(node) + ' --preflight-only'],
                            text=True, capture_output=True, check=True, timeout=150)
    (out / f'{node}-preflight.log').write_text(result.stdout + result.stderr)
for node in NODES:
    command = launch_command(node)
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', node, command],
                            text=True, capture_output=True, check=True, timeout=150)
    (out / f'{node}-launch.log').write_text(result.stdout + result.stderr)
    print('STARTED', node, flush=True)
observers = []
try:
    for node in NODES:
        stream = (out / f'{node}-telemetry.log').open('w')
        process = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', node,
                                    f'echo OBSERVER_PID=$$; exec bash {REMOTE}/observe.sh'],
                                   stdout=stream, stderr=subprocess.STDOUT)
        observers.append((node, process, stream))
    subprocess.run([sys.executable, str(ROOT / 'watch-startup.py'), str(out)], check=True)
    if args.arm in ('precision-release', 'ratio1-diag', 'selection-replay', 'selection-transplant', 'mhc-expanded-capture'):
        # One reviewed marker per worker rank, before any request reaches the release boot; the
        # ratio-1 pin marker must appear on the variant only.
        import precision_release
        from run_ratio1_arm import pin_marker_problems
        for node in NODES:
            marker = subprocess.run(['ssh', '-o', 'BatchMode=yes', node,
                                     f'podman logs {NAME} 2>&1 | grep -F -e DS41-PRECISION-APPLIED -e DS41-RATIO1-EXTEND-PIN || true'],
                                    text=True, capture_output=True, check=True, timeout=60).stdout
            (out / f'{node}-precision-marker.log').write_text(marker)
            if precision_release.marker_problems(marker, node):
                raise RuntimeError(f'{node}: missing, duplicate or incorrect worker precision marker')
            if pin_marker_problems(marker, args.arm == 'ratio1-diag'):
                raise RuntimeError(f'{node}: ratio-1 pin marker missing on the variant or present on a release arm')
        print('PRECISION-RELEASE-MARKERS-PASS', flush=True)
    repro = subprocess.run([sys.executable, str(ROOT / 'probe_repeatability.py'),
                    '--corpus', str(ROOT / 'receipts/determinism-corpus.json'),
                    '--out', str(out / 'repeatability'), '--lengths', args.lengths,
                    '--repeats', '6'], check=False)
    if args.ops_trace:
        report_path = out / 'repeatability/report.json'
        if not report_path.is_file():
            raise RuntimeError('Reproducer failed without a complete report; do not trace')
        report = json.loads(report_path.read_text())
        if {r['length'] for r in report} != set(map(int, args.lengths.split(','))) or repro.returncode not in (0, 1):
            raise RuntimeError('Reproducer incomplete or unexpected failure')
        print('REPRO-FINISHED', repro.returncode, report, flush=True)
        trace_command = [sys.executable, str(ROOT / 'run_attention_traces.py'), '--operators', '--prefix', args.trace_prefix]
        if args.trace_pairs:
            trace_command += ['--pairs', args.trace_pairs]
        if args.dense_replay:
            trace_command += ['--dense-replay']
        subprocess.run(trace_command, check=True)
    else:
        repro.check_returncode()
finally:
    for node, process, stream in observers:
        stream.flush()
        lines = (out / f'{node}-telemetry.log').read_text().splitlines()
        first = lines[0] if lines else ''
        if first.startswith('OBSERVER_PID=') and first.split('=', 1)[1].isdigit():
            pid = first.split('=', 1)[1]
            # Only signal our recorded observer PID if it still executes our script.
            cmd = f'if ps -p {pid} -o args= | grep -Fq "{REMOTE}/observe.sh"; then kill -TERM {pid}; fi'
            subprocess.run(['ssh', '-o', 'BatchMode=yes', node, cmd], timeout=30, check=True)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=30)
        stream.close()
    for node in NODES:
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', node,
                                 f'podman logs {NAME} 2>&1; journalctl -k --since "{since}" --no-pager'],
                                text=True, capture_output=True, timeout=60)
        (out / f'{node}-final.log').write_text(result.stdout + result.stderr)
